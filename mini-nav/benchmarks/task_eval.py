from typing import Dict, Literal, cast

import lancedb
import pyarrow as pa
import torch
from database import db_manager
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BitImageProcessorFast

# 数据集配置：数据集名称 -> (HuggingFace ID, 图片列名, 标签列名)
DATASET_CONFIG: Dict[str, tuple[str, str, str]] = {
    "CIFAR-10": ("uoft-cs/cifar10", "img", "label"),
    "CIFAR-100": ("uoft-cs/cifar100", "img", "fine_label"),
}


def _get_table_name(dataset: str, model_name: str) -> str:
    """Generate database table name from dataset and model name.

    Args:
        dataset: Dataset name, e.g. "CIFAR-10".
        model_name: Model name, e.g. "Dinov2".

    Returns:
        Formatted table name, e.g. "cifar10_dinov2".
    """
    ds_part = dataset.lower().replace("-", "")
    model_part = model_name.lower()
    return f"{ds_part}_{model_part}"


def _infer_vector_dim(
    processor: BitImageProcessorFast,
    model: nn.Module,
    sample_image,
) -> int:
    """Infer model output vector dimension via a single forward pass.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        sample_image: A sample image for dimension inference.

    Returns:
        Vector dimension.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        inputs = processor(images=sample_image, return_tensors="pt")
        inputs.to(device)
        output = model(inputs)

    # output shape: [1, dim]
    return output.shape[-1]


def _build_eval_schema(vector_dim: int) -> pa.Schema:
    """Build PyArrow schema for evaluation database table.

    Args:
        vector_dim: Feature vector dimension.

    Returns:
        PyArrow schema with id, label, and vector fields.
    """
    return pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("label", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


@torch.no_grad()
def _establish_eval_database(
    processor: BitImageProcessorFast,
    model: nn.Module,
    table: lancedb.table.Table,
    dataloader: DataLoader,
) -> None:
    """Extract features from training images and store them in a database table.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        table: LanceDB table to store features.
        dataloader: DataLoader for the training dataset.
    """
    device = next(model.parameters()).device
    model.eval()

    global_idx = 0
    for batch in tqdm(dataloader, desc="Building eval database"):
        imgs = batch["img"]
        labels = batch["label"]

        # 预处理并推理
        inputs = processor(imgs, return_tensors="pt")
        inputs.to(device)
        outputs = model(inputs)  # [B, dim]

        # 整个batch一次性转到CPU
        features = cast(torch.Tensor, outputs).cpu()
        labels_list = labels.tolist()

        # 逐条写入数据库
        batch_size = len(labels_list)
        table.add(
            [
                {
                    "id": global_idx + j,
                    "label": labels_list[j],
                    "vector": features[j].numpy(),
                }
                for j in range(batch_size)
            ]
        )
        global_idx += batch_size


@torch.no_grad()
def _evaluate_recall(
    processor: BitImageProcessorFast,
    model: nn.Module,
    table: lancedb.table.Table,
    dataloader: DataLoader,
    top_k: int,
) -> tuple[int, int]:
    """Evaluate Recall@K by searching the database for each test image.

    For each batch, features are extracted in one forward pass and moved to CPU,
    then each sample is searched individually against the database.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        table: LanceDB table to search against.
        dataloader: DataLoader for the test dataset.
        top_k: Number of top results to retrieve.

    Returns:
        A tuple of (correct_count, total_count).
    """
    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Evaluating Recall@{top_k}"):
        imgs = batch["img"]
        labels = batch["label"]

        # 批量前向推理
        inputs = processor(imgs, return_tensors="pt")
        inputs.to(device)
        outputs = model(inputs)  # [B, dim]

        # 整个batch一次性转到CPU
        features = cast(torch.Tensor, outputs).cpu()
        labels_list = labels.tolist()

        # 逐条搜索并验证
        for j in range(len(labels_list)):
            feature = features[j].tolist()
            true_label = labels_list[j]

            # 搜索 top_k 最相似结果
            results = (
                table.search(feature)
                .select(["label", "_distance"])
                .limit(top_k)
                .to_polars()
            )

            # 检查 top_k 中是否包含正确标签
            retrieved_labels = results["label"].to_list()
            if true_label in retrieved_labels:
                correct += 1
            total += 1

    return correct, total


def task_eval(
    processor: BitImageProcessorFast,
    model: nn.Module,
    dataset: Literal["CIFAR-10", "CIFAR-100"],
    model_name: str,
    top_k: int = 10,
    batch_size: int = 64,
) -> float:
    """Evaluate model Recall@K accuracy on a dataset using vector retrieval.

    Workflow:
        1. Create or open a database table named by dataset and model.
        2. Build database from training set features (skip if table exists).
        3. Evaluate on test set: extract features in batches, search top_k,
           check if correct label appears in results.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        dataset: Dataset name.
        model_name: Model name, used for table name generation.
        top_k: Number of top similar results to retrieve.
        batch_size: Batch size for DataLoader.

    Returns:
        Recall@K accuracy (0.0 ~ 1.0).

    Raises:
        ValueError: If dataset name is not supported.
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset}. Only support: {list(DATASET_CONFIG.keys())}."
        )
    hf_id, img_col, label_col = DATASET_CONFIG[dataset]

    # 加载数据集
    train_dataset = load_dataset(hf_id, split="train")
    test_dataset = load_dataset(hf_id, split="test")

    # 生成表名，推断向量维度
    table_name = _get_table_name(dataset, model_name)
    vector_dim = _infer_vector_dim(processor, model, train_dataset[0][img_col])
    expected_schema = _build_eval_schema(vector_dim)
    existing_tables = db_manager.db.list_tables().tables

    # 如果旧表 schema 不匹配（如 label 类型变更），删除重建
    if table_name in existing_tables:
        old_table = db_manager.db.open_table(table_name)
        if old_table.schema != expected_schema:
            print(f"Table '{table_name}' schema mismatch, rebuilding.")
            db_manager.db.drop_table(table_name)
            existing_tables = []

    if table_name in existing_tables:
        # 表已存在且 schema 匹配，跳过建库
        print(f"Table '{table_name}' already exists, skipping database build.")
        table = db_manager.db.open_table(table_name)
    else:
        # 创建新表
        table = db_manager.db.create_table(table_name, schema=expected_schema)

        # 使用 DataLoader 批量建库
        train_loader = DataLoader(
            train_dataset.with_format("torch"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        _establish_eval_database(processor, model, table, train_loader)

    # 使用 DataLoader 批量评估
    test_loader = DataLoader(
        test_dataset.with_format("torch"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    correct, total = _evaluate_recall(processor, model, table, test_loader, top_k)

    accuracy = correct / total
    print(f"\nRecall@{top_k} on {dataset} with {model_name}: {accuracy:.4f}")
    print(f"Correct: {correct}/{total}")

    return accuracy
