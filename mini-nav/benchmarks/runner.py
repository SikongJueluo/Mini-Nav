"""Benchmark runner for executing evaluations."""

from pathlib import Path
from typing import Any

import lancedb
from benchmarks.datasets import HuggingFaceDataset, LocalDataset
from benchmarks.tasks import get_task
from configs.models import BenchmarkConfig, DatasetSourceConfig
from rich.console import Console
from rich.table import Table

console = Console()


def create_dataset(config: DatasetSourceConfig) -> Any:
    """Create a dataset instance from configuration.

    Args:
        config: Dataset source configuration.

    Returns:
        Dataset instance.

    Raises:
        ValueError: If source_type is not supported.
    """
    if config.source_type == "huggingface":
        return HuggingFaceDataset(
            hf_id=config.path,
            img_column=config.img_column,
            label_column=config.label_column,
        )
    elif config.source_type == "local":
        return LocalDataset(
            local_path=config.path,
            img_column=config.img_column,
            label_column=config.label_column,
        )
    else:
        raise ValueError(
            f"Unsupported source_type: {config.source_type}. "
            f"Supported types: 'huggingface', 'local'"
        )


def _get_table_name(config: BenchmarkConfig, model_name: str) -> str:
    """Generate database table name from config and model name.

    Args:
        config: Benchmark configuration.
        model_name: Model name for table naming.

    Returns:
        Formatted table name.
    """
    prefix = config.model_table_prefix
    # Use dataset path as part of table name (sanitize)
    dataset_name = Path(config.dataset.path).name.lower().replace("-", "_")
    return f"{prefix}_{dataset_name}_{model_name}"


def _ensure_table(
    config: BenchmarkConfig,
    model_name: str,
    vector_dim: int,
) -> lancedb.table.Table:
    """Ensure the LanceDB table exists with correct schema.

    Args:
        config: Benchmark configuration.
        model_name: Model name for table naming.
        vector_dim: Feature vector dimension.

    Returns:
        LanceDB table instance.
    """
    import pyarrow as pa
    from database import db_manager

    table_name = _get_table_name(config, model_name)

    # Build expected schema
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("label", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )

    db = db_manager.db
    existing_tables = db.list_tables().tables

    # Check if table exists and has correct schema
    if table_name in existing_tables:
        table = db.open_table(table_name)
        if table.schema != schema:
            console.print(
                f"[yellow]Table '{table_name}' schema mismatch, rebuilding.[/yellow]"
            )
            db.drop_table(table_name)
            table = db.create_table(table_name, schema=schema)
    else:
        table = db.create_table(table_name, schema=schema)

    return table


def _print_benchmark_info(
    config: BenchmarkConfig, vector_dim: int, table_name: str, table_count: int
) -> None:
    """Print benchmark configuration info using Rich table.

    Args:
        config: Benchmark configuration.
        vector_dim: Feature vector dimension.
        table_name: Database table name.
        table_count: Number of entries in the table.
    """
    table = Table(title="Benchmark Configuration", show_header=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Dataset", f"{config.dataset.source_type} - {config.dataset.path}")
    table.add_row("Model Output Dimension", str(vector_dim))
    table.add_row("Table Name", table_name)
    table.add_row("Table Entries", str(table_count))

    console.print(table)


def run_benchmark(
    model: Any,
    processor: Any,
    config: BenchmarkConfig,
    model_name: str = "model",
) -> dict[str, Any]:
    """Run benchmark evaluation.

    Workflow:
        1. Create dataset from configuration
        2. Create benchmark task from configuration
        3. Build evaluation database from training set
        4. Evaluate on test set

    Args:
        model: Feature extraction model.
        processor: Image preprocessor.
        config: Benchmark configuration.
        model_name: Model name for table naming.

    Returns:
        Dictionary containing evaluation results.

    Raises:
        ValueError: If benchmark is not enabled in config.
    """
    # Create dataset
    console.print(
        f"[cyan]Loading dataset:[/cyan] {config.dataset.source_type} - {config.dataset.path}"
    )
    dataset = create_dataset(config.dataset)

    # Get train and test splits
    train_dataset = dataset.get_train_split()
    test_dataset = dataset.get_test_split()

    if train_dataset is None or test_dataset is None:
        raise ValueError(
            f"Dataset {config.dataset.path} does not have train/test splits"
        )

    # Infer vector dimension from a sample
    sample = train_dataset[0]
    sample_image = sample["img"]

    from utils.feature_extractor import infer_vector_dim

    vector_dim = infer_vector_dim(processor, model, sample_image)
    console.print(f"[cyan]Model output dimension:[/cyan] {vector_dim}")

    # Ensure table exists with correct schema
    table = _ensure_table(config, model_name, vector_dim)
    table_name = _get_table_name(config, model_name)

    # Check if database is already built
    table_count = table.count_rows()
    if table_count > 0:
        console.print(
            f"[yellow]Table '{table_name}' already has {table_count} entries, skipping database build.[/yellow]"
        )
    else:
        # Create and run benchmark task
        task = get_task(config.task.type, top_k=config.task.top_k)
        console.print(
            f"[cyan]Building database[/cyan] with {len(train_dataset)} training samples..."
        )
        task.build_database(model, processor, train_dataset, table, config.batch_size)

    # Run evaluation (results with Rich table will be printed by the task)
    task = get_task(config.task.type, top_k=config.task.top_k)
    console.print(f"[cyan]Evaluating[/cyan] on {len(test_dataset)} test samples...")
    results = task.evaluate(model, processor, test_dataset, table, config.batch_size)

    return results
