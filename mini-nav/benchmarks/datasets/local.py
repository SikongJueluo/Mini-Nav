"""Local dataset loader for benchmark evaluation."""

from pathlib import Path
from typing import Any, Optional

from ..base import BaseDataset


class LocalDataset(BaseDataset):
    """Dataset loader for local datasets."""

    def __init__(
        self,
        local_path: str,
        img_column: str = "image_path",
        label_column: str = "label",
    ):
        """Initialize local dataset loader.

        Args:
            local_path: Path to local dataset directory or CSV file.
            img_column: Name of the image path column.
            label_column: Name of the label column.
        """
        self.local_path = Path(local_path)
        self.img_column = img_column
        self.label_column = label_column
        self._train_dataset: Optional[Any] = None
        self._test_dataset: Optional[Any] = None

    def _load_csv_dataset(self) -> tuple[Any, Any]:
        """Load dataset from CSV file.

        Expected CSV format:
            label,image_path,x1,y1,x2,y2
            "class_name","path/to/image.jpg",100,200,300,400

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        import pandas as pd

        from torch.utils.data import Dataset as TorchDataset

        # Load CSV file
        df = pd.read_csv(self.local_path)

        # Create a simple dataset class
        class CSVDataset(TorchDataset):
            def __init__(self, dataframe: pd.DataFrame, img_col: str, label_col: str):
                self.df = dataframe.reset_index(drop=True)
                self.img_col = img_col
                self.label_col = label_col

            def __len__(self) -> int:
                return len(self.df)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                row = self.df.iloc[idx]
                return {
                    "img": row[self.img_col],
                    "label": row[self.label_col],
                }

        # Split into train/test (80/20)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        self._train_dataset = CSVDataset(train_df, self.img_column, self.label_column)
        self._test_dataset = CSVDataset(test_df, self.img_column, self.label_column)

        return self._train_dataset, self._test_dataset

    def _load_directory_dataset(self) -> tuple[Any, Any]:
        """Load dataset from directory structure.

        Expected structure:
            local_path/
                train/
                    class_name_1/
                        image1.jpg
                        image2.jpg
                    class_name_2/
                        image1.jpg
                test/
                    class_name_1/
                        image1.jpg

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        from torch.utils.data import Dataset as TorchDataset
        from PIL import Image

        class DirectoryDataset(TorchDataset):
            def __init__(self, root_dir: Path, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.samples = []
                self.label_map = {}

                # Build label map
                classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
                self.label_map = {cls: idx for idx, cls in enumerate(classes)}

                # Build sample list
                for cls_dir in root_dir.iterdir():
                    if cls_dir.is_dir():
                        label = self.label_map[cls_dir.name]
                        for img_path in cls_dir.iterdir():
                            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                                self.samples.append((img_path, label))

            def __len__(self) -> int:
                return len(self.samples)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                img_path, label = self.samples[idx]
                image = Image.open(img_path).convert("RGB")
                return {"img": image, "label": label}

        train_dir = self.local_path / "train"
        test_dir = self.local_path / "test"

        if train_dir.exists():
            self._train_dataset = DirectoryDataset(train_dir)
        if test_dir.exists():
            self._test_dataset = DirectoryDataset(test_dir)

        return self._train_dataset, self._test_dataset

    def get_train_split(self) -> Any:
        """Get training split of the dataset.

        Returns:
            Training dataset.
        """
        if self._train_dataset is None:
            if self.local_path.suffix.lower() == ".csv":
                self._load_csv_dataset()
            else:
                self._load_directory_dataset()
        return self._train_dataset

    def get_test_split(self) -> Any:
        """Get test/evaluation split of the dataset.

        Returns:
            Test dataset.
        """
        if self._test_dataset is None:
            if self.local_path.suffix.lower() == ".csv":
                self._load_csv_dataset()
            else:
                self._load_directory_dataset()
        return self._test_dataset
