"""Dataset loading and preprocessing helpers for experiment pipelines."""

from typing import Tuple

import numpy as np

from .config import DATA_DIR


DatasetSplits = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class DataRepository:
    """Load datasets and provide train-derived standardization utilities."""

    def __init__(self, data_dir=DATA_DIR) -> None:
        self.data_dir = data_dir

    def load_dataset(self, dataset_name: str) -> DatasetSplits:
        """Load a pre-split dataset stored as a .npz bundle."""
        data_path = self.data_dir / f"{dataset_name}.npz"
        with np.load(data_path) as data:
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_val = data["X_val"]
            y_val = data["y_val"]
            X_test = data["X_test"]
            y_test = data["y_test"]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def load_digits_fixed_split(self) -> DatasetSplits:
        """Load digits data and apply fixed split indices provided by starter data."""
        data_path = self.data_dir / "digits_data.npz"
        split_path = self.data_dir / "digits_split_indices.npz"

        with np.load(data_path) as data:
            X = data["X"]
            y = data["y"]

        with np.load(split_path) as split:
            train_idx = split["train_idx"]
            val_idx = split["val_idx"]
            test_idx = split["test_idx"]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def standardize_splits(
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize using train-set statistics and apply to val/test."""
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std = np.where(std < 1e-12, 1.0, std)

        X_train_s = (X_train - mean) / std
        X_val_s = (X_val - mean) / std
        X_test_s = (X_test - mean) / std
        return X_train_s, X_val_s, X_test_s
