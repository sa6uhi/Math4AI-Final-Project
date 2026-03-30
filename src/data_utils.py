from typing import Tuple

import numpy as np

from .paths import DATA_DIR


DatasetSplits = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class DataRepository:
    def __init__(self, data_dir=DATA_DIR) -> None:
        self.data_dir = data_dir

    def load_dataset(self, dataset_name: str) -> DatasetSplits:
        data_path = self.data_dir / f"{dataset_name}.npz"
        with np.load(data_path) as data:
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_val = data["X_val"]
            y_val = data["y_val"]
            X_test = data["X_test"]
            y_test = data["y_test"]
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def standardize_splits(
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std = np.where(std < 1e-12, 1.0, std)

        X_train_s = (X_train - mean) / std
        X_val_s = (X_val - mean) / std
        X_test_s = (X_test - mean) / std
        return X_train_s, X_val_s, X_test_s
