import numpy as np
from typing import Tuple

class PCAMath:
  @staticmethod
  def compute_pca(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_mean = np.mean(X_train, axis=0)
        X_centered = X_train - train_mean
        
        # SVD: X = U S Vt
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return Vt, S, train_mean

    @staticmethod
    def project(X: np.ndarray, Vt: np.ndarray, train_mean: np.ndarray, m: int) -> np.ndarray:
        # Use the TRAIN mean to center the input data (Val or Test)
        X_centered = X - train_mean
        return np.dot(X_centered, Vt[:m, :].T)
