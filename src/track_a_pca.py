import numpy as np
from typing import Tuple

class PCAMath:
  @staticmethod
  def compute_pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #Center the data
    X_centered=X-np.mean(X,axis=0)

    #SVD
    _, S, Vt=np.linalg.svd(X_centered, full_matrices=False)

    #Explained Variance Ratio
    exp_var_ratio=(S**2)/np.sum(S**2)

    return Vt, S, exp_var_ratio

  @staticmethod
  def project(X:np.ndarray, Vt:np.ndarray, m:int) -> np.ndarray:
      X_centered=X-np.mean(X,axis=0)
      return np.dot(X_centered, Vt[:m, :].T)
