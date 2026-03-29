from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    preds_grid = predict_fn(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, preds_grid, levels=2, alpha=0.25, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=18, edgecolors="k", linewidths=0.2, cmap="coolwarm")
    plt.title(title)
    plt.xlabel("x1 (standardized)")
    plt.ylabel("x2 (standardized)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
