import matplotlib.pyplot as plt
import numpy as np
from src.data_utils import DataRepository
from src.track_a_pca import PCAMath
from src.models import SoftmaxRegressionClassifier
from src.trainers import SoftmaxTrainer
from src.config import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs

def run_track_a():
    ensure_output_dirs()
    repo=DataRepository()
    X_train, y_train, X_val,y_val, X_test, y_test = repo.load_digits_fixed_split()

    #Compute PCA
    Vt, S, exp_var=PCAMath.compute_pca(X_train)

    #Scree Plot
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, 11), exp_var[:10], alpha=0.7, label='Individual')
    plt.step(range(1, 11), np.cumsum(exp_var)[:10], where='mid', label='Cumulative')
    plt.title("Scree Plot (Top 10 Components)")
    plt.ylabel("Variance Ratio")
    plt.savefig(FIGURES_DIR / "track_a_scree.png")

    #2D Visualization
    X_2d=PCAMath.project(X_train,Vt,2)
    plt.figure(figsize=(7, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap='tab10', s=5, alpha=0.6)
    plt.colorbar(label='Digit Class')
    plt.title("2D PCA Projection of Digits")
    plt.savefig(FIGURES_DIR / "track_a_viz_2d.png")
