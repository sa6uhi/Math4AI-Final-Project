import matplotlib.pyplot as plt
import numpy as np
from src.track_a_pca import PCAMath
from src.data_utils import DataRepository
from src.models import SoftmaxRegressionClassifier
from src.trainers import SoftmaxTrainer
from src.config import FIGURES_DIR, RESULTS_DIR, get_hyperparams_config

def run_track_a():
    repo = DataRepository()
    X_train, y_train, X_val, y_val, X_test, y_test = repo.load_digits_fixed_split()
    
    # 1. Compute PCA on Train only
    Vt, S, train_mean = PCAMath.compute_pca(X_train)
    exp_var = (S**2) / np.sum(S**2)

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

    # 2. Get Consistent Hyperparams
    hparams = get_hyperparams_config("digits", "softmax")

    table_rows = []
    for m in [10, 20, 40]:
        # FIXED: Passing train_mean to ensure zero data leakage
        X_tr_m = PCAMath.project(X_train, Vt, train_mean, m)
        X_te_m = PCAMath.project(X_test, Vt, train_mean, m)
        X_va_m = PCAMath.project(X_val, Vt, train_mean, m)
        
        model = SoftmaxRegressionClassifier(input_dim=m, n_classes=10)
        model.initialize(seed=42)
        
        # Use central tunables
        trainer = SoftmaxTrainer(
            epochs=int(hparams["epochs"]),
            learning_rate=float(hparams["learning_rate"]),
            lambda_reg=float(hparams["lambda_reg"]),
            batch_size=int(hparams["batch_size"])
        )
        
        trainer.fit(model, X_tr_m, y_train, X_va_m, y_val)
        _, acc = model.evaluate(X_te_m, y_test)
        
        # FIXED: Numeric CSV format (no 'm=' prefix)
        table_rows.append(f"{m},{acc:.4f}")

    # Write cleaned CSV
    with open(RESULTS_DIR / "track_a_comparison.csv", "w") as f:
        f.write("m,test_accuracy\n" + "\n".join(table_rows))

if __name__ == "__main__":
    run_track_a()
