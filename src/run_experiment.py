import argparse

from .data_utils import load_dataset, standardize_splits
from .io_utils import save_history_csv
from .models import (
    evaluate_hidden_layer,
    evaluate_softmax,
    predict_hidden_layer,
    predict_softmax,
)
from .paths import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs
from .plotting import plot_decision_boundary
from .trainers import train_hidden_layer, train_softmax

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Math4AI synthetic experiment.")
    parser.add_argument("--dataset", choices=["moons", "linear_gaussian"], required=True)
    parser.add_argument("--model", choices=["softmax", "hidden_layer"], required=True)
    return parser.parse_args()


def get_hyperparams(dataset: str, model: str) -> dict:
    if model == "softmax":
        return {
            "epochs": 900 if dataset == "moons" else 200,
            "learning_rate": 0.1,
            "lambda_reg": 0.01,
        }

    return {
        "hidden_dim": 10,
        "epochs": 1200 if dataset == "moons" else 300,
        "learning_rate": 0.3,
        "lambda_reg": 0.01,
    }


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset)
    X_train, X_val, X_test = standardize_splits(X_train, X_val, X_test)

    hparams = get_hyperparams(args.dataset, args.model)

    if args.model == "softmax":
        W, b, history = train_softmax(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=hparams["epochs"],
            learning_rate=hparams["learning_rate"],
            lambda_reg=hparams["lambda_reg"],
            seed=SEED,
        )
        test_loss, test_acc = evaluate_softmax(X_test, y_test, W, b)
        predict_fn = lambda X: predict_softmax(X, W, b)
        title = f"{args.dataset.replace('_', ' ').title()}: Softmax Decision Boundary"
    else:
        W1, b1, W2, b2, history = train_hidden_layer(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_dim=hparams["hidden_dim"],
            epochs=hparams["epochs"],
            learning_rate=hparams["learning_rate"],
            lambda_reg=hparams["lambda_reg"],
            seed=SEED,
        )
        test_loss, test_acc = evaluate_hidden_layer(X_test, y_test, W1, b1, W2, b2)
        predict_fn = lambda X: predict_hidden_layer(X, W1, b1, W2, b2)
        title = f"{args.dataset.replace('_', ' ').title()}: 1-Hidden-Layer Decision Boundary"

    metrics_path = RESULTS_DIR / f"{args.dataset}_{args.model}_metrics.csv"
    figure_path = FIGURES_DIR / f"{args.dataset}_{args.model}_boundary.png"

    save_history_csv(history, metrics_path)
    plot_decision_boundary(X_test, y_test, predict_fn, figure_path, title)

    print("-" * 40)
    print(f"FINAL TEST RESULTS ({args.dataset.upper()} + {args.model.upper()})")
    print(f"Test Cross-Entropy: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figure: {figure_path}")
    print("-" * 40)


if __name__ == "__main__":
    main()
