import argparse
from typing import Dict

from .data_utils import DataRepository
from .io_utils import MetricsWriter
from .models import HiddenLayerClassifier, SoftmaxRegressionClassifier
from .paths import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs
from .plotting import DecisionBoundaryPlotter
from .trainers import HiddenLayerTrainer, SoftmaxTrainer

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Math4AI synthetic experiment.")
    parser.add_argument("--dataset", choices=["moons", "linear_gaussian"], required=True)
    parser.add_argument("--model", choices=["softmax", "hidden_layer"], required=True)
    return parser.parse_args()


class ExperimentRunner:
    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self.data_repo = DataRepository()
        self.metrics_writer = MetricsWriter()
        self.plotter = DecisionBoundaryPlotter()

    @staticmethod
    def get_hyperparams(dataset: str, model: str) -> Dict[str, float]:
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

    def run(self, dataset: str, model_type: str) -> None:
        ensure_output_dirs()

        X_train, y_train, X_val, y_val, X_test, y_test = self.data_repo.load_dataset(dataset)
        X_train, X_val, X_test = self.data_repo.standardize_splits(X_train, X_val, X_test)

        hparams = self.get_hyperparams(dataset, model_type)

        if model_type == "softmax":
            model = SoftmaxRegressionClassifier(
                input_dim=X_train.shape[1],
                n_classes=int(y_train.max()) + 1,
            )
            model.initialize(self.seed)
            trainer = SoftmaxTrainer(
                epochs=int(hparams["epochs"]),
                learning_rate=float(hparams["learning_rate"]),
                lambda_reg=float(hparams["lambda_reg"]),
            )
            history = trainer.fit(model, X_train, y_train, X_val, y_val)
            test_loss, test_acc = model.evaluate(X_test, y_test)
            predict_fn = model.predict
            title = f"{dataset.replace('_', ' ').title()}: Softmax Decision Boundary"
        else:
            model = HiddenLayerClassifier(
                input_dim=X_train.shape[1],
                hidden_dim=int(hparams["hidden_dim"]),
                output_dim=int(y_train.max()) + 1,
            )
            model.initialize(self.seed)
            trainer = HiddenLayerTrainer(
                epochs=int(hparams["epochs"]),
                learning_rate=float(hparams["learning_rate"]),
                lambda_reg=float(hparams["lambda_reg"]),
            )
            history = trainer.fit(model, X_train, y_train, X_val, y_val)
            test_loss, test_acc = model.evaluate(X_test, y_test)
            predict_fn = model.predict
            title = f"{dataset.replace('_', ' ').title()}: 1-Hidden-Layer Decision Boundary"

        metrics_path = RESULTS_DIR / f"{dataset}_{model_type}_metrics.csv"
        figure_path = FIGURES_DIR / f"{dataset}_{model_type}_boundary.png"

        self.metrics_writer.save_history_csv(history, metrics_path)
        self.plotter.plot_decision_boundary(X_test, y_test, predict_fn, figure_path, title)

        print("-" * 40)
        print(f"FINAL TEST RESULTS ({dataset.upper()} + {model_type.upper()})")
        print(f"Test Cross-Entropy: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved figure: {figure_path}")
        print("-" * 40)


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(seed=SEED)
    runner.run(args.dataset, args.model)


if __name__ == "__main__":
    main()
