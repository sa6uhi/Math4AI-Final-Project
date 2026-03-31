"""Experiment orchestration for single runs and repeated-seed summaries."""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import (
    CLI_RUN_DESCRIPTION,
    DATASET_CHOICES,
    DEFAULT_SEED,
    FIGURE_FILE_TEMPLATE,
    METRICS_FILE_TEMPLATE,
    MODEL_CHOICES,
    NORMAL_Z_95,
    OUTPUT_SEPARATOR,
    FIGURES_DIR,
    RESULTS_DIR,
    REPEATED_SEED_SUMMARY_HEADER,
    REPEATED_SUMMARY_FILE_TEMPLATE,
    REPEAT_SEEDS_DEFAULT,
    ensure_output_dirs,
    parse_seed_list,
    T_CRIT_95,
    get_hyperparams_config,
)
from .data_utils import DataRepository
from .io_utils import MetricsWriter
from .models import HiddenLayerClassifier, SoftmaxRegressionClassifier
from .plotting import DecisionBoundaryPlotter
from .trainers import HiddenLayerTrainer, SoftmaxTrainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CLI_RUN_DESCRIPTION)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeat-seeds", type=str, default=REPEAT_SEEDS_DEFAULT)
    return parser.parse_args()


class ExperimentRunner:
    """Run one experiment configuration and persist metrics/figures."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self.data_repo = DataRepository()
        self.metrics_writer = MetricsWriter()
        self.plotter = DecisionBoundaryPlotter()

    @staticmethod
    def get_hyperparams(dataset: str, model: str) -> Dict[str, float | int]:
        """Return default hyperparameters for a dataset/model pair."""
        return get_hyperparams_config(dataset, model)

    @staticmethod
    def _ci95(values: List[float]) -> Dict[str, float]:
        """Compute mean, sample std, and two-sided 95% CI for scalar values.

        Uses Student's t critical values for small samples and a 1.96 normal
        approximation for df > 30.
        """
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        if arr.size <= 1:
            return {
                "mean": mean,
                "std": 0.0,
                "ci_low": mean,
                "ci_high": mean,
            }

        std = float(np.std(arr, ddof=1))
        df = arr.size - 1
        t_critical = T_CRIT_95.get(df, NORMAL_Z_95)
        margin = t_critical * std / np.sqrt(arr.size)
        return {
            "mean": mean,
            "std": std,
            "ci_low": mean - margin,
            "ci_high": mean + margin,
        }

    def run(self, dataset: str, model_type: str, seed: int | None = None) -> Dict[str, float | Path]:
        """Run a single dataset/model experiment and write output artifacts."""
        ensure_output_dirs()

        seed_to_use = self.seed if seed is None else seed

        if dataset == "digits":
            X_train, y_train, X_val, y_val, X_test, y_test = self.data_repo.load_digits_fixed_split()
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = self.data_repo.load_dataset(dataset)
            X_train, X_val, X_test = self.data_repo.standardize_splits(X_train, X_val, X_test)

        hparams = self.get_hyperparams(dataset, model_type)

        if model_type == "softmax":
            model = SoftmaxRegressionClassifier(
                input_dim=X_train.shape[1],
                n_classes=int(y_train.max()) + 1,
            )
            model.initialize(seed_to_use)
            trainer = SoftmaxTrainer(
                epochs=int(hparams["epochs"]),
                learning_rate=float(hparams["learning_rate"]),
                lambda_reg=float(hparams["lambda_reg"]),
                batch_size=int(hparams["batch_size"]),
                seed=seed_to_use,
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
            model.initialize(seed_to_use)
            trainer = HiddenLayerTrainer(
                epochs=int(hparams["epochs"]),
                learning_rate=float(hparams["learning_rate"]),
                lambda_reg=float(hparams["lambda_reg"]),
                batch_size=int(hparams["batch_size"]),
                seed=seed_to_use,
            )
            history = trainer.fit(model, X_train, y_train, X_val, y_val)
            test_loss, test_acc = model.evaluate(X_test, y_test)
            predict_fn = model.predict
            title = f"{dataset.replace('_', ' ').title()}: 1-Hidden-Layer Decision Boundary"

        metrics_path = RESULTS_DIR / METRICS_FILE_TEMPLATE.format(
            dataset=dataset,
            model_type=model_type,
            seed=seed_to_use,
        )
        figure_path = FIGURES_DIR / FIGURE_FILE_TEMPLATE.format(
            dataset=dataset,
            model_type=model_type,
            seed=seed_to_use,
        )

        self.metrics_writer.save_history_csv(history, metrics_path)
        if X_test.shape[1] == 2:
            self.plotter.plot_decision_boundary(X_test, y_test, predict_fn, figure_path, title)

        print(OUTPUT_SEPARATOR)
        print(f"FINAL TEST RESULTS ({dataset.upper()} + {model_type.upper()} + seed={seed_to_use})")
        print(f"Test Cross-Entropy: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Saved metrics: {metrics_path}")
        if X_test.shape[1] == 2:
            print(f"Saved figure: {figure_path}")
        print(OUTPUT_SEPARATOR)

        return {
            "dataset": dataset,
            "model": model_type,
            "seed": seed_to_use,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "metrics_path": metrics_path,
            "figure_path": figure_path,
        }

    def run_repeated_seeds(self, dataset: str, model_type: str, seeds: List[int]) -> Path:
        """Run multiple seeds and save aggregate metrics with 95% CI bounds."""
        rows = []
        losses = []
        accuracies = []

        for seed in seeds:
            out = self.run(dataset=dataset, model_type=model_type, seed=seed)
            losses.append(float(out["test_loss"]))
            accuracies.append(float(out["test_accuracy"]))

        loss_stats = self._ci95(losses)
        acc_stats = self._ci95(accuracies)

        rows.append([
            "test_cross_entropy",
            loss_stats["mean"],
            loss_stats["std"],
            loss_stats["ci_low"],
            loss_stats["ci_high"],
            len(seeds),
        ])
        rows.append([
            "test_accuracy",
            acc_stats["mean"],
            acc_stats["std"],
            acc_stats["ci_low"],
            acc_stats["ci_high"],
            len(seeds),
        ])

        out_path = RESULTS_DIR / REPEATED_SUMMARY_FILE_TEMPLATE.format(
            dataset=dataset,
            model_type=model_type,
        )
        np.savetxt(
            out_path,
            np.array(rows, dtype=object),
            delimiter=",",
            header=REPEATED_SEED_SUMMARY_HEADER,
            comments="",
            fmt="%s",
        )

        print(f"Saved repeated-seed summary: {out_path}")
        return out_path


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(seed=args.seed)

    if args.repeat_seeds:
        seeds = parse_seed_list(args.repeat_seeds)
        runner.run_repeated_seeds(args.dataset, args.model, seeds)
        return

    runner.run(args.dataset, args.model, seed=args.seed)


if __name__ == "__main__":
    main()
