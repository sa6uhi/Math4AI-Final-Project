"""Central configuration for experiment defaults and tunable constants."""

from pathlib import Path
from typing import Dict, Final, List, Tuple

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = BASE_DIR / "data"
RESULTS_DIR: Final[Path] = BASE_DIR / "results"
FIGURES_DIR: Final[Path] = BASE_DIR / "figures"


def ensure_output_dirs() -> None:
    """Create output directories used by experiment runs if they do not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Core defaults and CLI policy
# -----------------------------------------------------------------------------

DEFAULT_SEED: Final[int] = 42
DATASET_CHOICES: Final[Tuple[str, ...]] = ("moons", "linear_gaussian", "digits")
MODEL_CHOICES: Final[Tuple[str, ...]] = ("softmax", "hidden_layer")
CLI_RUN_DESCRIPTION: Final[str] = "Run one Math4AI synthetic experiment."

REPEAT_SEEDS_DEFAULT: Final[str] = ""
SEED_LIST_DELIMITER: Final[str] = ","

METRICS_HISTORY_HEADER: Final[str] = "epoch,train_loss,val_loss,val_accuracy"
REPEATED_SEED_SUMMARY_HEADER: Final[str] = "metric,mean,std,ci_low,ci_high,n"

METRICS_FILE_TEMPLATE: Final[str] = "{dataset}_{model_type}_seed{seed}_metrics.csv"
FIGURE_FILE_TEMPLATE: Final[str] = "{dataset}_{model_type}_seed{seed}_boundary.png"
REPEATED_SUMMARY_FILE_TEMPLATE: Final[str] = "{dataset}_{model_type}_repeated_seed_summary.csv"

OUTPUT_SEPARATOR: Final[str] = "-" * 40

DEFAULT_BATCH_RUNS: Final[Tuple[Tuple[str, str], ...]] = (
    ("linear_gaussian", "softmax"),
    ("linear_gaussian", "hidden_layer"),
    ("moons", "hidden_layer"),
    ("moons", "softmax"),
)


# -----------------------------------------------------------------------------
# Two-sided 95% t critical values for df=1..30.
# -----------------------------------------------------------------------------

T_CRIT_95: Final[Dict[int, float]] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

NORMAL_Z_95: Final[float] = 1.96

# -----------------------------------------------------------------------------
# Default hyperparameters keyed by (dataset, model_type).
# -----------------------------------------------------------------------------

HYPERPARAMS: Final[Dict[Tuple[str, str], Dict[str, float | int]]] = {
    ("digits", "softmax"): {
        "epochs": 200,
        "learning_rate": 0.05,
        "lambda_reg": 1e-4,
        "batch_size": 64,
    },
    ("digits", "hidden_layer"): {
        "hidden_dim": 32,
        "epochs": 200,
        "learning_rate": 0.05,
        "lambda_reg": 1e-4,
        "batch_size": 64,
    },
    ("moons", "softmax"): {
        "epochs": 900,
        "learning_rate": 0.1,
        "lambda_reg": 0.01,
        "batch_size": 256,
    },
    ("linear_gaussian", "softmax"): {
        "epochs": 200,
        "learning_rate": 0.1,
        "lambda_reg": 0.01,
        "batch_size": 256,
    },
    ("moons", "hidden_layer"): {
        "hidden_dim": 10,
        "epochs": 1200,
        "learning_rate": 0.3,
        "lambda_reg": 0.01,
        "batch_size": 256,
    },
    ("linear_gaussian", "hidden_layer"): {
        "hidden_dim": 10,
        "epochs": 300,
        "learning_rate": 0.3,
        "lambda_reg": 0.01,
        "batch_size": 256,
    },
}


def get_hyperparams_config(dataset: str, model: str) -> Dict[str, float | int]:
    """Return hyperparameters for a supported dataset/model pair."""
    key = (dataset, model)
    if key not in HYPERPARAMS:
        raise ValueError(f"Unsupported dataset/model pair: {dataset}/{model}")
    return HYPERPARAMS[key].copy()


def parse_seed_list(raw: str) -> List[int]:
    """Parse comma-separated seed values and drop empty entries."""
    return [int(item.strip()) for item in raw.split(SEED_LIST_DELIMITER) if item.strip()]
