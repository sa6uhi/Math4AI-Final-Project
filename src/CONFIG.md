# CONFIG

This document describes every configuration variable in `src/config.py` and why it exists.

## Path settings

- `BASE_DIR`: Absolute project root used to derive all other internal directories.
- `DATA_DIR`: Folder containing `.npz` datasets and split index files.
- `RESULTS_DIR`: Folder where metrics CSV files are written.
- `FIGURES_DIR`: Folder where generated plots are written.
- `ensure_output_dirs()`: Creates `RESULTS_DIR` and `FIGURES_DIR` if missing so runs do not fail on first execution.

## Core defaults and CLI policy

- `DEFAULT_SEED`: Canonical seed for deterministic experiments.
- `DATASET_CHOICES`: Allowed dataset names for CLI validation.
- `MODEL_CHOICES`: Allowed model names for CLI validation.
- `CLI_RUN_DESCRIPTION`: Shared CLI description text for script and module entrypoints.
- `REPEAT_SEEDS_DEFAULT`: Default empty repeated-seed argument.
- `SEED_LIST_DELIMITER`: Delimiter used by `parse_seed_list`.
- `parse_seed_list(raw)`: Parses comma-separated seed strings and discards empty tokens.

## Output format and naming

- `METRICS_HISTORY_HEADER`: CSV header for per-epoch history files.
- `REPEATED_SEED_SUMMARY_HEADER`: CSV header for repeated-seed summaries.
- `METRICS_FILE_TEMPLATE`: Naming pattern for per-seed metrics output.
- `FIGURE_FILE_TEMPLATE`: Naming pattern for per-seed figure output.
- `REPEATED_SUMMARY_FILE_TEMPLATE`: Naming pattern for repeated-seed summary output.
- `OUTPUT_SEPARATOR`: Shared text separator for console output blocks.

## Batch run defaults

- `DEFAULT_BATCH_RUNS`: Canonical dataset/model pairs used by `scripts/run_all_experiments.py`.

## Statistical constants

- `T_CRIT_95`: Two-sided Student's t critical values for 95% CI at df=1..30.
- `NORMAL_Z_95`: Fallback z critical value (1.96) for larger sample sizes.

## Hyperparameter defaults

The `HYPERPARAMS` map stores settings keyed by `(dataset, model_type)`. This keeps one source of truth for all experiment scripts.

### Why these values

### `digits + softmax`
- `epochs=200`: Enough optimization steps to converge on normalized digit vectors without overlong runtime.
- `learning_rate=0.05`: More conservative than synthetic settings because digits are higher dimensional and gradients are noisier.
- `lambda_reg=1e-4`: Light regularization to avoid over-constraining linear separability on pixel features.
- `batch_size=64`: Stable minibatch gradient estimate while still computationally efficient.

### `digits + hidden_layer`
- `hidden_dim=32`: Small-but-expressive hidden width for non-linear representation without high overfitting risk.
- `epochs=200`, `learning_rate=0.05`, `lambda_reg=1e-4`, `batch_size=64`: Mirrors softmax digits defaults for fair model comparison.

### `moons + softmax`
- `epochs=900`: Linear model needs more iterations on a non-linearly separable toy dataset.
- `learning_rate=0.1`: Faster progress for low-dimensional synthetic data.
- `lambda_reg=0.01`: Stronger regularization to stabilize training on small synthetic splits.
- `batch_size=256`: Full/near-full-batch behavior for small datasets.

### `linear_gaussian + softmax`
- `epochs=200`: Linearly separable structure converges quickly.
- `learning_rate=0.1`, `lambda_reg=0.01`, `batch_size=256`: Same scale as moons softmax for consistent baseline behavior.

### `moons + hidden_layer`
- `hidden_dim=10`: Sufficient non-linear capacity for curved class boundaries while keeping model simple.
- `epochs=1200`: Extra optimization budget for non-linear fitting.
- `learning_rate=0.3`: Higher rate works well on low-dimensional toy data with tanh hidden layer.
- `lambda_reg=0.01`, `batch_size=256`: Regularized and stable small-data training.

### `linear_gaussian + hidden_layer`
- `hidden_dim=10`: Keeps architecture comparable to moons hidden-layer baseline.
- `epochs=300`: More than linear model but less than moons due to easier geometry.
- `learning_rate=0.3`, `lambda_reg=0.01`, `batch_size=256`: Consistent synthetic hidden-layer defaults.

## Access pattern

Use `get_hyperparams_config(dataset, model)` to retrieve a copy of configuration values.

This avoids accidental in-place edits to the global `HYPERPARAMS` map during experiments.
