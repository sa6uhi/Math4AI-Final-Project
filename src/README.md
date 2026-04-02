# src

Implementation code for Math4AI experiments.

## Modules

- `config.py`: centralized defaults, paths, output templates, and hyperparameter map
- `data_utils.py`: `DataRepository` for dataset loading and standardization
- `models.py`: `SoftmaxRegressionClassifier` and `HiddenLayerClassifier`
- `trainers.py`: `SoftmaxTrainer` and `HiddenLayerTrainer`
- `plotting.py`: `DecisionBoundaryPlotter`
- `io_utils.py`: `MetricsWriter`
- `run_experiment.py`: `ExperimentRunner` orchestration class
- `track_a_pca.py`: PCA/SVD utilities used by Track A workflow

## Notes

- Synthetic datasets (`moons`, `linear_gaussian`) are standardized using train-set statistics.
- Digits uses the fixed split indices provided in starter data.
- Config definitions and rationale are documented in `CONFIG.md`.
- Repeated-seed summaries are written as CSV with columns:
	`metric,mean,std,ci_low,ci_high,n`

## Example usage

Run one seeded synthetic experiment from module entrypoint:

```bash
python -m src.run_experiment --dataset linear_gaussian --model softmax --seed 42
```

Run one seeded digits experiment:

```bash
python -m src.run_experiment --dataset digits --model softmax --seed 42
python -m src.run_experiment --dataset digits --model hidden_layer --optimizer adam --seed 42
```

Run repeated seeds and generate CI summary:

```bash
python -m src.run_experiment --dataset linear_gaussian --model softmax --repeat-seeds "42,43,44,45,46"
python -m src.run_experiment --dataset digits --model softmax --repeat-seeds "42,43,44,45,46"
python -m src.run_experiment --dataset digits --model hidden_layer --optimizer adam --repeat-seeds "42,43,44,45,46"
```

Run hidden-layer optimizer ablation:

```bash
python -m src.run_experiment --dataset digits --model hidden_layer --ablate-hidden-optimizers --seed 42
```

Run Track A script entrypoint:

```bash
python scripts/run_track_a.py
```
