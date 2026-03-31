# src

Implementation code for Math4AI experiments.

## Modules

- `data_utils.py`: `DataRepository` for dataset loading and standardization
- `models.py`: `SoftmaxRegressionClassifier` and `HiddenLayerClassifier`
- `trainers.py`: `SoftmaxTrainer` and `HiddenLayerTrainer`
- `plotting.py`: `DecisionBoundaryPlotter`
- `io_utils.py`: `MetricsWriter`
- `run_experiment.py`: `ExperimentRunner` orchestration class

## Notes

- Synthetic datasets (`moons`, `linear_gaussian`) are standardized using train-set statistics.
- Digits uses the fixed split indices provided in starter data.
- Repeated-seed summaries are written as CSV with columns:
	`metric,mean,std,ci_low,ci_high,n`

## Example usage

Run one seeded synthetic experiment:

```bash
python -m src.run_experiment --dataset linear_gaussian --model softmax --seed 42
```

Run one seeded digits experiment:

```bash
python -m src.run_experiment --dataset digits --model softmax --seed 42
```

Run repeated seeds and generate CI summary:

```bash
python -m src.run_experiment --dataset linear_gaussian --model softmax --repeat-seeds "42,43,44,45,46"
```
