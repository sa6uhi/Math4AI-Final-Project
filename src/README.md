# src

Implementation code for Math4AI experiments.

## Modules

- `data_utils.py`: dataset loading and train-fit standardization
- `models.py`: softmax/math helpers and evaluation utilities
- `trainers.py`: training loops for softmax and one-hidden-layer models
- `plotting.py`: decision-boundary plotting helper
- `io_utils.py`: metrics CSV export helper
- `run_experiment.py`: CLI runner for synthetic experiments

## Run

From project root:

- `python -m src.run_experiment --dataset linear_gaussian --model softmax`
- `python -m src.run_experiment --dataset linear_gaussian --model hidden_layer`
- `python -m src.run_experiment --dataset moons --model softmax`
- `python -m src.run_experiment --dataset moons --model hidden_layer`
