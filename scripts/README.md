# scripts

Command-line entry points for running experiments.

## Files

- `run_experiment.py`: Run one experiment or repeated-seed summary.
- `run_all_experiments.py`: Batch helper for baseline experiment runs.

## Common commands

Run one synthetic experiment:

```bash
python -m scripts.run_experiment --dataset linear_gaussian --model softmax --seed 42
```

Run one digits experiment:

```bash
python -m scripts.run_experiment --dataset digits --model softmax --seed 42
```

Run repeated seeds and write summary CSV:

```bash
python -m scripts.run_experiment --dataset linear_gaussian --model softmax --repeat-seeds "42,43,44,45,46"
```

Run the default synthetic batch:

```bash
python -m scripts.run_all_experiments
```
