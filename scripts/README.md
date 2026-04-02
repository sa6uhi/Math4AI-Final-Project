# scripts

Command-line entry points for running experiments.

## Files

- `run_experiment.py`: Run one experiment or repeated-seed summary.
- `run_all_experiments.py`: Batch helper for baseline experiment runs.
- `run_track_a.py`: Run Track A PCA workflow for digits.

## Common commands

Run commands from the project root and use module form (`python -m ...`).

Run one synthetic experiment:

```bash
python -m scripts.run_experiment --dataset linear_gaussian --model softmax --seed 42
```

Run one digits experiment:

```bash
python -m scripts.run_experiment --dataset digits --model softmax --seed 42
python -m scripts.run_experiment --dataset digits --model hidden_layer --optimizer adam --seed 42
```

Run repeated seeds and write summary CSV:

```bash
python -m scripts.run_experiment --dataset linear_gaussian --model softmax --repeat-seeds "42,43,44,45,46"
python -m scripts.run_experiment --dataset digits --model softmax --repeat-seeds "42,43,44,45,46"
python -m scripts.run_experiment --dataset digits --model hidden_layer --optimizer adam --repeat-seeds "42,43,44,45,46"
```

Run hidden-layer optimizer ablation (SGD, Momentum, Adam):

```bash
python -m scripts.run_experiment --dataset digits --model hidden_layer --ablate-hidden-optimizers --seed 42
```

Run the default synthetic batch:

```bash
python -m scripts.run_all_experiments
```

Run Track A PCA workflow:

```bash
python -m scripts.run_track_a
```
