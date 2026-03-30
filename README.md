# Math4AI Final Project

From Linear Scores to a Single Hidden Layer: a mathematical study of when linear models are sufficient and when a one-hidden-layer neural network adds useful representational power.

## Team

- Sabuhi Nazarov
- Agshin Fataliyev

## Project Goal

This project follows the Math4AI capstone framing:

- implement core models from scratch in NumPy
- compare softmax regression and a one-hidden-layer tanh network
- analyze linear Gaussian and moons synthetic datasets
- save reproducible metrics and decision-boundary figures

## Repository Structure

- data: provided datasets (linear_gaussian.npz, moons.npz, digits files)
- src: implementation code (data loading, models, trainers, plotting, experiment runner)
- scripts: runnable entry scripts that call `src` classes
- results: generated per-epoch metrics CSV files
- figures: generated decision-boundary figures
- container: container setup (Containerfile and compose.yaml)
- requirements.txt: Python dependencies

## Prerequisites

- Python 3.11+ (project tested with Python 3.13)
- pip
- Optional for container run: Docker with compose

## Quick Start (Local Python venv)

1. Clone repository and enter project root.

```bash
git clone git@github.com:sa6uhi/Math4AI-Final-Project.git
cd Math4AI-Final-Project
```

2. Create and activate virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Run all four synthetic experiments.

```bash
python -m scripts.run_all_experiments
```

5. Check outputs.

- CSV metrics in results
- decision-boundary figures in figures

## Single Experiment Command

Run one configuration when needed:

```bash
python -m scripts.run_experiment --dataset linear_gaussian --model softmax
python -m scripts.run_experiment --dataset linear_gaussian --model hidden_layer
python -m scripts.run_experiment --dataset moons --model softmax
python -m scripts.run_experiment --dataset moons --model hidden_layer
```

## Container Run (Docker compose)

From project root:

```bash
docker compose -f container/compose.yaml up --build
docker compose -f container/compose.yaml down
```

## Reproducibility Notes

- deterministic seed is set in the experiment pipeline
- train-fit standardization is applied consistently to train/validation/test for synthetic datasets
- metrics are saved per epoch with columns:
  - epoch
  - train_loss
  - val_loss
  - val_accuracy

## Expected Output Files

Metrics:

- results/linear_gaussian_softmax_metrics.csv
- results/linear_gaussian_hidden_layer_metrics.csv
- results/moons_softmax_metrics.csv
- results/moons_hidden_layer_metrics.csv

Figures:

- figures/linear_gaussian_softmax_boundary.png
- figures/linear_gaussian_hidden_layer_boundary.png
- figures/moons_softmax_boundary.png
- figures/moons_hidden_layer_boundary.png

## Troubleshooting

- If matplotlib cache warnings appear in containers, compose already sets writable HOME and MPLCONFIGDIR.
- If permissions fail on mounted outputs, run compose as configured in container/compose.yaml.
- If module import fails for src, make sure you run commands from project root.
- If `compose up` appears to print repeated experiment blocks, you are seeing accumulated logs from the same stopped container. Use `docker compose -f container/compose.yaml down -v && docker compose -f container/compose.yaml up --build` for a fresh run.
