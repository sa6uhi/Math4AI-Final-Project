# Container

Minimal container setup for running project experiments reproducibly.

## Files

- `Containerfile`: Builds a Python image with project dependencies.
- `compose.yaml`: Runs the project command in a container service.

## Quick Run

From the project root:

```bash
nerdctl compose -f container/compose.yaml up
```

Alternative with Docker:

```bash
docker compose -f container/compose.yaml up
```

Run only Track A PCA workflow:

```bash
nerdctl compose -f container/compose.yaml run --rm math4ai-track-a
```

Docker equivalent:

```bash
docker compose -f container/compose.yaml run --rm math4ai-track-a
```

## Notes

- The compose service runs batch experiments and writes outputs into mounted project folders.
- The `math4ai-track-a` service runs the Track A PCA pipeline and writes `results/track_a_comparison.csv` plus Track A figures.
- If you only want to validate config resolution:

```bash
nerdctl compose -f container/compose.yaml config
```
