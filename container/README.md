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

## Notes

- The compose service runs batch experiments and writes outputs into mounted project folders.
- If you only want to validate config resolution:

```bash
nerdctl compose -f container/compose.yaml config
```
