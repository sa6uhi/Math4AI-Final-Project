# figures

Generated plots from experiment runs.

## Typical files

- `<dataset>_<model>_seed<seed>_boundary.png`
- `track_a_scree.png`
- `track_a_viz_2d.png`

## Notes

- Decision boundary figures are generated only for 2D feature datasets.
- Digits experiments do not produce boundary plots because inputs are high-dimensional.
- Regenerate figures by rerunning the corresponding `python -m scripts.run_experiment ...` command.
- Track A figures are regenerated via `python scripts/run_track_a.py`.
