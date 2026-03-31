# results

Generated CSV metrics from experiment runs.

## Typical files

- `<dataset>_<model>_seed<seed>_metrics.csv`
- `<dataset>_<model>_repeated_seed_summary.csv`

## Metric history schema

`epoch,train_loss,val_loss,val_accuracy`

## Repeated-seed summary schema

`metric,mean,std,ci_low,ci_high,n`

Where:
- `metric` is `test_cross_entropy` or `test_accuracy`
- `n` is the number of seeds included in the summary
- CI columns represent a two-sided 95% confidence interval
