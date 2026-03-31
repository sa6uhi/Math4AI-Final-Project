from src.config import DEFAULT_BATCH_RUNS, DEFAULT_SEED
from src.run_experiment import ExperimentRunner


def main() -> None:
    runner = ExperimentRunner(seed=DEFAULT_SEED)
    for dataset, model_type in DEFAULT_BATCH_RUNS:
        runner.run(dataset=dataset, model_type=model_type, seed=DEFAULT_SEED)


if __name__ == "__main__":
    main()
