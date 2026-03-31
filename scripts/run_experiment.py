import argparse

from src.config import (
    CLI_RUN_DESCRIPTION,
    DATASET_CHOICES,
    DEFAULT_SEED,
    MODEL_CHOICES,
    REPEAT_SEEDS_DEFAULT,
    parse_seed_list,
)
from src.run_experiment import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CLI_RUN_DESCRIPTION)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeat-seeds", type=str, default=REPEAT_SEEDS_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(seed=args.seed)

    if args.repeat_seeds:
        seeds = parse_seed_list(args.repeat_seeds)
        runner.run_repeated_seeds(dataset=args.dataset, model_type=args.model, seeds=seeds)
        return

    runner.run(dataset=args.dataset, model_type=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
