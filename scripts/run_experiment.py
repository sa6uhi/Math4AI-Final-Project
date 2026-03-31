import argparse

from src.config import (
    CLI_RUN_DESCRIPTION,
    DATASET_CHOICES,
    DEFAULT_SEED,
    HIDDEN_OPTIMIZER_DEFAULT,
    MODEL_CHOICES,
    OPTIMIZER_CHOICES,
    REPEAT_SEEDS_DEFAULT,
    parse_seed_list,
)
from src.run_experiment import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CLI_RUN_DESCRIPTION)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--optimizer", choices=OPTIMIZER_CHOICES, default=HIDDEN_OPTIMIZER_DEFAULT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeat-seeds", type=str, default=REPEAT_SEEDS_DEFAULT)
    parser.add_argument("--ablate-hidden-optimizers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(seed=args.seed)

    if args.ablate_hidden_optimizers:
        runner.run_hidden_optimizer_ablation(dataset=args.dataset, seed=args.seed)
        return

    if args.repeat_seeds:
        seeds = parse_seed_list(args.repeat_seeds)
        runner.run_repeated_seeds(
            dataset=args.dataset,
            model_type=args.model,
            seeds=seeds,
            optimizer=args.optimizer,
        )
        return

    runner.run(dataset=args.dataset, model_type=args.model, seed=args.seed, optimizer=args.optimizer)


if __name__ == "__main__":
    main()
