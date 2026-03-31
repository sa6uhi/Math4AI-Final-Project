import argparse

from src.run_experiment import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Math4AI synthetic experiment.")
    parser.add_argument("--dataset", choices=["moons", "linear_gaussian", "digits"], required=True)
    parser.add_argument("--model", choices=["softmax", "hidden_layer"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-seeds", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(seed=args.seed)

    if args.repeat_seeds:
        seeds = [int(item.strip()) for item in args.repeat_seeds.split(",") if item.strip()]
        runner.run_repeated_seeds(dataset=args.dataset, model_type=args.model, seeds=seeds)
        return

    runner.run(dataset=args.dataset, model_type=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
