import argparse

from src.run_experiment import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Math4AI synthetic experiment.")
    parser.add_argument("--dataset", choices=["moons", "linear_gaussian"], required=True)
    parser.add_argument("--model", choices=["softmax", "hidden_layer"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(seed=42)
    runner.run(dataset=args.dataset, model_type=args.model)


if __name__ == "__main__":
    main()
