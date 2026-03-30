from src.run_experiment import ExperimentRunner


def main() -> None:
    runner = ExperimentRunner(seed=42)
    runs = [
        ("linear_gaussian", "softmax"),
        ("linear_gaussian", "hidden_layer"),
        ("moons", "hidden_layer"),
        ("moons", "softmax"),
    ]
    for dataset, model_type in runs:
        runner.run(dataset=dataset, model_type=model_type)


if __name__ == "__main__":
    main()
