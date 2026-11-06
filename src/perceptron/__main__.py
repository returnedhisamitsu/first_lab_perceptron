"""Command-line interface for Perceptron."""

import argparse

from perceptron import Perceptron, generate_dataset, plot_dataset, plot_errors


def create_parser() -> argparse.ArgumentParser:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train a perceptron on a synthetic dataset."
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to generate.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    return parser


def main() -> None:
    """Run the perceptron training pipeline."""

    parser = create_parser()
    args = parser.parse_args()

    try:
        X, y = generate_dataset(args.samples)

        model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)

        model.train(X, y)

        accuracy = model.score(X, y)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"Accuracy: {accuracy:.2%}")

    plot_dataset(X, y, model)
    plot_errors(model.errors_per_epoch)


if __name__ == "__main__":
    main()
