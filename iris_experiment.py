"""Run Perceptron experiment on the Iris dataset."""

from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from perceptron import Perceptron, load_iris_dataset, plot_dataset, plot_errors  # type: ignore[import-untyped]


def main() -> None:
    X, y = load_iris_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Perceptron(
        n_features=2,
        learning_rate=0.1,
        epochs=50,
    )

    model.train(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(f"Test accuracy: {accuracy:.2%}")

    plot_dataset(X_test, y_test, model)
    plot_errors(model.errors_per_epoch)


if __name__ == "__main__":
    main()
