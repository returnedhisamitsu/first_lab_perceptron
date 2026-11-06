import math

import matplotlib.pyplot as plt

from perceptron import Perceptron


def plot_dataset(X: list[list[float]], y: list[int], model: Perceptron) -> None:
    """Plot a dataset with class labels and the model's decision boundary."""
    if not X:
        raise ValueError("Dataset must not be empty")

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    for point in X:
        if not isinstance(point, list):
            raise TypeError("Each point must be a list")

        if len(point) != 2:
            raise ValueError("Each point must contain exactly 2 coordinates")

        for coordinate in point:
            if type(coordinate) not in (int, float):
                raise TypeError("Coordinates must be numbers")

            if not math.isfinite(coordinate):
                raise ValueError("Coordinates must be finite")

    for target in y:
        if type(target) is not int:
            raise TypeError("Class labels must be integers")

        if target not in (0, 1):
            raise ValueError("Class labels must be 0 or 1")

    if len(model.w) != 2:
        raise ValueError("Model must have exactly 2 weights")

    for weight in model.w:
        if not math.isfinite(weight):
            raise ValueError("Weights must be finite")

    if not math.isfinite(model.bias):
        raise ValueError("Bias must be finite")

    if model.w[0] == 0 and model.w[1] == 0:
        raise ValueError("Both weights cannot be zero")

    class_0 = [point for point, target in zip(X, y) if target == 0]
    class_1 = [point for point, target in zip(X, y) if target == 1]

    plt.scatter(
        [point[0] for point in class_0],
        [point[1] for point in class_0],
        label="Class 0",
    )
    plt.scatter(
        [point[0] for point in class_1],
        [point[1] for point in class_1],
        label="Class 1",
    )

    x_min, x_max = plt.xlim()
    x_values = [x_min, x_max]

    w0, w1 = model.w
    if w1 != 0:
        y_values = [-(w0 * x + model.bias) / w1 for x in x_values]

        plt.plot(x_values, y_values, label="Decision boundary")
    elif w0 != 0:
        x_boundary = -model.bias / w0

        plt.axvline(x_boundary, label="Decision boundary")
    else:
        raise ValueError("Cannot plot decision boundary: both weights are zero.")

    plt.xlim(x_min, x_max)

    plt.xlabel("Ox")
    plt.ylabel("Oy")
    plt.title("Perceptron dataset")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_errors(errors_per_epoch: list[int]) -> None:
    """Plot the number of errors during each training epoch."""
    if not errors_per_epoch:
        raise ValueError("Errors list must not be empty")

    for error in errors_per_epoch:
        if type(error) is not int:
            raise TypeError("Errors must be integers")

        if error < 0:
            raise ValueError("Errors cannot be negative")

    epochs = list(range(1, len(errors_per_epoch) + 1))
    plt.plot(epochs, errors_per_epoch, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Number of errors")
    plt.title("Training errors per epoch")
    plt.grid(True)
    plt.show()
