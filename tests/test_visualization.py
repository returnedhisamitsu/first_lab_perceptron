from collections.abc import Iterator

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from perceptron import Perceptron
from perceptron.visualization import plot_dataset, plot_errors


@pytest.fixture(autouse=True)
def prevent_gui(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def test_plot_dataset() -> None:
    model = Perceptron(2)
    model.w = [1.0, -1.0]
    model.bias = 0.0

    X = [[-1.0, -1.0], [1.0, 1.0]]
    y = [0, 1]

    plot_dataset(X, y, model)

    axes = plt.gca()

    assert len(axes.collections) == 2
    assert len(axes.lines) == 1


def test_plot_dataset_vertical_boundary() -> None:
    model = Perceptron(2)
    model.w = [1.0, 0.0]
    model.bias = -2.0

    X = [[-1.0, -1.0], [1.0, 1.0]]
    y = [0, 1]

    plot_dataset(X, y, model)

    assert len(plt.gca().lines) == 1


def test_plot_dataset_invalid_length() -> None:
    model = Perceptron(2)

    with pytest.raises(ValueError):
        plot_dataset([[1.0, 2.0]], [], model)


def test_plot_dataset_invalid_point() -> None:
    model = Perceptron(2)

    with pytest.raises(ValueError):
        plot_dataset([[1.0]], [0], model)


def test_plot_dataset_invalid_label() -> None:
    model = Perceptron(2)

    with pytest.raises(ValueError):
        plot_dataset([[1.0, 2.0]], [2], model)


def test_plot_dataset_zero_weights() -> None:
    model = Perceptron(2)
    model.w = [0.0, 0.0]
    model.bias = 0.0

    with pytest.raises(ValueError):
        plot_dataset([[1.0, 2.0]], [0], model)


def test_plot_errors() -> None:
    errors = [10, 5, 2, 0]

    plot_errors(errors)

    line = plt.gca().lines[0]

    assert np.asarray(line.get_xdata()).tolist() == [1, 2, 3, 4]
    assert np.asarray(line.get_ydata()).tolist() == errors


def test_plot_errors_empty() -> None:
    with pytest.raises(ValueError):
        plot_errors([])


def test_plot_errors_negative() -> None:
    with pytest.raises(ValueError):
        plot_errors([5, 3, -1])
