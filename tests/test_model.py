import pytest

from perceptron import Perceptron


def test_activate() -> None:
    assert Perceptron.activate(5) == 1
    assert Perceptron.activate(-5) == 0
    assert Perceptron.activate(0) == 1


def test_predict() -> None:
    perceptron = Perceptron(2)
    perceptron.w = [1.0, -1.0]
    perceptron.bias = 0
    assert perceptron.predict([2.0, 3.0]) == 0
    assert perceptron.predict([3.0, 2.0]) == 1


def test_train() -> None:
    perceptron = Perceptron(2)

    X = [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]]
    y = [0, 0, 1, 1]

    perceptron.train(X, y)
    assert perceptron.errors_per_epoch[-1] == 0
    assert perceptron.score(X, y) == 1.0


def test_score() -> None:
    perceptron = Perceptron(1)
    perceptron.w = [1.0]
    perceptron.bias = 0.0
    weights_before = perceptron.w.copy()
    bias_before = perceptron.bias

    X = [[-2.0], [-1.0], [1.0], [2.0]]

    y = [0, 0, 1, 0]

    assert perceptron.score(X, y) == 0.75
    assert perceptron.w == weights_before
    assert perceptron.bias == bias_before


def test_constructor_validation() -> None:
    # Constructor: incorrect types
    with pytest.raises(TypeError):
        Perceptron(2.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        Perceptron(2, "rick")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        Perceptron(2, 0.1, 5.5)  # type: ignore[arg-type]

    # Constructor: incorrect values
    with pytest.raises(ValueError):
        Perceptron(-5)

    with pytest.raises(ValueError):
        Perceptron(2, 0.1, 0)

    with pytest.raises(ValueError):
        Perceptron(2, 0.0)

    with pytest.raises(ValueError):
        Perceptron(2, float("inf"))

    with pytest.raises(ValueError):
        Perceptron(2, float("nan"))


def test_train_validation() -> None:
    model = Perceptron(1)
    # Train: empty datasets
    with pytest.raises(ValueError):
        model.train([], [0])

    with pytest.raises(ValueError):
        model.train([[1.0]], [])

    # Train: mismatched dataset sizes
    with pytest.raises(ValueError):
        model.train([[1.0], [2.0]], [0])

    # Train: incorrect target type
    with pytest.raises(TypeError):
        model.train([[1.0]], ["0"])  # type: ignore[list-item]

    # Train: incorrect target value
    with pytest.raises(ValueError):
        model.train([[1.0]], [2])

    # Train: incorrect number of features
    with pytest.raises(ValueError):
        model.train([[1.0, 2.0]], [0])

    # Train: incorrect feature type
    with pytest.raises(TypeError):
        model.train([["hello"]], [0])  # type: ignore[list-item]

    # Train: non-finite feature
    with pytest.raises(ValueError):
        model.train([[float("inf")]], [0])


def test_score_validation() -> None:
    model = Perceptron(1)

    # Empty datasets
    with pytest.raises(ValueError):
        model.score([], [0])

    with pytest.raises(ValueError):
        model.score([[1.0]], [])

    # Mismatched dataset sizes
    with pytest.raises(ValueError):
        model.score([[1.0], [2.0]], [0])

    # Incorrect target type
    with pytest.raises(TypeError):
        model.score([[1.0]], ["0"])  # type: ignore[list-item]

    with pytest.raises(TypeError):
        model.score([[1.0]], [1.0])  # type: ignore[list-item]

    # Incorrect target value
    with pytest.raises(ValueError):
        model.score([[1.0]], [2])

    # Incorrect number of features
    with pytest.raises(ValueError):
        model.score([[1.0, 2.0]], [0])

    # Incorrect feature type
    with pytest.raises(TypeError):
        model.score([["hello"]], [0])  # type: ignore[list-item]

    # Non-finite feature
    with pytest.raises(ValueError):
        model.score([[float("nan")]], [0])
