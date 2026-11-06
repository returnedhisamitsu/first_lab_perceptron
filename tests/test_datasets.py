import pytest

from perceptron import generate_dataset, load_iris_dataset


def test_generate_dataset() -> None:
    with pytest.raises(TypeError):
        generate_dataset(5.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        generate_dataset(1)

    with pytest.raises(TypeError):
        generate_dataset(100, sigma="0.5")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        generate_dataset(100, sigma=0)

    with pytest.raises(ValueError):
        generate_dataset(100, sigma=-1)

    with pytest.raises(ValueError):
        generate_dataset(100, sigma=float("inf"))

    with pytest.raises(ValueError):
        generate_dataset(100, sigma=float("nan"))

    X, y = generate_dataset(201, 42)
    assert len(X) == 201
    assert len(y) == 201
    assert y.count(0) == 100
    assert y.count(1) == 101


def test_dataset_reproducibility() -> None:
    X1, y1 = generate_dataset(200, 42)
    X2, y2 = generate_dataset(200, 42)

    assert X1 == X2
    assert y1 == y2


def test_dataset_separation() -> None:
    X, y = generate_dataset(200, 42)

    for point, target in zip(X, y):
        s = 3 * point[0] - 2 * point[1]

        if target == 0:
            assert s < 0
        else:
            assert s >= 0


def test_load_iris_dataset() -> None:
    X, y = load_iris_dataset()

    assert len(X) == 100
    assert len(y) == 100

    assert y.count(0) == 50
    assert y.count(1) == 50

    assert all(len(point) == 2 for point in X)


def test_load_iris_dataset_values() -> None:
    X, y = load_iris_dataset()

    assert all(isinstance(value, float) for point in X for value in point)

    assert set(y) == {0, 1}
