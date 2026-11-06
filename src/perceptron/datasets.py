import random


def generate_dataset(
    samples: int, seed: int | None = None
) -> tuple[list[list[float]], list[int]]:
    """Generate a synthetic linearly separable binary classification dataset."""
    if type(samples) is not int:
        raise TypeError("samples must be an integer")

    if samples < 2:
        raise ValueError("samples must be at least 2")

    rng = random.Random(seed)

    class_0_samples = samples // 2
    class_1_samples = samples - class_0_samples

    dataset: list[tuple[list[float], int]] = []

    for _ in range(class_0_samples):
        while True:
            x_cord = rng.gauss(-3, 0.5)
            y_cord = rng.gauss(2, 0.5)
            s = 3 * x_cord - 2 * y_cord
            if s < 0:
                dataset.append(([x_cord, y_cord], 0))
                break

    for _ in range(class_1_samples):
        while True:
            x_cord = rng.gauss(3, 0.5)
            y_cord = rng.gauss(-2, 0.5)
            s = 3 * x_cord - 2 * y_cord
            if s >= 0:
                dataset.append(([x_cord, y_cord], 1))
                break

    rng.shuffle(dataset)
    X = [point for point, target in dataset]
    y = [target for point, target in dataset]
    return X, y
