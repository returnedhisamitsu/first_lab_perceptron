"""This module defines the Perceptron class, which can be trained
on a dataset and evaluated using accuracy."""


class Perceptron:
    """A simple binary classification model."""

    def __init__(self, n_features: int, learning_rate: float = 0.1, epochs: int = 1000):
        """Initialize the perceptron with the given parameters."""
        if not isinstance(n_features, int):
            raise TypeError(
                "number of features must be integer"
            )
        if not isinstance(epochs, int):
            raise TypeError(
                "epochs must be an integer"
            )
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                "learning rate must be an integer or float"
            )
        if n_features <= 0:
            raise ValueError(
                "number of features must be positive"
            )
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive"
            )
        if learning_rate <= 0.0:
            raise ValueError(
                "learning rate must be positive"
            )

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = [0.0] * n_features
        self.bias = 0.0
        self.errors_per_epoch: list[int] = []

    def predict(self, point: list[float]) -> int:
        """Predict the class of a sample using the current weights and bias."""
        if len(point) != self.n_features:
            raise ValueError(
                "Number of point features must match n_features in constructor"
            )
        z = self.bias
        for i in range(self.n_features):
            z += self.w[i] * point[i]
        return Perceptron.activate(z)

    @staticmethod
    def activate(z: float) -> int:
        """Convert the linear output into a binary class label."""
        if z >= 0:
            return 1
        else:
            return 0

    def find_error(self, point: list[float], target: int) -> int:
        """Calculate the difference between the target and predicted labels."""
        return target - self.predict(point)

    def train(self, X: list[list[float]], y: list[int]) -> None:
        """Train the perceptron using the provided labeled dataset."""
        if not X:
            raise ValueError(
                "Can't train perceptron without points"
            )
        if not y:
            raise ValueError(
                "Can't train perceptron without targets"
            )
        if len(X) != len(y):
            raise ValueError(
                "Number of points must match number of targets"
            )
        for target in y:
            if target not in {0, 1}:
                raise ValueError(
                    "Target values must be either 0 or 1"
                )
        for point in X:
            if len(point) != self.n_features:
                raise ValueError(
                    "Number of point features must match n_features in constructor"
                )

        self.errors_per_epoch = []
        for _ in range(self.epochs):
            mistakes = 0
            for point, target in zip(X, y):
                e = self.find_error(point, target)
                if e != 0:
                    mistakes += 1
                    for i, feature in enumerate(point):
                        self.w[i] += self.learning_rate * e * feature
                    self.bias += e * self.learning_rate
            self.errors_per_epoch.append(mistakes)
            if mistakes == 0:
                break

    def score(self, X: list[list[float]], y: list[int]) -> float:
        """Calculate the accuracy of the model on the given dataset."""
        if not X:
            raise ValueError(
                "Cannot calculate accuracy on an empty dataset"
            )
        if not y:
            raise ValueError(
                "Can't calculate score of current model without targets"
            )
        if len(X) != len(y):
            raise ValueError(
                "Number of points must match number of targets"
            )
        for target in y:
            if target not in {0, 1}:
                raise ValueError(
                    "Target values must be either 0 or 1"
                )
        for point in X:
            if len(point) != self.n_features:
                raise ValueError(
                    "Number of point features must match n_features in constructor"
                )
        correct = 0
        number_of_tests = len(X)
        for point, target in zip(X, y):
            if self.predict(point) == target:
                correct += 1
        return correct / number_of_tests
