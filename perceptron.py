class Point:
    __slots__ = ('x1', 'x2', 'target')

    def __init__(self, x1, x2, target):
        self.x1 = x1
        self.x2 = x2
        self.target = target


class Perceptron:

    def __init__(self, learning_rate=0.1):
        self.w1 = 0.0
        self.w2 = 0.0
        self.b = 0.0
        self.learning_rate = learning_rate

    def predict(self, point):
        z = self.w1 * point.x1 + self.w2 * point.x2 + self.b
        if z > 0:
            return 1
        else:
            return 0

    def find_error(self, point):
        return point.target - self.predict(point)

    def train(self, points: list[Point]):
        epoch = 0
        while epoch < 1000:
            mistakes = 0
            for point in points:
                error = self.find_error(point)
                if error != 0:
                    self.w1 += error * self.learning_rate * point.x1
                    self.w2 += error * self.learning_rate * point.x2
                    self.b += error * self.learning_rate
                    mistakes += 1
            if mistakes == 0:
                break
            epoch += 1

points = [
    Point(1, 0, 1),
    Point(-0.5, 0, 0),
    Point(2, 1, 1),
    Point(-2, 1, 0)
]

perceptron = Perceptron()
perceptron.train(points)
print(f"{perceptron.w1:.2f}x1 + {perceptron.w2:.2f}x2 + {perceptron.b:.2f} = 0 is our result")