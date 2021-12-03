import math

import numpy as np
from matplotlib import pyplot as plt


class SingleActivation:
    @staticmethod
    def step_function(x: float):
        return 1 if x > 0 else 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def relu(x):
        return max(0, x)


class NumpyActivation(SingleActivation):
    @staticmethod
    def step_function(x):
        y = x > 0
        return y.astype(np.int32)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)


class DrawActivation:
    @staticmethod
    def step_function(x):
        y = NumpyActivation.step_function(x)

        plt.plot(x, y)
        plt.ylim(-0.1, 1.1)

        plt.show()

    @staticmethod
    def sigmoid(x):
        y = NumpyActivation.sigmoid(x)

        plt.plot(x, y)
        plt.ylim(-0.1, 1.1)

        plt.show()

    @staticmethod
    def relu(x):
        y = NumpyActivation.relu(x)

        plt.plot(x, y)
        # plt.ylim(-0.1, 1.1)

        plt.show()


def test_step_function():
    x1 = 10
    y1 = SingleActivation.step_function(x1)
    x2 = -3
    y2 = SingleActivation.step_function(x2)

    print("Single Activation <step-function>")
    print(f"x1: {x1} -> y1: {y1}\nx2: {x2} -> y2: {y2}")

    x = np.array([-1.0, 1.0, 2.0])
    y = NumpyActivation.step_function(x)
    print("Numpy Activation <step-function>")
    print(f"x: {x} -> y: {y}")

    DrawActivation.step_function(np.arange(-5, 5, 0.1))


def test_sigmoid():
    x1 = 10
    y1 = SingleActivation.sigmoid(x1)
    x2 = -3
    y2 = SingleActivation.sigmoid(x2)

    print("Single Activation <sigmoid>")
    print(f"x1: {x1} -> y1: {y1}\nx2: {x2} -> y2: {y2}")

    x = np.array([-1.0, 1.0, 2.0])
    y = NumpyActivation.sigmoid(x)
    print("Numpy Activation <sigmoid>")
    print(f"x: {x} -> y: {y}")

    DrawActivation.sigmoid(np.arange(-5, 5, 0.1))


def test_relu():
    x1 = 10
    y1 = SingleActivation.relu(x1)
    x2 = -3
    y2 = SingleActivation.relu(x2)

    print("Single Activation <relu()>")
    print(f"x1: {x1} -> y1: {y1}\nx2: {x2} -> y2: {y2}")

    x = np.array([-1.0, 1.0, 2.0])
    y = NumpyActivation.relu(x)
    print("Numpy Activation <relu()>")
    print(f"x: {x} -> y: {y}")

    DrawActivation.relu(np.arange(-5, 5, 0.1))


if __name__ == '__main__':
    test_step_function()
    test_sigmoid()
    test_relu()
