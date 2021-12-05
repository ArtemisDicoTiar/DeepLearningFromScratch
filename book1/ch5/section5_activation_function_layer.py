import numpy as np

from book1.ch5.section4_simple_layers import Layer


class RELU:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)

        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, d_out):
        d_out[self.mask] = 0
        dx = d_out
        return dx


class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, d_out):
        dx = d_out * self.out * (1 - self.out)
        return dx


def test_relu():
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(x)
    mask = (x <= 0)
    print(mask)

    relu = RELU()
    y = relu.forward(x)
    print(y)

    rL_ry = np.array([[1.0, -0.5], [-2.0, 3.0]])
    b_y = relu.backward(rL_ry)
    print(b_y)


if __name__ == '__main__':
    test_relu()
