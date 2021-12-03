# 다중 퍼셉트론을 써보자
import numpy as np


def NOT(x):
    return 1 if x == 0 else 0


def AND(x1, x2):
    x = np.array([x1, x2])

    # weight
    w = np.array([0.5, 0.5])
    # bias
    b = -0.7
    # weight & bias 의 sign 만 변경

    y = np.sum(w * x) + b
    return 0 if y <= 0 else 1


def OR(x1, x2):
    x = np.array([x1, x2])

    # weight
    w = np.array([0.5, 0.5])
    # bias
    b = -0.2
    # weight & bias 의 sign 만 변경

    y = np.sum(w * x) + b
    return 0 if y <= 0 else 1


def XOR(x1, x2):
    """

    :param x1:
    :param x2:
    :return:
    00 -> 0
    01 -> 1
    10 -> 1
    11 -> 0
    """
    # x1 XOR x2 = x1 * x2' + x1' * x2
    # single layer perceptron으로 설명하지 못하는 것들을 다중으로 쌓게 되면 가능하게 할 수 있다.

    s1 = AND(x1, NOT(x2))
    s2 = AND(NOT(x1), x2)

    y = OR(s1, s2)

    # print(f"x1: {x1}, x2: {x2}, s1: {s1}, s2: {s2}, y: {y}")

    return 0 if y <= 0 else 1


if __name__ == '__main__':
    print(XOR(0, 0))  # -> 0
    print(XOR(0, 1))  # -> 0
    print(XOR(1, 0))  # -> 0
    print(XOR(1, 1))  # -> 1
