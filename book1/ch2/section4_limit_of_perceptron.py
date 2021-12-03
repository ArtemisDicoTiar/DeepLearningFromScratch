# XOR을 만들어보자!
import numpy as np


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
    # 기존의 구분 0, 1 을 구분해주는 함수는 직선의 형태로 나온다
    # 하지만 이번에 결과 0, 1을 구분해주는 함수는 절대 단일 직선 함수로는 안된다.
    # 비선형으로 구분하면 되지 않나?
    # -> 다중 퍼셉트론을 써보자!

    x = np.array([x1, x2])
    w = np.array([...])
    b = np.array([...])

    y = np.sum(x * w) + b

    return 0 if y <= 0 else 1


if __name__ == '__main__':
    print(XOR(0, 0))  # -> 0
    print(XOR(0, 1))  # -> 0
    print(XOR(1, 0))  # -> 0
    print(XOR(1, 1))  # -> 1

