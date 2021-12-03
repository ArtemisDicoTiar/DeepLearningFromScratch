import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])

    # weight
    w = np.array([0.5, 0.5])
    # bias
    b = -0.7

    y = np.sum(w * x) + b
    return 0 if y <= 0 else 1


def NAND(x1, x2):
    x = np.array([x1, x2])

    # weight
    w = np.array([-0.5, -0.5])
    # bias
    b = +0.7
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


if __name__ == '__main__':
    print("======= AND =======")
    print(AND(0, 0))  # -> 0
    print(AND(0, 1))  # -> 0
    print(AND(1, 0))  # -> 0
    print(AND(1, 1))  # -> 1
    print()
    # 행렬을 적용한 형태도 잘 작동함!

    print("======= NAND =======")
    print(NAND(0, 0))  # -> 1
    print(NAND(0, 1))  # -> 1
    print(NAND(1, 0))  # -> 1
    print(NAND(1, 1))  # -> 0
    print()
    # nand도 잘 작동! (and의 정 반대 작동)

    print("======= OR =======")
    print(OR(0, 0))  # -> 0
    print(OR(0, 1))  # -> 1
    print(OR(1, 0))  # -> 1
    print(OR(1, 1))  # -> 1
    print()
    # or도 잘 작동!

