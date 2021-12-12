import numpy as np


def sect_4_1():
    x = np.random.rand(10, 1, 28, 28)
    print(x.shape)      # 10, 1, 28, 28
    print(x[0].shape)   # 1, 28, 28
    print(x[1].shape)   # 1, 28, 28

    # first data > first channel
    # x[0, 0] = x[0][0]
    print(x[0, 0].shape)      # 28, 28


if __name__ == '__main__':
    sect_4_1()
