import matplotlib.pyplot as plt
import numpy as np


def numerical_diff_1(f, x):
    # 일반적인 미분 정의
    h = 10e-50
    return (f(x + h) - f(x)) / h


def numerical_diff_2(f, x, which=None):
    # 중심 차분, 중앙 차분
    h = 1e-4
    if which is not None:
        delta_plus = x[which] + h
        delta_minus = x[which] - h
        xs_plus = x.copy()
        xs_minus = x.copy()

        xs_plus[which] = delta_plus
        xs_minus[which] = delta_minus
        return (f(xs_plus) - f(xs_minus)) / (2 * h)

    return (f(x + h) - f(x - h)) / (2 * h)


def func_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def func_2(x):
    return x[0] ** 2 + x[1] ** 2


def round_func_2_by_x0(x):
    return 2 * x[0] + x[1] ** 2


def round_func_2_by_x1(x):
    return x[0] ** 2 + 2 * x[1]


def plot_2d_func(f):
    x = np.arange(0.0, 20.0, 0.1)
    y = f(x)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(x, y)
    plt.show()


def plot_3d_func(f):
    x0 = np.arange(0.0, 20.0, 0.1)
    x1 = np.arange(0.0, 20.0, 0.1)
    y = f([x0, x1])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot([x0, x1], y)
    plt.show()


def test_func_1():
    plot_2d_func(func_1)
    print(numerical_diff_2(func_1, 5))
    print(numerical_diff_2(func_1, 10))


def test_func_2():
    # plot_3d_func(func_2)
    print(numerical_diff_2(func_2, [3.0, 4.0], which=0))
    print(numerical_diff_2(func_2, [3.0, 4.0], which=1))


if __name__ == '__main__':
    test_func_1()
    test_func_2()
