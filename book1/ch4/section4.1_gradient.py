import numpy as np


def func_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


if __name__ == '__main__':
    print(numerical_gradient(func_2, np.array([3.0, 4.0])))
    print(numerical_gradient(func_2, np.array([0.0, 2.0])))
    print(numerical_gradient(func_2, np.array([3.0, 0.0])))

    print(gradient_descent(func_2, init_x=np.array([-3.0, 4.0]), lr=0.0001, step_num=100))
