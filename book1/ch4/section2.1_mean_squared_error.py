import numpy as np


def sum_of_squared_error(pred, ans):
    return np.sum((pred - ans) ** 2) / 2


def mean_squared_error(pred, ans):
    return np.sum((pred - ans) ** 2) / len(pred)


y_prob = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

print(sum_of_squared_error(y_prob, t))
print(mean_squared_error(y_prob, t))
