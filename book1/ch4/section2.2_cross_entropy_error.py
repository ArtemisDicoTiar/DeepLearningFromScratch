import numpy as np


def cross_entropy_error(pred, ans):
    # pred값이 0이면 로그에 0을 넣게 되어 -inf발생
    # 이걸 막기 위해 적당히 작은 값인 델타를 pred에 더해줌.
    delta = 1e-7
    return - np.sum(ans * np.log(pred + delta))


y_prob1 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
y_prob2 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

print(cross_entropy_error(y_prob1, t))
print(cross_entropy_error(y_prob2, t))
