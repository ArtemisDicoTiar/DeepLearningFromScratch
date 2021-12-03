# 일반적으로 회귀에서는 항등함수(identity function)을
#         분류에서는 소프트맥스(softmax)를 사용한다.
import numpy as np


def softmax_eq(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmax_enh(x):
    # 위 소프트 맥스로 계산하면 오버플로우 문제가 발생한다.
    # exp(x) 태울때 e^1000을 넘게 되면 inf가 발생한다.

    # solution
    # 임의의 수 C를 분모, 분자에 모두 곱한다
    # 그 곱한 C를 exp 안으로 넣는다
    # 그러면 log_C가 안으로 들어가지고 그 값을 C'이라 하자
    # 그 C'이 전체 인풋의 최대값의 마이너스를 취하면 exp 해서 inf로 발산하는 걸 막을 수 있다
    C_prime = - np.max(x)
    return np.exp(x + C_prime) / np.sum(np.exp(x + C_prime))


if __name__ == '__main__':
    # 오 다행인게 오버플로우 발생하면 오버플로우 떳다고 알려주네

    # RuntimeWarning: overflow encountered in exp
    #   return np.exp(x) / np.sum(np.exp(x))

    # RuntimeWarning: invalid value encountered in true_divide
    #   return np.exp(x) / np.sum(np.exp(x))

    a = np.array([0.3, 2.9, 4.0, 10000000, -100000300])
    y_eq = softmax_eq(a)
    y_enh = softmax_enh(a)

    print(y_eq, np.sum(y_eq))
    print(y_enh, np.sum(y_enh))

