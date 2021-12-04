import numpy as np

from book1.ch3.section5_output_layer import softmax_enh


class SimpleNet:
    @staticmethod
    def _cross_entropy_error(pred, ans):
        # pred값이 0이면 로그에 0을 넣게 되어 -inf발생
        # 이걸 막기 위해 적당히 작은 값인 델타를 pred에 더해줌.
        delta = 1e-7
        return - np.sum(ans * np.log(pred + delta))

    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax_enh(z)

        loss = self._cross_entropy_error(y, t)

        return loss


def test_simple_net():
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(x)
    print(p)
    print(np.argmax(p))

    # answer label
    t = np.array([0, 0, 1])
    loss = net.loss(x, t)
    print(loss)


def test_grad():
    def numerical_gradient(f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 값 복원
            it.iternext()

        return grad

    net = SimpleNet()

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(x)
    print(p)
    print(np.argmax(p))

    # answer label
    t = np.array([0, 0, 1])

    f = lambda w: net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW)


if __name__ == '__main__':
    # test_simple_net()
    test_grad()