"""
for _ in range(step):
    1. mini batch
    2. grad calc
    3. param update
end for
"""
import keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

from book1.ch3.section2_activation_functions import NumpyActivation
from book1.ch3.section5_output_layer import softmax_enh


def cross_entropy_error(pred, ans):
    # pred값이 0이면 로그에 0을 넣게 되어 -inf발생
    # 이걸 막기 위해 적당히 작은 값인 델타를 pred에 더해줌.
    delta = 1e-7
    return - np.sum(ans * np.log(pred + delta))


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


class TwoLayerNet:
    def __init__(self,
                 input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.params = dict()

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = NumpyActivation.sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        z2 = NumpyActivation.sigmoid(a2)

        y = softmax_enh(z2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=-1)
        t = np.argmax(t, axis=-1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_grad(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = dict()
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


def mini_batch():
    global loss, acc
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784) / 255
    X_test = X_test.reshape(10000, 784) / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    loss_history = list()
    accuracy_history = list()

    # hyper params
    mini_epochs = 20
    train_size = X_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    # dnn setting
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    print("====== Training Started ======")
    for _ in range(mini_epochs):
        # get batch
        batch_indices = np.random.choice(train_size, batch_size)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        # get grad
        grad = network.numerical_grad(X_batch, y_batch)

        # params update
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # learning history record
        loss = network.loss(X_batch, y_batch)
        loss_history.append(loss)

        acc = network.accuracy(X_batch, y_batch)
        accuracy_history.append(acc)

        print(".", end='')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(loss_history)
    plt.show()

    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.plot(accuracy_history)
    plt.show()


if __name__ == '__main__':
    mini_batch()

