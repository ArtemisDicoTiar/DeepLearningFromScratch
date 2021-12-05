# 이전에 구한 nn에 적용된
# numerical grad 구하는 부분을
# back_prop 으로 변경


"""
for _ in range(step):
    1. mini batch
    2. grad calc
    3. param update
end for
"""
from collections import OrderedDict

import keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

from book1.ch3.section2_activation_functions import NumpyActivation
from book1.ch3.section5_output_layer import softmax_enh
from book1.ch5.section5_activation_function_layer import RELU
from book1.ch5.section6_affine_softmax_layer import Affine, SoftmaxWithLoss


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

        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = RELU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.outputLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # forward
    def loss(self, x, t):
        y = self.predict(x)
        return self.outputLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=-1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # legacy
    def numerical_grad(self, x, t):
        def loss(x, t):
            def predict(x):
                W1, W2 = self.params['W1'], self.params['W2']
                b1, b2 = self.params['b1'], self.params['b2']

                a1 = np.dot(x, W1) + b1
                z1 = NumpyActivation.sigmoid(a1)

                a2 = np.dot(z1, W2) + b2
                z2 = NumpyActivation.sigmoid(a2)

                y = softmax_enh(z2)

                return y

            y = predict(x)

            return cross_entropy_error(y, t)

        loss_W = lambda W: loss(x, t)

        grads = dict()
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # back_prop
        d_out = 1
        d_out = self.outputLayer.backward(d_out)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            d_out = layer.backward(d_out)

        grads = dict()

        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


def mini_batch():
    global loss, acc
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784) / 255
    X_test = X_test.reshape(10000, 784) / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # hyper params
    train_size = X_train.shape[0]
    batch_size = 100

    # dnn setting
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    batch_indices = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_indices]
    y_batch = y_train[batch_indices]

    # get grad
    num_grad = grad = network.numerical_grad(X_batch, y_batch)
    back_grad = network.gradient(X_batch, y_batch)

    print(f"W1: {np.average(num_grad['W1'] - back_grad['W1'])}, "
          f"b1: {np.average(num_grad['b1'] - back_grad['b1'])}, "
          f"W2: {np.average(num_grad['W2'] - back_grad['W2'])}, "
          f"b2: {np.average(num_grad['b2'] - back_grad['b2'])}")
    # diff 가 매우 작다 -> back prop으로 해도 numerical grad 만큼의 성능이 나온다는 것!


if __name__ == '__main__':
    mini_batch()
