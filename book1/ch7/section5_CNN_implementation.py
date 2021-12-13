"""
Overall network:
    input: mnist hand written number image
        shape: N, 1, 28, 28
            N: Batch size
            1: channel size
            28, 28: height, width

    hidden layer / network:
        ## network init weight distribution -> param

        Conv
            # of filters
            size of filter
            stride
            pad
        ReLU
        Pooling

        Affine
            Conv block result size -> hidden size
        ReLU

        Affine
            hidden size -> output size
            output size: 10
        Softmax

    output: classified number in 0~9
"""
import pickle
from collections import OrderedDict

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from matplotlib import pyplot as plt

from book1.ch5.section5_activation_function_layer import RELU
from book1.ch5.section6_affine_softmax_layer import Affine, SoftmaxWithLoss
from book1.ch7.section4_3_conv_layer_implementation import Convolution
from book1.ch7.section4_4_pooling_layer_implementation import Pooling
from book1.common.trainer import Trainer


class SimpleConvNet:
    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv_param=None,
                 hidden_size=100,
                 output_size=10,
                 weight_init_std=0.01
                 ):
        self.weight_init_std = weight_init_std
        self.output_size = output_size
        self.hidden_size = hidden_size

        if conv_param is None:
            conv_param = {
                'filter_num': 30,
                'filter_size': 5,
                'stride': 1,
                'pad': 0,
            }

        self.filter_num = conv_param['filter_num']
        self.filter_size = conv_param['filter_size']
        self.filter_pad = conv_param['pad']
        self.filter_stride = conv_param['stride']

        self.channel_size = input_dim[0]
        self.input_size = input_dim[1]

        self.conv_output_size = (self.input_size - self.filter_size + 2 * self.filter_pad) / self.filter_stride + 1
        self.pool_output_size = int(self.filter_num * (self.conv_output_size / 2) * (self.conv_output_size / 2))

        # ========== params ========== #
        self.params = self._init_params()

        # ========== layers ========== #
        self.layers, self.output_layer = self._init_layers()

    def _init_params(self):
        params = dict()

        # bias는 초기에 없다고 설정
        params['W1'] = self.weight_init_std * np.random.randn(self.filter_num, self.channel_size, self.filter_size,
                                                              self.filter_size)
        params['b1'] = np.zeros(self.filter_num)

        params['W2'] = self.weight_init_std * np.random.randn(self.pool_output_size, self.hidden_size)
        params['b2'] = np.zeros(self.hidden_size)

        params['W3'] = self.weight_init_std * np.random.randn(self.hidden_size, self.output_size)
        params['b3'] = np.zeros(self.output_size)

        return params

    def _init_layers(self):
        layers = OrderedDict()

        layers['Conv1'] = Convolution(W=self.params['W1'], b=self.params['b1'],
                                      stride=self.filter_stride, pad=self.filter_pad)
        layers['Relu1'] = RELU()
        layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        layers['Affine1'] = Affine(W=self.params['W2'], b=self.params['b2'])
        layers['Relu2'] = RELU()

        layers['Affine2'] = Affine(W=self.params['W3'], b=self.params['b3'])

        output_layer = SoftmaxWithLoss()

        return layers, output_layer

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.output_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        d_out = 1
        d_out = self.output_layer.backward(d_out)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_out = layer.backward(d_out)

        grads = dict()

        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db

        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db

        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


def train_convnet():
    def _get_data(flatten: bool = False, one_hot_encode: bool = True):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

        if flatten:
            X_train = X_train.reshape(60000, 784) / 255
            X_test = X_test.reshape(10000, 784) / 255

        if one_hot_encode:
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

        return (X_train, y_train), (X_test, y_test)

    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = _get_data(flatten=False)

    # 시간이 오래 걸릴 경우 데이터를 줄인다.
    # x_train, t_train = x_train[:5000], t_train[:5000]
    # x_test, t_test = x_test[:1000], t_test[:1000]

    max_epochs = 20

    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=10, weight_init_std=0.01)

    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    # 매개변수 보존
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    # 그래프 그리기
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    train_convnet()
