import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.datasets import mnist

from book1.ch3.section2_activation_functions import NumpyActivation
from book1.ch3.section5_output_layer import softmax_enh


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784) / 255
    X_test = X_test.reshape(10000, 784) / 255

    return X_test, y_test


def init_network():
    with open("./sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']

    A1 = np.dot(x, W1) + B1
    Z1 = NumpyActivation.sigmoid(A1)

    A2 = np.dot(Z1, W2) + B2
    Z2 = NumpyActivation.sigmoid(A2)

    A3 = np.dot(Z2, W3) + B3
    y = softmax_enh(A3)

    return y


def infer_without_batch():
    x, ans = get_data()
    network = init_network()

    acc_cnt = 0
    accs = list()
    for i in range(len(x)):
        y = predict(network, x[i])
        pred = np.argmax(y)

        if pred == ans[i]:
            acc_cnt += 1
        accs.append(100 * acc_cnt / (i + 1))

        print(f"current accuracy: {100 * acc_cnt / (i + 1)} ({i + 1}/{len(x)})")

    print(f"Accuracy: {acc_cnt / len(x) * 100} (Final)")

    plt.plot(accs)
    plt.show()


def infer_with_batch():
    x, ans = get_data()
    network = init_network()

    batch_size = 1000

    acc_cnt = 0
    accs = list()
    for i in range(0, len(x), batch_size):
        y = predict(network, x[i: i + batch_size])
        pred = np.argmax(y, axis=1)

        acc_cnt += np.sum(pred == ans[i: i + batch_size])

        accs.append(100 * acc_cnt / (i + batch_size))

        print(f"current accuracy: {100 * acc_cnt / (i + batch_size)} ({(i + batch_size)}/{len(x)})")

    print(f"Accuracy: {acc_cnt / len(x) * 100} (Final)")

    plt.plot(accs)
    plt.show()


if __name__ == '__main__':
    # infer_without_batch()
    infer_with_batch()
