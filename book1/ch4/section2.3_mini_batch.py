import keras.utils
import numpy as np
from keras.datasets import mnist


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784) / 255
    X_test = X_test.reshape(10000, 784) / 255

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]

    delta = 1e-7
    # 이건 one hot encoding 이 되어 있을 때
    return - np.sum(t * np.log(y + delta)) / batch_size

    # 이건 one hot encoding 이 안되어 있을 때
    # return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = get_data()

    train_size = X_train.shape[0]
    batch_size = 10

    batch_indices = np.random.choice(train_size, batch_size)

    x_batch = X_train[batch_indices]
    y_batch = y_train[batch_indices]

    aug_ans = np.random.uniform(size=y_batch.size)

    print(cross_entropy_error(aug_ans, y_batch))
