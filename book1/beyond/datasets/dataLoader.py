from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

from book1.beyond.utils.data import Data, DataSet


def get_mnist_handwritten(normalise: bool = True,
                          categorise: bool = True,
                          as_data: bool = False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if normalise:
        X_train = X_train.reshape(60000, 784) / 255
        X_test = X_test.reshape(10000, 784) / 255

    if categorise:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    if as_data:
        return Data(
            train=DataSet(x=X_train, y=y_train),
            test=DataSet(x=X_test, y=y_test)
        )

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = get_mnist_handwritten()
    print(X_test)
