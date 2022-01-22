import numpy as np
import torch
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from torch.utils.data import TensorDataset, DataLoader

from book1.beyond.utils.data import Data, DataSet


def get_mnist_handwritten(flatten: bool = False,
                          normalise: bool = True,
                          categorise: bool = True,
                          as_data: bool = False,
                          as_dataloader: bool = False,
                          batch_size: int = 32):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape((60000, 1, 28, 28))
    X_test = X_test.reshape((10000, 1, 28, 28))

    if flatten:
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

    if normalise:
        X_train = X_train / 255
        X_test = X_test / 255

    if categorise:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    if as_data:
        return Data(
            train=DataSet(x=X_train, y=y_train),
            test=DataSet(x=X_test, y=y_test)
        )
    if as_dataloader:
        train_dataloader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)),
                                      batch_size=batch_size)
        test_dataloader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)),
                                     batch_size=batch_size)

        return Data(
            train=train_dataloader,
            test=test_dataloader
        )

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = get_mnist_handwritten()
    print(X_test)
