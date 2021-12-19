from dataclasses import dataclass

import numpy as np


@dataclass
class DataSet:
    x: np.ndarray = None
    y: np.ndarray = None


@dataclass
class Data:
    train: DataSet = None
    test: DataSet = None


if __name__ == '__main__':
    data = Data()
