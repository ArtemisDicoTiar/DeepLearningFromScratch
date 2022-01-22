from dataclasses import dataclass
from typing import Union

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class DataSet:
    x: np.ndarray = None
    y: np.ndarray = None


@dataclass
class Data:
    train: Union[DataSet, DataLoader] = None
    test: Union[DataSet, DataLoader] = None


if __name__ == '__main__':
    data = Data()
