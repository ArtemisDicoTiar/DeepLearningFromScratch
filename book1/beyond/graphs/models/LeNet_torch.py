from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F


class LeNet1Torch(Module):
    def __init__(self):
        super().__init__()
        self.layers = self._build_net()

    def _build_net(self):
        layers = namedtuple('layers',
                            field_names=[
                                'conv1',
                                'conv2',

                                'fc'
                            ])

        self.conv1 = layers.conv1 = nn.Conv2d(1, 4, kernel_size=(5, 5))  # 1개 이미지에서 4개 피쳐 맵으로 (5, 5) 커널 이용
        self.conv2 = layers.conv2 = nn.Conv2d(4, 12, kernel_size=(5, 5))  # 4개 피쳐 맵에서 12개 피쳐 맵으로 (5, 5) 커널 이용
        self.fc = layers.fc = nn.Linear(in_features=12 * 4 * 4, out_features=10)  # 12*4*4를 flatten한 후 10개 피쳐로 연결

        return layers

    def forward(self, x):
        x = F.avg_pool2d(  # conv1 -> relu -> 커널 2*2로 avg_pool
            F.relu(self.layers.conv1(x)),
            kernel_size=(2, 2)
        )
        x = F.avg_pool2d(  # conv2 -> relu -> 커널 2*2로 avg_pool
            F.relu(self.layers.conv2(x)),
            kernel_size=(2, 2)
        )
        x = torch.flatten(x, start_dim=1)  # fc 연결을 위해 (i, j, k)를 (i, j*k, 1)로 flatten
        x = self.layers.fc(x)  # fc 레이어 연결

        return x


class LeNet4Torch(Module):
    def __init__(self):
        super().__init__()
        self.layers = self._build_net()

    def _build_net(self):
        layers = namedtuple('layers',
                            field_names=[
                                'conv1',
                                'conv2',

                                'fc1',
                                'fc2'
                            ])

        self.conv1 = layers.conv1 = nn.Conv2d(1, 4, kernel_size=(5, 5))  # 1개 이미지에서 4개 피쳐 맵으로 (5, 5) 커널 이용
        self.conv2 = layers.conv2 = nn.Conv2d(4, 16, kernel_size=(5, 5))  # 4개 피쳐 맵에서 16개 피쳐 맵으로 (5, 5) 커널 이용
        self.fc1 = layers.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  # 16*5*5 를 120개 피쳐로 연결
        self.fc2 = layers.fc2 = nn.Linear(in_features=120, out_features=10)  # 120 -> 10 fc

        return layers

    def forward(self, x):
        x = F.avg_pool2d(
            F.relu(self.layers.conv1(x)),
            kernel_size=(2, 2)
        )
        x = F.avg_pool2d(
            F.relu(self.layers.conv2(x)),
            kernel_size=(2, 2)
        )
        x = torch.flatten(x)
        x = F.relu(self.layers.fc1(x))
        x = self.layers.fc2(x)

        return x


class LeNet5Torch(Module):
    def __init__(self):
        super().__init__()
        self.layers = self._build_net()

    def _build_net(self):
        layers = namedtuple('layers',
                            field_names=[
                                'conv1',
                                'conv2',

                                'fc1',
                                'fc2',
                                'fc3'
                            ])

        self.conv1 = layers.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))  # 1개 이미지에서 6개 피쳐 맵으로 (5, 5) 커널 이용
        self.conv2 = layers.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))  # 4개 피쳐 맵에서 16개 피쳐 맵으로 (5, 5) 커널 이용
        self.fc1 = layers.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  # 16*5*5 를 120개 피쳐로 연결
        self.fc2 = layers.fc2 = nn.Linear(in_features=120, out_features=84)  # 120 -> 84 fc
        self.fc3 = layers.fc3 = nn.Linear(in_features=84, out_features=10)  # 84 -> 10 fc

        return layers

    def forward(self, x):
        x = F.avg_pool2d(
            F.relu(self.layers.conv1(x)),
            kernel_size=(2, 2)
        )
        x = F.avg_pool2d(
            F.relu(self.layers.conv2(x)),
            kernel_size=(2, 2)
        )
        x = torch.flatten(x)
        x = F.relu(self.layers.fc1(x))
        x = F.relu(self.layers.fc2(x))
        x = self.layers.fc3(x)

        return x


if __name__ == '__main__':
    lenet = LeNet1Torch()
    lenet.train()
