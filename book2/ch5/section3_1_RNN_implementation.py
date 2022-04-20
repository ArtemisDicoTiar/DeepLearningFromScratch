from typing import List, Union

import numpy as np


class RNN:
    def __init__(self, Wx, Wh, b):
        """
        :parameter x: (N, D)
        :param Wx: (D, H)
        :parameter h_prev: (N, H)
        :param Wh: (H, H)
        :param b: (H, )
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        # t: (N, H)
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)
        self.cache = (x, h_prev, h_next)
        # h_next: (N, H)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_nex = self.cache

        # tanh.back = 1 - tanh^2
        # (N, H)
        dt = dh_next * (1 - dh_next**2)
        # (N, H) -> (H, )
        db = np.sum(dt, axis=0)
        # (H, H) = (H, ?) * (N, H)
        # therefore, ? is N. Then, left input become transpose of h_prev
        dWh = np.matmul(h_prev.T, dt)
        # (N, H) = (?, ?) * (N, H) * (?, ?)
        # to achieve same dim then (H, H) must be multiplied after
        dh_prev = np.matmul(dt, Wh.T)
        # (D, H) = (D, N) * (N, H)
        dWx = np.matmul(x.T, dt)
        # (N, D) = (?, ?) * (N, H) * (?, ?)
        # to achieve (N, D) then (H, D) must be multiplied after
        dx = np.matmul(dt, Wx.T)

        # 여기에 ...은 왜 붙었지...?
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

