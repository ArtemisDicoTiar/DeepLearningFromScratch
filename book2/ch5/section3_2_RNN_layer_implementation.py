"""
이 블럭의 인풋은 Xs, 아웃풋은 Hs이다.
"""
import numpy as np

from book2.ch5.section3_1_RNN_implementation import RNN


class RNNBlock:
    def __init__(self, Wx, Wh, b, stateful=False):
        """
        RNN block (partial RNN layer)
        :param Wx:
        :param Wh:
        :param b:
        :param stateful: output hidden vector should be held?
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        # this hidden vector is given from previous block
        # as Truncated RNN is applied for this block
        # to reduce the complexity of computing
        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, Xs):
        Wx, Wh, b = self.params
        N, T, D = Xs.shape  # (N: batch_size, T: time_size, D: input_size)
        D, H = Wx.shape

        self.layers = []
        Hs = np.empty((N, T, H), dtype=float)

        if not self.stateful or self.h is None:
            # 상태가 저장으로 설정되지 않은 레이어거나
            # 상태가 저장되어 있지 않으면
            self.h = np.zeros((N, H), dtype=float)

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(Xs[:, t, :], self.h)
            Hs[:, t, :] = self.h
            self.layers.append(layer)

        return Hs

    def backward(self, dHs):
        Wx, Wh, b = self.params
        N, T, H = dHs.shape
        D, H = Wx.shape

        dXs = np.empty((N, T, D), dtype=float)
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 이전 cell에서 나오는 dh를 더해 나가야함.
            # 왜냐면 다음 셀에서 들어오는 거 하나
            # t 번째 dhs 하나
            dx, dh = layer.backward(dHs[:, t, :] + dh)
            dXs[:, t, :] = dx

            # block에서 관리하는 grads는 전체 셀에서의 grad를 다 더해야함.
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        # 이걸 왜... 하는 거지?
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dXs

