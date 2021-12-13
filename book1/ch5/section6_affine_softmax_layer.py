import numpy as np

from book1.ch3.section5_output_layer import softmax_enh


class Affine:
    def __init__(self, W, b):
        self.original_x_shape = None
        self.W = W
        self.b = b

        self.x = None

        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, d_out):
        # d_out: (N, B.shape)
        dx = np.dot(d_out, self.W.T)
        self.dW = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis=0)  # d_out shape -> B.shape

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    @staticmethod
    def _softmax(x):
        return softmax_enh(x)

    @staticmethod
    def _cross_entropy_error(y, t):
        delta = 1e-7
        return - np.sum(t * np.log(y + delta))

    def forward(self, x, t):
        self.y = self._softmax(x)
        self.t = t
        self.loss = self._cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, d_out=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
