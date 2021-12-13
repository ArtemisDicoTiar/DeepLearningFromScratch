import numpy as np

from book1.ch7.section4_2_im2col_data_flatten import im2col, col2im


def exp():
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, 1, 0)

    print(col1.shape)

    x2 = np.random.rand(10, 3, 7, 7)
    col2 = im2col(x2, 5, 5, 1, 0)

    print(col2.shape)


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # x -> col
        col = im2col(x, FH, FW, self.stride, self.pad)
        # reshape to (FN, 기존의 데이터 크기 // FN)
        col_w = self.W.reshape(FN, -1).T

        # output calc
        out = np.dot(col, col_w) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # 넘파이에서 transpose는 축의 순서 변경이다
        # transpose에 적히는 수가 기존의 축의 순서를 의미한다.

        self.x = x
        self.col = col
        self.col_W = col_w

        return out

    def backward(self, d_out):
        FN, C, FH, FW = self.W.shape
        d_out = d_out.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(d_out, axis=0)
        self.dW = np.dot(self.col.T, d_out)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        d_col = np.dot(d_out, self.col_W.T)
        dx = col2im(d_col, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


if __name__ == '__main__':
    exp()
