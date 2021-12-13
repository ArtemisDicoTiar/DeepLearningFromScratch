# Objective:
#   3차원을 2차원으로 변형후
#   긱 row 별 pooling 적용
#   reshape to original dim
import numpy as np

from book1.ch7.section4_2_im2col_data_flatten import im2col, col2im


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.x = None
        self.arg_max = None
        self.pad = pad
        self.stride = stride
        self.pool_w = pool_w
        self.pool_h = pool_h

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 3 dim to 2 dim
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # get max in each row (col -> 0, row -> 1)
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # reshape to 3 dim
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, d_out):
        d_out = d_out.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        d_max = np.zeros((d_out.size, pool_size))
        d_max[np.arange(self.arg_max.size), self.arg_max.flatten()] = d_out.flatten()
        d_max = d_max.reshape(d_out.shape + (pool_size,))

        d_col = d_max.reshape(d_max.shape[0] * d_max.shape[1] * d_max.shape[2], -1)
        d_x = col2im(d_col, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return d_x


