import numpy as np

X = np.array([1, 2])
print(X.shape)  # (1, 2)
print(X)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)  # (2, 3)
print(W)

B = 0.3

XW = np.dot(X, W)
print(XW.shape)  # (, 3)
print(XW)

Y = XW + B
print(Y.shape)   # (, 3)
print(Y)
