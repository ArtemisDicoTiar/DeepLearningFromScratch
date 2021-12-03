import numpy as np

A = np.array([1, 2, 3, 4])

print(A)
# >>> [1 2 3 4]
print(np.ndim(A))
# >>> 1
print(A.shape)
# >>> (4,)
print(A.shape[0])
# >>> 4

# x by y matrix (x: the number of rows, y: the number of columns)
B = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print(B)
# >>>
# [[1 2]
#  [3 4]
#  [5 6]]
print(np.ndim(B))
# >>> 2
print(B.shape)
# >>> (3, 2)
print(B.shape[0])
# >>> 3
