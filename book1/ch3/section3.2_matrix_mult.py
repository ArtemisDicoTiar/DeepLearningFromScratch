import numpy as np

print("------------ (2, 2) X (2, 2) = (2,2) ------------")
A = np.array([[1, 2], [3, 4]])
print(A.shape)
print(A)

B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(B)

# inner product
y1 = np.dot(A, B)
print(y1)

# element-wise product
y2 = A * B
print(y2)

print("------------ (2, 3) X (3, 2) = (2,2) ------------")
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)
print(A)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)
print(B)

# inner product
y1 = np.dot(A, B)
print(y1)

# element-wise product (UN-AVAILABLE)
# y2 = A * B
# print(y2)



