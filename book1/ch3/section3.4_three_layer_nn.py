import numpy as np

from book1.ch3.section2_activation_functions import NumpyActivation


def identity_func(x):
    return x


# ========== Layer 1 ========== #
X = np.array([1.0, 0.5])

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)  # (1, 2)
print(W1.shape)  # (2, 3)
print(B1.shape)  # (1, 3)

A1 = np.dot(X, W1) + B1
print(A1.shape)  # (1, 3)
print(A1)

# logits
Z1 = NumpyActivation.sigmoid(A1)
print(Z1.shape)  # (1, 3)
print(Z1)

# ========== Layer 2 ========== #
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)  # (1, 3)
print(W2.shape)  # (3, 2)
print(B2.shape)  # (1, 2)

A2 = np.dot(Z1, W2) + B2
print(A2.shape)  # (1, 2)
print(A2)

Z2 = NumpyActivation.sigmoid(A2)
print(Z2.shape)  # (1, 2)
print(Z2)

# ========== Layer 3 ========== #
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

print(Z2.shape)  # (1, 2)
print(W3.shape)  # (2, 2)
print(B3.shape)  # (1, 2)

A3 = np.dot(Z2, W3) + B3
print(A3.shape)  # (1, 2)
print(A3)

Y = identity_func(A3)
print(Y.shape)  # (1, 2)
print(Y)


def init_network():
    network = dict()
    network['W1'] = W1
    network['B1'] = B1
    network['W2'] = W2
    network['B2'] = B2
    network['W3'] = W3
    network['B3'] = B3

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']

    A1 = np.dot(x, W1) + B1
    Z1 = NumpyActivation.sigmoid(A1)

    A2 = np.dot(Z1, W2) + B2
    Z2 = NumpyActivation.sigmoid(A2)

    A3 = np.dot(Z2, W3) + B3
    y = identity_func(A3)

    return y


print("-------- NETWORK USED --------")
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
