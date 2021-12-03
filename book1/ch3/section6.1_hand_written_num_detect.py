import numpy as np
from PIL import Image
from keras.datasets import mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

print(X_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000,)

print(X_test.shape)     # (10000, 28, 28)
print(y_test.shape)     # (10000,)

print()

tmp_img = X_train[0]
tmp_label = y_train[0]
print(tmp_label)

print(tmp_img.shape)
tmp_img = tmp_img.reshape(28, 28)
print(tmp_img.shape)

img_show(tmp_img)
