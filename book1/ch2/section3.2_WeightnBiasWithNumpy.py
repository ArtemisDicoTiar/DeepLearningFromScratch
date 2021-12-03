"""
3.1에서 만든 형태는 직관적이지만 행렬의 형태가 더 알아보기 쉽다
바이어스를 적용해보자
"""

import numpy as np

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7

wx = np.sum(w * x)  # 0.5
print(wx)

y = wx + b  # -0.19999999999999996
print(y)

