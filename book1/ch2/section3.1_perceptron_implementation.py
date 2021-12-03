def AND(x1: int, x2: int):
    # weight and critical point
    w1, w2, theta = 0.5, 0.5, 0.7

    y = x1 * w1 + x2 * w2
    return 0 if y <= theta else 1


if __name__ == '__main__':
    print(AND(0, 0))    # -> 0
    print(AND(0, 1))    # -> 0
    print(AND(1, 0))    # -> 0
    print(AND(1, 1))    # -> 1
    # AND 게이트 가 생각한 대로 잘 작동해 줌!
