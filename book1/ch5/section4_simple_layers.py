class Layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        raise NotImplemented

    def backward(self, d_out):
        raise NotImplemented


class MultLayer(Layer):

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, d_out):
        dx = d_out * self.y
        dy = d_out * self.x

        return dx, dy


class AddLayer(Layer):

    def forward(self, x, y):
        self.x = x
        self.y = y

        return x + y

    def backward(self, d_out):
        dx = d_out * 1
        dy = d_out * 1

        return dx, dy


def test_mult_layer():
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MultLayer()
    mul_tax_layer = MultLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    cost = mul_tax_layer.forward(apple_price, tax)

    print(cost)

    d_price = 1
    d_apple_price, d_tax = mul_tax_layer.backward(d_price)
    d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)

    print(d_apple, d_apple_num, d_tax)


def test_complicate_layer():
    apple = 100
    apple_count = 2
    orange = 150
    orange_count = 3

    tax_rate = 1.1

    mult_apple_layer = MultLayer()
    mult_orange_layer = MultLayer()
    add_price_layer = AddLayer()
    mult_tax_layer = MultLayer()

    apple_price = mult_apple_layer.forward(apple, apple_count)
    orange_price = mult_orange_layer.forward(orange, orange_count)

    total_price = add_price_layer.forward(apple_price, orange_price)

    tax_included_price = mult_tax_layer.forward(total_price, tax_rate)

    print(apple, apple_count, apple_price)
    print(orange, orange_count, orange_price)

    print(total_price, tax_rate, tax_included_price)

    d_cost = 1

    d_total_price, d_tax = mult_tax_layer.backward(d_cost)

    d_apple_price, d_orange_price = add_price_layer.backward(d_total_price)

    d_apple, d_apple_num = mult_apple_layer.backward(d_apple_price)
    d_orange, d_orange_num = mult_orange_layer.backward(d_orange_price)

    print(d_cost, d_tax, d_total_price)

    print(d_apple_price, d_apple, d_apple_num)
    print(d_orange_price, d_orange, d_orange_num)


if __name__ == '__main__':
    # test_mult_layer()
    test_complicate_layer()
