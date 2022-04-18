import numpy as np

from book2.ch2.section3_1_preprocess_corpus import preprocess
from book2.ch3.section3_1_context_n_target import create_contexts_target
from book2.ch3.section3_2_onehot_context import convert2onehot
from book2.utils.layers import MatMul, SoftmaxWithLoss
from book2.utils.optimizer import Adam
from book2.utils.trainer import Trainer


class SimpleCBOW:
    def __init__(self, vocab_size: int, hidden_size: int):
        V, H = vocab_size, hidden_size

        # weights init
        # 왜 0.01을 곱해줬지?
        # random.randn 은 정규분포를 따른다. (X ~ N(0, 1))
        # sigma * randn + mu = N(mu, sigma^2)
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
        # 이러면 전체 범위가 0 ~ 40이 된다.
        # 어쩌면 np에서의 설명이 잘 못 된거 일지도...?
        # 그냥 X~N(0,1) 의 값에 0.01 곱해서 줄여놓았다고 봐야할듯.
        # 그래서 왜? 곱한 건데?
        # imageNet 논문에서 N(0,1) 노이즈를 썼더니 좋더라 그래서 인듯
        # https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
        W_in = 0.01 * np.random.randn(V, H).astype(float)
        W_out = 0.01 * np.random.randn(H, V).astype(float)

        # layers init
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # collecting
        layers = [self.in_layer0, self.in_layer1, self.out_layer]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # distribution of words
        self.word2vec = W_in

    def forward(self, contexts, target):
        """
        forward the CBOW model
        :param contexts:
            3 dim np array
            (vocab_size, window_size, sentence_length)
        :param target:
            2 dim np array
            (vocab_siz, sentence_length)
        :return:
        """
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5

        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None


if __name__ == '__main__':
    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    text = "You say goodbye and I say hello."
    corpus, word2id, id2word = preprocess(text)

    vocab_size = len(word2id)

    contexts, target = create_contexts_target(corpus, window_size)
    contexts = convert2onehot(contexts, len(word2id))
    target = convert2onehot(target, len(word2id))

    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, target, max_epoch, batch_size)

    trainer.plot()

    word2vec = model.word2vec
    for word_id, word in id2word.items():
        print(word, word2vec[word_id])
