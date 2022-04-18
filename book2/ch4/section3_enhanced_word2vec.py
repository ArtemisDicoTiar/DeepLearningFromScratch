import pickle
from pathlib import Path
from typing import List

import numpy as np

from book2.ch4.section2_6n7_negative_sampling import NegativeSamplingLoss
from book2.dataset import ptb
from book2.utils.layers import Embedding
from book2.utils.optimizer import Adam
from book2.utils.trainer import Trainer
from book2.utils.uitl import create_contexts_target


class CBOW:
    def __init__(self, vocab_size: int, hidden_size: int, window_size: int, corpus):
        """

        :param vocab_size:   어휘 사이즈
        :param hidden_size:  히든 벡터 사이즈
        :param window_size:  맥락 몇개?
        :param corpus:       어휘 컬렉션
        """
        V, H = vocab_size, hidden_size

        # weights
        W_in = 0.01 * np.random.randn(V, H).astype(float)
        W_out = 0.01 * np.random.randn(V, H).astype(float)

        # layers
        self.layers: List = []
        for _ in range(2 * window_size):
            # 앞뒤로 봐야하기 때문에 2배
            # 임베딩 레이어를 윈도우 크기만큼 넣기
            self.layers.append(Embedding(W_in))

        self.ns_loss = NegativeSamplingLoss(W_out, corpus)

        layers = [self.ns_loss] + self.layers
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word2vecs: np.array = W_in

    def forward(self, contexts, target):
        h = 0
        for idx, layer in enumerate(self.layers):
            h += layer.forward(contexts[:, idx])
        h *= 1 / len(self.layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.layers)
        for layer in self.layers:
            # 여기 레이어가 가장 첫번째 이므로 여기까지 역전파하면 끝.
            layer.backward(dout)

        return None


if __name__ == '__main__':
    # hyper params
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    # read data
    corpus, word2id, id2word = ptb.load_data("train")
    vocab_size = len(word2id)

    contexts, target = create_contexts_target(corpus, window_size)

    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    word2vecs = model.word2vecs

    params = {
        'word2vecs': word2vecs.astype(np.float16),
        'word2id': word2id,
        'id2word': id2word
    }

    pkl_file = 'cbow_params.pkl'
    with Path(f"./{pkl_file}").open("wb") as f:
        pickle.dump(params, f, -1)
