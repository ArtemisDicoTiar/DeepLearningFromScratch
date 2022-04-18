import collections

import numpy as np

from book2.utils.layers import SigmoidWithLoss, Embedding

GPU = False


def np_random():
    print(np.random.choice(10))
    print(np.random.choice(10))

    words = ["you", "say", "goodbye", "I", "hello", "."]

    print(np.random.choice(words))

    # 랜덤하게 5개
    print(np.random.choice(words, size=5))

    # 랜덤하게 5개, 중복 제외
    print(np.random.choice(words, size=5, replace=False))

    # 확률 기반으로 선택
    p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
    print(np.random.choice(words, p=p))


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layer = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h: np.array, target: np.array):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # positive
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layer[0].forward(score, correct_label)

        # negative
        for idx in range(1, self.sample_size + 1):
            negative_target = negative_sample[:, idx - 1]
            score = self.embed_dot_layers[idx].forward(h, negative_target)
            negative_label = np.zeros(batch_size, dtype=np.int32)
            loss += self.loss_layer[idx].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for loss_layer, embed_layer in zip(self.loss_layer, self.embed_dot_layers):
            dscore = loss_layer.backward(dout)
            dh += embed_layer.backward(dscore)

        return dh


if __name__ == '__main__':
    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    # 이 지수 값은 실험적으로 결정된 거 같다. 상황에 맞게 수정할 수 있는 하이퍼 파라미터
    power = 0.75
    sample_size = 2

    sampler = UnigramSampler(corpus, power, sample_size)
    target = np.array([1, 3, 0])
    negative_sample = sampler.get_negative_sample(target)
    print(negative_sample)
