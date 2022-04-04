"""
Dimension reduction
말 그대로 차원을 줄이는 거다.
핵심은 중요한 정보는 최대한 살리고 필요없는 거는 버리고

Intuitive한 eg는 데이터 분포에서 데이터의 분포를 잘 보여주는 regression line을 그리면
데이터에서 중요한 정보는 최대한 유지하되 정보의 크기를 줄일 수 있다.

여기서 사용한 차원 감소방법은 Singular Value Decomposition (SVD)이다.
X = USV^T
X 행렬을 U, S, V로 분해하는 게 핵심이다.
U와 V는 orthogonal mat
S는 diagonal mat

U: 현재의 context에서는 단어 공간으로 취급 할 수 있다.
S: singular value가 큰 순서로 나열되어 있다.
    singular value (특잇값) -> 해당 축의 중요도

방법: 중요도가 낮은 원소 (특잇값이 작은 축)을 제거

"""
import numpy as np
from matplotlib import pyplot as plt

from book2.ch2.section3_1_preprocess_corpus import preprocess
from book2.ch2.section3_2_distribution_repr_of_words import create_co_matrix
from book2.ch2.section4_1_improve_stat_methods import ppmi

if __name__ == '__main__':
    np.set_printoptions(precision=2)

    text = "You say goodbye and I say hello."
    corpus, word2id, id2word = preprocess(text)
    vocab_size = len(word2id)
    C = create_co_matrix(corpus, vocab_size)

    W = ppmi(C)

    U, S, V = np.linalg.svd(W)
    print(C)
    print("=" * 70)
    print(U)
    print("=" * 70)
    print(S)
    print("=" * 70)
    print(V)
    print("=" * 70)

    print(U[0, :2])
    print("=" * 70)

    for word, word_id in word2id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.show()

