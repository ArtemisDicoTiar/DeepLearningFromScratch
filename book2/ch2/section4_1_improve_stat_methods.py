"""
앞선 섹션에서 구현한 방식으로 벡터를 표현하는 건 단순히 "발생"이라는 점을 이용한다는 것에 문제가 있다.
가령 영어에서는 "a"나 "the"를 굉장히 자주 사용하는 데 그러면 the car... 을 분석하다보면 car는 the와 높은 관련이 있다고 나오는 데
이게 과연 관련이 있는 걸까?

이를 해결하기 위한 방법이 Point-wise Mutual Information (PMI)다.

수식은 log_2 ( P(x, y) / {P(x) * P(y)} ) 다
두개의 단어가 각각 나타날 확률을 곱하고 그 값을 둘다 동시에 나타날 확률에 나눠준다.
로그 스케일을 취해준건 위 수식을 잘 정리하면 코퍼스의 단어개수에 비례해질텐데 이 큰 수를 작게 반영하기 위함으로 보인다.

다만 둘다 나타나지 않게 되면 P(x, y) = 0이고, 그러면 log2(0) -> -inf이므로
이를 방지할 수 있게 다음을 고안한다.
Positive PMI = max(0, PMI(x, y)) 이다. 단순하다.
생각해보면 음의 값을 가질정도로 작은 값이라는 건 PMI가 말도 안되게 무시할 만한 정보라는 뜻이니깐


"""
import numpy as np

from book2.ch2.section3_1_preprocess_corpus import preprocess
from book2.ch2.section3_2_distribution_repr_of_words import create_co_matrix


def pmi(C: np.array, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]

    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = pmi

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print(f"{100 * cnt / total}% done..")

    return M


def ppmi(C, verbose=False, eps=1e-8):
    pmi_mat = pmi(C, verbose, eps)
    pmi_mat[pmi_mat < 0] = 0
    return pmi_mat


if __name__ == '__main__':
    text = "You say goodbye and I say hello."
    corpus, word2id, id2word = preprocess(text)
    vocab_size = len(word2id)
    C = create_co_matrix(corpus, vocab_size)

    W0 = pmi(C)
    W1 = ppmi(C)

    np.set_printoptions(precision=3)
    print(C)
    print("=" * 70)
    print(W0)
    print("=" * 70)
    print(W1)

    """
    여기서 문제1
    PPMI 방식의 접근은 corpus가 커지면 (즉 단어의 개수가 늘면) 그대로 차원의 증가로 이어진다.
    이는 그리 현실적인 방법이 아니다. (메모리 어떻게 할건데...)
    + 문제2
    대부분의 행렬 정보가 0이다. 즉 meaningless인 정보가 많다는 거다.
    
    + 문제3
    이런 행렬은 노이즈를 조금만 줘도 (dirty data, abusing) 행렬이 쉽게 흔들린다.
    
    => Sol: 벡터의 차원감소
    """