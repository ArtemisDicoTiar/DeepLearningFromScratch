"""
.2 단어의 분산 표현
색을 벡터로 표현하듯 단어도 벡터로 표현해보자. 어떻게?
이런 표현 방식을 NLP에서 분산표현 (Distributional Representation)이라고 한다.
    보통 단어 (word or term)의 표현은 fixed length of dense vector로 된다.

.3 분포 가설 (Distributional Hypothesis)
단어를 벡터로 표현하는 여러 연구들 모두 여기에 기반을 둔다.

H: 쉽게 말해 단어가 사용된 context가 의미를 형성한다는 것.
이때 context는 해당 단어 주변에 사용된 단어를 본다.
eg.
    window_size = 2이면
    you say goodbye and i say hello
    라는 문장에서 goodbye의 context는
    [you, say, and, i]가 된다.
    이건 단순히 예시다. 좌측 윈도우만, 우측 윈도우만 보기도 한다.

.4 동시발생 행렬 (Co-occurrence dist)
통계적으로 단어의 발생 횟수를 세어 보는 방법이다.

각 단어의 발생 빈도를 벡터로 표현한다.
eg. 위와 동일한 문장으로 say의 벡터는
    you say goodbye and i hello .
   [1   0   1       0   1 1     0]
   의 형태를 띄게 된다.
   이를 문장의 모든 단어로 표기 하게 되면 행렬로 표기가 되는 데 이를 co-occurrence matrix라고 한다.

.5 벡터간 유사도 (Cosine similarity)
각 벡터간 유사한 정도는 코사인 시밀러리티로 구하면된다.
정규화 (L2 Norm)된 두 벡터의 내적을 하면 구할 수 있다.
다만 zero division 을 막아야하기 때문에 각 값에 아주 작은 값 epsilon을 더해준다. (1e-8)
1e-8의 값이 된 이유는 이정도로 작은 값은 부동소수점 연산에서 반올림되어 흡수되기 때문이다.
쉽게 말해 0이 아닌 값에서는 그 값에 유의미한 변화를 주지 않을 정도의 영향이며, 0인 경우, 0으로 나눠지는 것을 막게끔 아주 작은 값이 있어준다는 것이다.


"""
import numpy as np

from book2.ch2.section3_1_preprocess_corpus import preprocess


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_sim(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    return np.dot(nx, ny)


def most_similar(query, word2id, id2word, word_matrix, top=5):
    if query not in word2id:
        raise ValueError(f"Cannot find {query}.")

    query_id = word2id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id2word)
    similarity = np.zeros(vocab_size)

    for i in range(vocab_size):
        similarity[i] = cos_sim(word_matrix[i], query_vec)

    for idx, sim_idx in enumerate(similarity.argsort()[::-1]):
        if idx >= top:
            break

        word = id2word[sim_idx]
        if word == query:
            continue

        print(f"{query} <--> {word} => {similarity[sim_idx]}")


if __name__ == '__main__':
    text = "You say goodbye and I say hello."
    corpus, word2id, id2word = preprocess(text)
    vocab_size = len(word2id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[word2id['you']]
    c1 = C[word2id['i']]

    print(cos_sim(c0, c1))
    most_similar("you", word2id, id2word, C, top=5)
