import numpy as np

from book2.ch2.section3_1_preprocess_corpus import preprocess
from book2.ch3.section3_1_context_n_target import create_contexts_target


def convert2onehot(ary: np.array, max_id: int) -> np.array:
    """
    (np.arange(ary.max() + 1) == np.array([[1], [3]]))
    array([[False,  True, False, False, False, False, False],
           [False, False, False,  True, False, False, False]])
    """
    return (np.arange(max_id) == ary[..., None]).astype(int)


if __name__ == '__main__':
    text = "You say goodbye and I say hello."
    corpus, word2id, id2word = preprocess(text)
    print(corpus)
    print(id2word)

    contexts, target = create_contexts_target(corpus, 1)

    contexts = convert2onehot(contexts, len(word2id))
    target = convert2onehot(target, len(word2id))
    print(contexts)
    print(target)

