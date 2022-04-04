import numpy as np

from book2.ch2.section3_1_preprocess_corpus import preprocess


def create_contexts_target(corpus: list, window_size: int = 1):
    target = corpus[window_size:-window_size]
    contexts = []
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


if __name__ == '__main__':
    text = "You say goodbye and I say hello."
    corpus, word2id, id2word = preprocess(text)
    print(corpus)
    print(id2word)

    contexts, target = create_contexts_target(corpus, 1)
    print(contexts)
    print(target)
