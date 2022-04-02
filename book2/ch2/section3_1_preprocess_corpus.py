import numpy as np

text = "You say goodbye and I say hello."
text = text \
    .lower() \
    .replace('.', ' .')
print(text)

words = text.split(" ")
print(words)

id2word = dict(
    tuple(enumerate(set(words)))
)
word2id = {
    v: k
    for (k, v) in id2word.items()
}

print(word2id)
print(id2word)

print(id2word[1])
print(word2id['hello'])

corpus = np.array(
    list(
        word2id[w] for w in words
    )
)
print(corpus)


def preprocess(text: str):
    """

    :param text:
    :return: np.array(corpus), dict(word2id), dict(id2word)
    """
    text = text \
        .lower() \
        .replace(".", " .")

    words = text.split(" ")

    id2word = dict(
        tuple(enumerate(set(words)))
    )
    word2id = {
        v: k
        for (k, v) in id2word.items()
    }

    corpus = np.array(
        list(
            word2id[w] for w in words
        )
    )

    return corpus, word2id, id2word


text = "You say goodbye and I say hello."
corpus, word2id, id2word = preprocess(text)

print(corpus, word2id, id2word)
