"""
PTB (Penn Treebank) 데이터셋을 사용해보자
word2vec 발명자인 토마스 미콜로프의 웹페이지에서 받을수 있다고 한다.

"""
from book2.dataset import ptb

if __name__ == '__main__':
    corpus, word2id, id2word = ptb.load_data("train")

    print('말뭉치 크기:', len(corpus))
    print('corpus[:30]:', corpus[:30])
    print()
    print('id_to_word[0]:', id2word[0])
    print('id_to_word[1]:', id2word[1])
    print('id_to_word[2]:', id2word[2])
    print()
    print("word_to_id['car']:", word2id['car'])
    print("word_to_id['happy']:", word2id['happy'])
    print("word_to_id['lexus']:", word2id['lexus'])
