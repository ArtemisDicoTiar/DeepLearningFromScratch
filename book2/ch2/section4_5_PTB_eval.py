import numpy as np
from sklearn.utils.extmath import randomized_svd

from book2.ch2.section3_2_distribution_repr_of_words import create_co_matrix, most_similar
from book2.ch2.section4_1_improve_stat_methods import ppmi
from book2.dataset import ptb

window_size = 2
word2vec_size = 100

corpus, word2id, id2word = ptb.load_data('train')
vocab_size = len(word2id)

print("co-occurrence calculating...")
C = create_co_matrix(corpus, vocab_size, window_size)

print("PPMI calculating...")
W = ppmi(C, verbose=True)

# print("SVD0 calculating...")
# U0, S0, V0 = np.linalg.svd(W)

print("SVD1 calculating...")
U1, S1, V1 = randomized_svd(W, n_components=word2vec_size, n_iter=5, random_state=None)

# word_vecs0 = U0[:, :word2vec_size]
word_vecs1 = U1[:, :word2vec_size]

queries = ['you', 'year', 'car', 'toyota', 'apple', 'pear']
#
# print("From SVD0...")
# for query in queries:
#     most_similar(query, word2id, id2word, word_vecs0, top=5)

print("From SVD1...")
for query in queries:
    most_similar(query, word2id, id2word, word_vecs1, top=5)

"""
you <--> i => 0.8380627036094666
you <--> we => 0.8274505138397217
you <--> they => 0.6783696413040161
you <--> me => 0.5875997543334961
year <--> month => 0.8792139887809753
year <--> week => 0.8729401230812073
year <--> day => 0.7365939021110535
year <--> decade => 0.6950209736824036
car <--> auto => 0.7141766548156738
car <--> truck => 0.6640236973762512
car <--> luxury-car => 0.6066495776176453
car <--> disk-drive => 0.551403284072876
toyota <--> kuwait => 0.7149404287338257
toyota <--> aeroflot => 0.6527894139289856
toyota <--> itel => 0.6165465712547302
toyota <--> pennzoil => 0.581954300403595
apple <--> imperial => 0.4731031060218811
apple <--> syndrome => 0.4292067289352417
apple <--> disks => 0.41299542784690857
apple <--> linear => 0.41286778450012207
"""


