from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import os, gensim, numpy

WORK_PATH = 'wiki-pages-text/'

sentence1 = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
sentence2 = 'nikolaj coster-waldau work fox broadcast company'

def Word2VecSim(claim, sentence):

    word2vec_model = Word2Vec(size=10, min_count=0, window=10)
    if sentence == '':
        return 0
    reader = [claim, sentence]
    word2vec_model.build_vocab(reader)
    word2vec_model.train(reader, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
    claimVec = sent2vec(word2vec_model, reader[0])
    sentenceVec = sent2vec(word2vec_model, reader[1])
    return similarity(claimVec, sentenceVec)

def sent2vec(model, words):

    vect_list = []
    for w in words:
        try:
            vect_list.append(model.wv[w])
        except:
            continue
    vect_list = numpy.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect / numpy.sqrt((vect ** 2).sum())

def similarity(a_vect, b_vect):
    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = None
    for a, b in zip(a_vect, b_vect):
        dot_val += a * b
        a_norm += a ** 2
        b_norm += b ** 2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm * b_norm) ** 0.5)
    return cos

if __name__ == '__main__':

    print(Word2VecSim(sentence1, sentence2))
