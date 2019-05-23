from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import os, gensim, numpy

sentence1 = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
sentence2 = 'nikolaj coster-waldau work fox broadcast company'

class BuildVector:
    def __init__(self):
        self.word2vec_model = gensim.models.Word2Vec.load('word2vec.model')
        self.doc2vec_model = gensim.models.Doc2Vec.load('doc2vec.model')

    def Doc2VecSim(self, claim, sentence):
        if sentence == '':
            return 0
        reader = [claim, sentence]
        #self.word2vec_model.build_vocab(reader)
        #self.word2vec_model.train(reader, total_examples=self.word2vec_model.corpus_count, epochs=self.word2vec_model.epochs)
        claimVec =self.doc2vec_model.infer_vector(claim.split())
        sentenceVec = self.doc2vec_model.infer_vector(sentence.split())
        return self.similarity(claimVec, sentenceVec)

    def Word2VecSim(self, claim, sentence):
        if sentence == '':
            return 0
        reader = [claim, sentence]
        claimVec = self.sent2vec(self.word2vec_model, reader[0])
        sentenceVec = self.sent2vec(self.word2vec_model, reader[1])
        return self.similarity(claimVec, sentenceVec)

    def sent2vec(self, model, words):

        vect_list = []
        for w in words:
            try:
                vect_list.append(model.wv[w])
            except:
                continue
        vect_list = numpy.array(vect_list)
        vect = vect_list.sum(axis=0)
        return vect / numpy.sqrt((vect ** 2).sum())

    def similarity(self, a_vect, b_vect):
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