from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from datetime import datetime

import os

if __name__ == '__main__':
    word2vec_model = Word2Vec(size=30, min_count=0, window=10)
    word2vec_model.build_vocab(['a'])
    word2vec_model.train(['a'], total_examples=word2vec_model.corpus_count, epochs=word2vec_model.epochs)
    root = 'wiki-pages-text/'
    start = datetime.now()
    end = datetime.now()
    try:
        for root, dirnames, filenames in os.walk(top='wiki-pages-text/'):
            for filename in filenames:
                file = open(root + filename, encoding="utf8")
                print ("training " + filename)
                lines = file.readlines()
                word2vec_model.build_vocab(lines, update=True)
                word2vec_model.train(lines, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.epochs)
                print(datetime.now() - end)
                end = datetime.now()
    except Exception as e:
        print ("Failed in BuildW2VModel:" + str(e))
    print(datetime.now() - start)
    word2vec_model.save('word2vec.model')