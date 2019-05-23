import multiprocessing
from datetime import datetime
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from os import listdir
from os.path import isfile, join
TaggedDocument = gensim.models.doc2vec.TaggedDocument

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(doc.split(),[self.labels_list[idx]])

if __name__ == '__main__':
    docLabels = [f for f in listdir("wiki-pages-text") if f.endswith('.txt')]
    data = []
    try:
        for doc in docLabels:
            print(doc)
            data.append(open('wiki-pages-text/' + doc, encoding="utf8").read())
    except Exception as e:
        print ("Failed in BuildD2VModels:" + str(e))

    it = LabeledLineSentence(data, docLabels)

    model = gensim.models.Doc2Vec(vector_size=100, window=10, min_count=1, workers=11, alpha=0.025, min_alpha=0.025)

    print("start building vocab")
    model.build_vocab(it)

    data = []

    print("start training")
    model.train(documents = it, total_examples=model.corpus_count, epochs=model.epochs)

    model.save('doc2vec.doc2vec_model')
