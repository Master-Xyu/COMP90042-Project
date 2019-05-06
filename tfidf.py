from collections import defaultdict
import os
from gensim import corpora, models, similarities
from pprint import pprint
from matplotlib import pyplot as plt
import logging

def PrintDictionary(dictionary):
    token2id = dictionary.token2id
    dfs = dictionary.dfs
    token_info = {}
    for word in token2id:
        token_info[word] = dict(
            word = word,
            id = token2id[word],
            freq = dfs[token2id[word]]
        )
    token_items = token_info.values()
    token_items = sorted(token_items, key = lambda x:x['id'])
    print('The info of dictionary: ')
    pprint(token_items)
    print('--------------------------')

#PrintDictionary(dictionary)

def get_tfidf(claim, sentences):
    stoplist = set('for a of the and to in \n . , ? ! \' \" -lrb- -rrb- ;'.split())

    texts = [[word for word in sentence.lower().split() if word not in stoplist] for sentence in sentences]
    texts.append([word for word in claim.lower().split() if word not in stoplist])
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    corpus_lsi = lsi_model[corpus_tfidf]

    corpus_simi_matrix = similarities.MatrixSimilarity(corpus_lsi)

    PrintDictionary(dictionary)
    print(claim.lower().split())

    test_bow = dictionary.doc2bow(word for word in claim.lower().split() if word not in stoplist)
    test_tfidf = tfidf_model[test_bow]
    test_lsi = lsi_model[test_tfidf]
    test_simi = corpus_simi_matrix[test_lsi]
    print(list(enumerate(test_simi)))
    return list(enumerate(test_simi))
