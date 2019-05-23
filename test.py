from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import load_model
from tools import line_query, rebuild_line, rebuildSentences, get_prefix
from itertools import combinations
from BuildVector import Word2VecSim
from collections import Counter
import json
from SearchFiles import searcher
import numpy as np


if __name__ == '__main__':
    model = load_model('vertification.h5')
    load_f  = open("devset.json", 'r')
    load_dict = json.load(load_f)
    load_f.close()
    m = 0
    s = searcher()

    claim = 'Brad Wilk helped co-found Rage in 1962.'
    sentences = ['Wilk started his career as a drummer for Greta in 1990 , and helped co-found Rage with Tom Morello and Zack de la Rocha in August 1991 .']
    claim, newSentence = rebuildSentences(claim, sentences)
    print(Word2VecSim(claim, newSentence))
    print(claim)
    print(newSentence)
    print(model.predict(np.array([[len(claim) / 100, Word2VecSim(claim, newSentence)]],  dtype=float)))