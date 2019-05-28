from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import load_model
from tools import line_query, rebuild_line, rebuildSentences, get_prefix
from itertools import combinations
from BuildVector import BuildVector
from collections import Counter
import json
from SearchFiles import searcher
import numpy as np
from datetime import datetime
from Vertification import vertify

if __name__ == '__main__':
    s = searcher()
    model = load_model('vertification.h5')
    vector_model = BuildVector()
    with open("train.json", 'r') as load_f:
        load_dict = json.load(load_f)
        num = 0
        for key in load_dict.keys():
            evidences = []
            searchResults = []
            term = load_dict[key]
            label, evidence = vertify(term['claim'], model, s, vector_model)
            if label != term['label']:
                print(term['claim'])
                print(term['label'])
                print(term['evidence'])
                print(label)
                print(evidence)
                break