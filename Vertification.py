import json
from SearchFiles import searcher
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import load_model
from test import line_query, rebuild_line, rebuildSentences, get_prefix
from itertools import combinations
from word2vec import Word2VecSim

def vertify(claim, model):
    s = searcher()
    results = s.runQuery(claim)
    allEvidences = []
    inputs = []

    #find 10 evidences by querying the claim
    for result in results:
        sentence = line_query(result)
        allEvidences.append(sentence)
    results = []
    newClaim, newEvidence = rebuildSentences(claim,[])
    lenofNewClaim = len(newClaim.split())

    #get all input circumstances
    for i in range(0, len(allEvidences)+1):
        inputs.append(list(combinations(allEvidences, i)))

    #get the results of all inputs
    for term in inputs:
        result = [[]]
        text = ''
        for line in term:
            result[0].append(get_prefix(line))
            text += rebuild_line(line)
        newClaim, newEvidence = rebuildSentences(claim, [])
        result.append(model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)]))

    #TODO: processing all results and return evidences and label

if __name__ == '__main__':
    model = load_model('vertification.h5')
    load_f  = open("train.json", 'r')
    write_f = open("train.json", 'w')
    load_dict = json.load(load_f)
    for key in load_dict.keys():
        try:
            label = 0
            term = load_dict[key]
            label, evidence = vertify(term['claim'], model)
            load_dict[key]['label'] = label
            load_dict[key]['evidence'] = evidence
        except Exception as e:
            print ("Failed in vertification:" + str(e))

    write_f.write(json.dumps(load_dict))
    load_f.close()
    write_f.close()