from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import load_model
from tools import line_query, rebuild_line, rebuildSentences, get_prefix
from itertools import combinations
from word2vec import Word2VecSim
from collections import Counter
import json
from SearchFiles import searcher
import numpy as np

def vertify(claim, model, s):

    results = s.runQuery(claim)
    allEvidences = []
    inputs = []

    #find i evidences by querying the claim
    for i in range(0,5):
        sentence = line_query(results[i])
        allEvidences.append(sentence)
    results = []
    newClaim, newEvidence = rebuildSentences(claim,[])
    lenofNewClaim = len(newClaim.split())

    #get all input circumstances
    for i in range(1, len(allEvidences)+1):
        inputs.append(list(combinations(allEvidences, i)))

    #build input array from all inputs
    predictions = []
    input_evidences = []
    input_array = []
    for outer in inputs:
        for term in outer:
            result = []
            text = []
            for line in term:
                result.append(get_prefix(line))
                text.append(rebuild_line(line))
            input_evidences.append(result)
            newClaim, newEvidence = rebuildSentences(claim, text)
            input_array.append([lenofNewClaim / 100,  Word2VecSim(newClaim, newEvidence)])

    predictResults = model.predict(np.array(input_array,  dtype=float))
    '''
    for result in list(predictResults):
        print(result)
    '''
    #decide label based on support or refute is larger, if none of them then no evidence
    for predictResult in predictResults:
        if predictResult[0] > predictResult[1] and predictResult[0] > predictResult[2]:
            predictions.append(0)
        elif predictResult[1] > predictResult[0] and predictResult[1] > predictResult[2]:
            predictions.append(1)
        elif predictResult[2] > predictResult[0] and predictResult[2] > predictResult[1]:
            predictions.append(-1)
        else:
            print("Error in deciding label.")

    #decide the final label according to amount
    c = Counter(predictions)
    if c[1] != 0 and c[-1] != 0:
        if c[1] >= c[-1]:
            label = 1
        else:
            label = -1
    elif c[1] != 0:
        label = 1
    elif c[-1] != 0:
        label = -1
    else:
        return "NOT ENOUGH INFO", []

    shortest_input = -1
    highest_score = 0.0
    shortest_length = 100
    for i in range(0, len(predictions)):
        if predictions[i] == label:
            if label == 1:
                if predictResults[i][1] > highest_score:
                    shortest_input = i
                    shortest_length = len(input_evidences[i])
                    highest_score = predictResults[i][1]
                elif predictResults[i][1] == highest_score:
                    if len(input_evidences[i]) < shortest_length:
                        shortest_input = i
                        shortest_length = len(input_evidences[i])
            else:
                if predictResults[i][2] > highest_score:
                    shortest_input = i
                    shortest_length = len(input_evidences[i])
                    highest_score = predictResults[i][2]
                elif predictResults[i][2] == highest_score:
                    if len(input_evidences[i]) < shortest_length:
                        shortest_input = i
                        shortest_length = len(input_evidences[i])
    if label == 1:
        return "SUPPORTS", input_evidences[shortest_input]
    else:
        return "REFUTES", input_evidences[shortest_input]


if __name__ == '__main__':
    model = load_model('vertification.h5')
    load_f  = open("test-unlabelled.json", 'r')
    out_f = open('testoutput.json', 'w')
    load_dict = json.load(load_f)
    out_dict = {}
    load_f.close()
    m = 0
    s = searcher()
    right = 0
    for key in load_dict.keys():
        try:
            label = 0
            term = load_dict[key]
            label, evidence = vertify(term['claim'], model, s)
            #print(term['claim'], ';', term['evidence'], ';', term['label'], ';', label, ';', evidence)
            print(term['claim'], ';', label, ';', evidence)
            out_dict[key] = {}
            out_dict[key]['claim'] = term['claim']
            out_dict[key]['label'] = label
            out_dict[key]['evidence'] = evidence
            #if label == term['label']:
                #right += 1
            #m+=1
            #if m > 4:
                #print("Precision = " + str(right/5))
                #break

        except Exception as e:
            print("Failed in vertification:" + str(e))
            out_dict[key] = {}
            out_dict[key]['claim']    = term['claim']
            out_dict[key]['label']    = "NOT ENOUGH INFO"
            out_dict[key]['evidence'] = []

    out_f.write(json.dumps(out_dict))
