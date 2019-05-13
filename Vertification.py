import json
from SearchFiles import searcher
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import load_model
from test import getLinefromFile, removePrefix, rebuildSentences

def vertify(claim, model):
    s = searcher()
    results = s.runQuery(claim)
    sentences = []
    for result in results:
        sentence = getLinefromFile()
        sentences.append(removePrefix(sentence))
    allEvidence = []
    newClaim, newEvidence = rebuildSentences(claim,[])
    lenofNewClaim = len(newClaim.split())
    #1 sentense(s)
    for i1 in range(0,10):
        allEvidence.append([[i1], model.predict([lenofNewClaim, Word2VecSim(newClaim, sentences[i1])])])
        # 2 sentense(s)
        for i2 in range(i1,10):
            newEvidence = [sentences[i1],sentences[i2]]
            allEvidence.append([[i1, i2], model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])
            # 3 sentense(s)
            for i3 in range(i2,10):
                newEvidence = [sentences[i1], sentences[i2], sentences[i3]]
                allEvidence.append([[i1, i2, i3], model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])
                # 4 sentense(s)
                for i4 in range(i3, 10):
                    newEvidence = [sentences[i1], sentences[i2], sentences[i3], sentences[i4]]
                    allEvidence.append(
                        [[i1, i2, i3, i4], model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])
                    # 5 sentense(s)
                    for i5 in range(i4, 10):
                        newEvidence = [sentences[i1], sentences[i2], sentences[i3], sentences[i4], sentences[i5]]
                        allEvidence.append(
                            [[i1, i2, i3, i4, i5], model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])
                    # 6 sentense(s)
                    newEvidence = [sentences[i] for i in range(0,10) if i != i1 and i != i2 and i != i3 and i != i4]
                    allEvidence.append(
                        [[i for i in range(0,10) if i != i1 and i != i2 and i != i3 and i != i4],
                         model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])
                # 7 sentense(s)
                newEvidence = [sentences[i] for i in range(0, 10) if i != i1 and i != i2 and i != i3]
                allEvidence.append(
                    [[i for i in range(0, 10) if i != i1 and i != i2 and i != i3],
                     model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])
            # 8 sentense(s)
            newEvidence = [sentences[i] for i in range(0, 10) if i != i1 and i != i2]
            allEvidence.append(
                [[i for i in range(0, 10) if i != i1 and i != i2],
                 model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])
        # 9 sentense(s)
        newEvidence = [sentences[i] for i in range(0, 10) if i != i1]
        allEvidence.append(
            [[i for i in range(0, 10) if i != i1],
             model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])
    # 10 sentense(s)
    newEvidence = list(sentences)
    allEvidence.append(
        [[i for i in range(0, 10)],
         model.predict([lenofNewClaim, Word2VecSim(newClaim, newEvidence)])])

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