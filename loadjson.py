from SearchFiles import searcher
from tfidf import get_tfidf
from test import rebuildSentences
from word2vec import Word2VecSim
import json
from test import term_query, line_query, rebuild_line
from itertools import combinations

if __name__ == '__main__':
    s = searcher()
    out_f = open('train_output.txt', 'w')
    with open("train.json",'r') as load_f:
        load_dict = json.load(load_f)
        m = 0
        resultTerm={}
        index = 0
        output = {}
        for key in load_dict.keys():
            try:
                resultTerm['key']=key
                label = 0
                term = load_dict[key]

                allEvidences = []
                evidences = []
                allCombinations = []

                #find all real evidences
                if len(term['evidence']) != 0:
                    for doc in term['evidence']:
                        results = s.runTermQuery(doc[0] + ' ' + str(doc[1]))
                        line = term_query([doc[0],doc[1]], results)
                        line = rebuild_line(line)
                        evidences.append(line)
                        allEvidences.append(line)

                if len(evidences) > 9:
                    continue
                #insert other evidences to fullfill 10 evidences
                line = term['claim']
                line = line.replace('\\', ' ')
                line = line.replace('/', ' ')
                results = s.runQuery(line)
                for result in results:
                    if len(allEvidences) > 9:
                        break
                    line = line_query(result)
                    if line in evidences:
                        continue
                    allEvidences.append(rebuild_line(line))
                for i in range(1, len(results)+1):
                    allCombinations.append(list(combinations(allEvidences, i)))
                #fine the label of this term
                if term['label'] == "SUPPORTS":
                    label = 1
                elif term['label'] == "NOT ENOUGH INFO":
                    label = 0
                elif term['label'] == "REFUTES":
                    label = -1

                newClaim, realSentence = rebuildSentences(term['claim'], evidences)
                resultTerm['label'] = label
                resultTerm['length'] = len(newClaim.split())
                resultTerm['similarity'] = Word2VecSim(newClaim, realSentence)
                index += 1
                output[index] = resultTerm
                print(newClaim, ';', realSentence, ';', resultTerm['similarity'], resultTerm['label'])
                resultTerm = {}

                #for all combinations, get the simmilarity and the label
                preNewSentences = set()
                numOfNoEvidence = 0
                for com1 in allCombinations:

                    for com in com1:
                        newClaim, newSentence = rebuildSentences(term['claim'], com)

                        #discard duplicated state
                        if newSentence not in preNewSentences:
                            preNewSentences.add(newSentence)
                        else:
                            continue

                        if set(evidences).issubset(set(com)) or newSentence == realSentence:
                            term_label = label
                        else:
                            term_label = 0

                        #control the num of no evidence combinations
                        if term_label == 0:
                            numOfNoEvidence+=1
                            if numOfNoEvidence > 4:
                                continue

                        resultTerm['label'] = term_label
                        resultTerm['length']=len(newClaim.split())
                        resultTerm['similarity']=Word2VecSim(newClaim, newSentence)
                        index += 1
                        output[index] = resultTerm
                        print(newClaim, ';', newSentence, ';', resultTerm['similarity'], resultTerm['label'])
                        resultTerm = {}

            except Exception as e:
                print ("Failed in loadjson:" + str(e))

            m += 1
            if m > 2000:
                break

        out_f.write(json.dumps(output))
            #TODO: verify whether it works and improve efficency
    '''
    with open('train_output.txt', 'w') as out_f:
        for term in output:
            out_f.write(term +'\n')
    '''
