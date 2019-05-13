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
        i = 0
        result={}
        for key in load_dict.keys():
            output = []
            try:
                result['key']=key
                label = 0
                term = load_dict[key]

                allEvidences = []
                evidences = []
                combinations = []

                #find all real evidences
                if len(term['evidence']) != 0:
                    for doc in term['evidence']:
                        results = s.runTermQuery(doc[0] + ' ' + str(doc[1]))
                        line = term_query([doc[0],doc[1]], results)
                        line = rebuild_line(line)
                        evidences.append(line)
                        allEvidences.append(line)

                #insert other evidences to fullfill 10 evidences
                results = s.runQuery(term['claim'])
                for result in results:
                    if len(allEvidences) > 9:
                        break
                    if line in evidences:
                        continue
                    allEvidences.append(rebuild_line(line_query(result)))

                for i in range(1, len(results)+1):
                    combinations.append(list(combinations(allEvidences, i)))

                #fine the label of this term
                if term['label'] == "SUPPORTS":
                    label = 1
                elif term['label'] == "NOT ENOUGH INFO":
                    label = 0
                elif term['label'] == "REFUTES":
                    label = -1

                #for all combinations, get the simmilarity and the label
                for com in combinations:
                    text = []
                    for line in com:
                        text.append(rebuild_line(line))
                    newClaim, newSentence = rebuildSentences(term['claim'], text)
                    if set(evidences).issubset(set(allEvidences)):
                        term_label = label
                    else:
                        term_label = 0
                    result['label'] = term_label
                    result['length']=len(newClaim.split())
                    result['similarity']=Word2VecSim(newClaim, newSentence)
                    output.append(json.dumps(result))
                    print(newClaim,';', newSentence, ';', Word2VecSim(newClaim, newSentence), label)

            except Exception as e:
                print ("Failed in loadjson:" + str(e))

            for term in output:
                out_f.write(term + '\n')

            i += 1
            if i > 9:
                break
            #TODO: verify whether it works and improve efficency
    '''
    with open('train_output.txt', 'w') as out_f:
        for term in output:
            out_f.write(term +'\n')
    '''
