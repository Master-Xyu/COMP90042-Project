from SearchFiles import searcher
from tfidf import get_tfidf
from test import rebuildSentences, removePrefix, term_query
from word2vec import Word2VecSim
import json



if __name__ == '__main__':
    s = searcher()
    with open("train.json",'r') as load_f:
        load_dict = json.load(load_f)
        i = 0

        for key in load_dict.keys():
            try:
                label = 0
                term = load_dict[key]
                text = []
                if len(term['evidence']) == 0:
                    continue
                for doc in term['evidence']:
                    results = s.runTermQuery(doc[0] + ' ' + str(doc[1]))
                    line = term_query([doc[0],doc[1]], results)
                    text.append(removePrefix(line))
                newClaim, newSentence = rebuildSentences(term['claim'], text)
                if term['label'] == "SUPPORTS":
                    label = 1
                elif term['label'] == "NOT ENOUGH INFO":
                    label = 0
                elif term['label'] == "REFUTES":
                    label = -1
                print(newClaim,';', newSentence, ';', Word2VecSim(newClaim, newSentence), label)
            except Exception as e:
                print ("Failed in loadjson:" + str(e))
            i+=1
            if i > 9:
                break


