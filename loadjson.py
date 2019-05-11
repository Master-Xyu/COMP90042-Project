from SearchFiles import searcher
from tfidf import get_tfidf
from test import rebuildSentences
from word2vec import Word2VecSim
import json

def term_query(item, querys):
    for query in querys:
        file = open("wiki-pages-text/" + query[0], encoding="utf8")
        for i in range(0,int(query[1])):
            line = file.readline()
        while line.split()[0] == item[0]:
            if str(item[1]) == line.split()[1]:
                file.close()
                return line
            line = file.readline()
    '''
    file = open("wiki-pages-text/wiki-096.txt", encoding="utf8")
    i = 0
    while True:
        i+=1
        line = file.readline()
        if line.split()[0] == 'The_Ten_Commandments_-LRB-1956_film-RRB-':
            print(i)
    '''
    print("No matching line error.")

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
                    temp_line = line.split()
                    word = temp_line[0]
                    temp_line.pop(0)
                    temp_line.pop(0)
                    temp_line.insert(0, word.replace('_', ' '))
                    line = ''
                    for word in temp_line:
                        line += word + ' '
                    text.append(line)
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


