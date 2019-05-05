from SearchFiles import searcher
from tfidf import get_tfidf
import json

def term_query(item, querys):
    for query in querys:
        file = open("wiki-pages-text/" + query[0], encoding="utf8")
        line_num = 0
        while True:
            line = file.readline()
            if line_num == int(query[1]):
                file.close()
                break
            line_num += 1
        if item in line:
            return line
    print("No matching line error.")

if __name__ == '__main__':
    with open("train.json",'r') as load_f:
        load_dict = json.load(load_f)
        for key in load_dict.keys():
            term = load_dict[key]
            print(load_dict[key])
            break
        s = searcher()
        text = []
        for doc in term['evidence']:
            results = s.runQuery(doc[0] + " " + str(doc[1]))
            line = term_query(doc[0] + " " + str(doc[1]), results)
            temp_line = line.split()
            word = temp_line[0]
            temp_line.pop(0)
            temp_line.pop(0)
            temp_line.insert(0, word.replace('_', ' '))
            line = ''
            for word in temp_line:
                line += word + ' '
            text.append(line)
            #text.append(term_query(doc[0] + " " + str(doc[1]), results))
        print(text)
        sim = get_tfidf(term['claim'], text)

