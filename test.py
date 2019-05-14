from tfidf import get_tfidf
import nltk
from nltk.corpus import wordnet
from word2vec import Word2VecSim

sentence1 = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
sentence2 = ['Fox Broadcasting Company The Fox Broadcasting Company -LRB- often shortened to Fox and stylized as FOX -RRB- is an American English language commercial broadcast television network that is owned by the Fox Entertainment Group subsidiary of 21st Century Fox . ', 'Nikolaj Coster-Waldau He then played Detective John Amsterdam in the short-lived Fox television series New Amsterdam -LRB- 2008 -RRB- , as well as appearing as Frank Pike in the 2009 Fox television film Virtuality , originally intended as a pilot . ']
sentence3 = 'Adrienne Bailon is an accountant. Adrienne Bailon is an accountant.'
sentence4 = ['Adrienne Bailon Adrienne Eliza Houghton -LRB- née Bailon ; born October 24 , 1983 -RRB- is an American singer-songwriter , recording artist , actress , dancer and television personality . ' ,
             'Adrienne Bailon Adrienne Eliza Houghton née Bailon is an']

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def rebuildSentences(claim, sentences):
    stoplist = set('for of the and to in with on \n . , ? ! \' \" -lrb- -rrb- ;'.split())

    claimWords = nltk.word_tokenize(claim)
    claimWords = [lemmatize(word.lower()) for word in claimWords if word not in stoplist]
    syns = {}
    for word in claimWords:
        syns[word] = wordnet.synsets(word)
    allwords = {}
    allwords[''] = list(claimWords)
    for key in syns.keys():
        allwords[key] = []
        for syn in syns[key]:
            for lem in syn.lemmas():
                allwords[key].append(str(lem).split('.')[3][:-2])
        allwords[key] = list(set(allwords[key]))

    newSentences = []
    for sentence in sentences:
        senwords = nltk.word_tokenize(sentence)
        senwords = [lemmatize(word.lower()) for word in senwords if word not in stoplist]
        newSentenceWords = []
        for word in senwords:
            for key in allwords.keys():
                if word in allwords[key]:
                    if key == '':
                        newSentenceWords.append(word)
                    else:
                        newSentenceWords.append(key)
        newSentence = ''
        newSentenceWords = list(set(newSentenceWords))
        for word in newSentenceWords:
            newSentence += word + ' '
        newSentences.append(newSentence)

    newSentence = ''
    for sentence in newSentences:
        newSentence += sentence
    newWords = list(set(newSentence.split()))

    newSentence = ''

    for word in claimWords:
        for newWord in newWords:
            if newWord == word:
                newSentence += word + ' '

    newClaim = ''
    for word in claimWords:
        newClaim += word + ' '
    return newClaim, newSentence

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
    print("No matching error!")

def line_query(query):
    try:
        file = open("wiki-pages-text/" + query[0], encoding="utf8")
        for i in range(0, int(query[1])):
            line = file.readline()
        return line
    except Exception as e:
        print ("Failed in line_query:" + str(e))

def rebuild_line(line):
    temp_line = line.split()
    #word = temp_line[0]
    temp_line.pop(0)
    temp_line.pop(0)
    #temp_line.insert(0, word.replace('_', ' '))
    line = ''
    for word in temp_line:
        line += word + ' '
    return line

def get_prefix(line):
    prefix = []
    prefix.append([line.split()[0], line.split()[1]])
    return prefix
if __name__ == '__main__':
    newClaim, newSentence = rebuildSentences(sentence1, sentence2)
    print(Word2VecSim(newClaim, newSentence))

'''
get_tfidf(sentence1, newSentences)
allwords, newSentences = rebuildSentences(sentence3, sentence4)
print(newSentences)
get_tfidf(sentence3, newSentences)
'''
