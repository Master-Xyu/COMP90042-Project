from tfidf import get_tfidf
import nltk
from nltk.corpus import wordnet
from BuildVector import BuildVector

sentence1 = "Roman Atwood is a content creator."
sentence2 = ['Roman_Atwood 0 Roman Bernard Atwood -LRB- born May 28 , 1983 -RRB- is an American YouTube personality , comedian , vlogger and pranker .', 'Roman_Atwood 3 He also has another YouTube channel called `` RomanAtwood '' , where he posts pranks .']
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
    stoplist = set('for of the and to in with on \n . , ? ! \' \" -lrb- -rrb- ( ) ;'.split())

    claimWords = nltk.word_tokenize(claim)
    claimWords = [lemmatize(word.lower()) for word in claimWords if word not in stoplist]
    claimWords = list(set(claimWords))
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
        sentence = sentence.replace("-LRB-", ' ')
        sentence = sentence.replace("-RRB", ' ')
        newSentences.append(sentence.replace('_', ' '))
    sentences = newSentences
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

    #arrange in order
    for word in claimWords:
        apended = False
        for newWord in newWords:
            if newWord == word:
                newSentence += word + ' '
                apended = True
                break
        if not apended:
            newSentence += "awww "

    newClaim = ''
    for word in claimWords:
        newClaim += word + ' '

    '''
    #insert new word to make the new sentense have same length with claim
    if len(nltk.word_tokenize(newSentence)) < len(claimWords):
        for i in range(0,len(claimWords) - len(nltk.word_tokenize(newSentence))):
            newSentence += " awwww"
    '''
    '''
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence):
                if word.lower() in stoplist:
                    continue
                insertedWord = lemmatize(word.lower())
                for key in allwords.keys():
                    if insertedWord in allwords[key]:
                        break
                    else:
                        newSentence += insertedWord + ' '
                        if(len(nltk.word_tokenize(newSentence)) == len(claimWords)):
                            return newClaim, newSentence
    '''
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
        file.close()
    print("No matching error!")

def line_query(query):
    try:
        file = open("wiki-pages-text/" + query[0], encoding="utf8")
        for i in range(0, int(query[1])):
            line = file.readline()
        file.close()
        return line
    except Exception as e:
        print ("Failed in line_query:" + str(e))
        file.close()

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
    return [line.split()[0], line.split()[1]]

if __name__ == '__main__':
    newClaim, newSentence = rebuildSentences(sentence1, sentence2)
    print(newClaim)
    print(newSentence)