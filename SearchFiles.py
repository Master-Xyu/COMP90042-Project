#!/usr/bin/env python
from pip._vendor.distlib.compat import raw_input

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene


from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import TermQuery
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher

from tools import line_query


"""
This script is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it
will search the Lucene index in the current directory called 'index' for the
search query entered against the 'contents' field.  It will then display the
'path' and 'name' fields for each of the hits it finds in the index.  Note that
search.close() is currently commented out because it causes a stack overflow in
some cases.
"""

class searcher:
    
    def __init__(self):
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        directory = SimpleFSDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
        self.searcher = IndexSearcher(DirectoryReader.open(directory))
        self.analyzer = StandardAnalyzer()

    def runQuery(self, claim):
        #print("Searching for:" + command)
        query = QueryParser("content", self.analyzer).parse(claim)
        scoreDocs = self.searcher.search(query, 10).scoreDocs
        #print(len(scoreDocs) , "total matching documents.")
        results = []
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            #print(' name:' + doc.get("name") + " line:" + doc.get("line"))
            results.append([doc.get("name"), doc.get("line"), doc.get('termName')])
        '''
        for result in results:
           print(line_query(result))
        '''
        return results


    def runTermQuery(self, claim):
        query = QueryParser("termName", self.analyzer).parse(claim)
        scoreDocs = self.searcher.search(query, 10).scoreDocs
        #print(len(scoreDocs) , "total matching documents.")
        results = []
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            #print(' name:' + doc.get("name") + " line:" + doc.get("line"))
            results.append([doc.get("name"), doc.get("line")])
        '''
        for result in results:
            print(line_query(result))
        '''
        return results


if __name__ == '__main__':
    #lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene' + lucene.VERSION)
    s = searcher()
    while True:
        print("Hit enter with no input to quit.")
        command = raw_input("Query:")
        if command == '':
            break
        s.runTermQuery("Roman_Atwood 3")
