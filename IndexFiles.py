#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
from datetime import datetime

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory

"""
This class is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""

class Ticker(object):
    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)


def indexDocs(storeDir, analyzer):
    if not os.path.exists(storeDir):
        os.mkdir(storeDir)

    store = SimpleFSDirectory(Paths.get(storeDir))
    analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(store, config)
    root = "wiki-pages-text/"
    t1 = FieldType()
    t1.setStored(True)
    t1.setTokenized(False)
    t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

    t2 = FieldType()
    t2.setStored(False)
    t2.setTokenized(True)
    t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    for root, dirnames, filenames in os.walk(top = root):
        print(root, dirnames, filenames)
        for filename in filenames:
            if not filename.endswith('.txt'):
                continue
            print ("adding" + filename)
            try:
                path = os.path.join(root, filename)
                file = open(path, encoding="utf8")
                i = 0
                #contents = file.read()
                termName = ''
                while True:
                    i+=1
                    line = file.readline()
                    doc = Document()
                    if not line:
                        break
                    if termName != line.split()[0]:
                        #print(termName)
                        termName = line.split()[0]
                        doc.add(Field("name", filename, t1))
                        doc.add(Field("line", i, t1))
                        doc.add(Field("contents", termName, t2))
                    else:
                        continue

                    writer.addDocument(doc)
                file.close()
                """
                doc = Document()
                doc.add(Field("name", filename, t1))
                doc.add(Field("path", root, t1))
                if len(contents) > 0:
                    doc.add(Field("contents", contents, t2))
                else:
                    print ("warning: no content in " + filename)
                writer.addDocument(doc)
                """
            except Exception as e:
                print ("Failed in indexDocs:" + str(e))
    ticker = Ticker()
    print('commit index')
    threading.Thread(target=ticker.run).start()
    writer.commit()
    writer.close()
    ticker.tick = False
    print ('done')

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print ('lucene' + lucene.VERSION)
    start = datetime.now()
    try:
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        indexDocs(os.path.join(base_dir, INDEX_DIR),
                   StandardAnalyzer())
        end = datetime.now()
        print (end - start)
    except Exception as e:
        print ("Failed: " + str(e))
        raise e
