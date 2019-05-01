import IndexFiles

if __name__ == '__main__':

    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print ('lucene' + lucene.VERSION)
    start = datetime.now()
    try:
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        indexer = IndexFiles('wiki-pages-text\\', os.path.join(base_dir, INDEX_DIR),
                   StandardAnalyzer())
        indexer.IndexFiles()

        end = datetime.now()
        print (end - start)
    except Exception as e:
        print ("Failed: " + str(e))
        raise e