import numpy as np
import time
import glob
import w3lib.html
from w3lib.html import remove_tags
import re
import codecs
from scrapy.utils.markup import remove_tags
from sklearn.model_selection import train_test_split

global unique_words

from data_cleaner import *
import os.path
import multiprocessing as mp
from functools import partial
import time

if __name__=="__main__":

    begin = time.time()
    words_frequency = {}
    data_set = "WebKB"
    categories = ['course', 'faculty', 'project', 'student']

    clean_train_documents = []
    y_train = []

    y_test = []
    clean_test_documents = []
    flag_debug_always_restart = False

    if os.path.exists("data/WebKB/my_WEBKB_train.txt") and os.path.exists("data/WebKB/my_WEBKB_train_VOCAB.txt") \
            and os.path.exists("data/WebKB/my_WEBKB_train.txt") and not (flag_debug_always_restart):
        ## Open the file with read only permission

        print("Haha! Data already cleaned!")

        f = codecs.open('data/WebKB/my_WEBKB_train_VOCAB.txt', "r", encoding="utf-8")
        unique_words = [x.strip('\n') for x in f.readlines()]
        f.close()

        f = codecs.open('data/WebKB/my_WEBKB_train.txt', "r", encoding="utf-8")
        train = [x.strip('\n') for x in f.readlines()]
        f.close()

        num_documents = len(train)

        for i in xrange(0, num_documents):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            line = train[i].split('\t')

            if line[1].split(" ") > 1:
                y_train.append(line[0])

                for n, w in enumerate(line[1].split(' ')):
                    if w not in words_frequency:
                        words_frequency[w] = 1
                    else:
                        words_frequency[w] = words_frequency[w] + 1

                clean_train_documents.append(line[1])

        # unique_words = list(words_frequency.keys())
        # unique_words = [k for (k,v) in words_frequency.items() if v>1]

        ## Open the file with read only permit
        f = codecs.open('data/WebKB/my_WEBKB_test.txt', "r", encoding="utf-8")
        test = [x.strip('\n') for x in f.readlines()]
        f.close()

        num_test_documents = len(test)

        for i in xrange(0, num_test_documents):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            line = test[i].split('\t')

            if line[1].split(" ") > 1:
                y_test.append(line[0])
                clean_test_documents.append(line[1])
    else:
        raw_data_all = []
        y_all = []
        schools = ['cornell', 'misc', 'texas', 'washington', 'wisconsin']
        for school in schools:
            for i, cat in enumerate(categories):
                path = 'data/WebKB/' + cat + '/' + school + '/*'
                files = glob.glob(path)
                for fpath in files:
                    '''
                    f = open(fpath, 'r')
                    raw_data_all.append(f.read())
                    y_all.append(i)
                    f.close()
                    '''  # try this next time?
                    with codecs.open(fpath, "rb", encoding='utf-8', errors='ignore') as fdata:
                        raw_data_all.append(fdata.read())
                    y_all.append(i)
        print raw_data_all[:1]
        print("Data size: %d" % len(raw_data_all))

        pool = mp.Pool(processes=4)  # we use 4 cores
        # unique_words = pool.map(findVocab, raw_data_all)
        unique_words = findVocab(raw_data_all)

        print("At %ds, Size of vocabulary:%d." % (time.time() - begin, len(unique_words)))
        # unique_words = findVocab(raw_data_all)

        # clean the documents by removing the tags
        cleaned_data_all = []
        for t in raw_data_all:
            cleaned_data_all.append(remove_tags(t))  # (unicode(t, errors='ignore')))
        print cleaned_data_all[:1]
        print("At %ds, all document are cleaned." % (time.time() - begin))

        # parse the documents

        partial_work = partial(parseXmlStopStemRem1by1, unique_words=unique_words)
        parsed_data_all = pool.map(partial_work, cleaned_data_all)
        # parsed_data_all = parseXmlStopStemRem(cleaned_data_all, unique_words)
        print("At %ds, all document parsed." % (time.time() - begin))
        print parsed_data_all[:2]

        train_data, test_data, y_train_all, y_test_all = \
            train_test_split(parsed_data_all, y_all, test_size=0.33, random_state=42)

        # Back up!
        f = codecs.open('data/WebKB/my_WEBKB_train_VOCAB.txt', "w", encoding="utf-8")
        for item in unique_words:
            f.write("%s\n" % item)
        f.close()
        f = codecs.open('data/WebKB/my_WEBKB_train.txt', "w", encoding="utf-8")
        for i, doc in enumerate(train_data):
            s = doc.split(" ")
            # print s
            if len(set(s)) > 1:
                clean_train_documents.append(doc)
                f.write(str(categories[y_train_all[i]]) + "\t" + doc + "\n")
                y_train.append(y_train_all[i])
        f.close()
        f = codecs.open('data/WebKB/my_WEBKB_test.txt', "w", encoding="utf-8")
        for i, doc in enumerate(test_data):
            s = doc.split(" ")
            if len(set(s)) > 1:
                f.write(str(categories[y_test_all[i]]) + "\t" + doc + "\n")
                clean_test_documents.append(doc)
                y_test.append(y_test_all[i])
        f.close()
        print("At %ds, all data backed up:" % (time.time() - begin))

    # Get the number of documents based on the dataframe column size

    print "Unique words:" + str(len(unique_words))

    num_documents = len(clean_train_documents)
    print "Length of train data:" + str(num_documents)

    num_test_documents = len(clean_test_documents)
    print "Length of test data:" + str(num_test_documents)

    b = 0.003
    idf_pars = ["no"]
    kcore_par = "A0"
    classifier_par = "svm"

    path = "/home/sunxiaoyu/Graph-Based-TC-master/webkb/"

    sliding_windows = [5]
    method = "node2vec"
    # sliding_windows = range(2,3)

