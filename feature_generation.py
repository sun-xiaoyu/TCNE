from sklearn import preprocessing
import numpy as np


def get_feature_matrix(emb, corpus, combining):
    le = preprocessing.LabelEncoder()
    if combining == "avg":
        if corpus == "20NG":
            with open("data/20NG/20ng-train-stemmed.txt") as f:
                collection = f.readlines()
                n = len(collection)
                m = emb["phone"].shape[0]
                x_train = np.zeros((n, m))
                y_train = [""]*n
                for i, a_line in enumerate(collection):
                    doc = a_line.split()
                    y_train[i] = doc[0]
                    doc_vecs = []
                    for word in doc[1:]:
                        if word in emb.keys():
                            doc_vecs.append(emb[word])
                    x_train[i] = np.mean(doc_vecs, axis=0)
                y_train = le.fit_transform(y_train)
            with open("data/20NG/20ng-test-stemmed.txt") as f:
                collection = f.readlines()
                n = len(collection)
                m = emb["phone"].shape[0]
                x_test = np.zeros((n, m))
                y_test = [""]*n
                for i, a_line in enumerate(collection):
                    doc = a_line.split()
                    y_test[i] = doc[0]
                    doc_vecs = []
                    for word in doc[1:]:
                        if word in emb.keys():
                            doc_vecs.append(emb[word])
                    x_test[i] = np.mean(doc_vecs, axis=0)
                y_test = le.fit_transform(y_test)
            return x_train, y_train, x_test, y_test
