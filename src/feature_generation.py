from sklearn import preprocessing
from src.graph_to_embedding import *
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
                y_test = le.transform(y_test)
        else:
            with open("data/%s/%s-train-stemmed.txt" % (corpus, corpus)) as f:
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
            with open("data/%s/%s-test-stemmed.txt" % (corpus, corpus)) as f:
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
                y_test = le.transform(y_test)
        x_train[np.isnan(x_train)] = 0
        x_test[np.isnan(x_test)] = 0
        return x_train, y_train, x_test, y_test, le.classes_
            

    if combining == "w_avg":
        if corpus == "20NG":
            with open("data/20NG/20ng-train-stemmed.txt" % (corpus, corpus)) as f:
                collection = f.readlines()
                n = len(collection)
                m = emb["phone"].shape[0]
                x_train = np.zeros((n, m))
                y_train = [""]*n
                for i, a_line in enumerate(collection):
                    doc = a_line.split()
                    y_train[i] = doc[0]
                    words = set(doc[1:])
                    doc_vecs = []
                    for word in words:
                        if word in emb.keys():
                            doc_vecs.append(emb[word])
                    x_train[i] = np.mean(doc_vecs, axis=0)
                y_train = le.fit_transform(y_train)
            with open("data/20NG/20ng-test-stemmed.txt" % (corpus, corpus)) as f:
                collection = f.readlines()
                n = len(collection)
                m = emb["phone"].shape[0]
                x_test = np.zeros((n, m))
                y_test = [""]*n
                for i, a_line in enumerate(collection):
                    doc = a_line.split()
                    y_test[i] = doc[0]
                    words = set(doc[1:])
                    doc_vecs = []
                    for word in words:
                        if word in emb.keys():
                            doc_vecs.append(emb[word])
                    x_test[i] = np.mean(doc_vecs, axis=0)
                y_test = le.transform(y_test)
        else:
            with open("data/%s/%s-train-stemmed.txt") as f:
                collection = f.readlines()
                n = len(collection)
                m = emb["phone"].shape[0]
                x_train = np.zeros((n, m))
                y_train = [""]*n
                for i, a_line in enumerate(collection):
                    doc = a_line.split()
                    y_train[i] = doc[0]
                    words = set(doc[1:])
                    doc_vecs = []
                    for word in words:
                        if word in emb.keys():
                            doc_vecs.append(emb[word])
                    x_train[i] = np.mean(doc_vecs, axis=0)
                y_train = le.fit_transform(y_train)
            with open("data/%s/%s-test-stemmed.txt") as f:
                collection = f.readlines()
                n = len(collection)
                m = emb["phone"].shape[0]
                x_test = np.zeros((n, m))
                y_test = [""]*n
                for i, a_line in enumerate(collection):
                    doc = a_line.split()
                    y_test[i] = doc[0]
                    words = set(doc[1:])
                    doc_vecs = []
                    for word in words:
                        if word in emb.keys():
                            doc_vecs.append(emb[word])
                    x_test[i] = np.mean(doc_vecs, axis=0)
                y_test = le.transform(y_test)
        x_train[np.isnan(x_train)] = 0
        x_test[np.isnan(x_test)] = 0
        return x_train, y_train, x_test, y_test, le.classes_

if __name__ == '__main__':
    corpus = "20NG"
    # graph_filepath = build_graph("20NG")
    graph_filepath = "data/graphs_saved/20NG.edgelist"
    g = Graph()
    print("Reading...")
    g.read_edgelist(filename=graph_filepath, weighted=True, directed=False)
    methods = ['node2vec', 'deepWalk', 'line']
    for method in methods[1:]:
        emb = graph_to_embedding(g, method, corpus)
        X_train, y_train, X_test, y_test = get_feature_matrix(emb, corpus, "avg")