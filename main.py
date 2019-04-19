from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import networkx as nx
import build_graph
from graph_to_embedding import graph_to_embedding
from sklearn.linear_model import LogisticRegression
from src.graph import Graph
from build_graph import build_col_graph_from_train
from graph_to_embedding import *
from feature_generation import *
import pickle

def Time(s):
    str = time.asctime(time.localtime())+". At %d s, " % (time.time() - start) + s + '\n'
    print(str,end='')
    return str


if __name__ == '__main__':
    global start
    methods = ['deepWalk', 'line', 'node2vec']
    combining = 'avg'

    # parameter sensitivity on emb_dim
    # corpus = "20NG"
    # graph_filepath = build_col_graph_from_train(corpus)
    # g = Graph()
    # g.read_edgelist(filename=graph_filepath, weighted=True, directed=False)
    # for emb_dim in [4,8,16,32,64,128,256][2:]:
    #     for method in methods:

    # parameter sensitivity on p and q
    corpus = "webkb"
    graph_filepath = build_col_graph_from_train(corpus)
    g = Graph()
    g.read_edgelist(filename=graph_filepath, weighted=True, directed=False)
    plist = [0.1, 0.3, 1, 3]
    method = 'node2vec'
    emb_dim = 128
    for p in plist:
        for q in plist:
            start = time.time()
            # DEBUG
            if emb_dim == 16 and method == 'line' or method == 'deepWalk':
                continue
            print(corpus, emb_dim, method)

            # GRAPH TO EMBEDDINGS
            report = "%s %d %s\n"%(corpus, emb_dim, method)
            emb = graph_to_embedding(g, method, corpus, emb_dim)
            report += Time("node(word) embeddings trained/read from file.")

            # EMBEDDING TO FEATURE MATRIX
            X_train, y_train, X_test, y_test, categories = get_feature_matrix(emb, corpus, combining)
            report += Time("feature matrix generated")

            fmodelpath = "data/model_saved/%s_%s_%s_%d.model" % (corpus, method, combining, emb_dim)
            if os.path.exists(fmodelpath):
                with open(fmodelpath, "rb") as fmodel:
                    forest = pickle.load(fmodel)
                    print("Model read from file.")
            else:
                svc = svm.LinearSVC()
                parameters = [{'C': [0.01, 0.1, 1, 10, 100, 1000]}]
                clf = GridSearchCV(svc, parameters, n_jobs=-1, cv=10)
                print("Training the classifier...")
                clf.fit(X_train,y_train)
                forest = clf.fit(X_train, y_train)
                with open(fmodelpath ,"wb") as fmodel:
                    pickle.dump(forest, fmodel)
            report += Time("svm model trained/read from file")

            pred_train = forest.predict(X_train)

            # training score
            score = accuracy_score(y_train, pred_train)
            # score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
            acc = "Accuracy in training set:" + str(score)
            mac = "Macro:" + str(metrics.precision_recall_fscore_support(y_train, pred_train, average='macro'))
            mic = "Micro:" + str(metrics.precision_recall_fscore_support(y_train, pred_train, average='micro'))
            met = metrics.classification_report(y_train, pred_train, target_names=categories, digits=4)

            report += "\nFeatures shape:" + str(X_train.shape) + "\n"
            report += '\n'.join([acc, mac, mic, met])

            pred_test = forest.predict(X_test)

            # testing score
            score = accuracy_score(y_test, pred_test)
            # score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
            acc = "Accuracy in testing set:" + str(score)
            mac = "Macro test:" + str(metrics.precision_recall_fscore_support(y_test, pred_test, average='macro'))
            mic = "Micro test:" + str(metrics.precision_recall_fscore_support(y_test, pred_test, average='micro'))
            met = metrics.classification_report(y_test, pred_test, target_names=categories, digits=4)
            report += '\n'+'\n'.join([acc, mac, mic, met])
            report += Time("all done.")

            print(report)
            path_results = "results/%s_%s_%s_%d.txt" % (corpus, method, combining, emb_dim)
            with open(path_results,'w') as fout:
                fout.write(report)
