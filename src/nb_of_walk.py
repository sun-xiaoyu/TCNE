from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from build_graph import build_col_graph_from_train
from feature_generation import *
import pickle

def Time(s):
    str = time.asctime(time.localtime())+". At %d s, " % (time.time() - start) + s + '\n'
    print(str,end='')
    return str


# if __name__ == '__main__':
global start
# os.chdir("..")
methods = ['deepWalk', 'line', 'node2vec']
combining = 'avg'

# parameter sensitivity on walk length
corpus = "20NG"
corpus = "webkb"
graph_filepath = build_col_graph_from_train(corpus)
g = Graph()
emb_dim = 128
g.read_edgelist(filename=graph_filepath, weighted=True, directed=False)
for nb_walks in [1, 2, 4, 8, 16, 32, 64, 80, 128]:
    for method in methods[:1]:
        start = time.time()
        path_results = "results/nb_walks/%s_%s_%s_%d_%d.txt" % (corpus, method, combining, emb_dim, nb_walks)
        if os.path.exists(path_results):
            continue

        print(corpus, emb_dim, method)

        # GRAPH TO EMBEDDINGS
        report = "%s %d %s %d \n" % (corpus, emb_dim, method, nb_walks)
        emb = graph_to_embedding(g, method, corpus, emb_dim, nb_walks=nb_walks)
        report += Time("node(word) embeddings trained/read from file.")

        # EMBEDDING TO FEATURE MATRIX
        X_train, y_train, X_test, y_test, categories = get_feature_matrix(emb, corpus, combining)
        report += Time("feature matrix generated")

        fmodelpath = "data/model_saved/%s_%s_%s_%d_%d_%d.model" % (corpus, method, combining, emb_dim, 80, nb_walks)
        if os.path.exists(fmodelpath):
            with open(fmodelpath, "rb") as fmodel:
                forest = pickle.load(fmodel)
                report += Time("Model read from file.")
        else:
            svc = svm.LinearSVC()
            parameters = [{'C': [0.01, 0.1, 1, 10, 100, 1000]}]
            clf = GridSearchCV(svc, parameters, n_jobs=25, cv=10)
            print("Training the classifier...")
            clf.fit(X_train, y_train)
            forest = clf.fit(X_train, y_train)
            with open(fmodelpath ,"wb") as fmodel:
                pickle.dump(forest, fmodel)
        report += Time("svm model trained from file")

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
        with open(path_results,'w') as fout:
            fout.write(report)
