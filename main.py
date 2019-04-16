from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import build_graph
import graph_to_embedding
from sklearn.linear_model import LogisticRegression
from src.graph import Graph
import build_graph
from graph_to_embedding import *
from feature_generation import *
import pickle

if __name__ == '__main__':
    corpus = "20NG"
    # graph_filepath = build_graph("20NG")
    graph_filepath = "data/graphs_saved/20NG.edgelist"
    g = Graph()
    print("Reading...")
    g.read_edgelist(filename=graph_filepath, weighted=True, directed=False)
    methods = ['node2vec', 'deepWalk', 'line']
    for method in methods[:1]:
        emb = graph_to_embedding(g, method, corpus)
        X_train, y_train, X_test, y_test, categories = get_feature_matrix(emb, corpus, "avg")

        fmodelpath = "data/model_saved/%s_128.model" % method
        if os.path.exists(fmodelpath):
            with open(fmodelpath, "rb") as fmodel:
                forest = pickle.load(fmodel)
        else:
            svc = svm.LinearSVC()
            parameters = [{'C': [0.01, 0.1, 1, 10, 100, 1000]}]
            clf = GridSearchCV(svc, parameters, n_jobs=-1, cv=10)
            print("Training the classifier...")
            clf.fit(X_train,y_train)
            forest = clf.fit(X_train, y_train)
            with open(fmodelpath ,"wb") as fmodel:
                pickle.dump(forest, fmodel)

        pred_train = forest.predict(X_train)

        # training score
        score = accuracy_score(y_train, pred_train)
        # score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
        acc = "Accuracy in training set:" + str(score)
        mac = "Macro:" + str(metrics.precision_recall_fscore_support(y_train, pred_train, average='macro'))
        mic = "Micro:" + str(metrics.precision_recall_fscore_support(y_train, pred_train, average='micro'))

        report = "\nFeatures shape:" + str(X_train.shape) + "\n"
        report += '\n'.join([acc, mac, mic])
        path_results = "results/" + corpus + "/" + method + "_" + time.asctime(time.localtime(time.time())) + '_'

        pred_test = forest.predict(X_test)

        # testing score
        score = accuracy_score(y_test, pred_test)
        # score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
        acc = "Accuracy in testing set:" + str(score)
        mac = "Macro test:" + str(metrics.precision_recall_fscore_support(y_test, pred_test, average='macro'))
        mic = "Micro test:" + str(metrics.precision_recall_fscore_support(y_test, pred_test, average='micro'))
        met = metrics.classification_report(y_test, pred_test, target_names=categories, digits=4)
        report += '\n'+'\n'.join([acc, mac, mic, met])

        print(report)
        path_results = "results/" + corpus + "/" + method + "_" + time.asctime(time.localtime(time.time())) + '.txt'
        with open(path_results,'w') as fout:
            fout.write(report)

        '''
        ## Statistical Significance
        base_test = []
        with open("tfidf_predictions_False.txt", "r") as f:
            for line in f:
                base_test.append(int(line))

        n_trials = np.sum(base_test != pred_test)
        n_succ = 0
        p = 0.05
        for count_elem, y_elem in enumerate(pred_test):
            if y_elem == y_test[count_elem] and y_test[count_elem] != base_test[count_elem]:
                n_succ += 1

        p_value = scipy.stats.binom_test(n_succ, n_trials)
        sign_bool = p_value < p
        print
        "Significance:" + str(p_value) + " " + str(sign_bool)

        text_file.write("\n" + "Features shape:" + str(features.shape) + "\n")
        text_file.write(acc + "\n" + mac + "\n" + mic + "\n" + met + "\n\n")
        text_file.write("Significance:" + str(p_value) + " " + str(sign_bool) + "\n")
        text_file.close()

        all_results.write(
            s_res + " Accuracy:" + str(score) + " Significance:" + str(p_value) + " " + str(sign_bool) + "\n")

        accs.append(score)
        f1s.append(metrics.f1_score(y_test, pred_test, average='macro'))
        
        if args.label_file and args.method != 'gcn':
            vectors = model.vectors
            X, Y = read_node_label(args.label_file)
            print("Training classifier using {:.2f}% nodes...".format(
                args.clf_ratio * 100))
            clf = Classifier(vectors=vectors, clf=LogisticRegression())
            clf.split_train_evaluate(X, Y, args.clf_ratio)
            Initialize a Random Forest classifier with 100 trees
            clf = RandomForestClassifier(n_estimators = 100)
            if classifier_par=="svm":
                svc = svm.LinearSVC()
                parameters = [{'C':[0.01,0.1,1,10,100,1000]}]
                clf = GridSearchCV(svc, parameters,n_jobs=-1,cv=10)
            elif classifier_par=="log":
                clf = SGDClassifier(loss="log")
            elif classifier_par=="cnn":
                clf = SGDClassifier(loss="log")
        '''
