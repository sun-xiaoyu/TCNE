import build_graph
import graph_to_embedding
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    graph_filepath = build_graph("20NG")
    emb, emb_filepath = graph_to_embedding(graph_filepath)
    train, test = get_feature_matrix(emb,)
    if args.label_file and args.method != 'gcn':
        vectors = model.vectors
        X, Y = read_node_label(args.label_file)
        print("Training classifier using {:.2f}% nodes...".format(
            args.clf_ratio * 100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio)