from __future__ import print_function
from src.graph import Graph
from src import node2vec
from src import line

g = Graph()
print("Reading...")
args = {}
if args["graph_format"] == 'adjlist':
    g.read_adjlist(filename=args["input"])
elif args["graph_format"] == 'edgelist':
    g.read_edgelist(filename=args["input"], weighted=args["weighted"],
                    directed=args["directed"])
if args["method"] == 'node2vec':
    model = node2vec.Node2vec(graph=g, path_length=args["walk_length"],
                              num_paths=args["number_walks"], dim=args["representation_size"],
                              workers=args["workers"], p=args["p"], q=args["q"], window=args["window_size"])
elif args["method"] == 'line':
    if args["label_file"] and not args["no_auto_save"]:
        model = line.LINE(g, epoch=args["epochs"], rep_size=args["representation_size"], order=args["order"],
                          label_file=args["label_file"], clf_ratio=args["clf_ratio"])
    else:
        model = line.LINE(g, epoch=args["epochs"],
                          rep_size=args["representation_size"], order=args["order"])
elif args["method"] == 'deepWalk':
    model = node2vec.Node2vec(graph=g, path_length=args["walk_length"],
                              num_paths=args["number_walks"], dim=args["representation_size"],
                              workers=args["workers"], window=args["window_size"], dw=True)