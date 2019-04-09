from __future__ import print_function
from src.graph import Graph
from src import node2vec
from src import line
import time
import os

def save_embeddings(model, output_path, method, nb_workers, start):
    t2 = time.time()
    if method != "line":
        print("\nrunning time for %s with %d workers is %f" % (method, nb_workers, t2 - start))
    else:
        print("\nrunning time for %s is %f" % (method, t2 - start))
    if method != 'gcn':
        print("Saving embeddings...")
        print(output_path)
        model.save_embeddings(output_path)


corpus = "20NG"
t0 = time.time()
g = Graph()
print("Reading...")
methods = ['node2vec', 'deepWalk', 'line']

args = {}
graph_filepath = "data/graphs_saved/20NG.edgelist"
method = "deepWalk"
args["graph_format"] = 'edgelist'
walk_length = 80
nb_walks = 10
emb_dim = 128
window_size = 10
nb_workers = 3
output_path = ""
model = None


if args["graph_format"] == 'edgelist':
    g.read_edgelist(filename=graph_filepath, weighted=True, directed=False)


for method in methods:
    start = time.time()
    if method == 'node2vec':
        p = 10
        q = 0.5
        model = node2vec.Node2vec(graph=g, path_length=walk_length,
                                  num_paths=nb_walks, dim=emb_dim,
                                  workers=3, p=p, q=q, window=window_size)
        output_path = "data/emb_saved/%s_%s_%d_%d_%d_%d_%.2f_%.2f.emb" % \
                      (corpus, method, emb_dim, nb_walks, walk_length, window_size, p, q)

    elif method == 'line':
        order = 3
        nb_epochs = 10
        model = line.LINE(g, epoch=nb_epochs, rep_size=emb_dim, order=order)
        output_path = "data/emb_saved/%s_%s_%d_%d.emb" % \
                      (corpus, method, emb_dim, order)

    elif method == 'deepWalk':
        output_path = "data/emb_saved/%s_%s_%d_%d_%d_%d.emb" % \
                      (corpus, method, emb_dim, nb_walks, walk_length, window_size)
        if os.path.exists(output_path):
            pass
            # return(read_emb(output_path))
        else:
            pass
            model = node2vec.Node2vec(graph=g, path_length=walk_length,
                                      num_paths=nb_walks, dim=emb_dim,
                                      workers=nb_workers, window=window_size, dw=True)

    save_embeddings(model, output_path, method, nb_workers, start)

# embeddings = model.vectors
# print(embeddings["phone"])
