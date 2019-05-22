from __future__ import print_function
from gensim.models import Word2Vec
import walker
import json
import os


class Node2vec(object):

    def __init__(self, graph, corpus, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):

        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph

        walk_path = "data/graphs_saved/%s_%d_%d_%.1f,%.1f" % (corpus, path_length, num_paths, p, q)
        if os.path.exists(walk_path):
            with open(walk_path, 'r') as fin:
                sentences = json.load(fin)
            print("Random walks read from file.")
        else:

            if dw:
                self.walker = walker.BasicWalker(graph, workers=kwargs["workers"])
            else:
                self.walker = walker.Walker(
                    graph, p=p, q=q, workers=kwargs["workers"])
                print("Preprocess transition probs...")
                self.walker.preprocess_transition_probs()
            sentences = self.walker.simulate_walks(
                num_walks=num_paths, walk_length=path_length)
            with open(walk_path, 'w') as fout:
                json.dump(sentences, fout)
            print("Random walks written to " + walk_path)

        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
