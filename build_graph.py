import networkx as nx
import pickle
import time


def Time(s):
    print(time.asctime(time.localtime())+". At %d s, " % (time.time() - start) + s)


def build_col_graph_from_corpus(nb_chunk):
    G = nx.Graph()
    for i in range(1, nb_chunk + 1):
        fin = "cleaned_wiki/cleaned-wiki-chunk_%d.txt" % i  # todo
        with open(fin, 'r') as f:
            s = f.read().split()
            length = len(s)
            for k, word in enumerate(s):
                if not G.has_node(word):
                    G.add_node(word, count=1)
                else:
                    G.node[word]['count'] += 1
                    for j in range(sliding_window):
                        try:
                            if k + j + 1 >= length:
                                break
                            next_word = s[k + j + 1]

                            if not G.has_node(next_word):
                                G.add_node(next_word, count=0)

                            if not G.has_edge(word, next_word):
                                G.add_edge(word, next_word, weight=1)
                                # G.edge[word][next_word]['w2vec'] = 0.01
                            else:
                                G.edge[word][next_word]['weight'] += 1
                        except:
                            print("aaa")
        if i % 1 == 0:
            Time("%d chunk dealt." % i)
            print(nx.info(G))


def build_col_graph_from_train(filepath,sliding_window):
    G = nx.Graph()
    with open(filepath, 'r') as f:
        docs = f.readlines()
        print("Total number of documents: %d"%(len(docs)))
        for doc in docs:
            s = doc.split()
            s = s[1:]
            length = len(s)
            for k, word in enumerate(s):
                if not G.has_node(word):
                    G.add_node(word, count=1)
                else:
                    G.node[word]['count'] += 1
                    for j in range(sliding_window):
                        try:
                            if k + j + 1 >= length:
                                break
                            next_word = s[k + j + 1]

                            if not G.has_node(next_word):
                                G.add_node(next_word, count=0)

                            if not G.has_edge(word, next_word):
                                G.add_edge(word, next_word, weight=1)
                                # G.edge[word][next_word]['w2vec'] = 0.01
                            else:
                                G[word][next_word]['weight'] += 1
                        except:
                            print("UNKNOWN ERROR")
    return G

# print len(G.node)

# print list(G.nodes())[1:10]
# print list(G.edges())[1:10]

# with open("large_graph%d.data"%(nb_chunk),'wb') as fout:
#     pickle.dump(G,fout)
# nx.write_gpickle(G, "large_graph%d.pickle"%(nb_chunk))
# Time("Graph saved in pickle. All Done")


if __name__ == '__main__':
    sliding_window = 2
    global start
    start = time.time()
    Time("Here we go!")
    corpus = "20NG"
    prefix = "data/" + corpus + '/'
    filename = "20ng-train-stemmed.txt"
    filepath = prefix + filename

    # Buld graph from training data
    G = build_col_graph_from_train(filepath,sliding_window)
    print(nx.info(G))
    G.name = "G"

    # save graph
    Time("Graph built")
    nx.write_weighted_edgelist(G, "Graphs_saved/" + corpus + ".edgelist")
    Time("Graph saved in weighted edgelist.")

    # # save reduced graph
    # nodesLowerThan5 = [x for x, y in G.nodes(data=True) if y['count'] < 5]
    # G.remove_nodes_from(nodesLowerThan5)
    # G.name = "G_reduced"
    # print(nx.info(G))
    # nx.write_weighted_edgelist(G, "Graphs_saved/" + corpus + "_reduced.edgelist")
    # Time("Reduced graph saved in weighted edgelist.")

