import networkx as nx
import pickle
import time
G = nx.Graph()
nb_chunk = 6936
nb_chunk = 6936
sliding_window = 2

def Time(s):
    print(time.asctime(time.localtime())+". At %d s, " % (time.time() - start) + s)

start = time.time()
Time("HH")
for i in range(1,nb_chunk+1):
    fin = "cleaned_wiki/cleaned-wiki-chunk_%d.txt" % i
    with open(fin,'r') as f:
        s = f.read().split()
        length = len(s)
        for k,word in enumerate(s):
            if not G.has_node(word):
                G.add_node(word, count=1)
            else:
                G.node[word]['count'] += 1
                for j in xrange(sliding_window):
                    try:
                        if k+j+1 >= length:
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
    if i %1 == 0:
        Time("%d chunk dealt." % i)
        print(nx.info(G))

# print len(G.node)
print(nx.info(G))
# print list(G.nodes())[1:10]
# print list(G.edges())[1:10]
Time("Graph built")
# with open("large_graph%d.data"%(nb_chunk),'wb') as fout:
#     pickle.dump(G,fout)
# nx.write_gpickle(G, "large_graph%d.pickle"%(nb_chunk))
# Time("Graph saved in pickle. All Done")
nodesLowerThan5 = [x for x,y in G.nodes(data=True) if y['count'] < 5]
G.remove_nodes_from(nodesLowerThan5)
nx.write_weighted_edgelist(G, "Graphs_saved/large_graph_reduced_%d.edgelist" % nb_chunk)
Time("Graph saved in weighted edgelist. All Done")
