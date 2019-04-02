import build_graph
import graph_to_embedding

if __name__ == '__main__':
    build_graph("20NG")
    graph_to_embedding(build_graph("20NG"))