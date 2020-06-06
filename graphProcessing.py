import numpy as np
import networkx as nx

# Code for loading graphs from the "Benchmark Data Sets for Graph Kernels" repository,
# https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

def load_graph(fileName_graph,fileName_indicators,fileName_labels):
    # Input: three filenames of files from the website.
    # Output: List of graph objects and their corresponding labels.

    # Load in edges from the graph file.
    edges = []

    for line in open(fileName_graph):
        split_line = line.split(',')
        edges.append(tuple([int(split_line[0]),int(split_line[1])]))

    # Load in the indicators

    indicators = []

    for line in open(fileName_indicators):
        indicators.append(int(line))

    indicators = np.array(indicators)

    num_graphs = max(indicators)

    # Create a large graph object for all of the graphs.
    bigGraph = nx.Graph()
    bigGraph.add_edges_from(edges)

    # Load in the labels

    labels = []

    for line in open(fileName_labels):
        labels.append(int(line))

    # Create a large adjacency matrix for all of the graphs.
    # This will be subdivided into a list of small adjacency matrices.
    big_adjacency_matrix = nx.to_numpy_array(bigGraph)

    adjacency_matrices = []

    for j in range(num_graphs):
        inds = np.where(indicators == j+1)[0]
        MIN = min(inds)
        MAX = max(inds)
        adjacency_matrices.append(big_adjacency_matrix[MIN:MAX+1,MIN:MAX+1])

    # Define a graph object for each small adjacency matrix.

    graphs = []

    for j in range(num_graphs):
        graphs.append(nx.from_numpy_array(adjacency_matrices[j]))

    return graphs, labels
