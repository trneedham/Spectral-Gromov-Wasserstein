"""
Code to visualize Sturm geodesics in the 'space of spaces'.
"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
from gromovWassersteinAveraging import *

"""
Function to draw graphs with edge weights.
"""

def draw_weighted_graph(G,nodePos):
    # Inputs:
    #   weighted Graph G as networkx graph object
    #   node positions, either as a dictionary or as a numpy matrix of size (number of nodes) x 2

    # Draw nodes
    nx.draw_networkx_nodes(G,pos = nodePos)

    # Create a list of unique edgeweights
    all_weights = []

    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight'])

    unique_weights = list(set(all_weights))

    # Draw the edges
    for weight in unique_weights:
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr)
                          in G.edges(data=True) if edge_attr['weight']==weight]

        width = weight*len(G.nodes)*3.0/sum(all_weights)

        nx.draw_networkx_edges(G,pos = nodePos,edgelist=weighted_edges,width=width)

    plt.axis('off')

"""
Functions to split nodes based on an optimal coupling between graphs.
"""

def split_nodePos_one_point(vec,vec_index,nodePos_matrix):
    node_location = nodePos_matrix[vec_index]
    nonzeros = len(vec[vec != 0])
    for j in range(nonzeros-1):
        nodePos_matrix = np.insert(nodePos_matrix,[vec_index],node_location,axis = 0)
    return nodePos_matrix

def split_nodePos_all_points(bad_vecs, vec_indices, num_copies, nodePos_matrix):
    for j in range(len(vec_indices)):
        nodePos_matrix = split_nodePos_one_point(bad_vecs[j],vec_indices[j],nodePos_matrix)
        vec_indices = [index+num_copies[j]-1 for index in vec_indices]
    return nodePos_matrix

def split_cost_and_probability_by_row_insert_nodes(coup,C,p,nodePos_matrix):
    bad_rows, row_indices, num_copies = find_bad_rows(coup)
    C = split_matrix_all_points(row_indices, num_copies, C)
    p = split_probability_all_points(bad_rows, row_indices, num_copies, p)
    nodePos_matrix = split_nodePos_all_points(bad_rows,row_indices,num_copies,nodePos_matrix)
    return C, p, nodePos_matrix

def split_cost_and_probability_by_column_insert_nodes(coup,C,p,nodePos_matrix):
    bad_columns, column_indices, num_copies = find_bad_columns(coup)
    bad_columns = bad_columns.T
    C = split_matrix_all_points(column_indices, num_copies, C)
    p = split_probability_all_points(bad_columns, column_indices, num_copies, p)
    nodePos_matrix = split_nodePos_all_points(bad_columns, column_indices,num_copies,nodePos_matrix)
    return C, p, nodePos_matrix

def split_cost_coupling_probabilities_insert_nodes(coup, C1, C2, p1, p2,
                                                   nodePos1_matrix, nodePos2_matrix, thresh=1):
    coup = normalized_threshold(coup, p1, thresh)
    C1, p1, nodePos1_matrix = split_cost_and_probability_by_row_insert_nodes(coup,C1,p1,nodePos1_matrix)
    C2, p2, nodePos2_matrix = split_cost_and_probability_by_column_insert_nodes(coup,C2,p2,nodePos2_matrix)
    coup = split_coupling(coup)
    return coup, C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix

"""
Align graphs via optimal coupling
"""

def align_graphs(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix):
    opt_coup, log = gromov_wasserstein_asym(C1,C2,p1,p2)
    coup, C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = split_cost_coupling_probabilities_insert_nodes(opt_coup, C1, C2, p1, p2,
                                                                                nodePos1_matrix, nodePos2_matrix, thresh=1)
    perm = 1*(coup != 0)
    C2 = np.matmul(np.matmul(perm,C2),perm.T)
    nodePos2_matrix = np.matmul(perm,nodePos2_matrix)

    return C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix

"""
Geodesic drawing. Need to preprocess and align matched nodes between the graphs.
"""

def preprocess_nodes(nodePos_matrix):

    # Center the nodes
    mean = np.mean(nodePos_matrix, axis = 0)
    nodePos_matrix = nodePos_matrix - mean

    # Rescale to have total norm 1
    nodePos_matrix = nodePos_matrix/np.linalg.norm(nodePos_matrix)

    return nodePos_matrix

def align_nodes(nodePos1_matrix,nodePos2_matrix):
    # Inputs must be matrices of the SAME SIZE!
    # We're assuming nodes have been pre-registered before this step

    # Preprocess by centering and rescaling
    nodePos1_matrix = preprocess_nodes(nodePos1_matrix)
    nodePos2_matrix = preprocess_nodes(nodePos2_matrix)

    # Align the centered and rescaled data using SVD
    H = np.matmul(nodePos1_matrix.T,nodePos2_matrix)
    U, S, Vt = np.linalg.svd(H, full_matrices=True)
    R = np.matmul(Vt.T,U.T)

    nodePos1_matrix = np.transpose(np.matmul(R,nodePos1_matrix.T))

    return nodePos1_matrix, nodePos2_matrix

def draw_geodesic(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix):

    C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = align_graphs(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix)

    nodePos1_matrix, nodePos2_matrix = align_nodes(nodePos1_matrix,nodePos2_matrix)

    num_steps = 10

    ts = np.linspace(0,1,num_steps)

    fig = plt.figure(figsize = (5*num_steps,20))

    for j in range(num_steps):

        t = ts[j]

        nodePos_matrix = (1-t)*nodePos1_matrix + t*nodePos2_matrix

        C = (1-t)*C1 + t*C2

        # To accentuate weights in the pictures:
        C = 10 * C**2

        G = nx.from_numpy_array(C)

        plt.subplot(2,num_steps/2,j+1)

        draw_weighted_graph(G,nodePos_matrix)

    return

"""
Repeating the above, but also depict node weights by size.
"""

def fix_probability_vector(p,nodePos_matrix):
    p_new = np.zeros(len(p))
    for j in range(len(p)):
        num_copies = len(list(np.where((nodePos_matrix == (nodePos_matrix[j][0], nodePos_matrix[j][1])).all(axis=1))[0]))
        p_new[j] = num_copies*p[j]
        # p = p/sum(p)
    return p_new

def draw_node_weighted_graph(G,p,nodePos):
    # Inputs:
    #   weighted Graph G as networkx graph object
    #   probability vector with length (number of nodes)
    #   node positions, either as a dictionary or as a numpy matrix of size (number of nodes) x 2


    # Draw nodes
    nx.draw_networkx_nodes(G,pos = nodePos,node_size=5000*p)

    # Create a list of unique edgeweights
    all_weights = []

    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight'])

    unique_weights = list(set(all_weights))

    # Draw the edges
    for weight in unique_weights:
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr)
                          in G.edges(data=True) if edge_attr['weight']==weight]

        width = weight*len(G.nodes)*3.0/sum(all_weights)

        nx.draw_networkx_edges(G,pos = nodePos,edgelist=weighted_edges,width=width)

    plt.axis('off')

def draw_geodesic_with_node_weights(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix):

    C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = align_graphs(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix)

    nodePos1_matrix, nodePos2_matrix = align_nodes(nodePos1_matrix,nodePos2_matrix)

    num_steps = 10

    ts = np.linspace(0,1,num_steps)

    fig = plt.figure(figsize = (5*num_steps,20))

    for j in range(num_steps):

        t = ts[j]

        nodePos_matrix = (1-t)*nodePos1_matrix + t*nodePos2_matrix

        C = (1-t)*C1 + t*C2

        # To accentuate weights in the pictures:
        C = 10 * C**2

        G = nx.from_numpy_array(C)

        plt.subplot(2,num_steps/2,j+1)

        p1_new = fix_probability_vector(p1,nodePos_matrix)

        draw_node_weighted_graph(G, p1_new, nodePos_matrix)

    return

"""
The above ideas can also be used to visualize Frechet means of graphs
in the Gromov-Wasserstein setting.
"""

def average_nodePos(CBase,pBase,CList,pList,nodePosList, seed = 0):

    alignedNodePosList = []

    # Define Node positions for the mean network.
    # Only to get sizes correct, doesn't actually matter.

    nodePosBase_matrix = np.zeros([CBase.shape[0],2])

    for j in range(len(CList)):
        C = CList[j]
        p = pList[j]
        nodePos_matrix = nodePosList[j]

        CBase, C, pBase, p, nodePosBase_matrix, nodePos_matrix = align_graphs(CBase,C,pBase,p,
                                                                              nodePosBase_matrix,nodePos_matrix)

        alignedNodePosList.append(nodePos_matrix)

    nodePos_matrix_seed = alignedNodePosList[seed]

    for j in range(len(CList)):

        nodePos_matrix = alignedNodePosList[j]
        nodePos_matrix, nodePos_matrix_seed = align_nodes(nodePos_matrix, nodePos_matrix_seed)

        alignedNodePosList[j] = nodePos_matrix

    meanNodePos_matrix = 1/len(CList)*sum(alignedNodePosList)

    return meanNodePos_matrix

def draw_filtered_graph(C,p,nodePos_matrix,filter_depth, rounding_places = 4):

    p_vals = np.unique(np.round(p,rounding_places))
    if filter_depth > len(p_vals):
        filter_depth = p_vals[-1]
    p_vals = np.concatenate((np.array([0]),p_vals),axis = 0)
    filtration_value = p_vals[filter_depth]+10**(-rounding_places)
    inds = p > filtration_value
    C0 = C[inds,:]
    CNew = C0[:,inds]
    pNew = p[inds]
    nodePosNew = nodePos_matrix[inds,:]

    draw_node_weighted_graph(nx.from_numpy_array(CNew),pNew,nodePosNew)

"""
Pictures look better if we filter out nodes with low weight.
"""

def draw_node_weighted_graph_with_threshold(G, p, nodePos, threshold = 0.5, nodeSizeFlag = False):
    # Inputs:
    #   weighted Graph G as networkx graph object
    #   probability vector with length (number of nodes)
    #   node positions, either as a dictionary or as a numpy matrix of size (number of nodes) x 2
    #   threshold: Will try to downplay nodes whose weight is < threshold*max_weight.
    #   Will also downplay edges containing nodes with this property.

    heavy_inds = list(np.where(p > threshold*np.max(p))[0])
    light_inds = list(np.where(p <= threshold*np.max(p))[0])


#     node_colors = len(G.nodes)*['o']

#     for j in range(len(G.nodes)):
#         if j in heavy_inds:
#             node_colors[j] = 1
#         else:
#             node_colors[j] = 2

    # Draw nodes
#     nx.draw_networkx_nodes(G,pos = nodePos, node_color = node_colors, node_size=5000*p)

    if nodeSizeFlag:
        nx.draw_networkx_nodes(G,pos = nodePos, nodelist = heavy_inds, alpha = 1, node_size=5000*p)
        nx.draw_networkx_nodes(G,pos = nodePos, nodelist = light_inds, alpha = 0.2, node_size=1000*p)
    else:
        nx.draw_networkx_nodes(G,pos = nodePos, nodelist = heavy_inds, alpha = 1)
        nx.draw_networkx_nodes(G,pos = nodePos, nodelist = light_inds, alpha = 0.2)

    # Create a list of unique edgeweights
    all_weights = []

    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight'])

    unique_weights = list(set(all_weights))

    # Draw the edges
    for weight in unique_weights:
        heavy_weighted_edges = [(node1,node2) for (node1,node2,edge_attr)
                          in G.edges(data=True) if edge_attr['weight']==weight
                               and node1 in heavy_inds and node2 in heavy_inds]

        light_weighted_edges = [(node1,node2) for (node1,node2,edge_attr)
                          in G.edges(data=True) if edge_attr['weight']==weight
                               and node1 in light_inds or node2 in light_inds]

        width = weight*len(G.nodes)*3.0/sum(all_weights)

        nx.draw_networkx_edges(G,pos = nodePos,edgelist=heavy_weighted_edges,alpha = 1,width=width)
        nx.draw_networkx_edges(G,pos = nodePos,edgelist=light_weighted_edges,alpha = 0.05,width=width)

    plt.axis('off')

def draw_geodesic_with_threshold(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,threshold = 0.5, nodeSizeFlag = False):

    C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = align_graphs(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix)

    nodePos1_matrix, nodePos2_matrix = align_nodes(nodePos1_matrix,nodePos2_matrix)

    num_steps = 10

    ts = np.linspace(0,1,num_steps)

    fig = plt.figure(figsize = (5*num_steps,20))

    for j in range(num_steps):

        t = ts[j]

        nodePos_matrix = (1-t)*nodePos1_matrix + t*nodePos2_matrix

        C = (1-t)*C1 + t*C2

        # To accentuate weights in the pictures:
        C = 10 * C**2

        G = nx.from_numpy_array(C)

        plt.subplot(2,num_steps/2,j+1)

        p1_new = fix_probability_vector(p1,nodePos_matrix)

        draw_node_weighted_graph_with_threshold(G, p1_new, nodePos_matrix, threshold, nodeSizeFlag)

    return

"""
For visualizing interpolations between
graphs via a given coupling.
The results are not necessarily geodesics, but
are useful for visualizing graph matchings.
"""

def align_graphs_fixed_coupling(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup):
    coup, C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = split_cost_coupling_probabilities_insert_nodes(opt_coup, C1, C2, p1, p2,
                                                                                nodePos1_matrix, nodePos2_matrix, thresh=1)
    perm = 1*(coup != 0)
    C2 = np.matmul(np.matmul(perm,C2),perm.T)
    nodePos2_matrix = np.matmul(perm,nodePos2_matrix)

    return C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix

def draw_geodesic_with_node_weights_fixed_coupling(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup):

    C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = align_graphs_fixed_coupling(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup)

    nodePos1_matrix, nodePos2_matrix = align_nodes(nodePos1_matrix,nodePos2_matrix)

    num_steps = 10

    ts = np.linspace(0,1,num_steps)

    fig = plt.figure(figsize = (5*num_steps,20))

    for j in range(num_steps):

        t = ts[j]

        nodePos_matrix = (1-t)*nodePos1_matrix + t*nodePos2_matrix

        C = (1-t)*C1 + t*C2

        # To accentuate weights in the pictures:
        C = 10 * C**2

        G = nx.from_numpy_array(C)

        plt.subplot(2,num_steps/2,j+1)

        p1_new = fix_probability_vector(p1,nodePos_matrix)

        draw_node_weighted_graph(G, p1_new, nodePos_matrix)

    return
