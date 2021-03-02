import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ot
import gromovWassersteinAveraging as gwa
from geodesicVisualization import *
from spectralGW import *

from graphProcessing import load_graph

import warnings
warnings.filterwarnings("ignore")


import random
import time
from scipy import linalg
import pandas as pd

plt.rcParams["figure.figsize"] = (6,1)

def draw_geodesic_with_node_weights_fixed_coupling(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup):

    C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = align_graphs_fixed_coupling(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup)

    nodePos1_matrix, nodePos2_matrix = align_nodes(nodePos1_matrix,nodePos2_matrix)

    num_steps = 5

    ts = np.linspace(0,1,num_steps)

    fig = plt.figure(figsize = (10*num_steps,10))

    for j in range(num_steps):

        t = ts[j]

        nodePos_matrix = (1-t)*nodePos1_matrix + t*nodePos2_matrix

        C = (1-t)*C1 + t*C2

        # To accentuate weights in the pictures:
        C = 10 * C**2

        G = nx.from_numpy_array(C)

        plt.subplot(1,num_steps,j+1)

        p1_new = fix_probability_vector(p1,nodePos_matrix)

        draw_node_weighted_graph(G, p1_new, nodePos_matrix)

    return fig


def draw_geodesic_with_node_weights_fixed_coupling_v2(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup):

    C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = align_graphs_fixed_coupling(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup)

    nodePos1_matrix, nodePos2_matrix = align_nodes(nodePos1_matrix,nodePos2_matrix)

    num_steps = 5

    ts = np.linspace(0,1,num_steps)

    fig = plt.figure(figsize = (10*num_steps,10))

    for j in range(num_steps):

        t = ts[j]

        nodePos_matrix = (1-t)*nodePos1_matrix + t*nodePos2_matrix

        C = (1-t)*C1 + t*C2

        # To accentuate weights in the pictures:
        C = 10 * C**2

        G = nx.from_numpy_array(C)

        plt.subplot(1,num_steps,j+1)
        
        if j == 0:
            p1_new = p1
        elif j==num_steps-1:
            p1_new = p2
        else:
            p1_new = fix_probability_vector(p1,nodePos_matrix)

        draw_node_weighted_graph(G, p1_new, nodePos_matrix)

    return fig


# Binary tree matching
num_nodes_1 = 10
num_nodes_2 = 20
distribution_exponent = 0.1

G1 = nx.balanced_tree(2,5)
G2 = nx.balanced_tree(2,4)

nodePos1 = nx.kamada_kawai_layout(G1)
nodePos2 = nx.kamada_kawai_layout(G2)

nodePos_matrix1 = np.array(list(nodePos1.values()))
nodePos_matrix2 = np.array(list(nodePos2.values()))

p1 = node_distribution(G1,0,distribution_exponent)
p2 = node_distribution(G2,0,distribution_exponent)

num_steps = 100
num_skips = 100

# Sample probability polytope
A, b = gw_equality_constraints(p1,p2)
start = time.time()
Markov_steps = coupling_ensemble(A,b,p1,p2,num_steps,num_skips)
end = time.time()
#print('---Probability polytope for binary tree sampled in {:3.3f} seconds'.format(end-start))

A1 = nx.to_numpy_array(G1)
A2 = nx.to_numpy_array(G2)
# Find Optimal Couplings
opt_coups = []
losses = []
start = time.time()

for j in range(num_steps):
    G0 = Markov_steps[j]
    coup, log = gromov_wasserstein_asym_fixed_initialization(A1, A2, p1, p2, G0)
    losses.append(log['gw_dist'])
    opt_coups.append(coup)
    
end = time.time()
#print('Compute Time for binary trees of size',len(p1),'and',len(p2),':',end-start,'seconds')

fig = draw_geodesic_with_node_weights_fixed_coupling_v2(nx.to_numpy_array(G1),nx.to_numpy_array(G2),
                                               p1,p2,nodePos_matrix1,nodePos_matrix2,opt_coups[np.argmin(losses)])

fig.suptitle('Binary tree-Adj',fontsize=12)
fig.savefig('res_binary_tree-Adj.pdf',bbox_inches='tight')
# fig.set_size_inches(6,1)
# fig

t = 20

nodePos_matrix1, ppp1, lam1, phi1 = extract_HK_data_normalized_Laplacian(G1)
nodePos_matrix2, ppp2, lam2, phi2 = extract_HK_data_normalized_Laplacian(G2)

HK1 = heat_kernel(lam1, phi1, t)
HK2 = heat_kernel(lam2, phi2, t)

# Find Optimal Couplings
opt_coups = []
losses = []

start = time.time()

for j in range(num_steps):
    G0 = Markov_steps[j]
    coup, log = gromov_wasserstein_asym_fixed_initialization(HK1, HK2, p1, p2, G0)
    losses.append(log['gw_dist'])
    opt_coups.append(coup)
    
end = time.time()

# print('Compute Time for graphs of size',len(p1),'and',len(p2),':',end-start,'seconds')

# plt.figure()
fig = draw_geodesic_with_node_weights_fixed_coupling_v2(nx.to_numpy_array(G1),nx.to_numpy_array(G2),
                                               p1,p2,nodePos_matrix1,nodePos_matrix2,opt_coups[np.argmin(losses)])
fig.suptitle('Binary tree-HK20',fontsize=12)
fig.savefig('res_binary_tree-HK20.pdf',bbox_inches='tight')

## Cycle to circulant
num_nodes_1 = 20
num_nodes_2 = 20
distribution_exponent = 0

G1 = nx.cycle_graph(num_nodes_1)
G2 = nx.circulant_graph(num_nodes_2,[1,2,3,5])

nodePos1 = nx.kamada_kawai_layout(G1)
nodePos2 = nx.kamada_kawai_layout(G2)

nodePos_matrix1 = np.array(list(nodePos1.values()))
nodePos_matrix2 = np.array(list(nodePos2.values()))

p1 = node_distribution(G1,0,distribution_exponent)
p2 = node_distribution(G2,0,distribution_exponent)


num_steps = 100
num_skips = 1000

# Sample probability polytope
A, b = gw_equality_constraints(p1,p2)

start = time.time()
Markov_steps = coupling_ensemble(A,b,p1,p2,num_steps,num_skips)
end = time.time()

A1 = nx.to_numpy_array(G1)
A2 = nx.to_numpy_array(G2)

# Find Optimal Couplings
opt_coups = []
losses = []

start = time.time()

for j in range(num_steps):
    G0 = Markov_steps[j]
    coup, log = gromov_wasserstein_asym_fixed_initialization(A1, A2, p1, p2, G0)
    losses.append(log['gw_dist'])
    opt_coups.append(coup)
    
end = time.time()

print('Compute Time for graphs of size',len(p1),'and',len(p2),':',end-start,'seconds')
fig = draw_geodesic_with_node_weights_fixed_coupling(nx.to_numpy_array(G1),nx.to_numpy_array(G2),
                                               p1,p2,nodePos_matrix1,nodePos_matrix2,opt_coups[np.argmin(losses)])

fig.suptitle('Cycle to circulant-adj',fontsize=12)
fig.savefig('res_cycle_circulant-adj.pdf',bbox_inches='tight')
t = 20

nodePos_matrix1, ppp1, lam1, phi1 = extract_HK_data_normalized_Laplacian(G1)
nodePos_matrix2, ppp2, lam2, phi2 = extract_HK_data_normalized_Laplacian(G2)

HK1 = heat_kernel(lam1, phi1, t)
HK2 = heat_kernel(lam2, phi2, t)

# Find Optimal Couplings
opt_coups = []
losses = []

start = time.time()

for j in range(num_steps):
    G0 = Markov_steps[j]
    coup, log = gromov_wasserstein_asym_fixed_initialization(HK1, HK2, p1, p2, G0)
    losses.append(log['gw_dist'])
    opt_coups.append(coup)
    
end = time.time()
fig = draw_geodesic_with_node_weights_fixed_coupling(nx.to_numpy_array(G1),nx.to_numpy_array(G2),
                                               p1,p2,nodePos_matrix1,nodePos_matrix2,opt_coups[np.argmin(losses)])
fig.suptitle('Cycle to circulant-HK20',fontsize=12)
fig.savefig('res_cycle_circulant-HK20.pdf',bbox_inches='tight')

## IMDB graphs
graph_file = 'data/IMDB-BINARY_A.txt'
indicator_file = 'data/IMDB-BINARY_graph_indicator.txt'
label_file = 'data/IMDB-BINARY_graph_labels.txt'

graphs, labels = load_graph(graph_file,indicator_file,label_file)

total_num_graphs = len(graphs)

ind1 = 2
ind2 = 21
distribution_exponent = 0

G1 = graphs[ind1]
G2 = graphs[ind2]

nodePos1 = nx.kamada_kawai_layout(G1)
nodePos2 = nx.kamada_kawai_layout(G2)

nodePos_matrix1 = np.array(list(nodePos1.values()))
nodePos_matrix2 = np.array(list(nodePos2.values()))

p1 = node_distribution(G1,0,distribution_exponent)
p2 = node_distribution(G2,0,distribution_exponent)


num_steps = 100
num_skips = 100

# Sample probability polytope
A, b = gw_equality_constraints(p1,p2)

start = time.time()
Markov_steps = coupling_ensemble(A,b,p1,p2,num_steps,num_skips)
end = time.time()


A1 = nx.to_numpy_array(G1)
A2 = nx.to_numpy_array(G2)

# Find Optimal Couplings
opt_coups = []
losses = []

start = time.time()

for j in range(num_steps):
    G0 = Markov_steps[j]
    coup, log = gromov_wasserstein_asym_fixed_initialization(A1, A2, p1, p2, G0)
    losses.append(log['gw_dist'])
    opt_coups.append(coup)
    
end = time.time()


fig = draw_geodesic_with_node_weights_fixed_coupling(nx.to_numpy_array(G1),nx.to_numpy_array(G2),
                                               p1,p2,nodePos_matrix1,nodePos_matrix2,opt_coups[np.argmin(losses)])
fig.suptitle('IMDB matching-Adj',fontsize=12)
fig.savefig('res_IMDB_matching-Adj.pdf',bbox_inches='tight')

t = 20

nodePos_matrix1, ppp1, lam1, phi1 = extract_HK_data_normalized_Laplacian(G1)
nodePos_matrix2, ppp2, lam2, phi2 = extract_HK_data_normalized_Laplacian(G2)

HK1 = heat_kernel(lam1, phi1, t)
HK2 = heat_kernel(lam2, phi2, t)

# Find Optimal Couplings
opt_coups = []
losses = []

start = time.time()

for j in range(num_steps):
    G0 = Markov_steps[j]
    coup, log = gromov_wasserstein_asym_fixed_initialization(HK1, HK2, p1, p2, G0)
    losses.append(log['gw_dist'])
    opt_coups.append(coup)
    
end = time.time()

fig = draw_geodesic_with_node_weights_fixed_coupling(nx.to_numpy_array(G1),nx.to_numpy_array(G2),
                                               p1,p2,nodePos_matrix1,nodePos_matrix2,opt_coups[np.argmin(losses)])
fig.suptitle('IMDB matching-HK20',fontsize=12)
fig.savefig('res_IMDB_matching-HK20.pdf',bbox_inches='tight')

plt.show()