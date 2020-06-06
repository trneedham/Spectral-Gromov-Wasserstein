import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import time
import ot
from scipy import linalg
from scipy import sparse
import gromovWassersteinAveraging as gwa
import spectralGW as sgw
from geodesicVisualization import *
from GromovWassersteinFramework import *
import json
import pandas as pd

# Load the S-GWL code
import DataIO as DataIO
import EvaluationMeasure as Eval
import GromovWassersteinGraphToolkit as GwGt
import pickle
import warnings

from graphProcessing import load_graph

# Load modules for network partitioning experiments
import community
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from sklearn import metrics
from infomap import Infomap

warnings.filterwarnings("ignore")


def node_correctness(coup,perm_inv):
    thresh_coup = (coup > 1e-10).astype(int)
    NC = np.sum((10*perm_inv - thresh_coup) == 9)/np.sum(thresh_coup)
    
    return NC


        
    
# Load PROTEINS data
print('---Running PROTEINS experiment')

graph_file = 'data/PROTEINS_A.txt'
indicator_file = 'data/PROTEINS_graph_indicator.txt'
label_file = 'data/PROTEINS_graph_labels.txt'

graphs, labels = load_graph(graph_file,indicator_file,label_file)

total_num_graphs = len(graphs)

results_adj = []
times_adj = []
results_hk = []
times_hk = []

t = 10
distribution_exponent_adj = 0.001
distribution_offset_adj = 0.01

distribution_exponent_hk = 0
distribution_offset_hk = 0

ot_hyperpara_adj = {'loss_type': 'L2',  # the key hyperparameters of GW distance
       'ot_method': 'proximal',
       'beta': 5e-2,
       'outer_iteration': 200,
       # outer, inner iteration, error bound of optimal transport
       'iter_bound': 1e-30,
       'inner_iteration': 1,
       'sk_bound': 1e-30,
       'node_prior': 0,
       'max_iter': 200,  # iteration and error bound for calcuating barycenter
       'cost_bound': 1e-16,
       'update_p': False,  # optional updates of source distribution
       'lr': 0,
       'alpha': 0}

for j in range(total_num_graphs):
    
    ind1 = j
    G = graphs[ind1]
    
    perm = np.random.permutation(np.eye(len(G.nodes())))

    start = time.time()
    G_adj = nx.to_numpy_array(G)
    G_adj_perm = np.matmul(np.matmul(perm,G_adj),perm.T)

    cost_s = G_adj
    cost_t = G_adj_perm
    p = sgw.node_distribution(G,distribution_offset_adj,distribution_exponent_adj)
    q = np.matmul(p,perm.T)
    p_s = p.reshape(len(p),1)
    p_t = q.reshape(len(q),1)
    
    coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(cost_s, cost_t, p_s, p_t, ot_hyperpara_adj)

    end = time.time()
    
    times_adj.append(end-start)
    
    start = time.time()
    G_hk = sgw.undirected_normalized_heat_kernel(G,t)
    G_hk_perm = np.matmul(np.matmul(perm,G_hk),perm.T)
    
    p = sgw.node_distribution(G,distribution_offset_hk,distribution_exponent_hk)
    q = p
    
    coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_hk, G_hk_perm, p, q, loss_fun = 'square_loss', log = True)
    end = time.time()
    
    times_hk.append(end-start)
    
    perm_inv = np.linalg.inv(perm)

    results_hk.append(node_correctness(coup_hk,perm_inv))
    results_adj.append(node_correctness(coup_adj,perm_inv))
    
    if j%100 == 0:
        print('Trial',j,'done...')

print('---Run completed for PROTEINS')
        
gwl_protein_raw_mean = np.mean(results_adj)
gwl_protein_raw_std  = np.std(results_adj)
gwl_protein_raw_time = np.sum(times_adj)

print('Adj. Mean:',np.mean(results_adj),'+/-',np.std(results_adj))
print('Adj. Time:',np.sum(times_adj))

specgwl_protein_raw_mean = np.mean(results_hk)
specgwl_protein_raw_std  = np.std(results_hk)
specgwl_protein_raw_time = np.sum(times_hk)

print('HK Mean:',np.mean(results_hk),'+/-',np.std(results_hk))
print('HK Time:',np.sum(times_hk))



# Load ENZYMES data
print('---Running ENZYMES experiment')

graph_file = 'data/ENZYMES_A.txt'
indicator_file = 'data/ENZYMES_graph_indicator.txt'
label_file = 'data/ENZYMES_graph_labels.txt'

graphs, labels = load_graph(graph_file,indicator_file,label_file)

graphs = [G for G in graphs if len(G.nodes())>0] # Drop empty graphs

total_num_graphs = len(graphs)

# Start run | ENZYMES | Raw data |
graph_file = 'data/ENZYMES_A.txt'
indicator_file = 'data/ENZYMES_graph_indicator.txt'
label_file = 'data/ENZYMES_graph_labels.txt'

graphs, labels = load_graph(graph_file,indicator_file,label_file)

graphs = [G for G in graphs if len(G.nodes())>0] # Drop empty graphs

total_num_graphs = len(graphs)
        
    
results_adj = []
times_adj = []
results_hk = []
times_hk = []

t = 10
distribution_exponent_adj = 0.001
distribution_offset_adj = 0.01

distribution_exponent_hk = 0
distribution_offset_hk = 0

ot_hyperpara_adj = {'loss_type': 'L2',  # the key hyperparameters of GW distance
       'ot_method': 'proximal',
       'beta': 5e-2,
       'outer_iteration': 200,
       # outer, inner iteration, error bound of optimal transport
       'iter_bound': 1e-30,
       'inner_iteration': 1,
       'sk_bound': 1e-30,
       'node_prior': 0,
       'max_iter': 200,  # iteration and error bound for calcuating barycenter
       'cost_bound': 1e-16,
       'update_p': False,  # optional updates of source distribution
       'lr': 0,
       'alpha': 0}

for j in range(total_num_graphs):
    
    ind1 = j
    G = graphs[ind1]
    
    perm = np.random.permutation(np.eye(len(G.nodes())))

    start = time.time()
    G_adj = nx.to_numpy_array(G)
    G_adj_perm = np.matmul(np.matmul(perm,G_adj),perm.T)

    cost_s = G_adj
    cost_t = G_adj_perm
    p = sgw.node_distribution(G,distribution_offset_adj,distribution_exponent_adj)
    q = np.matmul(p,perm.T)
    p_s = p.reshape(len(p),1)
    p_t = q.reshape(len(q),1)
    
    coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(cost_s, cost_t, p_s, p_t, ot_hyperpara_adj)

    end = time.time()
    
    times_adj.append(end-start)
    
    start = time.time()
    G_hk = sgw.undirected_normalized_heat_kernel(G,t)
    G_hk_perm = np.matmul(np.matmul(perm,G_hk),perm.T)
    
    p = sgw.node_distribution(G,distribution_offset_hk,distribution_exponent_hk)
    q = p
    
    
    coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_hk, G_hk_perm, p, q, loss_fun = 'square_loss', log = True)
    
    end = time.time()
    
    times_hk.append(end-start)
    
    perm_inv = np.linalg.inv(perm)

    results_hk.append(node_correctness(coup_hk,perm_inv))
    results_adj.append(node_correctness(coup_adj,perm_inv))
    
    if j%100 == 0:
        print('Trial',j,'done...')
        
print('---Run completed for ENZYMES')
        
gwl_enzyme_raw_mean = np.mean(results_adj)
gwl_enzyme_raw_std  = np.std(results_adj)
gwl_enzyme_raw_time = np.sum(times_adj)

print('Adj. Mean:',np.mean(results_adj),'+/-',np.std(results_adj))
print('Adj. Time:',np.sum(times_adj))

specgwl_enzyme_raw_mean = np.mean(results_hk)
specgwl_enzyme_raw_std  = np.std(results_hk)
specgwl_enzyme_raw_time = np.sum(times_hk)

print('HK Mean:',np.mean(results_hk),'+/-',np.std(results_hk))
print('HK Time:',np.sum(times_hk))




# Load REDDIT data
      
fileName_graph = 'data/REDDIT-BINARY/REDDIT-BINARY_A.txt'
fileName_indicators = 'data/REDDIT-BINARY/REDDIT-BINARY_graph_indicator.txt'
fileName_labels = 'data/REDDIT-BINARY/REDDIT-BINARY_graph_labels.txt'

from scipy.sparse import coo_matrix

# Load in edges from the graph file.
edges = []

for line in open(fileName_graph):
    split_line = line.split(',')
    edges.append(tuple([int(split_line[0]),int(split_line[1])]))

edges = np.array(edges)

# Load in the indicators

indicators = []

for line in open(fileName_indicators):
    indicators.append(int(line))

indicators = np.array(indicators)

num_graphs = max(indicators)

# Create a big adjacency matrix as a sparse array

row = edges[:,0]
col = edges[:,1]
size = max([max(row),max(col)])
data = np.ones(row.shape[0])
adj = coo_matrix((data, (row, col)), shape=(size+1, size+1))
adj1 = adj.tocsc()

# For each graph indicator, create a small adjacency matrix, then a graph

graphs = []

for j in range(num_graphs):
    ind = j+1
    ind_locs = np.argwhere(indicators == ind).ravel()
    start = np.min(ind_locs)
    end = np.max(ind_locs)
    AdjMat = adj1[start:end+1,start:end+1].toarray()
    graphs.append(nx.from_numpy_array(AdjMat))


results_adj = []
times_adj = []
results_hk = []
times_hk = []

t = 10

distribution_exponent_adj = .0001
distribution_offset_adj = 0.001

distribution_exponent_hk = 0
distribution_offset_hk = 0

ot_hyperpara_adj = {'loss_type': 'L2',  # the key hyperparameters of GW distance
       'ot_method': 'proximal',
       'beta': 5e-2,
       'outer_iteration': 800,
       # outer, inner iteration, error bound of optimal transport
       'iter_bound': 1e-30,
       'inner_iteration': 1,
       'sk_bound': 1e-30,
       'node_prior': 0,
       'max_iter': 800,  # iteration and error bound for calcuating barycenter
       'cost_bound': 1e-16,
       'update_p': False,  # optional updates of source distribution
       'lr': 0,
       'alpha': 0}

for j in range(500):
    
    ind1 = j
    G = graphs[ind1]
    
    perm = np.random.permutation(np.eye(len(G.nodes())))

    start = time.time()
    G_adj = nx.to_numpy_array(G)
    G_adj_perm = np.matmul(np.matmul(perm,G_adj),perm.T)

    cost_s = G_adj
    cost_t = G_adj_perm
    p = sgw.node_distribution(G,distribution_offset_adj,distribution_exponent_adj)
    q = np.matmul(p,perm.T)
    p_s = p.reshape(len(p),1)
    p_t = q.reshape(len(q),1)
    
    #coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(cost_s, cost_t, p_s, p_t, ot_hyperpara_adj)
    coup_adj, log_adj = ot.gromov.gromov_wasserstein(G_adj, G_adj_perm, p, q, loss_fun = 'square_loss', log = True)

    end = time.time()
    
    times_adj.append(end-start)
    
    start = time.time()
    G_hk = sgw.undirected_normalized_heat_kernel(G,t)
    G_hk_perm = np.matmul(np.matmul(perm,G_hk),perm.T)
    
    p = sgw.node_distribution(G,distribution_offset_hk,distribution_exponent_hk)
    q = p
    
    coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_hk, G_hk_perm, p, q, loss_fun = 'square_loss', log = True)
    end = time.time()
    
    times_hk.append(end-start)
    
    perm_inv = np.linalg.inv(perm)

    results_hk.append(node_correctness(coup_hk,perm_inv))
    results_adj.append(node_correctness(coup_adj,perm_inv))
    
    if j%100 == 0:
        print('Trial',j,'done...')
      

print('---Run completed for REDDIT')
        
gwl_reddit_raw_mean = np.mean(results_adj)
gwl_reddit_raw_std  = np.std(results_adj)
gwl_reddit_raw_time = np.sum(times_adj)

print('Adj. Mean:',np.mean(results_adj),'+/-',np.std(results_adj))
print('Adj. Time:',np.sum(times_adj))

specgwl_reddit_raw_mean = np.mean(results_hk)
specgwl_reddit_raw_std  = np.std(results_hk)
specgwl_reddit_raw_time = np.sum(times_hk)

print('HK Mean:',np.mean(results_hk),'+/-',np.std(results_hk))
print('HK Time:',np.sum(times_hk))


# Load COLLAB data
fileName_graph = 'data/COLLAB/COLLAB_A.txt'
fileName_indicators = 'data/COLLAB/COLLAB_graph_indicator.txt'
fileName_labels = 'data/COLLAB/COLLAB_graph_labels.txt'
      
from scipy.sparse import coo_matrix

# Load in edges from the graph file.
edges = []

for line in open(fileName_graph):
    split_line = line.split(',')
    edges.append(tuple([int(split_line[0]),int(split_line[1])]))

edges = np.array(edges)

# Load in the indicators

indicators = []

for line in open(fileName_indicators):
    indicators.append(int(line))

indicators = np.array(indicators)

num_graphs = max(indicators)

# Create a big adjacency matrix as a sparse array

row = edges[:,0]
col = edges[:,1]
size = max([max(row),max(col)])
data = np.ones(row.shape[0])
adj = coo_matrix((data, (row, col)), shape=(size+1, size+1))
adj1 = adj.tocsc()

# For each graph indicator, create a small adjacency matrix, then a graph

graphs = []

for j in range(num_graphs):
    ind = j+1
    ind_locs = np.argwhere(indicators == ind).ravel()
    start = np.min(ind_locs)
    end = np.max(ind_locs)
    AdjMat = adj1[start:end+1,start:end+1].toarray()
    graphs.append(nx.from_numpy_array(AdjMat))


results_adj = []
times_adj = []
results_hk = []
times_hk = []

t = 10

distribution_exponent_adj = 0.01
distribution_offset_adj = .001

distribution_exponent_hk = 0
distribution_offset_hk = 0

ot_hyperpara_adj = {'loss_type': 'L2',  # the key hyperparameters of GW distance
       'ot_method': 'proximal',
       'beta': 5e-1,
       'outer_iteration': 800,
       # outer, inner iteration, error bound of optimal transport
       'iter_bound': 1e-30,
       'inner_iteration': 1,
       'sk_bound': 1e-30,
       'node_prior': 0,
       'max_iter': 800,  # iteration and error bound for calcuating barycenter
       'cost_bound': 1e-16,
       'update_p': False,  # optional updates of source distribution
       'lr': 0,
       'alpha': 0}

for j in range(1000):
    
    ind1 = j
    G = graphs[ind1]
    
    perm = np.random.permutation(np.eye(len(G.nodes())))

    start = time.time()
    G_adj = nx.to_numpy_array(G)
    G_adj_perm = np.matmul(np.matmul(perm,G_adj),perm.T)

    cost_s = G_adj
    cost_t = G_adj_perm
    p = sgw.node_distribution(G,distribution_offset_adj,distribution_exponent_adj)
    q = np.matmul(p,perm.T)
    p_s = p.reshape(len(p),1)
    p_t = q.reshape(len(q),1)
    
    #coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(cost_s, cost_t, p_s, p_t, ot_hyperpara_adj)
    coup_adj, log_adj = ot.gromov.gromov_wasserstein(G_adj, G_adj_perm, p, q, loss_fun = 'square_loss', log = True)

    end = time.time()
    
    times_adj.append(end-start)
    
    start = time.time()
    G_hk = sgw.undirected_normalized_heat_kernel(G,t)
    G_hk_perm = np.matmul(np.matmul(perm,G_hk),perm.T)
    
    p = sgw.node_distribution(G,distribution_offset_hk,distribution_exponent_hk)
    q = p
    
    coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_hk, G_hk_perm, p, q, loss_fun = 'square_loss', log = True)
    end = time.time()
    
    times_hk.append(end-start)
    
    perm_inv = np.linalg.inv(perm)

    results_hk.append(node_correctness(coup_hk,perm_inv))
    results_adj.append(node_correctness(coup_adj,perm_inv))
    
    if j%100 == 0:
        print('Trial',j,'done...')

print('---Run completed for COLLAB')
        
gwl_collab_raw_mean = np.mean(results_adj)
gwl_collab_raw_std  = np.std(results_adj)
gwl_collab_raw_time = np.sum(times_adj)

print('Adj. Mean:',np.mean(results_adj),'+/-',np.std(results_adj))
print('Adj. Time:',np.sum(times_adj))
print('Adj. Average Time:',np.sum(times_adj)/1000)

specgwl_collab_raw_mean = np.mean(results_hk)
specgwl_collab_raw_std  = np.std(results_hk)
specgwl_collab_raw_time = np.sum(times_hk)

print('HK Mean:',np.mean(results_hk),'+/-',np.std(results_hk))
print('HK Time:',np.sum(times_hk))
print('HK Average Time:',np.sum(times_hk)/1000)
      

      

tab_cols = ['Method          ','Proteins          ','Enzymes          ','Reddit          ','Collab          ']

tab_rows = []

tab_rows.append(['GWL (score)          ','{:2.2f}+/-{:2.2f}          '.format(gwl_protein_raw_mean,gwl_protein_raw_std),
                '{:2.2f}+/-{:2.2f}          '.format(gwl_enzyme_raw_mean,gwl_enzyme_raw_std),
                '{:2.2f}+/-{:2.2f}          '.format(gwl_reddit_raw_mean,gwl_reddit_raw_std),
                '{:2.2f}+/-{:2.2f}          '.format(gwl_collab_raw_mean,gwl_collab_raw_std)])

tab_rows.append(['GWL (runtime)          ','{:2.2f}            '.format(gwl_protein_raw_time),
                '{:2.2f}            '.format(gwl_enzyme_raw_time),
                '{:2.2f}            '.format(gwl_reddit_raw_time),
                '{:2.2f}            '.format(gwl_collab_raw_time)])


tab_rows.append(['SpecGWL (score)    ','{:2.2f}+/-{:2.2f}          '.format(specgwl_protein_raw_mean,specgwl_protein_raw_std),
                '{:2.2f}+/-{:2.2f}          '.format(specgwl_enzyme_raw_mean,specgwl_enzyme_raw_std),
                '{:2.2f}+/-{:2.2f}          '.format(specgwl_reddit_raw_mean,specgwl_reddit_raw_std),
                '{:2.2f}+/-{:2.2f}          '.format(specgwl_collab_raw_mean,specgwl_collab_raw_std)])

tab_rows.append(['SpecGWL (runtime)   ','{:2.2f}            '.format(specgwl_protein_raw_time),
                '{:2.2f}            '.format(specgwl_enzyme_raw_time),
                '{:2.2f}            '.format(specgwl_reddit_raw_time),
                '{:2.2f}            '.format(specgwl_collab_raw_time)])

tab = pd.DataFrame(tab_rows,columns=tab_cols)

print(tab)

tab.to_csv('res_node_correctness.txt',header=True, index=False, sep='\t')
