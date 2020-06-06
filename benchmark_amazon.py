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
import json

# Load the S-GWL code
import DataIO as DataIO
import EvaluationMeasure as Eval
import GromovWassersteinGraphToolkit as GwGt
import pickle
import warnings

# Load modules for network partitioning experiments
import community
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from sklearn import metrics
from infomap import Infomap

warnings.filterwarnings("ignore")

# dictionaries for holding results
scores = {}
runtimes = {}
avetimes = {}

# load data
f = open('data/amazon.p', 'rb')
database = pickle.load(f)
f.close()
G = database['G']
labels = database['labels']

num_nodes = G.number_of_nodes()
num_partitions = len(np.unique(labels))

idx2node = {}
for n in G.nodes:
    idx2node[n] = n

# create noisy version
nG = nx.Graph()

for n in G.nodes:
    nG.add_node(n)

for e in G.edges:
    nG.add_edge(e[0],e[1])
    
start_edges = nx.number_of_edges(nG)
    
# add noise
for j in range(int( 0.1*G.number_of_edges()  )):
    x1 = int(num_nodes * np.random.rand())
    x2 = int(num_nodes * np.random.rand())
    if database['labels'][x1] != database['labels'][x2]:
        nG.add_edge(x1, x2)

print('---{:3d} edges in raw version \n'.format(G.number_of_edges()))        
print('---Added {:d} edges to create noisy version \n'.format(nx.number_of_edges(nG)-start_edges))

    
print('---Data files loaded. Computing...\n')


def process_sgwl_amazon(cost_s,database,num_nodes,num_partitions,verbose=False):
    p_s = np.zeros((num_nodes, 1))
    p_s[:, 0] = np.sum(cost_s, axis=1) ** 0.001
    p_s /= np.sum(p_s)
    p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)

    ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
           'ot_method': 'proximal',
           'beta': 2e-7,
           'outer_iteration': 300,
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

    time_s = time.time()
    sub_costs, sub_probs, sub_idx2nodes, trans = GwGt.graph_partition_gd(cost_s,
                                                                      p_s,
                                                                      p_t,
                                                                      idx2node,
                                                                      ot_dict)
    est_idx = np.argmax(trans, axis=1)

    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
    
    if verbose:
        print('---Mutual information score = {:3.5f}'.format(mutual_info))

    return mutual_info

###########################################################
###########################################################
# Method: Fluid communities
###########################################################
# Raw data
if not nx.is_connected(G):
    #print('---Fluid community requires connected graph, skipping raw version---')
    scores['fluid-raw'] = 'failed'
    runtimes['fluid-raw'] = 'failed'
else:
    time_s = time.time()
    comp = asyn_fluidc(G.to_undirected(), k=num_partitions)
    list_nodes = [frozenset(c) for c in comp]
    est_idx = np.zeros((num_nodes,))
    for i in range(len(list_nodes)):
        for idx in list_nodes[i]:
            est_idx[idx] = i
    runtime = time.time() - time_s
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
    scores['fluid-raw'] = mutual_info
    runtimes['fluid-raw'] = runtime

# Noisy data
if not nx.is_connected(nG):
    print('---Fluid community requires connected graph, skipping noisy version---')
    scores['fluid-noisy'] = 'failed'
    runtimes['fluid-noisy'] = 'failed'    
else:
    time_s = time.time()
    comp = asyn_fluidc(nG.to_undirected(), k=num_partitions)
    list_nodes = [frozenset(c) for c in comp]
    est_idx = np.zeros((num_nodes,))
    for i in range(len(list_nodes)):
        for idx in list_nodes[i]:
            est_idx[idx] = i
    runtime = time.time() - time_s
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
    scores['fluid-noisy'] = mutual_info
    runtimes['fluid-noisy'] = runtime    


    
###########################################################
###########################################################
# Method: FastGreedy
###########################################################
# Raw
time_s = time.time()
list_nodes = list(greedy_modularity_communities(G))
est_idx = np.zeros((num_nodes,))
for i in range(len(list_nodes)):
    for idx in list_nodes[i]:
        est_idx[idx] = i
runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['fastgreedy-raw'] = mutual_info
runtimes['fastgreedy-raw'] = runtime 


# Noisy
time_s = time.time()
list_nodes = list(greedy_modularity_communities(nG))
est_idx = np.zeros((num_nodes,))
for i in range(len(list_nodes)):
    for idx in list_nodes[i]:
        est_idx[idx] = i
runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['fastgreedy-noisy'] = mutual_info
runtimes['fastgreedy-noisy'] = runtime 




###########################################################
###########################################################
# Method: Louvain
###########################################################
# Raw
time_s = time.time()
partition = community.best_partition(G)
est_idx = np.zeros((num_nodes,))
for com in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys()
                  if partition[nodes] == com]
    for idx in list_nodes:
        est_idx[idx] = com
runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['louvain-raw'] = mutual_info
runtimes['louvain-raw'] = runtime 

# Noisy
time_s = time.time()
partition = community.best_partition(nG)
est_idx = np.zeros((num_nodes,))
for com in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys()
                  if partition[nodes] == com]
    for idx in list_nodes:
        est_idx[idx] = com
runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['louvain-noisy'] = mutual_info
runtimes['louvain-noisy'] = runtime 




###########################################################
###########################################################
# Method: Infomap
###########################################################   
# Raw
time_s = time.time()
im = Infomap()
for node in G.nodes:
    im.add_node(node)
for edge in G.edges:
    im.add_link(edge[0], edge[1])
    im.add_link(edge[1],edge[0])
# Run the Infomap search algorithm to find optimal modules
im.run()
# print(f"Found {im.num_top_modules} modules with Infomap")
est_idx = np.zeros((num_nodes,))
for node in im.tree:
    if node.is_leaf:
        est_idx[node.node_id] = node.module_id

runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['infomap-raw'] = mutual_info
runtimes['infomap-raw'] = runtime 

# Noisy
print('---Running Infomap with noisy data---\n')
time_s = time.time()
im = Infomap()
for node in nG.nodes:
    im.add_node(node)
for edge in nG.edges:
    im.add_link(edge[0], edge[1])
    im.add_link(edge[1],edge[0])
# Run the Infomap search algorithm to find optimal modules
im.run()
# print(f"Found {im.num_top_modules} modules with Infomap")
est_idx = np.zeros((num_nodes,))
for node in im.tree:
    if node.is_leaf:
        est_idx[node.node_id] = node.module_id

runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['infomap-noisy'] = mutual_info
runtimes['infomap-noisy'] = runtime 


###########################################################
###########################################################
# Method: GWL
########################################################### 
# Raw
start = time.time()
cost = nx.adjacency_matrix(G).toarray()
mutual_info = process_sgwl_amazon(cost,database,num_nodes,num_partitions);
end = time.time()
scores['gwl-raw'] = mutual_info
runtimes['gwl-raw'] = end-start 


# adjacency, undirected, noisy
start = time.time()
cost = nx.adjacency_matrix(nG).toarray()
mutual_info = process_sgwl_amazon(cost,database,num_nodes,num_partitions);
end = time.time()
scores['gwl-noisy'] = mutual_info
runtimes['gwl-noisy'] = end-start


###########################################################
###########################################################
# Proposed method: SpecGWL
########################################################### 
# Raw
mis = []
rt = []
ts = [85+5/9]#np.linspace(80,90,10)
for t in ts:
    start = time.time()
    cost = sgw.undirected_normalized_heat_kernel(G,t)
    mutual_info = process_sgwl_amazon(cost,database,num_nodes,num_partitions);
    mis.append(mutual_info)
    end = time.time()
    rt.append(end-start)

# print('--- Raw data | SpecGWL | Best mutual information score: {:3.3f} | @t = {:3.3f} | average runtime per iteration = {:3.3f}'.format(max(mis), ts[np.argmax(mis)], np.mean(rt)))
scores['specgwl-raw'] = max(mis)
runtimes['specgwl-raw'] = sum(rt)
# avetimes['specgwl-raw'] = np.mean(rt)

# Noisy
mis = []
rt = []
ts = [7.0555556]#np.linspace(6.5,7.5,10)
for t in ts:
    start = time.time()
    cost = sgw.undirected_normalized_heat_kernel(nG,t)
    mi = process_sgwl_amazon(cost,database,num_nodes,num_partitions);
    mis.append(mi)
    end = time.time()
    rt.append(end-start)
    
# print('--- Noisy data | SpecGWL | Best mutual information score: {:3.3f} | @t = {:3.3f} | average runtime per iteration = {:3.3f}'.format(max(mis), ts[np.argmax(mis)], np.mean(rt)))
scores['specgwl-noisy'] = max(mis)
runtimes['specgwl-noisy'] = sum(rt)
# avetimes['specgwl-noisy'] = np.mean(rt)

print('Mutual information scores')
print(json.dumps(scores,indent=1))
print('Runtimes')
print(json.dumps(runtimes,indent=1))
# print('Average runtime of SpecGWL')
# print(json.dumps(avetimes,indent=1))

with open('res_benchmark_amazon.txt', 'w') as outfile:
    json.dump(['Adjusted mutual information scores',
               scores,
               'Runtimes',
               runtimes], outfile,indent=1)
