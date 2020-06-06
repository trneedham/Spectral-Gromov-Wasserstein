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

# load data (using [XLC] pickle file)
# with open('data/EmailEU_database.pkl', 'rb') as f:
#     database = pickle.load(f)
# num_nodes = 1005
# num_partitions = 42
# # cost = database['cost']
# # cost_s = 0.5*(cost/num_nodes + cost.T/num_nodes)

# G = nx.Graph()
# nG = nx.Graph()
# dG = nx.DiGraph()
# ndG = nx.DiGraph()

# for edge in database['edges']:
#     G.add_edge(edge[0], edge[1])
#     dG.add_edge(edge[0], edge[1])
#     nG.add_edge(edge[0], edge[1])
#     ndG.add_edge(edge[0], edge[1])

# load data 
f = open('data/eu-email.p', 'rb')
database = pickle.load(f)
f.close()
dG = database['G']
labels = database['labels']
num_nodes = dG.number_of_nodes()
num_partitions = len(np.unique(labels))
G = dG.to_undirected()
database['label'] = database['labels']


# create noisy version
nG = nx.Graph()
ndG = nx.DiGraph()

for n in G.nodes:
    nG.add_node(n)

for n in dG.nodes:
    ndG.add_node(n)
    
for e in G.edges:
    nG.add_edge(e[0],e[1])
    
for e in dG.edges:
    ndG.add_edge(e[0],e[1])
    
start_edges = nx.number_of_edges(dG)



# add noise
for j in range(int( 0.1*G.number_of_edges()  )):
    x1 = int(num_nodes * np.random.rand())
    x2 = int(num_nodes * np.random.rand())
    if database['label'][x1] != database['label'][x2]:
        nG.add_edge(x1, x2)
        ndG.add_edge(x1,x2)

print('---{:3d} edges in raw version \n'.format(dG.number_of_edges()))        
print('---Added {:d} edges to create noisy version \n'.format(nx.number_of_edges(ndG)-start_edges))


print('---Data files loaded. Computing...\n')


def process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=2e-7,verbose=False):
    p_s = np.zeros((num_nodes, 1))
    p_s[:, 0] = np.sum(cost, axis=1) ** 0.001
    p_s /= np.sum(p_s)

    p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)
    ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
               'ot_method': 'proximal',
               'beta': beta,
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

    sub_costs, sub_probs, sub_idx2nodes, trans = GwGt.graph_partition(cost,
                                                                      p_s,
                                                                      p_t,
                                                                      database['idx2node'],
                                                                      ot_dict)

    est_idx = np.argmax(trans, axis=1)


    mutual_info = metrics.adjusted_mutual_info_score(database['label'], est_idx)

    if verbose:
        print('Mutual information score = {:3.3f}'.format(mutual_info))
    return mutual_info

"""
###########################################################
###########################################################
# Method: Fluid communities (symmetrized)
###########################################################
# Raw data
if not nx.is_connected(G):
    #print('---Fluid community requires connected graph, skipping raw version---')
    scores['fluid-symmetrized-raw'] = 'failed'
    runtimes['fluid-symmetrized-raw'] = 'failed'
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
    scores['fluid-symmetrized-raw'] = mutual_info
    runtimes['fluid-symmetrized-raw'] = runtime

# Noisy data
if not nx.is_connected(nG):
    print('---Fluid community requires connected graph, skipping noisy version---')
    scores['fluid-symmetrized-noisy'] = 'failed'
    runtimes['fluid-symmetrized-noisy'] = 'failed'    
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
    scores['fluid-symmetrized-noisy'] = mutual_info
    runtimes['fluid-symmetrized-noisy'] = runtime  
    

    
###########################################################
###########################################################
# Method: FastGreedy (symmetrized)
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
scores['fastgreedy-symmetrized-raw'] = mutual_info
runtimes['fastgreedy-symmetrized-raw'] = runtime 


# Noisy
time_s = time.time()
list_nodes = list(greedy_modularity_communities(nG))
est_idx = np.zeros((num_nodes,))
for i in range(len(list_nodes)):
    for idx in list_nodes[i]:
        est_idx[idx] = i
runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['fastgreedy-symmetrized-noisy'] = mutual_info
runtimes['fastgreedy-symmetrized-noisy'] = runtime 


###########################################################
###########################################################
# Method: Louvain (symmetrized)
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
scores['louvain-symmetrized-raw'] = mutual_info
runtimes['louvain-symmetrized-raw'] = runtime 

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
scores['louvain-symmetrized-noisy'] = mutual_info
runtimes['louvain-symmetrized-noisy'] = runtime 


###########################################################
###########################################################
# Method: Infomap (symmetrized)
###########################################################   
# Raw
time_s = time.time()
im = Infomap()
for node in G.nodes:
    im.add_node(node)
for edge in G.edges:
    im.add_link(edge[0], edge[1])
    im.add_link(edge[1], edge[0])
# Run the Infomap search algorithm to find optimal modules
im.run()
# print(f"Found {im.num_top_modules} modules with Infomap")
est_idx = np.zeros((num_nodes,))
for node in im.tree:
    if node.is_leaf:
        est_idx[node.node_id] = node.module_id

runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['infomap-symmetrized-raw'] = mutual_info
runtimes['infomap-symmetrized-raw'] = runtime 

# Noisy
print('---Running Infomap with noisy data---\n')
time_s = time.time()
im = Infomap()
for node in nG.nodes:
    im.add_node(node)
for edge in nG.edges:
    im.add_link(edge[0], edge[1])
    im.add_link(edge[1], edge[0])
# Run the Infomap search algorithm to find optimal modules
im.run()
# print(f"Found {im.num_top_modules} modules with Infomap")
est_idx = np.zeros((num_nodes,))
for node in im.tree:
    if node.is_leaf:
        est_idx[node.node_id] = node.module_id

runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['infomap-symmetrized-noisy'] = mutual_info
runtimes['infomap-symmetrized-noisy'] = runtime 
    
###########################################################
###########################################################
# Method: Infomap (asymmetric)
###########################################################   
# Raw
time_s = time.time()
im = Infomap()
for node in dG.nodes:
    im.add_node(node)
for edge in dG.edges:
    im.add_link(edge[0], edge[1])
# Run the Infomap search algorithm to find optimal modules
im.run()
# print(f"Found {im.num_top_modules} modules with Infomap")
est_idx = np.zeros((num_nodes,))
for node in im.tree:
    if node.is_leaf:
        est_idx[node.node_id] = node.module_id

runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['infomap-asymmetric-raw'] = mutual_info
runtimes['infomap-asymmetric-raw'] = runtime 

# Noisy
print('---Running Infomap with noisy data---\n')
time_s = time.time()
im = Infomap()
for node in ndG.nodes:
    im.add_node(node)
for edge in ndG.edges:
    im.add_link(edge[0], edge[1])
# Run the Infomap search algorithm to find optimal modules
im.run()
# print(f"Found {im.num_top_modules} modules with Infomap")
est_idx = np.zeros((num_nodes,))
for node in im.tree:
    if node.is_leaf:
        est_idx[node.node_id] = node.module_id

runtime = time.time() - time_s
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx)
scores['infomap-asymmetric-noisy'] = mutual_info
runtimes['infomap-asymmetric-noisy'] = runtime 

"""
###########################################################
###########################################################
# Method: GWL, symmetrized
########################################################### 
# Raw
start = time.time()
cost = nx.adjacency_matrix(G).toarray().astype(np.float64)/num_nodes
mutual_info = process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=2e-7);
end = time.time()
scores['gwl-symmetrized-raw'] = mutual_info
runtimes['gwl-symmetrized-raw'] = end-start 


# Noisy
start = time.time()
cost = nx.adjacency_matrix(nG).toarray().astype(np.float64)/num_nodes
mutual_info = process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=2e-7);
end = time.time()
scores['gwl-symmetrized-noisy'] = mutual_info
runtimes['gwl-symmetrized-noisy'] = end-start


###########################################################
###########################################################
# Method: GWL, asymmetric
########################################################### 
# Raw
start = time.time()
cost = nx.adjacency_matrix(dG).toarray().astype(np.float64)/num_nodes
mutual_info = process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=2e-7);
end = time.time()
scores['gwl-asymmetric-raw'] = mutual_info
runtimes['gwl-asymmetric-raw'] = end-start 


# Noisy
start = time.time()
cost = nx.adjacency_matrix(ndG).toarray().astype(np.float64)/num_nodes
mutual_info = process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=2e-7);
end = time.time()
scores['gwl-asymmetric-noisy'] = mutual_info
runtimes['gwl-asymmetric-noisy'] = end-start

###########################################################
###########################################################
# Proposed method: SpecGWL (symmetrized)
########################################################### 
# Raw
mis = []
rt = []
ts = [7]#np.linspace(7,10,20)
for t in ts:
    start = time.time()
    cost = sgw.undirected_normalized_heat_kernel(G,t).astype(np.float64)/num_nodes
    mutual_info = process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=3e-8);
    mis.append(mutual_info)
    end = time.time()
    rt.append(end-start)

# print('--- Raw data | Symmetrized | SpecGWL | Best mutual information score: {:3.3f} | @t = {:3.3f} | average runtime per iteration = {:3.3f}'.format(max(mis), ts[np.argmax(mis)], np.mean(rt)))
scores['specgwl-symmetrized-raw'] = max(mis)
runtimes['specgwl-symmetrized-raw'] = sum(rt)
# avetimes['specgwl-symmetrized-raw'] = np.mean(rt)

# Noisy
mis = []
rt = []
ts = [7]#np.linspace(7,10,20)
for t in ts:
    start = time.time()
    cost = sgw.undirected_normalized_heat_kernel(nG,t).astype(np.float64)/num_nodes
    mi = process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=3e-8);
    mis.append(mi)
    end = time.time()
    rt.append(end-start)
    
# print('--- Noisy data | Symmetrized | SpecGWL | Best mutual information score: {:3.3f} | @t = {:3.3f} | average runtime per iteration = {:3.3f}'.format(max(mis), ts[np.argmax(mis)], np.mean(rt)))
scores['specgwl-symmetrized-noisy'] = max(mis)
runtimes['specgwl-symmetrized-noisy'] = sum(rt)
# avetimes['specgwl-symmetrized-noisy'] = np.mean(rt)

###########################################################
###########################################################
# Proposed method: SpecGWL (asymmetric)
########################################################### 
# Raw
mis = []
rt = []
ts = [7]#np.linspace(6.5,8,10)
for t in ts:
    start = time.time()
    cost = sgw.directed_heat_kernel(dG,t).astype(np.float64)/num_nodes
    mutual_info = process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=3e-8);
    mis.append(mutual_info)
    end = time.time()
    rt.append(end-start)

# print('--- Raw data | Asymmetric | SpecGWL | Best mutual information score: {:3.3f} | @t = {:3.3f} | average runtime per iteration = {:3.3f}'.format(max(mis), ts[np.argmax(mis)], np.mean(rt)))
scores['specgwl-asymmetric-raw'] = max(mis)
runtimes['specgwl-asymmetric-raw'] = sum(rt)
# avetimes['specgwl-asymmetric-raw'] = np.mean(rt)

# Noisy
mis = []
rt = []
ts = [7]#np.linspace(6,8,10)
for t in ts:
    start = time.time()
    cost = sgw.directed_heat_kernel(ndG,t).astype(np.float64)/num_nodes
    mi = process_sgwl_eu(cost,database,num_nodes,num_partitions,beta=3e-8);
    mis.append(mi)
    end = time.time()
    rt.append(end-start)
    
# print('--- Noisy data | Asymmetric | SpecGWL | Best mutual information score: {:3.3f} | @t = {:3.3f} | average runtime per iteration = {:3.3f}'.format(max(mis), ts[np.argmax(mis)], np.mean(rt)))
scores['specgwl-asymmetric-noisy'] = max(mis)
runtimes['specgwl-asymmetric-noisy'] = sum(rt)
# avetimes['specgwl-asymmetric-noisy'] = np.mean(rt)

print('Mutual information scores')
print(json.dumps(scores,indent=1))
print('Runtimes')
print(json.dumps(runtimes,indent=1))
# print('Average runtime of SpecGWL')
# print(json.dumps(avetimes,indent=1))

with open('res_benchmark_regularized_eu.txt', 'w') as outfile:
    json.dump(['Adjusted mutual information scores',
               scores,
               'Runtimes',
               runtimes], outfile,indent=1)