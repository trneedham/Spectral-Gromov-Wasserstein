## Script to run graph partitioning experiment on EU-email dataset

# Load packages

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
from GromovWassersteinGraphToolkit import *
import pickle
import warnings

# Load modules for network partitioning experiments
import community
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.community.quality import performance, coverage, modularity
from sklearn import metrics
from infomap import Infomap

# Breakpoint analysis package
# import ruptures as rpt

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

def graph_partition_gd2(cost_s, p_s, p_t,idx2node, ot_hyperpara, trans0=None):
    """
    ** May 19, 2020: Gradient descent version of graph_partition
    
    
    Achieve a single graph partition via calculating Gromov-Wasserstein discrepancy
    between the target graph and proposed one

    Args:
        cost_s: (n_s, n_s) adjacency matrix of source graph
        p_s: (n_s, 1) the distribution of source nodes
        p_t: (n_t, 1) the distribution of target nodes
        idx2node: a dictionary {key = idx of row in cost, value = name of node}
        ot_hyperpara: a dictionary of hyperparameters

    Returns:
        sub_costs: a dictionary {key: cluster idx,
                                 value: sub cost matrices}
        sub_probs: a dictionary {key: cluster idx,
                                 value: sub distribution of nodes}
        sub_idx2nodes: a dictionary {key: cluster idx,
                                     value: a dictionary mapping indices to nodes' names
        trans: (n_s, n_t) the optimal transport
    """
    cost_t = np.diag(p_t[:, 0])
    cost_s = np.asarray(cost_s)
    # cost_t = 1 / (1 + cost_t)
    trans, log = gwa.gromov_wasserstein_asym_fixed_initialization(cost_s, cost_t, p_s.flatten(), p_t.flatten(), trans0)
    d_gw = log['gw_dist']
    sub_costs, sub_probs, sub_idx2nodes = node_cluster_assignment(cost_s, trans, p_s, p_t, idx2node)
    return sub_costs, sub_probs, sub_idx2nodes, trans, d_gw

def get_partition(coup):
    
    est_idx = np.argmax(coup, axis=1)
    num_clusters = np.max(est_idx)
    
    partition = []
    
    for j in range(num_clusters+1):
        partition.append(set(np.argwhere(est_idx == j).T[0]))
        
    return partition

# dictionaries for holding results
scores = {}
runtimes = {}
avetimes = {}

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


# Load precomputed noisy version
save_name = "eu_sym_noise.txt"

with open(save_name, "rb") as fp:
    nG = pickle.load(fp)
    
save_name = "eu_asym_noise.txt"

with open(save_name, "rb") as fp:
    ndG = pickle.load(fp)


print('---Data files loaded. Computing...\n')


def process_sgwl_eu(cost,database,num_nodes,num_partitions,verbose=False):
    p_s = np.zeros((num_nodes, 1))
    p_s[:, 0] = np.sum(cost, axis=1) ** .001
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

    sub_costs, sub_probs, sub_idx2nodes, trans, d_gw = graph_partition_gd2(cost,
                                                                      p_s,
                                                                      p_t,
                                                                      database['idx2node'],
                                                                      ot_dict)

    est_idx = np.argmax(trans, axis=1)


    mutual_info = metrics.adjusted_mutual_info_score(database['label'], est_idx,  average_method='max')

    if verbose:
        print('Mutual information score = {:3.3f}'.format(mutual_info))
    return mutual_info, d_gw, trans

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
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,  average_method='max')
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
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,  average_method='max')
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
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,  average_method='max')
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
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,  average_method='max')
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
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
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
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
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
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,  average_method='max')
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
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,  average_method='max')
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
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,  average_method='max')
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
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,  average_method='max')
scores['infomap-asymmetric-noisy'] = mutual_info
runtimes['infomap-asymmetric-noisy'] = runtime 


###########################################################
###########################################################
# Method: GWL, symmetrized
########################################################### 
# Raw
start = time.time()
cost = nx.adjacency_matrix(G).toarray()
mutual_info,_,_ = process_sgwl_eu(cost,database,num_nodes,num_partitions);
end = time.time()
scores['gwl-symmetrized-raw'] = mutual_info
runtimes['gwl-symmetrized-raw'] = end-start 


# Noisy
start = time.time()
cost = nx.adjacency_matrix(nG).toarray()
mutual_info,_,_ = process_sgwl_eu(cost,database,num_nodes,num_partitions);
end = time.time()
scores['gwl-symmetrized-noisy'] = mutual_info
runtimes['gwl-symmetrized-noisy'] = end-start


###########################################################
###########################################################
# Method: GWL, asymmetric
########################################################### 
# Raw
start = time.time()
cost = nx.adjacency_matrix(dG).toarray()
mutual_info,_,_ = process_sgwl_eu(cost,database,num_nodes,num_partitions);
end = time.time()
scores['gwl-asymmetric-raw'] = mutual_info
runtimes['gwl-asymmetric-raw'] = end-start 


# Noisy
start = time.time()
cost = nx.adjacency_matrix(ndG).toarray()
mutual_info,_,_ = process_sgwl_eu(cost,database,num_nodes,num_partitions);
end = time.time()
scores['gwl-asymmetric-noisy'] = mutual_info
runtimes['gwl-asymmetric-noisy'] = end-start

###########################################################
###########################################################
# Method: SpecGWL
###########################################################

# Note that the GWL pipeline above takes the true number of clusters as input. 
# We now show how this number is estimated in the SpecGWL pipeline for 
# a bona fide unsupervised partitioning method.

def t_selection_pipeline_undirected_eu(G,ts,num_partitions,fraction_t_to_keep=0.25):
    
    mis = []
    coups = []
    d_gws = []
    rt = []
    
    for t in ts:
        start = time.time()
        cost = sgw.undirected_normalized_heat_kernel(G,t)
        mutual_info, d_gw, coup = process_sgwl_eu(cost,database,num_nodes,num_partitions)
        mis.append(mutual_info)
        coups.append(coup)
        d_gws.append(d_gw)
        end = time.time()
        rt.append(end-start)

    print('Couplings Computed')
    
    coverages = []

    for j in range(len(ts)):
        coup = coups[j]
        partition = get_partition(coup)
        coverages.append(coverage(G,partition))
        
    num_to_keep = int(np.round(fraction_t_to_keep*len(ts)))
    
    good_t_max = ts[np.argsort(coverages)][-num_to_keep:]
    good_t_grad = ts[np.argsort(np.abs(np.gradient(coverages)))][:num_to_keep]
    
    return mis, coups, d_gws, good_t_max, good_t_grad, rt

def t_selection_pipeline_directed_eu(G,ts,num_partitions,fraction_t_to_keep=0.25):
    
    mis = []
    coups = []
    d_gws = []
    rt = []
    
    for t in ts:
        start = time.time()
        cost = sgw.directed_heat_kernel(G,t)
        mutual_info, d_gw, coup = process_sgwl_eu(cost,database,num_nodes,num_partitions)
        mis.append(mutual_info)
        coups.append(coup)
        d_gws.append(d_gw)
        end = time.time()
        rt.append(end-start)

    print('Couplings Computed')
    
    coverages = []

    for j in range(len(ts)):
        coup = coups[j]
        partition = get_partition(coup)
        coverages.append(coverage(G,partition))
        
    num_to_keep = int(np.round(fraction_t_to_keep*len(ts)))
    
    good_t_max = ts[np.argsort(coverages)][-num_to_keep:]
    good_t_grad = ts[np.argsort(np.abs(np.gradient(coverages)))][:num_to_keep]
    
    return mis, coups, d_gws, good_t_max, good_t_grad, rt


# Keeping t fixed, do a grid search to estimate the number of clusters
num_clusts = list(range(5,45))
t = 20

cost = sgw.undirected_normalized_heat_kernel(G,t)

d_gws = []
mis = []
coverages = []
modularities = []

for j in num_clusts:
    mutual_info, d_gw, coup = process_sgwl_eu(cost,database,num_nodes,j)
    partition = get_partition(coup)
    mis.append(mutual_info)
    d_gws.append(d_gw)
    coverages.append(coverage(G,partition))
    modularities.append(modularity(G,partition))
    
# Estimate number of clusters
estimated_clusters_raw_sym = num_clusts[np.argmax(modularities)]
print('Number of Clusters:',estimated_clusters_raw_sym)

# Now perform modularity/coverage maximizing pipeline
ts = np.linspace(3,10,40)
mis, coups, d_gws, good_t_max, good_t_grad, rt = t_selection_pipeline_undirected_eu(G,ts,estimated_clusters_raw_sym)

coverages = []

for j in range(len(ts)):
    coup = coups[j]
    partition = get_partition(coup)
    coverages.append(coverage(G,partition))
    
eu_raw_sym_ami = mis[np.argmax(coverages)]
print('AMI for EU, Raw, Sym:',eu_raw_sym_ami)
print('Occurs at t-value:',ts[np.argmax(coverages)])
scores['specgwl-symmetric-raw'] = eu_raw_sym_ami
runtimes['specgwl-symmetric-raw'] = rt[np.argmax(coverages)]

## Repeat for undirected, noisy data

num_clusts = list(range(20,50))
t = 20

cost = sgw.undirected_normalized_heat_kernel(nG,t)

d_gws = []
mis = []
coverages = []
modularities = []

for j in num_clusts:
    mutual_info, d_gw, coup = process_sgwl_eu(cost,database,num_nodes,j)
    partition = get_partition(coup)
    mis.append(mutual_info)
    d_gws.append(d_gw)
    coverages.append(coverage(nG,partition))
    modularities.append(modularity(nG,partition))
    
estimated_clusters_noisy_sym = num_clusts[np.argmax(modularities)]
print('Number of Clusters:',estimated_clusters_noisy_sym)

ts = np.linspace(3,10,30)
mis, coups, d_gws, good_t_max, good_t_grad, rt = t_selection_pipeline_undirected_eu(nG,ts,estimated_clusters_noisy_sym)

coverages = []

for j in range(len(ts)):
    coup = coups[j]
    partition = get_partition(coup)
    coverages.append(coverage(nG,partition))
    
eu_noisy_sym_ami = mis[np.argmax(coverages)]
print('AMI for EU, Noisy, Sym:',eu_noisy_sym_ami)
print('Occurs at t-value:',ts[np.argmax(coverages)])
scores['specgwl-symmetric-noisy'] = eu_noisy_sym_ami
runtimes['specgwl-symmetric-noisy'] = rt[np.argmax(coverages)]

## Repeat for directed, raw data
num_clusts = list(range(10,30))
t = 20

cost = sgw.directed_heat_kernel(dG,t)

d_gws = []
mis = []
coverages = []
modularities = []

for j in num_clusts:
    mutual_info, d_gw, coup = process_sgwl_eu(cost,database,num_nodes,j)
    partition = get_partition(coup)
    mis.append(mutual_info)
    d_gws.append(d_gw)
    coverages.append(coverage(dG,partition))
    modularities.append(modularity(dG,partition))
    
estimated_clusters_raw_asym = num_clusts[np.argmax(modularities)]
print('Number of Clusters:',estimated_clusters_raw_asym)
    
ts = np.linspace(3,10,30)
mis, coups, d_gws, good_t_max, good_t_grad, rt = t_selection_pipeline_directed_eu(dG,ts,estimated_clusters_raw_asym)

coverages = []

for j in range(len(ts)):
    coup = coups[j]
    partition = get_partition(coup)
    coverages.append(coverage(dG,partition))
    
eu_raw_asym_ami = mis[np.argmax(coverages)]
print('AMI for EU, Raw, Asym:',eu_raw_asym_ami)
print('Occurs at t-value:',ts[np.argmax(coverages)])
scores['specgwl-asymmetric-raw'] = eu_raw_asym_ami
runtimes['specgwl-asymmetric-raw'] = rt[np.argmax(coverages)]

## Repeat for directed, noisy data

num_clusts = list(range(20,50))
t = 20

cost = sgw.directed_heat_kernel(ndG,t)

d_gws = []
mis = []
coverages = []
modularities = []

for j in num_clusts:
    mutual_info, d_gw, coup = process_sgwl_eu(cost,database,num_nodes,j)
    partition = get_partition(coup)
    mis.append(mutual_info)
    d_gws.append(d_gw)
    coverages.append(coverage(ndG,partition))
    modularities.append(modularity(ndG,partition))
    
estimated_clusters_noisy_asym = num_clusts[np.argmax(modularities)]
print('Number of Clusters:',estimated_clusters_noisy_asym)


ts = np.linspace(3,10,30)
mis, coups, d_gws, good_t_max, good_t_grad, rt = t_selection_pipeline_directed_eu(ndG,ts,estimated_clusters_noisy_asym)

coverages = []

for j in range(len(ts)):
    coup = coups[j]
    partition = get_partition(coup)
    coverages.append(coverage(ndG,partition))
    
eu_noisy_asym_ami = mis[np.argmax(coverages)]
print('AMI for EU, Noisy, Sym:',eu_noisy_asym_ami)
print('Occurs at t-value:',ts[np.argmax(coverages)])
scores['specgwl-asymmetric-noisy'] = eu_noisy_asym_ami
runtimes['specgwl-asymmetric-noisy'] = rt[np.argmax(coverages)]

print('Mutual information scores')
print(json.dumps(scores,indent=1))
print('Runtimes')
print(json.dumps(runtimes,indent=1))

with open('res_partition_eu.txt', 'w') as outfile:
    json.dump(['Adjusted mutual information scores',
               scores,
               'Runtimes',
               runtimes], outfile,indent=1)