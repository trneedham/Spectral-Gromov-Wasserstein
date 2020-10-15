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
from GromovWassersteinGraphToolkit import *
import json

# Import Graph Partitioning Packages
from infomap import Infomap

# Load the S-GWL code
import DataIO as DataIO
import EvaluationMeasure as Eval
import GromovWassersteinGraphToolkit as GwGt
import pickle
import warnings

# Load modules for network partitioning experiments
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.community.quality import performance, coverage, modularity
from sklearn import metrics

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.signal import find_peaks

"""
Define some helper functions
"""

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


"""
Main Experiment
"""

num_trials = 10

num_nodes = 1000
clique_size = 150
p_in = 0.5
ps_out = [0.08, 0.10, 0.12, 0.15]


ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
               'ot_method': 'proximal',
               'beta': 0.15,
               'outer_iteration': 2 * num_nodes,  # outer, inner iterations and error bound of optimal transport
               'iter_bound': 1e-30,
               'inner_iteration': 5,
               'sk_bound': 1e-30,
               'node_prior': 0.0001,
               'max_iter': 1,  # iteration and error bound for calcuating barycenter
               'cost_bound': 1e-16,
               'update_p': False,  # optional updates of source distribution
               'lr': 0,
               'alpha': 0}

# Range to search for optimal number of clusters over
num_clusts = list(range(3,10))

train_times = []
specGW_avg_amis = []
specGW_avg_times = []
GWL_avg_amis = []
GWL_avg_times = []
infoMap_avg_amis = []
infoMap_avg_times = []

for pn in range(len(ps_out)):

    print('Starting p_out index = ',pn)

    ##############################################
    # Training specGW
    ##############################################

    G = nx.gaussian_random_partition_graph(n=num_nodes, s=clique_size, v=8,
                                                   p_in=p_in, p_out=ps_out[pn], directed=True)

    p_s, cost_s, idx2node = DataIO.extract_graph_info(G)
    p_s = (p_s + 1) ** 0.01
    p_s /= np.sum(p_s)

    start = time.time()

    t = 10
    cost = sgw.directed_heat_kernel(G,t)

    modularities = []

    for j in num_clusts:
        p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=j)
        sub_costs, sub_probs, sub_idx2nodes, coup, d_gw = graph_partition_gd2(cost,
                                                                      p_s,
                                                                      p_t,
                                                                      idx2node,
                                                                      ot_dict)
        partition = get_partition(coup)
        modularities.append(modularity(G,partition))

    est_num_clust = num_clusts[np.argmax(modularities)]

    ts = np.linspace(5,15,10)
    modularities = []

    for t in ts:
        cost = sgw.directed_heat_kernel(G,t)
        p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=est_num_clust)
        sub_costs, sub_probs, sub_idx2nodes, coup, d_gw = graph_partition_gd2(cost,
                                                                      p_s,
                                                                      p_t,
                                                                      idx2node,
                                                                      ot_dict)
        partition = get_partition(coup)
        modularities.append(modularity(G,partition))

    est_t_value = ts[np.argmax(modularities)]

    end = time.time()

    training_time = end - start

    train_times.append(training_time)

    print('Time to Train:', training_time)
    print('Estimated Clusters:', est_num_clust)
    print('Estimated t value:', est_t_value)

    ##############################################
    # Main Experiment
    ##############################################

    gwl_amis = []
    gwl_times = []
    specGW_amis = []
    specGW_times = []
    infoMap_amis = []
    infoMap_times = []

    for j in range(num_trials):

        # Create Graph
        G = nx.gaussian_random_partition_graph(n=num_nodes, s=clique_size, v=5,
                                                   p_in=p_in, p_out=ps_out[pn], directed=True)

        gt = np.zeros((num_nodes,))

        for i in range(len(G.nodes)):
            gt[i] = G.nodes[i]['block']

        num_partitions = int(np.max(gt) + 1)

        p_s, cost_s, idx2node = DataIO.extract_graph_info(G)
        p_s = (p_s + 1) ** 0.01
        p_s /= np.sum(p_s)

        # Run SpecGW
        start = time.time()
        cost = sgw.directed_heat_kernel(G,est_t_value)
        p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=est_num_clust)
        sub_costs, sub_probs, sub_idx2nodes, coup, d_gw = graph_partition_gd2(cost,
                                                                      p_s,
                                                                      p_t,
                                                                      idx2node,
                                                                      ot_dict)
        est_idx = np.argmax(coup, axis=1)

        ami = metrics.adjusted_mutual_info_score(gt, est_idx, average_method='max')

        end = time.time()

        specGW_amis.append(ami)
        specGW_times.append(end - start)

        # print('SpecGW AMI:',ami,' Time:',end -start)

        # Run GWL
        start = time.time()

        sub_costs, sub_probs, sub_idx2nodes = GwGt.recursive_graph_partition(cost_s,
                                                                         p_s,
                                                                         idx2node,
                                                                         ot_dict,
                                                                         max_node_num=300)
        est_idx = np.zeros((num_nodes,))
        for n_cluster in range(len(sub_idx2nodes)):
            for key in sub_idx2nodes[n_cluster].keys():
                idx = sub_idx2nodes[n_cluster][key]
                est_idx[idx] = n_cluster

        ami = metrics.adjusted_mutual_info_score(gt, est_idx, average_method='max')
        end = time.time()

        gwl_amis.append(ami)
        gwl_times.append(end-start)

        # print('GWL AMI:',ami,' Time:',end -start)

        # Run InfoMap
        start = time.time()

        im = Infomap()

        for edge in G.edges:
            im.add_link(edge[0], edge[1])
        # Run the Infomap search algorithm to find optimal modules
        im.run()
        # print(f"Found {im.num_top_modules} modules with Infomap")
        est_idx = np.zeros((num_nodes,))
        for node in im.tree:
            if node.is_leaf:
                est_idx[node.node_id] = node.module_id

        ami = metrics.adjusted_mutual_info_score(gt, est_idx, average_method='max')

        end = time.time()

        infoMap_amis.append(ami)
        infoMap_times.append(end-start)

        # print('InfoMap AMI:',ami,' Time:',end -start)

    specGW_avg_amis.append(np.mean(specGW_amis))
    specGW_avg_times.append(np.mean(specGW_times))
    GWL_avg_amis.append(np.mean(gwl_amis))
    GWL_avg_times.append(np.mean(gwl_times))
    infoMap_avg_amis.append(np.mean(infoMap_amis))
    infoMap_avg_times.append(np.mean(infoMap_times))


print('Average AMIs:')
print('p_out','specGW','GWL','Infomap')
for j in range(len(ps_out)):
    print(ps_out[j],np.round(specGW_avg_amis,3)[j],np.round(GWL_avg_amis,3)[j],np.round(infoMap_avg_amis,3)[j])

print('Average times:')
print('p_out','specGW','GWL','Infomap')
for j in range(len(ps_out)):
    print(ps_out[j],np.round(specGW_avg_times,2)[j],np.round(GWL_avg_times,2)[j],np.round(infoMap_avg_times,2)[j])

## Store results
ami_p_out = []
ami_specGW = []
ami_GWL = []
ami_Infomap = []

times_p_out = []
times_specGW = []
times_GWL = []
times_Infomap = []

for j in range(len(ps_out)):
    ami_p_out.append(ps_out[j])
    ami_specGW.append(np.round(specGW_avg_amis,3)[j])
    ami_GWL.append(np.round(GWL_avg_amis,3)[j])
    ami_Infomap.append(np.round(infoMap_avg_amis,3)[j])
    
    times_p_out.append(ps_out[j])
    times_specGW.append(np.round(specGW_avg_times,2)[j])
    times_GWL.append(np.round(GWL_avg_times,2)[j])
    times_Infomap.append(np.round(infoMap_avg_times,2)[j])
    
res_ami = {}#pd.DataFrame()
res_ami['p_out'] = ami_p_out
res_ami['specGW'] = ami_specGW
res_ami['GWL'] = ami_GWL
res_ami['Infomap'] = ami_Infomap

res_times = {}#pd.DataFrame()
res_times['p_out'] = times_p_out
res_times['specGW'] = times_specGW
res_times['GWL'] = times_GWL
res_times['Infomap'] = times_Infomap

with open('res_randomGraphPartitioning.txt', 'w') as outfile:
    json.dump(['Average AMIs',
               res_ami,
               'Average times',
               res_times], outfile,indent=0)