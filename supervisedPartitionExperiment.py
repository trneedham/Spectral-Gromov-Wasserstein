import numpy as np
import scipy
import pandas as pd
import seaborn as sns
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
from sklearn import manifold
from sklearn.model_selection import train_test_split

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
from networkx.algorithms.community.quality import modularity
from sklearn import metrics
from infomap import Infomap

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold


warnings.filterwarnings("ignore")



def get_sbm(ns,ps,seed=None):
    # convert ps from 1d to 2d array
    n = len(ns)
    if n*(n+1)/2 != len(ps):
        print('Error: check size of ps')
        return None
    else:
        R,C = np.triu_indices(n)
        pm  = np.zeros((n,n))
        pm[R,C] = ps
        pm[C,R] = ps
        
    G = nx.stochastic_block_model(ns, pm,seed=seed)
    
    gt = []
    for i in range(len(ns)):
        for j in range(ns[i]):
            gt.append(i)
    
    return G,gt,pm

def get_gw_ami(G,t,gt):
    # G  -- graph
    # t  -- heat kernel scale parameter
    # gt -- ground truth 
    distribution_exponent_hk = 0.001
    distribution_offset_hk = 0

    C1 = sgw.undirected_normalized_heat_kernel(G,t)
    p1 = sgw.node_distribution(G,distribution_offset_hk,distribution_exponent_hk)
    p2 = np.ravel(GwGt.estimate_target_distribution({0: p1.reshape(-1,1)}, dim_t=len(np.unique(gt))))
    # Note that we are inserting prior information about the number of clusters
    
    C2 = np.diag(p2)
    coup, log = ot.gromov.gromov_wasserstein(C1, C2, p1, p2, loss_fun = 'square_loss', log = True)
    est_idx = np.argmax(coup, axis=1)
    
    ami = metrics.adjusted_mutual_info_score(est_idx,gt,average_method='max')
    comms = [set() for v in np.unique(est_idx)]
    for idx,val in enumerate(est_idx):
        comms[val].add(idx)
        
    mod = modularity(G,comms)
    return ami,mod

def get_adj_ami(G,gt):
    # G  -- graph
    # gt -- ground truth 
    
    distribution_exponent_hk = 0.001
    distribution_offset_hk = 0

    C1 = nx.adjacency_matrix(G).toarray()
    p1 = sgw.node_distribution(G,distribution_offset_hk,distribution_exponent_hk)
    p2 = np.ravel(GwGt.estimate_target_distribution({0: p1.reshape(-1,1)}, dim_t=len(np.unique(gt))))
    # Note that we are inserting prior information about the number of clusters
    
    C2 = np.diag(p2)
    coup, log = ot.gromov.gromov_wasserstein(C1, C2, p1, p2, loss_fun = 'square_loss', log = True)
    est_idx = np.argmax(coup, axis=1)
    
    ami = metrics.adjusted_mutual_info_score(est_idx,gt)
    
    return ami

def optimize_specgwl(train_G,train_gt,ts = np.linspace(0,20,20)):
    # Find the t that gives the largest sum of squared AMIs across train_G
    
    squared_amis = []
    for t in ts:
        tmp = []
        for idx,G in enumerate(train_G):
            ami, _ = get_gw_ami(G,t,train_gt[idx])
            tmp.append(ami)
            
        squared_amis.append(np.dot(tmp,tmp))
        
    best_t_idx = np.argmax(squared_amis)
    
    return best_t_idx, squared_amis

def optimize_specgwl_v2(train_G,train_gt,ts = np.linspace(0,20,20)):
    # Find the t that gives the largest sum of AMIs across train_G
    
    squared_amis = []
    for t in ts:
        tmp = []
        for idx,G in enumerate(train_G):
            ami, _ = get_gw_ami(G,t,train_gt[idx])
            tmp.append(ami)
            
        squared_amis.append(np.sum(tmp))
        
    best_t_idx = np.argmax(squared_amis)
    
    return best_t_idx, squared_amis

def get_benchmark_amis(G,gt):
    # Louvain
    louv = community.best_partition(G)
    louvc = []
    for idx,val in louv.items():
        louvc.append(val)

    louv_ami = metrics.adjusted_mutual_info_score(gt,louvc)
    
    # Fluid communities
    fluid = asyn_fluidc(G,2)
    list_nodes = [set(c) for c in fluid]
    est_idx = np.zeros((nx.number_of_nodes(G),))
    for i in range(len(list_nodes)):
        for idx in list_nodes[i]:
            est_idx[idx] = i

    fluid_ami = metrics.adjusted_mutual_info_score(gt,est_idx)
    
    # FastGreedy
    list_nodes = list(greedy_modularity_communities(G))
    est_idx = np.zeros((nx.number_of_nodes(G),))
    for i in range(len(list_nodes)):
        for idx in list_nodes[i]:
            est_idx[idx] = i

    fg_ami = metrics.adjusted_mutual_info_score(gt,est_idx)
    
    # Infomap
    im = Infomap()
    for node in G.nodes:
        im.add_node(node)
    for edge in G.edges:
        im.add_link(edge[0], edge[1])
        im.add_link(edge[1],edge[0])
    # Run the Infomap search algorithm to find optimal modules
    im.run()
    # print(f"Found {im.num_top_modules} modules with Infomap")
    est_idx = np.zeros((nx.number_of_nodes(G),))
    for node in im.tree:
        if node.is_leaf:
            est_idx[node.node_id] = node.module_id

    im_ami = metrics.adjusted_mutual_info_score(gt,est_idx)
    
    benchmark = {'Louvain':louv_ami,
            'Fluid':fluid_ami,
            'FastGreedy':fg_ami,
            'Infomap':im_ami}
    
    return benchmark


## DEFINING SBMS WITH FIXED SEEDS FOR REPRODUCIBILITY
n = 5
ts = np.linspace(1,10,30)

# Choose community sizes for each block
# ns = [35 for i in range(n)]
np.random.seed(0)
ns = np.random.randint(low=20,high=50,size=n)

Gs = []
As = []
gts = []

# Create random graphs

for i in range(10):
    # Set up edge densities
    p_sz = n*(n+1)/2
    np.random.seed(i)  # This is only for reproducibility
    p_arr = 0.35*np.random.rand(int(p_sz))

    k = n
    curr = 0
    while k>0:
        p_arr[curr] = 0.5
        curr = curr+k
        k -=1

    # Create SBM
    G,gt,pm = get_sbm(ns,p_arr,seed=0)
    
    # Store
    Gs.append(G)
    As.append(nx.adjacency_matrix(G).toarray())
    gts.append(gt)
    
    
## PLOT SBMS

# fig,axs = plt.subplots(2,5,figsize=(13,5),dpi=300)
fig,axs = plt.subplots(2,5)
axs = axs.flatten()

for i in range(len(As)):
    ax = axs[i]
    ax.imshow(As[i])
    
fig.tight_layout()


fig.suptitle('SBMs for supervised learning with leave-one-out cross-validation')
fig.savefig('res_supervisedPartitionExperiment.png',bbox_inches='tight',dpi=150)


## LOOCV step
loo = LeaveOneOut()
# kf = KFold(n_splits=len(Gs),random_state=0)

amis = []
louvs = []
fgs = []
fluids = []
ims = []
adjs = []

for train_index,test_index in loo.split(Gs):
    
    train_G = [Gs[v] for v in train_index]
    train_gt = [gts[v] for v in train_index]
    test_G = [Gs[v] for v in test_index]
    test_gt = [gts[v] for v in train_index]
    
    # Optimize
    best_t_idx, squared_amis = optimize_specgwl_v2(train_G,train_gt,ts)
#     print(ts[best_t_idx])
    
    # Evaluate
    G = test_G[0]
    gt = test_gt[0]
    ami,_ = get_gw_ami(G,ts[best_t_idx],gt)
    amis.append(ami)
    
    # Append benchmarks
    bench = get_benchmark_amis(G,gt)
    adj_ami = get_adj_ami(G,gt)
    
    louvs.append(bench['Louvain'])
    fluids.append(bench['Fluid'])
    fgs.append(bench['FastGreedy'])
    ims.append(bench['Infomap'])
    adjs.append(adj_ami)
    
    print('--- Evaluated on test set',test_index[0],'---\n')
    
sbm_df = pd.DataFrame()
sbm_df['SpecGWL'] = amis
sbm_df['GWL'] = adjs
sbm_df['Fluid'] = fluids
sbm_df['FastGreedy'] = fgs
sbm_df['Louvain'] = louvs
sbm_df['Infomap'] = ims

print(sbm_df)

print(sbm_df.mean())

with open('res_supervisedPartitionExperiment.txt', 'w') as outfile:
    sbm_df.round(4).to_csv(outfile,index=True,sep="\t")
    
plt.show()