import numpy as np
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

warnings.filterwarnings("ignore")

def get_sbm(ns,ps):
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
        
    G = nx.stochastic_block_model(ns, pm)
    
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
    
    ami = metrics.adjusted_mutual_info_score(est_idx,gt)
    comms = [set() for v in np.unique(est_idx)]
    for idx,val in enumerate(est_idx):
        comms[val].add(idx)
        
    mod = modularity(G,comms)
    
    return ami,mod


## Construct sequence of SBMs that are increasingly indistinguishable
ns = [75,75]
ps = [0.5,0.35,0.5]
ts = np.linspace(0,20,20)

amis = []
mods = []
pvals = []
tvals = []
iterate = []

As = []

p_range = np.arange(0.15,0.35,0.02)

# Calculate modularity and AMI

for iteration in range(10):
    
    for p in p_range:
        p_copy = ps.copy()
        p_copy[1] = p

        G,gt,pm = get_sbm(ns,p_copy)
        A = nx.adjacency_matrix(G).toarray()

        if iteration==0:
            As.append(A)

        for t in ts:
            ami, mod = get_gw_ami(G,t,gt)
            amis.append(ami)
            mods.append(mod)
            tvals.append(t)
            pvals.append(p)
            iterate.append(iteration)
    
sbm_df = pd.DataFrame()
sbm_df['t'] = tvals
sbm_df['off-diag-p'] = pvals
sbm_df['AMI'] = amis
sbm_df['Modularity'] = mods
sbm_df['iteration'] = iterate

fig,axs = plt.subplots(2,5)
axs = axs.flatten()

for i in range(len(As)):
    ax = axs[i]
    ax.imshow(As[i])
    
fig.suptitle('SBMs with increasing cross-block edge densities')
fig.tight_layout()

# fig.savefig('200927_partition_sbm_two.png',bbox_inches='tight',dpi=150)

melted = pd.melt(sbm_df,['off-diag-p','t','iteration'])

f = sns.FacetGrid(melted,col = 'off-diag-p',col_wrap=5,margin_titles=True)
fg = plt.gcf()
fg.dpi = 50

f.map_dataframe(sns.lineplot, x='t', y='value',hue='variable')
f.set_axis_labels("t", "value")
f.add_legend()
cn = [round(v,2) for v in f.col_names]

fg = plt.gcf()

fg.suptitle('AMI and Modularity peaks across scales')

axes = f.axes.flatten()
for i,val in enumerate(cn):
    axes[i].set_title("cross-block edge density = %2.2f" % cn[i])
    
plt.show()