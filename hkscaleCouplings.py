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
from sklearn import manifold

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

from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


def draw_geodesic_with_node_weights_fixed_coupling_v2(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup):

    C1, C2, p1, p2, nodePos1_matrix, nodePos2_matrix = align_graphs_fixed_coupling(C1,C2,p1,p2,nodePos1_matrix,nodePos2_matrix,opt_coup)

    nodePos1_matrix, nodePos2_matrix = align_nodes(nodePos1_matrix,nodePos2_matrix)

    num_steps = 5

    ts = np.linspace(0,1,num_steps)

    #fig = plt.figure(figsize = (10*num_steps,10))
    fig = plt.figure(figsize = (20,5))

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




print('---Computing...---\n')

# Load graphs and plot
with open('hkscaleGraphs.p', 'rb') as f:
    Gs = pickle.load(f)
    
graph1 = Gs[0]
graph2 = Gs[1]


plt.figure()
plt.subplot(1,2,1)
nx.draw_networkx(graph1, layout = nx.kamada_kawai_layout(graph1),node_color='xkcd:light blue')

plt.subplot(1,2,2)
nx.draw_networkx(graph2, layout = nx.kamada_kawai_layout(graph2),node_color='xkcd:light blue')

plt.suptitle('Graphs from Enzyme dataset')
plt.savefig('res_hkscaleCouplings_graphs.png',bbox_inches='tight',dpi=150)  

# Plot couplings
ts = np.linspace(0,50,100)
coups = []
dists = []

distribution_exponent_hk = 0
distribution_offset_hk = 0

p1 = sgw.node_distribution(graph1,distribution_offset_hk,distribution_exponent_hk)
p2 = sgw.node_distribution(graph2,distribution_offset_hk,distribution_exponent_hk)

for t in ts:
    graph1_hk = sgw.undirected_normalized_heat_kernel(graph1,t)
    graph2_hk = sgw.undirected_normalized_heat_kernel(graph2,t)
    coup_hk, log_hk = ot.gromov.gromov_wasserstein(graph1_hk, graph2_hk, p1, p2, loss_fun = 'square_loss', log = True)
    coups.append(coup_hk)
    dists.append(log_hk['gw_dist'])
    

fig, axs = plt.subplots(10,10,figsize=(10,10))
axs = axs.flatten()
for i in range(len(coups)):
    ax = axs[i]
    ax.imshow(coups[i],cmap='Blues')

fig.tight_layout()
# fig.dpi = 50
fig.suptitle('Couplings across scale parameters')
fig.savefig('res_hkscaleCouplings_couplings_all.png',bbox_inches='tight',dpi=300)  

# Set things up for tSNE
X = [np.ravel(v) for v in coups]
X = np.array(X)

X_ = TSNE(n_components=2,perplexity=15,random_state=2).fit_transform(X)

fig,axs=plt.subplots(1,1)

# ax = axs[0]
im = axs.scatter(X_[:,0],X_[:,1],c = np.arange(len(coups)),cmap='RdBu_r')
# fig.colorbar(im, orientation='horizontal')


# zip joins x and y coordinates in pairs
for idx,val in enumerate(zip(X_[:,0],X_[:,1])):
    x,y = val

    label = "{:d}".format(idx)

    # this method is called every kth point
    if idx in set((0,1,3,12,71,9,19,29,39,99,89)):#idx % 3 == 0:
    #idx in set((0,1,9,19,29,39,99,89)):#idx % 3 == 0:
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
        
# fig.savefig('200927_hkscale_tsne_labeled.png',bbox_inches='tight',dpi=150)
fig.dpi = 100
fig.suptitle('tSNE embedding of couplings | Labels indicate coupling indices ')
fig.savefig('res_hkscaleCouplings_tsne.png',bbox_inches='tight',dpi=150)  

print('---Still computing...---\n')

# Plot relevant couplings
fig, axs = plt.subplots(1,10,figsize=(20,3),dpi=300)
axs = axs.flatten()
idx = 0
for i in range(len(coups)):
    if i in set((1,3,9,12,19,29,39,71,89,99)):
        ax = axs[idx]
        ax.imshow(coups[i],cmap='Blues')
        idx+=1

fig.tight_layout()
# fig.savefig('200927_hkscale_tsne_couplings.png',bbox_inches='tight',dpi=300)
fig.dpi = 50
fig.savefig('res_hkscaleCouplings_couplings_select.png',bbox_inches='tight',dpi=150) 

# Set things up for geodesic visualization

nodePos1 = nx.kamada_kawai_layout(graph1)
nodePos2 = nx.kamada_kawai_layout(graph2)

nodePos_matrix1 = np.array(list(nodePos1.values()))
nodePos_matrix2 = np.array(list(nodePos2.values()))


idx = 3
t = ts[idx]
coup = coups[idx]

graph1_hk = sgw.undirected_normalized_heat_kernel(graph1,t)
graph2_hk = sgw.undirected_normalized_heat_kernel(graph2,t)

fig = draw_geodesic_with_node_weights_fixed_coupling_v2(graph1_hk,graph2_hk,p1,p2,nodePos_matrix1,nodePos_matrix2,coup)
# fig.savefig('200927_hkscale_3_geodesic.png',bbox_inches='tight',dpi=300)
fig.dpi = 50
fig.savefig('res_hkscaleCouplings_geodesic_3.png',bbox_inches='tight',dpi=150) 

print('---Still computing...---\n')

idx = 19
t = ts[idx]
coup = coups[idx]

graph1_hk = sgw.undirected_normalized_heat_kernel(graph1,t)
graph2_hk = sgw.undirected_normalized_heat_kernel(graph2,t)

fig = draw_geodesic_with_node_weights_fixed_coupling_v2(graph1_hk,graph2_hk,p1,p2,nodePos_matrix1,nodePos_matrix2,coup)
# fig.savefig('200927_hkscale_19_geodesic.png',bbox_inches='tight',dpi=300)
fig.dpi = 50
fig.savefig('res_hkscaleCouplings_geodesic_19.png',bbox_inches='tight',dpi=150)

print('---Still computing...---\n')

idx = 39
t = ts[idx]
coup = coups[idx]

graph1_hk = sgw.undirected_normalized_heat_kernel(graph1,t)
graph2_hk = sgw.undirected_normalized_heat_kernel(graph2,t)

fig = draw_geodesic_with_node_weights_fixed_coupling_v2(graph1_hk,graph2_hk,p1,p2,nodePos_matrix1,nodePos_matrix2,coup)
fig.tight_layout()
# fig.savefig('200927_hkscale_39_geodesic.png',bbox_inches='tight',dpi=300)
fig.dpi = 50
fig.savefig('res_hkscaleCouplings_geodesic_39.png',bbox_inches='tight',dpi=150)

print('---Done! Rendering ---\n')

plt.show()