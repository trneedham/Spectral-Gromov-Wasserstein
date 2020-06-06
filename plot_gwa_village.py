# Start by loading packages

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import time
import ot
import scipy
from scipy import linalg
from scipy import sparse
import gromovWassersteinAveraging as gwa
import spectralGW as sgw
from geodesicVisualization import *
import seaborn as sns
import pandas as pd
import json

# Load the S-GWL code
import DataIO as DataIO
import EvaluationMeasure as Eval
import GromovWassersteinGraphToolkit as GwGt
import pickle
import warnings


warnings.filterwarnings("ignore")

print('---All modules loaded')

# Load data

num_nodes = 1991
num_partitions = 12

# load data
with open('data/India_database.p', 'rb') as f:
    database = pickle.load(f)
G = nx.Graph()
nG = nx.Graph()
for i in range(num_nodes):
    G.add_node(i)
    nG.add_node(i)
for edge in database['edges']:
    G.add_edge(edge[0], edge[1])
    nG.add_edge(edge[0], edge[1])
    
start_edges = nx.number_of_edges(G)
    
# # add noise
# for j in range(int( 1*G.number_of_edges()  )):
#     x1 = int(num_nodes * np.random.rand())
#     x2 = int(num_nodes * np.random.rand())
#     if database['label'][x1] != database['label'][x2]:
#         nG.add_edge(x1, x2)

print('---Data loaded. {:3d} edges in raw version \n'.format(G.number_of_edges()))        
# print('---Added {:d} edges to create noisy version \n'.format(nx.number_of_edges(nG)-start_edges))

# Bootstrap based on betweenness centrality
print('---Computing betweenness centrality')

bc = nx.betweenness_centrality(G)
bcsorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)


# bootstrap
num_bootstrap = 10
size_bootstrap= 30

nodes = []
for n in database['idx2node'].values():
    if n not in nodes:
        nodes.append(n)
print('---Performing bootstrap. Selecting {:d} samples with {:d} nodes each'.format(num_bootstrap,size_bootstrap))

top = []
for n in bcsorted:
    if n[1] > 5e-4:
        top.append(n[0])
        
samples = []
AList = []
HKList3 = []
HKList7 = []
HKList11 = []
pList = []
lambdas=[]

for i in range(0,num_bootstrap):
    select = np.random.choice(top,size_bootstrap,replace=False)
    sG = nx.Graph()
    for n in select:
        sG.add_node(n)
    for e in G.edges():
        if e[0] in select and e[1] in select:
            sG.add_edge(e[0],e[1])
    samples.append(sG)
    
    pList.append(ot.unif(nx.number_of_nodes(sG)))
    AList.append(nx.adjacency_matrix(sG).toarray())
    HKList3.append(sgw.undirected_normalized_heat_kernel(sG,3))
    HKList7.append(sgw.undirected_normalized_heat_kernel(sG,7))
    HKList11.append(sgw.undirected_normalized_heat_kernel(sG,11))
    lambdas.append(1/num_bootstrap)
    

print('---Bootstrap completed. Computing GW averages')

# GW barycenter computation
N = size_bootstrap # size of targeted barycenter
p = ot.unif(N) #weights of targeted barycenter
num_runs = 10   # each call to gromov_barycenters gives random initialization,
                # we will iterate this several times

def run_frechet(CList):
    runtimes = []
    frechet_loss = []
    
    for i in range(num_runs):
        start = time.time()
        gwa_adj = ot.gromov.gromov_barycenters(N,CList,pList,p,lambdas,'square_loss',max_iter=100,tol=1e-3)
        end = time.time()
        runtimes.append(end-start)

        # Frechet loss computation
        gwds = []
        for s in range(num_bootstrap):
            T, log = ot.gromov.gromov_wasserstein(gwa_adj,CList[s],p,pList[s],'square_loss',log=True)
            gwds.append(log['gw_dist'])

        frechet_loss.append(1.0/num_bootstrap*sum([d**2 for d in gwds]))

        #print('---Finished run {:d} in {:3.3f} seconds'.format(i,end-start))
    
    return frechet_loss, runtimes

res_times = []
res_loss = []
res_loss_centered = []
representation = []

# Adjacency matrix
CList = AList
frechet_loss, runtimes = run_frechet(CList)
ave_loss = np.mean(frechet_loss)
for s in range(num_runs):
    res_times.append(runtimes[s]) 
    res_loss.append(frechet_loss[s])
    res_loss_centered.append(frechet_loss[s]-ave_loss)
    representation.append('adj')

print('---Finished run with adj')
# HK3
CList = HKList3
frechet_loss, runtimes = run_frechet(CList)
ave_loss = np.mean(frechet_loss)
for s in range(num_runs):
    res_times.append(runtimes[s]) 
    res_loss.append(frechet_loss[s])
    res_loss_centered.append(frechet_loss[s]-ave_loss)
    representation.append('HK3')

print('---Finished run with HK3')
# HK7
CList = HKList7
frechet_loss, runtimes = run_frechet(CList)
ave_loss = np.mean(frechet_loss)
for s in range(num_runs):
    res_times.append(runtimes[s]) 
    res_loss.append(frechet_loss[s])
    res_loss_centered.append(frechet_loss[s]-ave_loss)
    representation.append('HK7')
    
print('---Finished run with HK7')
# HK11
CList = HKList11
frechet_loss, runtimes = run_frechet(CList)
ave_loss = np.mean(frechet_loss)
for s in range(num_runs):
    res_times.append(runtimes[s]) 
    res_loss.append(frechet_loss[s])
    res_loss_centered.append(frechet_loss[s]-ave_loss)
    representation.append('HK11')
    
print('---Finished run with HK11')


res = {'representation':representation, 'loss':res_loss,
       'log-loss':np.log(res_loss), 'centered-loss':res_loss_centered,
       'runtime':res_times}

df = pd.DataFrame(res)

# Perform Bartlett tests 
a = df[df['representation']=='adj']['centered-loss']
b = df[df['representation']=='HK3']['centered-loss']
c = df[df['representation']=='HK7']['centered-loss']
d = df[df['representation']=='HK11']['centered-loss']

_,pab = scipy.stats.bartlett(a,b)
_,pac = scipy.stats.bartlett(a,c)
_,pad = scipy.stats.bartlett(a,d)

print('---p={:3.3e} for Bartlett test of adj-HK3. Significance level needed after Bonferroni correction = 0.017'.format(pab))

print('---p={:3.3e} for Bartlett test of adj-HK7. Significance level needed after Bonferroni correction = 0.017'.format(pac))

print('---p={:3.3e} for Bartlett test of adj-HK11. Significance level needed after Bonferroni correction = 0.017'.format(pad))


p_vals = ['---p={:3.3e} for Bartlett test of adj-HK3. Significance level needed after Bonferroni correction = 0.017'.format(pab),
          '---p={:3.3e} for Bartlett test of adj-HK7. Significance level needed after Bonferroni correction = 0.017'.format(pac),
          '---p={:3.3e} for Bartlett test of adj-HK11. Significance level needed after Bonferroni correction = 0.017'.format(pad)]


matplotlib.style.use('ggplot')
# plt.figure()
sns.catplot(x='representation',y='loss',data=df,kind='boxen')
plt.title('Village data: loss-representation')
plt.savefig('res_gwa-village-loss.pdf',dpi=300,format='pdf',bbox_inches='tight')


# plt.figure()
sns.catplot(x='representation',y='centered-loss',data=df,kind='boxen')
plt.title('Village data: centered loss-representation')
plt.savefig('res_gwa-village-closs.pdf',dpi=300,format='pdf',bbox_inches='tight')

# plt.figure()
sns.catplot(x='representation',y='runtime',data=df,kind='boxen')
plt.title('Village data: runtime-representation')
plt.savefig('res_gwa-village-runtime.pdf',dpi=300,format='pdf',bbox_inches='tight')

plt.show()

# Save results
tab_cols = ['Representation','Frechet loss average','Frechet loss variance','Runtime']
tab_rows = []
tab_rows.append(['Adj', 
                np.mean(df[df['representation']=='adj']['loss']),
                np.var(df[df['representation']=='adj']['loss']),
               np.mean(df[df['representation']=='adj']['runtime'])])

tab_rows.append(['HK3', 
                np.mean(df[df['representation']=='HK3']['loss']),
                np.var(df[df['representation']=='HK3']['loss']),
               np.mean(df[df['representation']=='HK3']['runtime'])])

tab_rows.append(['HK7', 
                np.mean(df[df['representation']=='HK7']['loss']),
                np.var(df[df['representation']=='HK7']['loss']),
               np.mean(df[df['representation']=='HK7']['runtime'])])

tab_rows.append(['HK11', 
                np.mean(df[df['representation']=='HK11']['loss']),
                np.var(df[df['representation']=='HK11']['loss']),
               np.mean(df[df['representation']=='HK11']['runtime'])])

res_tab = pd.DataFrame(tab_rows,columns=tab_cols)
res_tab.to_csv('res_gwa_village.txt',header=True, index=False, sep='\t')

with open('res_gwa_village.txt', 'a') as outfile:
    json.dump(p_vals,outfile,indent=1)