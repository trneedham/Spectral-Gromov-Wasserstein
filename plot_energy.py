import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ot
import gromovWassersteinAveraging as gwa
from geodesicVisualization import *
from spectralGW import *
from GromovWassersteinFramework import *

import random
import time
from scipy import linalg
import pandas as pd

from graphProcessing import load_graph
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# Load Data
graph_file = 'data/IMDB-BINARY_A.txt'
indicator_file = 'data/IMDB-BINARY_graph_indicator.txt'
label_file = 'data/IMDB-BINARY_graph_labels.txt'

graphs, labels = load_graph(graph_file,indicator_file,label_file)

total_num_graphs = len(graphs)


# Start run
print('---Full run will take ~40 minutes. Starting now...')

adj_data = []
hk_data = []

num_trials = 100

num_steps = 100
num_skips = 1000
distribution_exponent = 0.1

ts = [5,10,20,50]

full_trial_start = time.time()

for trial_ind in range(num_trials):
    
    print('Trial:', trial_ind)

    # Define Graphs
    ind1 = np.random.randint(total_num_graphs)
    ind2 = np.random.randint(total_num_graphs)
    
    print('Graphs',ind1,'and',ind2)

    G1 = graphs[ind1]
    G2 = graphs[ind2]

    p1 = node_distribution(G1,0,distribution_exponent)
    p2 = node_distribution(G2,0,distribution_exponent)

    # Sample probability polytope
    A, b = gw_equality_constraints(p1,p2)

    Markov_steps = coupling_ensemble(A,b,p1,p2,num_steps,num_skips)

    #### Run Adjacency Computation ####
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
    
    coup, log = gromov_wasserstein_asym_fixed_initialization(A1, A2, p1, p2, p1[:,None]*p2[None,:])
    
    row = [ind1,ind2,(np.max(losses)-np.min(losses))/np.min(losses)*100, 
           (log['gw_dist'] - np.min(losses))/np.min(losses)*100, end-start]

    #print('Adj')
    #print(row)
    
    adj_data.append(row)
    
    for t_ind in range(len(ts)):
        
        t = ts[t_ind]
        
        #### Run Heat Kernel Computation ####
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

        coup, log = gromov_wasserstein_asym_fixed_initialization(HK1, HK2, p1, p2, p1[:,None]*p2[None,:])

        row = [ind1,ind2,t,(np.max(losses)-np.min(losses))/np.min(losses)*100, 
               (log['gw_dist'] - np.min(losses))/np.min(losses)*100, end-start]

        #print('HK')
        #print(row)

        hk_data.append(row)
        
#     if trial_ind % 10 ==0:
#         print('--Currently at trial {:d}'.format(trial_ind))
        
full_trial_end = time.time()

print('---All trials completed in {:3.3f} seconds'.format(
    full_trial_end-full_trial_start))


# Prepare results in pandas dataframe
cols = ['Graph Index 1','Graph Index 2','Worst Rel. Error (%)',
 'Product Rel. Error (%)','Time']
results_adj = pd.DataFrame(adj_data, columns = cols)

cols = ['Graph Index 1','Graph Index 2','t Value','Worst Rel. Error (%)',
 'Product Rel. Error (%)','Time']
results_hk = pd.DataFrame(hk_data, columns = cols)


## Prepare results to display in table
results_hk5=results_hk[results_hk['t Value']==5]
results_hk10=results_hk[results_hk['t Value']==10]
results_hk20=results_hk[results_hk['t Value']==20]
# Time per iteration
tab_adj_time = np.mean(results_adj['Time']/num_steps)
tab_hk5_time = np.mean(results_hk[results_hk['t Value']==5]['Time']/num_steps)
tab_hk10_time = np.mean(results_hk10['Time']/num_steps)
tab_hk20_time = np.mean(results_hk20['Time']/num_steps)

# Worst error
results_adj_worst_filt = results_adj[results_adj['Worst Rel. Error (%)'] <=1e2]
results_adj_worst_filt = results_adj_worst_filt[results_adj_worst_filt['Worst Rel. Error (%)'] >=0]

results_hk5_worst_filt = results_hk5[results_hk5['Worst Rel. Error (%)'] <=1e2]
results_hk5_worst_filt = results_hk5_worst_filt[results_hk5_worst_filt['Worst Rel. Error (%)'] >=0]
results_hk10_worst_filt = results_hk10[results_hk10['Worst Rel. Error (%)'] <=1e2]
results_hk10_worst_filt = results_hk10_worst_filt[results_hk10_worst_filt['Worst Rel. Error (%)'] >=0]
results_hk20_worst_filt = results_hk20[results_hk20['Worst Rel. Error (%)'] <=1e2]
results_hk20_worst_filt = results_hk20_worst_filt[results_hk20_worst_filt['Worst Rel. Error (%)'] >=0]

# Product error
results_adj_prod_filt = results_adj[results_adj['Product Rel. Error (%)'] <=1e2]
results_adj_prod_filt = results_adj_prod_filt[results_adj_prod_filt['Product Rel. Error (%)'] >=0]

results_hk5_prod_filt = results_hk5[results_hk5['Product Rel. Error (%)'] <=1e2]
results_hk5_prod_filt = results_hk5_prod_filt[results_hk5_prod_filt['Product Rel. Error (%)'] >=0]
results_hk10_prod_filt = results_hk10[results_hk10['Product Rel. Error (%)'] <=1e2]
results_hk10_prod_filt = results_hk10_prod_filt[results_hk10_prod_filt['Product Rel. Error (%)'] >=0]
results_hk20_prod_filt = results_hk20[results_hk20['Product Rel. Error (%)'] <=1e2]
results_hk20_prod_filt = results_hk20_prod_filt[results_hk20_prod_filt['Product Rel. Error (%)'] >=0]

tab_cols=['Loss function','Time/Iter(s)','Worst Error(%)','Product Error(%)']
tab_rows=[]
tab_rows.append(['Adj', tab_adj_time, 
                 np.mean(results_adj_worst_filt['Worst Rel. Error (%)']),
                 np.mean(results_adj_prod_filt['Product Rel. Error (%)'])])

tab_rows.append(['Spec, t=5', tab_hk5_time, 
                 np.mean(results_hk5_worst_filt['Worst Rel. Error (%)']),
                 np.mean(results_hk5_prod_filt['Product Rel. Error (%)'])])

tab_rows.append(['Spec, t=10', tab_hk10_time, 
                 np.mean(results_hk10_worst_filt['Worst Rel. Error (%)']),
                 np.mean(results_hk10_prod_filt['Product Rel. Error (%)'])])

tab_rows.append(['Spec, t=20', tab_hk20_time, 
                 np.mean(results_hk20_worst_filt['Worst Rel. Error (%)']),
                 np.mean(results_hk20_prod_filt['Product Rel. Error (%)'])])
             
tab = pd.DataFrame(tab_rows, columns = tab_cols)

# Display tabulated results to terminal
print(tab)
tab.to_csv('res_energy.txt',header=True, index=False, sep='\t')


# Plot results with product initializations
# fig = plt.figure(figsize = (15,10))
fig = plt.figure()
ax = fig.add_subplot(111)

adj_prod_errors = results_adj['Product Rel. Error (%)']

highest_percentage = 20

xs = np.linspace(1,highest_percentage,highest_percentage)

ys = np.zeros([highest_percentage,])

for j in range(highest_percentage):
    
    x = xs[j]
    y = np.sum(adj_prod_errors > x)/len(adj_prod_errors)
    ys[j] = y

### t = 5

adj_prod_errors_5 = results_hk[results_hk['t Value'] == 5]['Product Rel. Error (%)']

ys_5 = np.zeros([highest_percentage,])

for j in range(highest_percentage):
    
    x = xs[j]
    y = np.sum(adj_prod_errors_5 > x)/len(adj_prod_errors_5)
    ys_5[j] = y
    
### t = 10

adj_prod_errors_10 = results_hk[results_hk['t Value'] == 10]['Product Rel. Error (%)']

ys_10 = np.zeros([highest_percentage,])

for j in range(highest_percentage):
    
    x = xs[j]
    y = np.sum(adj_prod_errors_10 > x)/len(adj_prod_errors_10)
    ys_10[j] = y
    
### t = 20

adj_prod_errors_20 = results_hk[results_hk['t Value'] == 20]['Product Rel. Error (%)']

ys_20 = np.zeros([highest_percentage,])

for j in range(highest_percentage):
    
    x = xs[j]
    y = np.sum(adj_prod_errors_20 > x)/len(adj_prod_errors_20)
    ys_20[j] = y
    
plt.plot(xs,ys*100)
plt.plot(xs,ys_5*100)
plt.plot(xs,ys_10*100)
plt.plot(xs,ys_20*100)

fontSizeAxes = 12 #25
fontSizeTitle = 12 #30

legend_labels = ['Adj','HK 5', 'HK 10', 'HK 20']
plt.legend(legend_labels, fontsize = fontSizeTitle)
ax.set_xlabel('Rel. Error (%)', fontsize = fontSizeAxes)
ax.set_ylabel('% Samples with Error > Rel. Error',fontsize = fontSizeAxes)
ax.tick_params(axis="x", labelsize=fontSizeAxes)
ax.tick_params(axis="y", labelsize=fontSizeAxes)
plt.title('Relative Errors of GW Loss with Product Initializations',fontsize = fontSizeTitle)
# plt.show()
#fig.savefig('ProductInitializationErrors.png')
fig.savefig('res_energy_ProductInitializationErrors.pdf',bbox_inches='tight')

# Plot results with MCMC initializations
# fig = plt.figure(figsize = (15,10))
fig = plt.figure()
ax = fig.add_subplot(111)

adj_prod_errors = results_adj['Worst Rel. Error (%)']

highest_percentage = 20

xs = np.linspace(1,highest_percentage,highest_percentage)

ys = np.zeros([highest_percentage,])

for j in range(highest_percentage):
    
    x = xs[j]
    y = np.sum(adj_prod_errors > x)/len(adj_prod_errors)
    ys[j] = y

### t = 5

adj_prod_errors_5 = results_hk[results_hk['t Value'] == 5]['Worst Rel. Error (%)']

ys_5 = np.zeros([highest_percentage,])

for j in range(highest_percentage):
    
    x = xs[j]
    y = np.sum(adj_prod_errors_5 > x)/len(adj_prod_errors_5)
    ys_5[j] = y
    
### t = 10

adj_prod_errors_10 = results_hk[results_hk['t Value'] == 10]['Worst Rel. Error (%)']

ys_10 = np.zeros([highest_percentage,])

for j in range(highest_percentage):
    
    x = xs[j]
    y = np.sum(adj_prod_errors_10 > x)/len(adj_prod_errors_10)
    ys_10[j] = y
    
### t = 20

adj_prod_errors_20 = results_hk[results_hk['t Value'] == 20]['Worst Rel. Error (%)']

ys_20 = np.zeros([highest_percentage,])

for j in range(highest_percentage):
    
    x = xs[j]
    y = np.sum(adj_prod_errors_20 > x)/len(adj_prod_errors_20)
    ys_20[j] = y
    
plt.plot(xs,ys*100)
plt.plot(xs,ys_5*100)
plt.plot(xs,ys_10*100)
plt.plot(xs,ys_20*100)

legend_labels = ['Adj','HK 5', 'HK 10', 'HK 20']
plt.legend(legend_labels, fontsize = fontSizeTitle)
ax.set_xlabel('Rel. Error (%)', fontsize = fontSizeAxes)
ax.set_ylabel('% Samples with Error > Rel. Error',fontsize = fontSizeAxes)
ax.tick_params(axis="x", labelsize=fontSizeAxes)
ax.tick_params(axis="y", labelsize=fontSizeAxes)
plt.title('Worst Relative Errors of GW Loss with Random Initializations',fontsize = fontSizeTitle)

fig.savefig('res_energy_RandomInitializationErrors.pdf',bbox_inches='tight')

plt.show()