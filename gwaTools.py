"""
Here are some utility functions
"""

import numpy as np


# Function cycleFrag
# Compute a fragmented cycle network, represented 
# as a Markov transition matrix
# Input:
# n -- starting number of nodes
# k -- number of nodes to delete
# Output:
# A new transition matrix on n-k nodes

def cycleFrag(n,k):
    import random
    wgt = 1/2 # either stay at one node,
    # or transition to the next

    # Initialize transition matrix
    P = np.zeros((n,n))
    row_max = P.shape[0]
    col_max = P.shape[1]
    for ii in np.arange(0,row_max):
        for jj in np.arange(0,col_max):
            if ii == jj or (ii+1)%n == jj%n:
                P[ii][jj] = wgt

    # Delete nodes (i.e. rows and columns)
    for ii in np.arange(0,k):
        tmp = random.randint(0,n-1-ii)
        P = np.delete(P,tmp,0)
        P = np.delete(P,tmp,1)

    # Renormalize
    row_ones = np.ones((n-k,1))
    row_marginal = np.matmul(P,row_ones)
    res = P/row_marginal
    return res

# A related function that accepts a list of nodes
# to kill
def cycleFrag_at(n,to_kill):
    import random
    wgt = 1/2 # either stay at one node,
    # or transition to the next

    # Initialize transition matrix
    P = np.zeros((n,n))
    row_max = P.shape[0]
    col_max = P.shape[1]
    for ii in np.arange(0,row_max):
        for jj in np.arange(0,col_max):
            if ii == jj or (ii+1)%n == jj%n:
                P[ii][jj] = wgt

    # Delete nodes (i.e. rows and columns)
    for ii in to_kill:
        P = np.delete(P,ii,0)
        P = np.delete(P,ii,1)

    # Renormalize
    k = len(to_kill)
    row_ones = np.ones((n-k,1))
    row_marginal = np.matmul(P,row_ones)
    res = P/row_marginal
    return res

# A related function that accepts a list of nodes
# to kill
def forwardCycleFrag_at(n,to_kill):
    import random
    wgt = 1 # either stay at one node,
    # or transition to the next

    # Initialize transition matrix
    P = np.zeros((n,n))
    row_max = P.shape[0]
    col_max = P.shape[1]
    for ii in np.arange(0,row_max):
        for jj in np.arange(0,col_max):
            if (ii+1)%n == jj%n:  #or ii == jj:
                P[ii][jj] = wgt

    # Delete nodes (i.e. rows and columns)
    for ii in to_kill:
        P = np.delete(P,ii,0)
        P = np.delete(P,ii,1)

    return P

