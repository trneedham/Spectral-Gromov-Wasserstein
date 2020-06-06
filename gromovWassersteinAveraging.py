"""
Gromov-Wasserstein Based Averaging

This code adapts and elaborates on the Gromov-Wasserstein distance code of Flamary (add link).
The theory is based on work of KT Sturm (add ref.).
"""



import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import ot
from sklearn import manifold
from sklearn.cluster import KMeans
from ot.optim import cg
from ot.optim import line_search_armijo

"""
The method for computing GW distance is based on a projected gradient descent algorithm.
"""

"""
We define some basic functions.
"""
def loss_fun(a,b):
    return (1/2.0)*(a-b)**2

def frobenius(A,B):
    return np.trace(np.matmul(np.transpose(A),B))

"""
The next block of code recreates Flamary's implementation for 
GW distance between symmetric cost matrices. The only difference is some 
simplification by ignoring the KL-divergence option.
"""
# Auxilliary function to implement the tensor product of [ref]
def init_matrix(C1, C2, T, p, q):

    def f1(a):
        return (a**2) / 2.0

    def f2(b):
        return (b**2) / 2.0

    def h1(a):
        return a

    def h2(b):
        return b

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


# Define the tensor product from [ref]
def tensor_product(constC, hC1, hC2, T):
    A = -np.dot(hC1, T).dot(hC2.T)
    tens = constC + A
    return tens

# Define the loss function for GW distance.
def gwloss(constC, hC1, hC2, T):
    tens = tensor_product(constC, hC1, hC2,T)
    return frobenius(tens,T)

# Define a helper function that computes gwloss from 
# a fixed initial coupling
def gwloss_init(C1, C2, p, q, G0):
    constC, hC1, hC2 = init_matrix(C1,C2,G0,p,q)
    return gwloss(constC, hC1, hC2,G0)

# Define the gradient of the GW loss function.
def gwggrad(constC, hC1, hC2,T):
    return 2 * tensor_product(constC, hC1, hC2, T)

# Compute GW distance via projected gradient descent.
def gromov_wasserstein(C1, C2, p, q, log=True):
    G0 = p[:, None] * q[None, :]

    constC, hC1, hC2 = init_matrix(C1,C2,G0,p,q)
    
    def f(G):
        return gwloss(constC, hC1, hC2,G)

    def df(G):
        return gwggrad(constC, hC1, hC2,G)

    if log:
        res, log = cg(p, q, 0, 1, f, df, G0, log=True)
        log['gw_dist'] = gwloss(constC, hC1, hC2, res)
        return res, log
    else:
        return cg(p, q, 0, 1, f, df, res)
    
""" 
We now define a modified version of GW distance for asymmetric cost matrices.
"""

# Define the loss function for symmetrized GW distance.
def gwloss_asym(constC, constCt, hC1, hC1t, hC2, hC2t, T):
    tens = (1/2.0)*tensor_product(constC, hC1, hC2,T) \
            + (1/2.0)*tensor_product(constCt, hC1t, hC2t,T)
    return frobenius(tens,T)

# Define the gradient of the symmetrized GW loss function.
def gwggrad_asym(constC, constCt, hC1, hC1t, hC2, hC2t, T):
    return tensor_product(constC, hC1, hC2, T) \
             + tensor_product(constCt, hC1t, hC2t,T)

# Compute symmetrized GW distance via projected gradient descent.
def gromov_wasserstein_asym(C1, C2, p, q, log=True):
    G0 = p[:, None] * q[None, :]

    C1t = np.transpose(C1)
    C2t = np.transpose(C2)
    
    constC, hC1, hC2 = init_matrix(C1,C2,G0,p,q)
    constCt, hC1t, hC2t = init_matrix(C1t,C2t,G0,p,q)
    
    def f(G):
        return gwloss_asym(constC, constCt, hC1, hC1t, hC2, hC2t, G)

    def df(G):
        return gwggrad_asym(constC, constCt, hC1, hC1t, hC2, hC2t, G)

    if log:
        res, log = gwa_cg(p, q, 0, 1, f, df, G0, log=True)
        log['gw_dist'] = gwloss(constC, hC1, hC2, res)
        return res, log
    else:
        return cg(p, q, 0, 1, f, df, res)
    
# Small modification to the original cg function in ot.optim
def gwa_cg(a, b, M, reg, f, df, G0=None, numItermax=200,
       stopThr=1e-9, verbose=False, log=False):
# Please refer to cg function in pot package for documentation

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = np.outer(a, b)
    else:
        G = G0

    def cost(G):
        return np.sum(M * G) + reg * f(G)

    f_val = cost(G)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}'.format(it, f_val, 0))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg * df(G)
        # set M positive
        Mi += Mi.min()

        # solve linear program
        Gc = ot.emd(a, b, Mi)

        deltaG = Gc - G

        # line search
        alpha, fc, f_val = line_search_armijo(cost, G, deltaG, Mi, f_val)
        
        # added May 19, 2020 to avoid multiplication between NoneType and Float
        if alpha is not None:
            G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0
        
        # extra stopping condition in gwa_cg
        # to avoid dividing by zero
        if f_val == 0:
            loop = 0

        delta_fval = (f_val - old_fval) / abs(f_val)
        if abs(delta_fval) < stopThr:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}'.format(
                    'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
            print('{:5d}|{:8e}|{:8e}'.format(it, f_val, delta_fval))

    if log:
        return G, log
    else:
        return G

"""
Code related to GW warm start using the Third Lower Bound (links)
"""


# GW with fixed initialization. Here we include an option to start at a particular coupling
# in the gradient descent process. 
def gromov_wasserstein_asym_fixed_initialization(C1, C2, p, q, G0, log=True):
    
    C1t = np.transpose(C1)
    C2t = np.transpose(C2)
    
    constC, hC1, hC2 = init_matrix(C1,C2,G0,p,q)
    constCt, hC1t, hC2t = init_matrix(C1t,C2t,G0,p,q)
    
    def f(G):
        return gwloss_asym(constC, constCt, hC1, hC1t, hC2, hC2t, G)

    def df(G):
        return gwggrad_asym(constC, constCt, hC1, hC1t, hC2, hC2t, G)

    if log:
        res, log = gwa_cg(p, q, 0, 1, f, df, G0, log=True)
        log['gw_dist'] = gwloss(constC, hC1, hC2, res)
        return res, log
    else:
        return gwa_cg(p, q, 0, 1, f, df, res)
    

# One way to initializate is to use the TLB. Here is some code to implement this.

def gen_inv(f,g,pf,pg):
    # Function gen_inv for computing the generalized inverses 
    # of real-valued distributions. 
    # Inputs: f,g real numbers; pf, pg the masses at each number
    # Output: Finv, Ginv 
    
    # [0] sort the distributions to make it easier to take a cumsum
    sort_idf = np.argsort(f) # sort indices
    sort_idg = np.argsort(g)
    
    sorted_f = f[sort_idf]
    sorted_g = g[sort_idg]
    
    sorted_pf = pf[sort_idf]
    sorted_pg = pg[sort_idg]
    
    cm_pf = np.cumsum(sorted_pf)
    cm_pg = np.cumsum(sorted_pg)
    
    # [1] Define the generalized inverses
    def Finv(r):
        return sorted_f[np.argmax(cm_pf >= r)]
    # For 0 <= r <= 1, np.argmax(cm_pf>=r) finds the index where 
    # the cumulative distribution exceeds r. sorted_f of this index 
    # returns the first real number whose sublevel set includes this much mass 
    
    def Ginv(r):
        return sorted_g[np.argmax(cm_pg >= r)]  
    
    return Finv, Ginv

def twoTLB(A,pA,B,pB,steps=50):
    # Function twoTLB 
    # Inputs: networks (A,pA), (B,pB)
    # Outputs: 
    # -- tlb: 2-TLB cost
    # -- tlbCoup: optimal coupling for 2-TLB
    
    import ot
    
    C = np.zeros([A.shape[1],B.shape[1]])
    r_Vals = np.linspace(0,1,steps)
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            out = 0
            Finv, Ginv = gen_inv(A[i,:],B[j,:],pA,pB)
            for r in range(0,steps-1):
                # Riemann integral
                out = out + (r_Vals[r+1]-r_Vals[r])*(Finv(r_Vals[r]) - Ginv(r_Vals[r]))**2
            C[i,j] = out
    
    tlb, log = ot.emd2(pA,pB,C, log = True, return_matrix=True)
    return tlb, log['G']

""" 
The functions below will be used to compute averages.
"""

# Expands each entry of C0 to a matrix with the size of C1. The resulting
# square matrix has size C0*C1
def expand(C0,C1):
    N0 = C0.shape[0]
    N1 = C1.shape[1]
    return np.repeat(np.repeat(C0,N1).reshape(N0,N0*N1),N1,axis=0)

# Tiles copies of C1 to produce a matrix of size C0*C1
def tile(C0,C1):
    N0 = C0.shape[0]
    return np.tile(C1,(N0,N0))

# Geodesic joining the networks in network space, via Sturm's formula
def geodesic(C0,p,C1,q,t):
    opt_coup, log = gromov_wasserstein_asym(C0,C1,p,q)
    C0new = expand(C0,C1)
    C1new = tile(C0,C1)
    def geod(t):
        return (1-t)*C0new + t*C1new
    return geod, opt_coup

""" Code for preprocessing coupling matrices and 
probability vectors"""

def coupling_threshold(coup, thresh = 1):
    sZ = coup.shape
    rows = sZ[0]
    cols = sZ[1]
    # obtain marginals
    p = np.matmul(coup,np.ones((cols,1)))
    q = np.matmul(np.ones((1,rows)),coup)
    
    # get min values (take first value if duplicates)
    a = .01*min(p)[0] 
    b = .01*min(q)[0]
    thresh = min(thresh,a,b)
    coup[np.abs(coup)<thresh] = 0
    return coup

def normalized_threshold(coup, p, thresh = 1):
    coup = coupling_threshold(coup, thresh)
    normalization_vector = np.array([p[j]/sum(coup[j,:]) for j in range(coup.shape[0])])
    coup = np.matmul(np.diag(normalization_vector),coup)
    return coup


def split_matrix_one_point(vec_index, num_copies, A):
    # Input: - (bad) row or column index (coming from the coupling matrix)
    #        - number of copies to make
    #        - arbitrary matrix,

    # Output: split matrix

    # [note]: kept legacy function as old_split_matrix_one_point; 
    # new version sets all the internal weights of the split nodes to the same value
    size = A.shape[0]
    
    # Create matrices to insert into C
    A1 = A[vec_index,:vec_index]
    A1 = np.tile(A1,[num_copies,1])
    A2 = A[vec_index,vec_index]*np.ones((num_copies,num_copies)) #[note] only change is here
    A3 = A[vec_index,vec_index+1:]
    A3 = np.tile(A3,[num_copies,1])
    
    B1 = A[:vec_index,vec_index]
    B1 = np.tile(B1,[num_copies,1]).T
    B3 = A[vec_index+1:,vec_index]
    B3 = np.tile(B3,[num_copies,1]).T
    
    # Insert zero rows 
    for j in range(num_copies-1):
        A = np.insert(A,vec_index+1,np.zeros((1,size)),axis=0)
    
    # Insert zero columns
    size = A.shape[0]
    
    for j in range(num_copies-1):
        A = np.insert(A.T,vec_index+1,np.zeros((1,size)),axis=0).T
    
    # Fill in Zeros
    A[vec_index:vec_index+num_copies,:vec_index] = A1
    A[vec_index:vec_index+num_copies,vec_index:vec_index+num_copies] = A2
    A[vec_index:vec_index+num_copies,vec_index+num_copies:] = A3
    
    A[:vec_index,vec_index:vec_index+num_copies] = B1
    A[vec_index+num_copies:,vec_index:vec_index+num_copies] = B3
    
    return A

def split_matrix_all_points(vec_indices,num_copies,A):
    # Input: - list of row indices to expand, should be in increasing order
    #        - list of number of copies to expand by for each index
    #        - arbitrary matrix A
    num_indices = len(vec_indices)
    for j in range(num_indices):
        A = split_matrix_one_point(vec_indices[j], num_copies[j], A)
        vec_indices = [index + num_copies[j] - 1 for index in vec_indices]
    return A

def find_bad_rows(coup):
    # bad rows: rows with more than one nonzero value, preventing the matrix from being a permutation
    row_indices = np.nonzero(np.count_nonzero(coup, axis=1) > 1)[0]
    bad_rows = coup[row_indices,:]
    num_bad_rows = len(row_indices)
    num_copies = []
    for j in range(num_bad_rows):
        bad_row = coup[row_indices[j],:]
        num_nonzero = np.count_nonzero(bad_row)
        num_copies.append(num_nonzero)
    return bad_rows, row_indices, num_copies

def split_probability_one_point(vec,vec_index,num_copies,p):
    nonzeros = vec[vec != 0]
    p = np.delete(p,vec_index)
    p = np.insert(p,vec_index,nonzeros)
    return p

def split_probability_all_points(bad_vecs, vec_indices, num_copies, p):
    for j in range(len(vec_indices)):
        p = split_probability_one_point(bad_vecs[j],vec_indices[j],num_copies[j],p)
        vec_indices = [index+num_copies[j]-1 for index in vec_indices]
    return p

def split_cost_and_probability_by_row(coup,C,p):
    bad_rows, row_indices, num_copies = find_bad_rows(coup)
    C = split_matrix_all_points(row_indices, num_copies, C)
    p = split_probability_all_points(bad_rows, row_indices, num_copies, p)
    return C, p

def find_bad_columns(coup):
    column_indices = np.nonzero(np.count_nonzero(coup, axis=0) > 1)[0]
    bad_columns = coup[:,column_indices]
    num_bad_columns = len(column_indices)
    num_copies = []
    for j in range(num_bad_columns):
        bad_column = coup[:,column_indices[j]]
        num_nonzero = np.count_nonzero(bad_column)
        num_copies.append(num_nonzero)
    return bad_columns, column_indices, num_copies

def split_cost_and_probability_by_column(coup,C,p):
    bad_columns, column_indices, num_copies = find_bad_columns(coup)
    bad_columns = bad_columns.T
    C = split_matrix_all_points(column_indices, num_copies, C)
    p = split_probability_all_points(bad_columns, column_indices, num_copies, p)
    return C, p

def split_row(row):
    row_length = len(row)
    num_nonzero = np.count_nonzero(row)
    nonzero_inds = np.nonzero(row)[0]
    split_rows = np.zeros((num_nonzero,row_length))
    for j in range(num_nonzero):
        split_rows[j,nonzero_inds[j]] = row[nonzero_inds[j]]
    return split_rows

def split_all_rows(bad_rows, row_indices, num_copies, coup):
    row_length = len(coup[0,:])
    for j in range(len(row_indices)):
        
        for k in range(num_copies[j]-1):
            coup = np.insert(coup,row_indices[j]+1,np.zeros((1,row_length)),axis=0)
        
        coup[range(row_indices[j],row_indices[j]+num_copies[j]),:] = split_row(bad_rows[j])
        
        row_indices = [index + num_copies[j] - 1 for index in row_indices]
    
    return coup

def split_column(column):
    column_length = len(column)
    num_nonzero = np.count_nonzero(column)
    nonzero_inds = np.nonzero(column)[0]
    split_columns = np.zeros((column_length, num_nonzero))
    for j in range(num_nonzero):
        split_columns[nonzero_inds[j],j] = column[nonzero_inds[j]]
    return split_columns

def split_all_columns(bad_columns, column_indices, num_copies, coup):
    column_length = len(coup[:,0])
    for j in range(len(column_indices)):
        
        for k in range(num_copies[j]-1):
            coup = np.insert(coup,column_indices[j]+1,np.zeros((1,column_length)),axis=1)
        
        coup[:,range(column_indices[j],column_indices[j]+num_copies[j])] = split_column(bad_columns[j])
        
        column_indices = [index + num_copies[j] - 1 for index in column_indices]
    
    return coup

def split_coupling(coup):
    bad_rows, row_indices, num_copies = find_bad_rows(coup)
    coup = split_all_rows(bad_rows, row_indices, num_copies, coup)
    bad_columns, column_indices, num_copies = find_bad_columns(coup)
    coup = split_all_columns(bad_columns.T, column_indices, num_copies, coup)
    return coup


def split_cost_coupling_probabilities(coup, C0, C1, p, q,thresh=1):
    coup = normalized_threshold(coup, p,thresh)
    C0, p = split_cost_and_probability_by_row(coup,C0,p)
    C1, q = split_cost_and_probability_by_column(coup,C1,q)
    coup = split_coupling(coup)
    return coup, C0, C1, p, q

""" Reimplement geodesics code with preceding code
for mass splitting according to the optimal coupling """

def geod(C0,C1,p,q):
    opt_coup, log = gromov_wasserstein_asym(C0,C1,p,q)
    coup, C0, C1, p, q = split_cost_coupling_probabilities(opt_coup, C0, C1, p, q)
    perm = 1*(coup != 0)
    C1new = np.matmul(np.matmul(perm,C1),perm.T)
    
    geod_curve = lambda t:(1-t)*C0 + t*C1new
    
    return geod_curve

def geod_plot(C0,C1,p,q):
    geod_curve = geod(C0,C1,p,q)
    
    fig = plt.figure(figsize = (20,8))
    
    for j in range(10):
        ax = fig.add_subplot(2,5,j+1)
        plt.imshow(geod_curve(np.linspace(0,1,10)[j]))
        ax.axis('off')

def geod_plot_shapes(C0,C1,p,q):
    geod_curve = geod(C0,C1,p,q)
    
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=2)
    labels = list(range(100))
    
    fig = plt.figure(figsize = (20,8))
    
    for j in range(10):
        ax = fig.add_subplot(2,5,j+1)
        C = geod_curve(np.linspace(0,1,10)[j])
        results = mds.fit(C)
        coords = results.embedding_
        plt.scatter(coords[:, 0], coords[:, 1], c = labels)
        plt.axis('equal')
        ax.axis('off')

""" Computing averages """

def exp_map(C0,vec):
    return C0 + vec

def log_map(C0, C1, p, q):  
    opt_coup, log = gromov_wasserstein_asym(C0,C1,p,q)
    coup, C0, C1, p, q = split_cost_coupling_probabilities(opt_coup, C0, C1, p, q)
    perm = 1*(coup != 0)
    C1 = np.matmul(np.matmul(perm,C1),perm.T)
    vec = -C0 + C1 #[note] this is the main step
    return C0, C1, p, q, vec, opt_coup

def log_map_simple(C0, C1, p, q, opt_coup):  
    # same as log_map, but with coupling supplied
    coup, C0, C1, p, q = split_cost_coupling_probabilities(opt_coup, C0, C1, p, q)
    perm = 1*(coup != 0)
    C1 = np.matmul(np.matmul(perm,C1),perm.T)
    vec = -C0 + C1 
    return C0, p, vec

def log_map_compressed(C0, C1, p, q, opt_coup):  
    # expand, permute, then compress
    coup_new, C0_new, C1_new, _, q_new = split_cost_coupling_probabilities(opt_coup, C0, C1, p, q)
    perm = 1*(coup_new != 0)
    C1_new = np.matmul(np.matmul(perm,C1_new),perm.T)
    q_new = np.matmul(perm,q_new)
    # uncompressed tangent vector
    vec = -C0_new + C1_new
    # compressing step. First build a cleaned-up version of opt_coup
    # that is not yet blown-up
    opt_coup = normalized_threshold(opt_coup, p,thresh=1)
    _, bad_row_indices, bad_row_num_copies = find_bad_rows(opt_coup)
    # note: coup_new is "cleaned up" version of opt_coup, so
    # we should use this to find bad rows

    # Get a vector of labels
    row_labels = []
    for i in range(0,C0.shape[0]):
        if i not in bad_row_indices:
            row_labels.append(i)
        else:
            idx = np.where(i == bad_row_indices)[0][0]
            #idx = np.where(1 in bad_row_indices)[0][0]
            to_append = [i]*bad_row_num_copies[idx]
            row_labels.extend(to_append)

    # convert to np array for easy indexing
    row_labels = np.asarray(row_labels)

    # Now create compressed tangent vector
    cpr_vec = np.zeros(C0.shape)
    for i in range(0,C0.shape[0]):
        idx_i = np.where(row_labels == i)[0]
        for j in range(0,C0.shape[0]):
            idx_j = np.where(row_labels == j)[0]
            #debugging
            #print('\nPrinting index i')
            # print(idx_i)
            # print('\nPrinting index j')
            # print(idx_j)
            tmp = vec[idx_i,:]
            tmp = tmp[:,idx_j]
            cpr_vec[i,j] = np.mean(tmp)

    return cpr_vec

def frechet_gradient_compressed(CList,pList,CBase,pBase,budget):
    numC = len(CList)
    tangent_vectors = []

    for j in range(numC):
        # check coupling first
        opt_coup, log = gromov_wasserstein_asym(CBase,CList[j],pBase,pList[j])
        # clean-up step
        opt_coup = normalized_threshold(opt_coup, pBase,thresh=1)


        # budget allocation
        _, bad_row_indices, bad_row_num_copies = find_bad_rows(opt_coup)
        row_req = opt_coup.shape[0] + (np.sum(bad_row_num_copies) - len(bad_row_num_copies))

        if log['gw_dist'] < 1e-10:
            vec = np.zeros((CBase.shape[0],CBase.shape[0]))
            tangent_vectors.append(vec)
        elif row_req > budget:
            vec = log_map_compressed(CBase,CList[j],pBase,pList[j],opt_coup)
            # not expanding anything
            tangent_vectors.append(vec)
        else:
            CBase, pBase, vec = log_map_simple(CBase,CList[j],pBase,pList[j],opt_coup)
            tangent_vectors.append(vec)
            # expand previous computed vectors
            for k in range(j):
                tangent_vectors[k] = split_matrix_all_points(bad_row_indices, bad_row_num_copies, tangent_vectors[k])

    gradient = (1/float(numC))*sum(tangent_vectors)
    return CBase, pBase, gradient
    



def frechet_gradient(CList,pList,CBase,pBase):
    numC = len(CList)
    tangent_vectors = []
    
    for j in range(numC):
        CBase, C1, pBase, q, vec, opt_coup = log_map(CBase,CList[j],pBase,pList[j])
        tangent_vectors.append(vec)
        
        # this is the part which expands all the previously computed tangent vectors
        for k in range(j):
            bad_rows, row_indices, num_copies = find_bad_rows(opt_coup)
            tangent_vectors[k] = split_matrix_all_points(row_indices, num_copies, tangent_vectors[k])
            
    gradient = (1/float(numC))*sum(tangent_vectors)
    
    return CBase, pBase, gradient



def frechet_loss(center,CList,pCenter,pList):
    dists = []
    for k in range(len(CList)):
        opt_coup, log = gromov_wasserstein_asym(center,CList[k],pCenter,pList[k])
        dists.append(log['gw_dist'])
    loss = 1.0/len(CList)*sum([d**2 for d in dists])
    return loss



def network_karcher_mean(CList,pList,maxIter = 50):
    CBase = CList[0]
    pBase = pList[0]
    Delta = 1000
    counter = 0
    Deltas = []
    loss_init = frechet_loss(CBase,CList,pBase,pList)
    Frechet_Loss = [loss_init]
    print(counter, loss_init)
    while Delta > 1e-15 and counter < maxIter:
        C0, pBase, gradient = frechet_gradient(CList,pList,CBase,pBase)
        CBase = exp_map(C0,gradient)
        Delta = np.linalg.norm(gradient)
        counter = counter + 1
        Deltas.append(Delta)
        #print(counter, Delta)
        curr_loss = frechet_loss(CBase, CList, pBase, pList)
        Frechet_Loss.append(curr_loss)
        print(counter, curr_loss)

    return CBase, Deltas, Frechet_Loss, pBase      


def network_karcher_mean_compressed(CList,pList,budget,maxIter = 50):
    CBase = CList[0]
    pBase = pList[0]
    Delta = 1000
    counter = 0
    Deltas = []
    loss_init = frechet_loss(CBase,CList,pBase,pList)
    Frechet_Loss = [loss_init]
    print(counter, loss_init)
    while Delta > 1e-10 and counter < maxIter:
        C0, pBase, gradient = frechet_gradient_compressed(CList,pList,CBase,pBase,budget)

        CBase = exp_map(C0,gradient)
        Delta = np.linalg.norm(gradient)
        counter = counter + 1
        Deltas.append(Delta)
        #print(counter, Delta)
        curr_loss = frechet_loss(CBase, CList, pBase, pList)
        Frechet_Loss.append(curr_loss)
        print(counter, curr_loss)

    return CBase, Deltas, Frechet_Loss 

# Karcher mean with backtracking line search optimization
#[note] added this to help with convergence
def network_karcher_mean_armijo(CList,pList,maxIter = 50):
    # performing backtracking line search
    CBase = CList[0]
    pBase = pList[0]
    Delta = 1000
    counter = 0
    loss_init = frechet_loss(CBase,CList,pBase,pList)
    Frechet_Loss = [loss_init]
    #Delta = loss_init
    print('Iter','Frechet_Loss')
    print(counter, loss_init)
    #while counter < maxIter: # This is a pretty arbitrary stopping condition
    while Delta > 1e-10*loss_init and counter < maxIter: # This is a pretty arbitrary stopping condition
        C0, pBase, gradient = frechet_gradient(CList,pList,CBase,pBase)
        # perform backtracking line search with following parameters:
        # currently at C0, descent direction = gradient, 
        # actual gradient = -gradient, loss function = frechet_loss
        
        f_val = frechet_loss(C0,CList,pBase,pList)
        
        def frechet_loss_at(Y):
            return frechet_loss(Y,CList,pBase,pList)
        
        # Optionally normalize descent direction. Our gradient is minus the
        # true gradient
        pk = np.divide(gradient,np.linalg.norm(gradient))
        # Can replace gradient below by pk
        alpha, fc, f_val = line_search_armijo(frechet_loss_at, C0, gradient, -gradient, f_val)  
        CBase = exp_map(C0, alpha*gradient) #step size satisfying Armijo condition    
        Frechet_Loss.append(f_val)
        
        Delta = abs(Frechet_Loss[-1]-Frechet_Loss[-2])
        counter = counter + 1
        print(counter, f_val)
    print('Initial Loss: '+str(Frechet_Loss[0]))
    print('Loss for Minimizer: '+str(Frechet_Loss[-1]))
    return CBase, pBase, Frechet_Loss

# Network Karcher mean with a schedule for changing
# the step size

def network_karcher_mean_armijo_sched(CBase, pBase, CList,pList,exploreIter = 20, maxIter = 50):
    # Take exploreIter full gradient steps at first,
    # then do backtracking line search up to maxIter
    Delta = 1000
    counter = 0
    loss_init = frechet_loss(CBase,CList,pBase,pList)
    Frechet_Loss = [loss_init]
    #Delta = loss_init
    print('Iter','Frechet_Loss')
    print(counter, loss_init)

    tmpC = [CBase]
    tmpP = [pBase]
    # Do full gradient steps
    while counter < exploreIter:
        CBase, pBase, gradient = frechet_gradient(CList,pList,CBase,pBase)
        print('current size ', CBase.shape)
        CBase = exp_map(CBase,gradient)
        tmpC.append(CBase)
        tmpP.append(pBase)
        f_val = frechet_loss(CBase, CList, pBase, pList)
        Frechet_Loss.append(f_val)
        counter = counter + 1
        print(counter, f_val, ' full steps')

    # Set CBase to be the best seed point
    idx = Frechet_Loss.index(min(Frechet_Loss))
    CBase = tmpC[idx]
    pBase = tmpP[idx]
    print('setting seed at ', idx)
    f_val = frechet_loss(CBase, CList, pBase, pList)
    print('seed loss is', f_val)

    #while counter < maxIter: # This is a pretty arbitrary stopping condition
    while Delta > 1e-10*loss_init and counter < maxIter: # This is a pretty arbitrary stopping condition
        C0, pBase, gradient = frechet_gradient(CList,pList,CBase,pBase)
        # perform backtracking line search with following parameters:
        # currently at CBase, descent direction = gradient, 
        # actual gradient = -gradient, loss function = frechet_loss
        def frechet_loss_at(Y):
            return frechet_loss(Y,CList,pBase,pList)
        
        # Optionally normalize descent direction. Our gradient is minus the
        # true gradient
        # gradSz = np.linalg.norm(gradient)
        #if gradSz > 0:
         #   pk = np.divide(gradient,np.linalg.norm(gradient)) 
        try:
            alpha, fc, f_val = line_search_armijo(frechet_loss_at, C0, gradient, -gradient, f_val)
        except:
            print('exception at iteration ', counter)
            f_val = frechet_loss_at(C0)
            print('error at current center is ', f_val)
            return C0, pBase, Frechet_Loss

        CBase = exp_map(CBase, alpha*gradient) #step size satisfying Armijo condition    
        Frechet_Loss.append(f_val)
        
        Delta = abs(Frechet_Loss[-1]-Frechet_Loss[-2])
        counter = counter + 1
        print(counter, f_val, 'step scale',alpha)
    print('Initial Loss: '+str(Frechet_Loss[0]))
    print('Loss for Minimizer: '+str(Frechet_Loss[-1]))
    return CBase, pBase, Frechet_Loss


# Network Karcher mean with a schedule for changing
# the step size and also a compression scheme between iterates

def network_karcher_mean_armijo_sched_compress(CBase, pBase, CList,pList,budget,exploreIter = 20, maxIter = 50):
    # Take exploreIter full gradient steps at first,
    # then do backtracking line search up to maxIter
    Delta = 1000
    counter = 0
    loss_init = frechet_loss(CBase,CList,pBase,pList)
    Frechet_Loss = [loss_init]
    #Delta = loss_init
    print('Iter','Frechet_Loss')
    print(counter, loss_init)

    tmpC = [CBase]
    tmpP = [pBase]
    # Do full gradient steps
    while counter < exploreIter:
        CBase, pBase, gradient = frechet_gradient_compressed(CList,pList,CBase,pBase,budget)
        print('current size ', CBase.shape)
        CBase = exp_map(CBase,gradient)
        tmpC.append(CBase)
        tmpP.append(pBase)
        f_val = frechet_loss(CBase, CList, pBase, pList)
        Frechet_Loss.append(f_val)
        counter = counter + 1
        print(counter, f_val, ' full gradient step')

    # Set CBase to be the best seed point
    idx = Frechet_Loss.index(min(Frechet_Loss))
    CBase = tmpC[idx]
    pBase = tmpP[idx]
    print('setting seed at ', idx)
    f_val = frechet_loss(CBase, CList, pBase, pList)
    print('seed loss is', f_val)

    #while counter < maxIter: # This is a pretty arbitrary stopping condition
    while Delta > 1e-10*loss_init and counter < maxIter: # This is a pretty arbitrary stopping condition
        C0, pBase, gradient = frechet_gradient_compressed(CList,pList,CBase,pBase,budget)
        # perform backtracking line search with following parameters:
        # currently at CBase, descent direction = gradient, 
        # actual gradient = -gradient, loss function = frechet_loss
        def frechet_loss_at(Y):
            return frechet_loss(Y,CList,pBase,pList)
        
        # Optionally normalize descent direction. Our gradient is minus the
        # true gradient
        # gradSz = np.linalg.norm(gradient)
        #if gradSz > 0:
         #   pk = np.divide(gradient,np.linalg.norm(gradient)) 
        try:
            alpha, fc, f_val = line_search_armijo(frechet_loss_at, C0, gradient, -gradient, f_val)
        except:
            print('exception at iteration ', counter)
            f_val = frechet_loss_at(C0)
            print('error at current center is ', f_val)
            return C0, pBase, Frechet_Loss

        CBase = exp_map(CBase, alpha*gradient) #step size satisfying Armijo condition    
        Frechet_Loss.append(f_val)
        
        Delta = abs(Frechet_Loss[-1]-Frechet_Loss[-2])
        counter = counter + 1
        print(counter, f_val, 'step scale',alpha)
    print('Initial Loss: '+str(Frechet_Loss[0]))
    print('Loss for Minimizer: '+str(Frechet_Loss[-1]))
    return CBase, pBase, Frechet_Loss

""" Network compression """

def network_compress(A,pA,k):
    # Input: 
    # A         -- square matrix (network) 
    # pA        -- probability measure on A
    # k         -- number of nodes to compress to
    # Output:
    # S         -- k x k compressed form of A

    # Procedure:
    # - for each node, create a vector of incoming 
    # and outgoing weights
    # - apply k-means clustering to the "space of weights"
    # 
    n = len(A)
    #concatenate
    V = np.concatenate((A, A.T), axis = 1)
    # perform kmeans
    kmeans = KMeans(n_clusters=k).fit(V)
    labels = kmeans.labels_
    means = kmeans.cluster_centers_
    # means is a k x 2n matrix. Columns n:2n are 
    # a transpose of 0:n, so it suffices to work with columns 0:n
    means = means[:,0:n]

    # get unique labels in the order in which they appear
    unique_l, indices = np.unique(labels, return_index=True)
    unsrt_idx = [labels[index] for index in sorted(indices)]

    # initialize compressed network
    S = np.zeros((k,k))
    pS = np.zeros(k)
    for ii in range(0,k):
        idx_i = labels[:] == unsrt_idx[ii]
        pi = pA[idx_i]
        pS[ii] = np.sum(pi)
        for jj in range(0,k):
            # S_{i,j} = (\sum_j means(i,j))/# of j labels
            idx_j = labels[:] == unsrt_idx[jj]
            tmp = A[idx_i,:]
            tmp = tmp[:,idx_j]
            S[ii,jj] = np.mean(tmp)

    return S, pS

def network_compress_list(CList,pList,k):
    # compress all networks in list
    resC = []
    resp = []
    for ii in range(0,len(CList)):
        C, p = network_compress(CList[ii],pList[ii],k)
        resC.append(C)
        resp.append(p)
    
    return resC, resp


def network_log_compress(A,B,pA,pB,coup,k):
    # Helper function for compressing a log map.
    # Takes two appropriately expanded networks, 
    # a permutation matrix matching the two networks,
    # and compresses all three objects. 
    # Date: September 17, 2019
    # Input: 
    # A         -- square matrix (network)
    # B         -- square matrix, same size as A 
    # pA        -- probability measure on A
    # pB        -- probability measure on B
    # coup      -- permutation matrix
    # k         -- number of nodes to compress to
    # Output:
    # sA,sB,spA,spB-- k x k compressed forms

    # Procedure:
    # - concatenate A, A.T, B, B.T
    # - apply k-means clustering to the rows
    # - collapse indices according to clustering
    # 
    #n = len(A)
    #concatenate
    V = np.concatenate((A, A.T, B, B.T), axis = 1)
    # perform kmeans
    kmeans = KMeans(n_clusters=k).fit(V)
    labels = kmeans.labels_
    #means = kmeans.cluster_centers_
    # means is a k x 2n matrix. Columns n:2n are 
    # a transpose of 0:n, so it suffices to work with columns 0:n
    #means = means[:,0:n]

    # get unique labels in the order in which they appear
    unique_l, indices = np.unique(labels, return_index=True)
    unsrt_idx = [labels[index] for index in sorted(indices)]

    # initialize compressed network
    sA = np.zeros((k,k))
    sB = np.zeros((k,k))
    spA= np.zeros(k)
    spB= np.zeros(k)
    
    # Question: can for-loops below be removed by just using
    # means output from the k-means step?
    for ii in range(0,k):
        idx_i = labels[:] == unsrt_idx[ii]
        piA = pA[idx_i]
        piB = pB[idx_i]
        spA[ii] = np.sum(piA)
        spB[ii] = np.sum(piB)
        for jj in range(0,k):
            # S_{i,j} = (\sum_j means(i,j))/# of j labels
            idx_j = labels[:] == unsrt_idx[jj]
            # first do A
            tmpA = A[idx_i,:]
            tmpA = tmpA[:,idx_j]
            sA[ii,jj] = np.mean(tmpA)
            # now for B
            tmpB = B[idx_i,:]
            tmpB = tmpB[:,idx_j]
            sB[ii,jj] = np.mean(tmpB)


    return sA, sB, spA, spB, unsrt_idx


""" Network pdist """
# Given a database of networks with probability
# vectors, create a pairwise GW-distance matrix

def network_pdist(CList, pList):
    n = len(CList)
    pdist = np.zeros((n,n))
    for ii in range(0,n):
        for jj in range(ii+1,n):
            _, log = gromov_wasserstein_asym(CList[ii],CList[jj],
                                                pList[ii],pList[jj])
            pdist[ii,jj] = log['gw_dist']
            print('done for ii',ii,'and jj', jj)

    pdist_full = pdist + pdist.T
    # Optionally save
    # sio.savemat('pdist.mat',{'pdist':pdist_full})
    return pdist_full
    

""" Helper functions """

def print_shapes(CList):
    # Print shapes of arrays in list
    for ii in range(0,len(CList)):
        print('entry',ii,'has shape', CList[ii].shape)


