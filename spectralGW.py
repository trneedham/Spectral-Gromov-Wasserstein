"""
Spectral Gromov-Wasserstein graph averaging.

Includes functions for matching graphs by their heat kernels as well as methods
for visualizing the matchings as graph interpolations. Also includes functions
for sampling the probability simplex via a Markov chain algorithm.
"""


import numpy as np
import ot
import networkx as nx
from scipy import linalg

# Produce a probability distribution on nodes of the network.

def node_distribution(G,a,b):
    # Input: Networkx graph object, parameters a and b
    # Output: probability vector representing distrubtion on the nodes of G.

    n = np.array([G.degree[j] for j in range(len(G.node))])
    q = (n+a)**b
    p = q/sum(q)

    return p

def extract_HK_data_normalized_Laplacian(G):

    # Node positions
    nodePos = nx.kamada_kawai_layout(G)
    nodePos_matrix = np.array(list(nodePos.values()))

    # Probability vector
    p = ot.unif(len(G.nodes))

    # Laplacian Eigen-data
    L = nx.normalized_laplacian_matrix(G).toarray()
    lam, phi = np.linalg.eigh(L)

    return nodePos_matrix, p, lam, phi

# Find eigenvalues and vectors for graph Laplacian
def laplacian_eig(G):
    # Input: Networkx graph
    # Output: eigenvalues and eigenvectors of graph laplacian
    L = nx.laplacian_matrix(G).toarray()
    lam, phi = np.linalg.eigh(L)

    return lam, phi

# Create heat kernel matrix from precomputed eigenvalues/tangent_vectors
def heat_kernel(lam,phi,t):
    # Input: eigenvalues and eigenvectors for normalized Laplacian, time parameter t
    # Output: heat kernel matrix

    u = np.matmul(phi,np.matmul(np.diag(np.exp(-t*lam)),phi.T))

    return u

def directed_heat_kernel(G,t):
    # Input: DiGraph G and time parameter t
    # Output: heat kernel matrix
    # Automatically computes directed laplacian matrix and then exponentiates

    L = np.asarray(nx.directed_laplacian_matrix(G))
    lam, phi = np.linalg.eigh(L)
    return heat_kernel(lam,phi,t)

def undirected_heat_kernel(G,t):
    # Input: Graph G and time parameter t
    # Output: heat kernel matrix
    # Automatically computes directed laplacian matrix and then exponentiates

    L = nx.laplacian_matrix(G).toarray()
    lam, phi = np.linalg.eigh(L)
    return heat_kernel(lam,phi,t)

def undirected_normalized_heat_kernel(G,t):
    # Input: Graph G and time parameter t
    # Output: heat kernel matrix
    # Automatically computes directed laplacian matrix and then exponentiates

    L = nx.normalized_laplacian_matrix(G).toarray()
    lam, phi = np.linalg.eigh(L)
    return heat_kernel(lam,phi,t)

"""
Markov chain polytope sampling
"""

def gw_equality_constraints(p,q):
    # Inputs: probability row vectors
    # Output: matrices A and b defining equality constraints

    m = len(p)
    n = len(q)

    A_p_type = np.zeros([m,m*n])
    b_p_type = p.reshape(m,1)

    for i in range(m):
        row = i*n*[0] + n*[1] + (n*m-(i*n+n))*[0]
        row = np.array(row)
        A_p_type[i,:] = row

    A_q_type = np.zeros([n,m*n])
    b_q_type = q.reshape(n,1)

    for j in range(n):
        row_pattern = np.zeros([1,n])
        row_pattern[0,j] = 1
        row = np.tile(row_pattern,m)
        A_q_type[j,:] = row

    A = np.concatenate((A_p_type,A_q_type), axis = 0)
    b = np.concatenate((b_p_type,b_q_type), axis = 0)

    return A, b

def project_mu(mu,A,b,P,product_mu):

    # Input: coupling-shaped matrix mu; equality constraints A,b from gw_equality_constraints
    #        function; product coupling of some probability measures p and q.
    #        P is a projection matrix onto row space of A.
    # Output: Orthogonal projection of mu onto the affine subspace determined by A,b.

    m = product_mu.shape[0]
    n = product_mu.shape[1]

    # Create the vector to actually project and reshape
    vec_to_project = mu - product_mu
    vec_to_project = vec_to_project.reshape(m*n,)

    # Project it
    vec_to_project = vec_to_project - np.matmul(P,vec_to_project)

    projected_mu = product_mu.reshape(m*n,) + vec_to_project

    projected_mu = projected_mu.reshape(m,n)

    return projected_mu

def markov_hit_and_run_step(A,b,P,p,q,mu_initial='Product'):
    # Input: equality constraints A,b from gw_equality_constraints; pair of
    #       probability vectors p, q; initialization
    #        P is a projection matrix onto row space of A.
    # Output: new coupling measure after a hit-and-run step.

    m = p.shape[0]
    n = q.shape[0]

    product_mu = p[:,None]*q[None,:]

    if mu_initial == 'Product':
        mu_initial = product_mu

    mu_initial = project_mu(mu_initial,A,b,P,product_mu)
    # Project to the affine subspace
    # We assume mu_initial already lives there, but this will help with accumulation of numerical error

    mu_initial = mu_initial.reshape(m*n,)

    # Choose a random direction
    direction = np.random.normal(size = m*n)

    # Project to subspace of admissible directions

    direction = direction - np.matmul(P,direction)

    # Renormalize

    direction = direction/np.linalg.norm(direction)

    # Determine how far to move while staying in the polytope - These are inequality bounds,
    # so we just need the entries to stay positive

    pos = direction > 1e-6
    neg = direction < -1e-6

    direction_pos = direction[pos]
    direction_neg = direction[neg]
    mu_initial_pos = mu_initial[pos]
    mu_initial_neg = mu_initial[neg]

    lower = np.max(-mu_initial_pos/direction_pos)
    upper = np.min(-mu_initial_neg/direction_neg)

    # Choose a random distance to move
    r = (upper - lower)*np.random.uniform() + lower

    mu_new = mu_initial + r*direction
    mu_new = mu_new.reshape(m,n)

    return mu_new

def coupling_ensemble(A,b,p,q,num_samples,num_skips,mu_initial = 'Product'):
    # Inputs: equality constraints A,b; probability vectors p,q; number of steps
    #         to take in the Markov chain; initialization
    # Output: Ensemble of couplings from the probability simplex.

    if mu_initial == 'Product':
        mu_initial = p[:,None]*q[None,:]

    # Find orthonormal basis for row space of A
    Q = linalg.orth(A.T)
    # Create projector onto the row space of A
    P = np.matmul(Q,Q.T)

    num_steps = num_samples*num_skips

    Markov_steps = []

    for j in range(num_steps):
        mu_new = markov_hit_and_run_step(A,b,P,p,q,mu_initial)
        mu_initial = mu_new
        if j%num_skips == 0:
            Markov_steps.append(mu_new)

    return Markov_steps
