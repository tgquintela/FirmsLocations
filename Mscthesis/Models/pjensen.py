
"""
"""

import numpy as np
#import networkx as nx
from scipy.spatial import KDTree

import time


def built_network(df, loc_vars, type_var, radius):
    """Function for building the network from the locations."""

    type_vals = list(df[type_var].unique())
    n_vals = len(type_vals)

    N_t = df.shape[0]
    N_x = np.array([np.sum(df[type_var] == type_v) for type_v in type_vals])

    retrieve_t, compute_t = 0, 0

    net = np.zeros((n_vals, n_vals))
    for i in range(n_vals):
        ##########
        t0 = time.time()
        ##########
        elements_i = np.where(df[type_var] == type_vals[i])[0]
        N_i = elements_i.shape[0]
        counts_i = compute_neigh_count(df, i, type_vals, loc_vars,
                                       type_var, radius)
        ##########
        retrieve_t += time.time()-t0
        t1 = time.time()
        ##########
        aux = compute_unorm_corrs(counts_i, i)
        ## Normalization
        cte2 = np.log10(np.divide(float(N_t-1), (N_i*(N_i-1))))
        cte2 = 0 if N_x[i] == 1 else cte2
        cte = np.log10(np.divide(float(N_t-N_i), (N_i*N_x)))
        cte[np.where(cte == -np.inf)] = 0
        cte[i] = cte2
        #net[i, :] = np.multiply(cte, aux)
        aux = cte + np.log10(aux)
        aux[np.where(aux == -np.inf)] = 0
        net[i, :] = aux

        ##########
        print "Finished %s in %f seconds." %(type_vals[i], time.time()-t0)
        compute_t += time.time()-t1
        ##########

    return net, type_vals, N_x, retrieve_t, compute_t


def compute_unorm_corrs(counts_i, i):
    """"""

    Nts = np.sum(counts_i, 1)
    unnorm_corrs = np.zeros((counts_i.shape[1]))

    for j in range(counts_i.shape[1]):
        if i == j:
            aux = np.divide(counts_i[:, i].astype(float)-1, Nts)
            unnorm_corrs[i] = np.sum(aux)

        else:
            aux = np.divide(counts_i[:, j].astype(float),
                            Nts-(counts_i[:, i]-1))
            unnorm_corrs[j] = np.sum(aux)

    return unnorm_corrs


def compute_neigh_count(df, j, type_vals, loc_vars, type_var, radius):
    """
    radius: expressed in kms.
    """

    kdtree = KDTree(df[loc_vars].as_matrix(), leafsize=10000)
    elements_j = np.where(df[type_var] == type_vals[j])[0]
    N_j = elements_j.shape[0]
    radius = radius/6371.009

    counts = np.zeros((N_j, len(type_vals)))
    for i in range(N_j):
        k = elements_j[i]
        neighs = kdtree.query_ball_point(df[loc_vars].as_matrix()[k], radius)
        vals = df[type_var][neighs]
        counts[i, :] = np.array([np.sum(vals == val) for val in type_vals])

    counts = counts.astype(int)
    return counts
