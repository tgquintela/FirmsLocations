
"""
"""

import numpy as np
import networkx as nx
from itertools import product, combinations


def computation_neighbourhood(i, df, loc_vars, type_var, type_vals):
    """"""
    indices = get_from_neighbourhood(point, coordinates)
    neighs = df.irow(indices)
    n_t = neighs.shape[0]

    if type(type_vals) != list:
        ns = np.sum(neighs == type_vals)
    else:
        ns = []
        for i in range(len(type_vals)):
            n_aux = np.sum(neighs == type_vals[i])
            ns.append(n_aux)
    return ns, n_t


def get_from_neighbourhood(point, coordinates):
    pass


def get_from_type(df, type_var, type_val):
    """Function to retrieve the indices of the rows with type equal to
    type_val.
    """
    indices = df[typevar][df[typevar] == type_val].index
    return indices


def self_interaction(df, loc_vars, type_var, type_val):
    """Computation of the self interactions."""
    ## 0. Computation of needed variables
    idxs = get_from_type(df, type_var, type_val)
    N_t = df.shape[0]
    N_a = idxs.shape[0]

    ## 1. Computation of the index
    C = np.log10((N_t-1)/float(N_a*(N_a-1)))
    suma = 0
    for i in idxs:
        n_a, n_t = computation_neighbourhood(i, df, loc_vars, type_var,
                                             type_val)
        suma = suma + n_a/float(n_t)
    a_AA = C*suma
    return a_AA


def x_interaction(df, loc_vars, type_var, type_vals):
    """Computation of the interaction between """
    ## 0. Computation of needed variables
    idxs_A = get_from_type(df, type_var[0])
    idxs_B = get_from_type(df, type_var[1])

    N_t = locations.shape[0]
    N_a = idxs_A.shape[0]
    N_b = idxs_B.shape[0]

    ## 1. Computation of the index
    C = np.log10((N_t-N_a)/float(N_a*N_b))
    suma = 0
    for i in idxs:
        ns, n_t = computation_neighbourhood(i, df, type_vals)
        n_a, n_b = ns
        suma = suma + n_b/float((n_t-n_a))

    a_AB = C*suma
    return a_AB


def built_network(df, loc_vars, type_var):
    """Function for building the network from the locations."""

    type_vals = list(df[type_var].unique())
    n_vals = len(type_vals)
    pairs = product(range(n_vals), range(n_vals))

    net = np.zeros((n_vals, n_vals))
    for p in pairs:
        if p[0] == p[1]:
            type_val = type_vals[p[0]]
            aux = self_interaction(df, loc_vars, type_var, type_val)
        else:
            type_val_pairs = [type_vals[p[0]], type_vals[p[1]]]
            aux = x_interaction(df, loc_vars, type_var, type_val_pairs)
        net[p[0], p[1]] = aux

    return net


def get_zetas(net, clusters):
    ## TODO with networkx
    for c in clusters:
        n_c = len(c)
        pairs = combinations(2, n_c)
        values = [net[p[0], p[1]] for p in pairs]
        pos_values = [val for val in values if val >= 0]
        neg_values = [val for val in values if val >= 0]
    pass
