
"""
Auxiliary functions
===================
Functions to perform general computations of statistics or transformations
useful for compute the models.

"""

import numpy as np


###############################################################################
############################# Auxiliary functions #############################
###############################################################################
def init_measure_compute(df, type_vars, loc_vars, radius, permuts):
    """Auxiliary function to prepare the initialization and preprocess of the
    required input variables.
    """

    # Global stats
    N_t = df.shape[0]
    N_x, type_vals = compute_global_counts(df, type_vars)
    n_vals = [len(type_vals[e]) for e in type_vals.keys()]

    # Replace to save memory
    repl = generate_replace(type_vals)

    type_arr = np.array(df[type_vars].replace(repl)).astype(int)
    type_arr = type_arr if len(type_arr) == 2 else type_arr.reshape((N_t, 1))
    df[type_vars] = type_arr

    # Preparing reindices
    reindices = reindices_creation(df, permuts)
    n_calc = reindices.shape[1]

    # Computation of the locations
    # indices
    indices = np.array(df.index)

    output = (df, type_vals, n_vals, N_t, N_x, reindices,
              n_calc, indices)
    return output


def reindices_creation(df, permuts):
    "Function to create reindices for permuting elements of the array."
    N_t = df.shape[0]
    reindex = np.array(df.index)
    reindex = reindex.reshape((N_t, 1))
    if permuts is not None:
        if type(permuts) == int:
            permuts = [np.random.permutation(N_t) for i in range(permuts)]
            permuts = np.vstack(permuts).T
            bool_ch = len(permuts.shape) == 1
            permuts = permuts.reshape((N_t, 1)) if bool_ch else permuts
        n_per = permuts.shape[1]
        permuts = [reindex[permuts[:, i]] for i in range(n_per)]
        permuts = np.hstack(permuts)
    reindex = [reindex] if permuts is None else [reindex, permuts]
    reindices = np.hstack(reindex)
    return reindices


def compute_global_counts(df, type_vars):
    "Compute counts of each values."

    N_x = {}
    type_vals = {}
    for var in type_vars:
        t_vals = sorted(list(df[var].unique()))
        aux_nx = [np.sum(df[var] == type_v) for type_v in t_vals]
        aux_nx = np.array(aux_nx)
        N_x[var], type_vals[var] = aux_nx, t_vals
    return N_x, type_vals


def mapping_typearr(type_arr, type_vars):
    maps = {}
    for i in range(type_arr.shape[1]):
        vals = np.unique(type_arr[:, i])
        maps[type_vars[i]] = dict(zip(vals, range(vals.shape[0])))
    return maps


def generate_replace(type_vals):
    "Generate the replace for use indices and save memory."
    repl = {}
    for v in type_vals.keys():
        repl[v] = dict(zip(type_vals[v], range(len(type_vals[v]))))
    return repl
