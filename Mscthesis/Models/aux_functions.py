
"""
Auxiliary functions
===================
Functions to perform general computations of statistics or transformations
useful for compute the models.

"""

import numpy as np
from Mscthesis.Preprocess.comp_complementary_data import \
    average_position_by_cp, counting_type_by_cp


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


def compute_aggregate_counts(df, agg_var, loc_vars, type_vars, reindices):
    ## Compute the tables
    locs = average_position_by_cp(df, agg_var, loc_vars)
    agg_values = list(locs.index)
    locs = locs.as_matrix()

    tables = {}
    axis = {}
    for col in type_vars:
        n_vals = df[col].unique().shape[0]
        aux = np.zeros((len(agg_values), n_vals, reindices.shape[1]))
        for i in range(reindices.shape[1]):
            # The type values
            aux_df = df.loc[:, [agg_var]+type_vars]
            aux2 = aux_df[type_vars].reindex(reindices[:, i]).as_matrix()
            aux_df[type_vars] = aux2
            table, cols = counting_type_by_cp(aux_df, agg_var, type_vars)
            aux[:, :, i] = table.as_matrix()

        tables[col] = aux
        axis[col] = {'rows': agg_values, 'columns': cols}

    return tables, axis, locs
