
"""
This module contains functions which helps in the computation of extra and
complementary data needed and computed from the known data.

TODO:
----
- Other functions not only count.

"""

# Counting process of each type


import numpy as np
import pandas as pd

from Mscthesis.Retrieve.density_assignation import general_density_assignation


###############################################################################
############################ Main functions counts ############################
###############################################################################
def compute_aggregate_counts(df, agg_var, loc_vars, type_vars, reindices):
    ## Compute the tables
    locs = average_position_by_aggvar(df, agg_var, loc_vars)
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
            table, cols = counting_type_by_aggvar(aux_df, agg_var, type_vars)
            aux[:, :, i] = table.as_matrix()

        tables[col] = aux
        axis[col] = {'rows': agg_values, 'columns': cols}

    return tables, axis, locs


def compute_aggregate_counts_grid(locs_grid, feat_arr, reindices):
    "Define the aggretriever information."
    # locs_grid, reindices, feat_arr
    from itertools import product
    u1, u2 = np.unique(locs_grid[:, 0]), np.unique(locs_grid[:, 1])
    N_calc = reindices.shape[1]
    n_vals = []
    for i in range(feat_arr.shape[1]):
        n_vals.append(np.unique(feat_arr[:, i]).shape[0])
    agglocs, aggfeatures = [], []
    for p in product(u1, u2):
        ## Function to check if it is all equal
        logi = locs_grid == p
        logi = np.logical_and(logi[:, 0], logi[:, 1])
        if logi.sum() > 0:
            ## Computation of counts for each permutation in a given cell p
            auxM = []
            for j in range(N_calc):
                idxs = reindices[:, j]
                aux = computation_aggregate_collapse_i(feat_arr[idxs[logi], :],
                                                       n_vals)
                aux = aux.reshape(aux.shape[0], 1)
                auxM.append(aux)
            ## Prepare outputs
            agglocs.append(p)
            auxM = np.concatenate(auxM, axis=1)
            auxM = auxM.reshape(auxM.shape[0], auxM.shape[1], 1)
            aggfeatures.append(auxM)
    ## Format output
    aggfeatures = np.concatenate(aggfeatures, axis=2)
    aggfeatures = np.swapaxes(np.concatenate(aggfeatures, axis=2), 2, 1)
    agglocs = np.array(agglocs)
    return agglocs, aggfeatures


###############################################################################
############################ Auxiliar counts by var ###########################
###############################################################################
def aggregate_by_var(empresas, agg_var, loc_vars, type_vars=None):
    """Function to aggregate variables by the selected variable considering a
    properly structured data.
    """
    ## Aggregation
    positions = average_position_by_aggvar(empresas, agg_var, loc_vars)
    if type_vars is not None:
        types = aggregate_by_typevar(empresas, agg_var, type_vars)
        df, cols = pd.concat([positions, types], axis=1)
        cols = {'types': cols}
        cols['positions'] = list(positions.columns)
    else:
        df = positions
        cols = {'positions': list(positions.columns)}

    return df, cols


def aggregate_by_typevar(empresas, agg_var, type_vars):
    "Function to aggregate only by type_var."
    type_vars = [type_vars] if type(type_vars) != list else type_vars
    df = counting_type_by_aggvar(empresas, agg_var, type_vars)
    cols = list(df.columns)
    return df, cols


def average_position_by_aggvar(df, aggvar, loc_vars):
    "Compute the pivot table to assign to cp a geographic coordinates."
    table = df.pivot_table(values=loc_vars, rows=aggvar, aggfunc=np.mean)
    return table


def counting_type_by_aggvar(df, aggvar, type_vars):
    "Compute the counting of types by "
    table = df[[aggvar] + type_vars].pivot_table(rows=aggvar, cols=type_vars,
                                                 aggfunc='count')
    table = table.fillna(value=0)
    cols = table.columns.get_level_values(1).unique()
    m = len(cols)
    table = table.loc[:, table.columns[:m]]
    table.columns = cols
    return table, cols


def std_type_by_aggvar(df, aggvar, loc_vars):
    "Compute the counting of types by "
    table = df.pivot_table(rows=aggvar, values=loc_vars,
                           aggfunc=np.std)
    table = table.fillna(value=0)
    table.columns = ['STD-X', 'STD-Y']
    return table


def mean_type_by_aggvar(df, aggvar, loc_vars):
    "Compute the counting of types by "
    table = df.pivot_table(rows=aggvar, values=loc_vars,
                           aggfunc=np.mean)
    table.columns = ['MEAN-X', 'MEAN-Y']
    table = table.fillna(value=0)
    return table


###############################################################################
############################# Auxiliar grid counts ############################
###############################################################################
def computation_aggregate_collapse_i(type_arr, n_vals):
    "Counting the types of each one."
    values = np.unique(type_arr[:, 0])
    counts_i = np.zeros(n_vals[0])
    for j in range(values.shape[0]):
        counts_i[values[j]] = (type_arr == values[j]).sum()
    return counts_i


###############################################################################
############################# Auxiliar grid counts ############################
###############################################################################
def compute_population_data(locs, pop, popvars, parameters):
    "Function to compute the correspondant population data to each point."

    ## 0. Computation of initial variables
    locs = np.array(locs)
    locs_pop = np.array(pop[popvars['loc_vars']])
    pop_pop = np.array(pop[popvars['pop_vars']])

    ## 1. Computation of assignation to point
    pop_assignation = general_density_assignation(locs, parameters, pop_pop,
                                                  locs_pop)

    return pop_assignation
