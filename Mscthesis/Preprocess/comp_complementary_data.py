
"""
This module contains functions which helps in the computation of extra and
complementary data needed and computed from the known data.
"""

# Counting process of each type


import numpy as np
import pandas as pd

from Mscthesis.Retrieve.density_assignation import general_density_assignation


def aggregate_by_var(empresas, agg_var, loc_vars, type_vars=None):
    """Function to aggregate variables by the selected variable considering a
    properly structured data.
    """
    ## Aggregation
    positions = average_position_by_cp(empresas, agg_var, loc_vars)
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
    df = counting_type_by_cp(empresas, agg_var, type_vars)
    cols = list(df.columns)
    return df, cols


def average_position_by_cp(df, cp_var, loc_vars):
    "Compute the pivot table to assign to cp a geographic coordinates."
    table = df.pivot_table(values=loc_vars, rows=cp_var, aggfunc=np.mean)
    return table


def counting_type_by_cp(df, cp_var, type_vars):
    "Compute the counting of types by "
    table = df[[cp_var] + type_vars].pivot_table(rows=cp_var, cols=type_vars,
                                                 aggfunc='count')
    table = table.fillna(value=0)
    cols = table.columns.get_level_values(1).unique()
    m = len(cols)
    table = table.loc[:, table.columns[:m]]
    table.columns = cols
    return table, cols


def std_type_by_cp(df, cp_var, loc_vars):
    "Compute the counting of types by "
    table = df.pivot_table(rows=cp_var, values=loc_vars,
                           aggfunc=np.std)
    table = table.fillna(value=0)
    table.columns = ['STD-X', 'STD-Y']
    return table


def mean_type_by_cp(df, cp_var, loc_vars):
    "Compute the counting of types by "
    table = df.pivot_table(rows=cp_var, values=loc_vars,
                           aggfunc=np.mean)
    table.columns = ['MEAN-X', 'MEAN-Y']
    table = table.fillna(value=0)
    return table


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
