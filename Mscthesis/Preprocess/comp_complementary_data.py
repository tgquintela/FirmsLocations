
"""
This module contains functions which helps in the computation of extra and
complementary data needed and computed from the known data.
"""

# Counting process of each type


import numpy as np


def average_position_by_cp(df, cp_var, loc_vars):
    "Compute the pivot table to assign to cp a geographic coordinates."
    table = df.pivot_table(values=loc_vars, rows=cp_var, aggfunc=np.mean)
    return table


def counting_type_by_cp(df, cp_var, type_var):
    "Compute the counting of types by "
    table = df[[cp_var, type_var]].pivot_table(rows=cp_var, cols=type_var,
                                               aggfunc='count')
    table = table.fillna(value=0)
    cols = table.columns.get_level_values(1).unique()
    m = len(cols)
    table = table.loc[:, table.columns[:m]]
    table.columns = cols
    return table


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
