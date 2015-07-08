
"""
Auxiliary functions to perform the main functions showed in this module.
"""

import numpy as np


def reverse_mapping(mapdict):
    "Function to reverse a mapping dict."
    r_mapdict = {}
    for e in mapdict.keys():
        mapdict[mapdict[e]] = e
    return r_mapdict

        df, type_vals, n_vals, N_t, N_x = aux[:5]
        reindices, n_calc, indices = aux[5:]


def numpy_applymap(mapdict):
    "Function to apply a mapdict to an numpy array."
    return
