
"""
Module oriented to retrieve columns related with cnae index.
"""

import numpy as np


def retrieve_cols(df, val, cols):
    lvl = len(val)
    res = df[cols][df['cnae'].apply(lambda x: x[:lvl]) == val]
    return res


def transform_cnae_col(cnae_col, lvl):
    """"""
    def f_trans(x):
        if type(x) == str:
            return int(x[:lvl])
        else:
            return int(str(int(x))[:lvl])

    lvl_n = len(str(cnae_col[1]))
    if lvl >= lvl_n:
        return cnae_col
    else:
        return cnae_col.apply(f_trans)


def transform_sector_col(sector_col):
    def f_trans(x):
        if x == 'servicios':
            return 2
        else:
            return 0
    return sector_col.apply(f_trans)


def unique_codes(categorical):
    types = np.unique(categorical)
    hashdict = dict(range(len(types)), types)
    return hashdict


def unique_codes_double(categorical0, categorical1):
    types0 = np.unique(categorical0)
    types_comb = []
    for t in types0:
        logi = categorical0 == t
        types1_t = np.unique(categorical1[logi])
        for t1 in types1_t:
            types_comb.append((t, t1))
    f = lambda x: types_comb.index(tuple(x))
    return types_comb, f
