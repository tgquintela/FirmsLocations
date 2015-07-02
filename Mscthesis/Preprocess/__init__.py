

"""
Module oriented to group all classes and functions which function is to
preprocess the data and prepare new data and structuctures to be use in the
processes we want to perform.

TODO:
-----
- Accept other functions or descriptors not only count...

"""

import numpy as np
import pandas as pd
from itertools import product
#from Mscthesis.IO.io_aggfile import read_agg
from comp_complementary_data import compute_aggregate_counts


class Aggregator():
    "Aggregate or read aggregate information."

    def __init__(self, filepath=None, typevars=None, vals=None):
        if filepath is None:
            typevars = format_typevars(typevars)
            self.vals = vals
            self.typevars = typevars
            self.bool_read_agg = False
        else:
            self.bool_read_agg = True
            typevars = format_typevars(typevars)
            self.typevars = typevars

    def retrieve_aggregation(self, df=None, reindices=None):
        "Main function for retrieving aggregation."
        if self.bool_read_agg:
            # TODO: Function to read file
            filepath, typevars = self.filepath, self.typevars
            agglocs, aggfeatures = read_aggregation(filepath, typevars)
        else:
            ## Correct inputs
            locs = df[self.typevars['loc_vars']].as_matrix()
            feat_arr = df[self.typevars['feat_vars']].as_matrix()
            if self.typevars['agg_var'] is None:
                agg_arr = None
            else:
                agg_arr = df[self.typevars['agg_var']].as_matrix()
            if reindices is None:
                N_t = locs.shape[0]
                reindices = np.array(range(N_t)).reshape((N_t, 1))
            if len(feat_arr.shape) == 1:
                feat_arr = feat_arr.reshape(feat_arr.shape[0], 1)
            ## Compute agglocs and aggfeatures
            agglocs, aggfeatures = create_aggregation(locs, agg_arr, feat_arr,
                                                      reindices, self.vals,
                                                      self.typevars)
        ## Format output
        agglocs = np.array(agglocs)
        ndim, N_t = len(agglocs.shape), agglocs.shape[0]
        agglocs = agglocs if ndim > 1 else agglocs.reshape((N_t, 1))
        return agglocs, aggfeatures


def create_aggregation(locs, agg_arr, feat_arr, reindices, vals=None,
                       typevars=None):
    "Create aggregation."
    if agg_arr is None:
        agg_arr = map_multivars2key(locs, vals=vals)
        typevars['agg_var'] = 'agglocs'

    typevars = format_typevars(typevars, locs.shape[1], feat_arr.shape[1])

    agg_var = typevars['agg_var']
    feat_vars, loc_vars = typevars['feat_vars'], typevars['loc_vars']

    df1 = pd.DataFrame(locs, columns=loc_vars)
    df2 = pd.DataFrame(agg_arr, columns=[agg_var])
    df3 = pd.DataFrame(feat_arr, columns=feat_vars)
    df = pd.concat([df1, df2, df3], axis=1)
    agg_desc, axis, locs = compute_aggregate_counts(df, agg_var, loc_vars,
                                                    feat_vars, reindices)
    return locs, agg_desc


def read_aggregation(filepath, typevars):
    ## TODO
    aggtable = read_agg(filepath)
    aggfeatures = aggtable[typevars['feat_vars']]
    agglocs = aggtable[typevars['loc_vars']]
    return agglocs, aggfeatures


def format_typevars(typevars, locs_dim=None, feats_dim=None):
    "Check typevars."
    if typevars is None:
        agg_var = 'agg'
        loc_vars = [chr(97+i) for i in range(locs_dim)]
        feat_vars = [str(i) for i in range(feats_dim)]
        typevars = {'feat_vars': feat_vars, 'loc_vars': loc_vars,
                    'agg_var': agg_var}
    if 'agg_var' not in typevars.keys():
        typevars['agg_var'] = None
    return typevars


def map_multivars2key(multi, vals=None):
    "Maps a multivariate discrete array to a integer."
    n_dim, N_t = len(multi.shape), multi.shape[0]
    if vals is None:
        vals = []
        for i in range(n_dim):
            aux = np.unique(multi[:, i])
            vals.append(aux)
    combs = product(*vals)
    map_arr = -1*np.ones(N_t)
    i = 0
    for c in combs:
        logi = np.ones(N_t).astype(bool)
        for j in range(n_dim):
            logi = np.logical_and(logi, multi[:, j] == c[j])
        map_arr[logi] = i
        i += 1
    map_arr = map_arr.astype(int)
    return map_arr
