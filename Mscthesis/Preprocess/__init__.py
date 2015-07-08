

"""
Module oriented to group all classes and functions which function is to
preprocess the data and prepare new data and structuctures to be use in the
processes we want to perform.

TODO:
-----

"""

import numpy as np
import pandas as pd
from itertools import product
#from Mscthesis.IO.io_aggfile import read_aggregation
from comp_complementary_data import compute_aggregate_counts,\
    average_position_by_aggarr
from preprocess_cols import generate_replace, transform_cnae_col
from Mscthesis.Geo_tools.geo_transformations import general_projection


class Firms_Preprocessor():
    "Special class to preprocess firms data."
    projection_values = None
    map_vars = None
    map_indices = None

    ## TODO: map indices

    def __init__(self, typevars):
        self.typevars = typevars
        self.map_vars = ['cnae', 'cp']

    def preprocess(self, empresas, cnae_lvl=2, method_proj='ellipsoidal', radians=False):
        "Function to preprocess firms data."
        ## 0. Set vars
        finantial_vars = [e for e in self.typevars['feat_vars'] if e != 'cnae']
        loc_vars = self.typevars['loc_vars']
        self.projection_values = [loc_vars, method_proj, True, radians]
        # 1. Indices
        self.map_indices = zip(list(empresas.index), range(empresas.shape[0]))
        empresas.index = range(empresas.shape[0])
        # 2. Location transformation
        empresas[loc_vars] = general_projection(empresas, loc_vars,
                                                method=method_proj,
                                                inverse=False,
                                                radians=radians)
        ## 3. Feature array
        # TODO: temporal line (filter of none values of cp)
        #logi = empresas['cp'].isnull()
        #logi = np.logical_not(logi)
        #empresas = empresas.loc[logi, :]
        #################################
        # generate replacement in discrete vars
        t_vals = {'cnae': sorted(list(empresas['cnae'].unique())),
                  'cp': sorted(list(empresas['cp'].unique()))}
        self.map_info = generate_replace(t_vals)
        # cnae variable
        empresas.loc[:, 'cnae'] = transform_cnae_col(empresas['cnae'], cnae_lvl)
        #empresas['cnae'] = empresas['cnae'].replace(self.map_vars['cnae']).astype(int)
        # cp variable
        #empresas.loc[:, 'cp'] = empresas.loc[:, 'cp'].replace(self.map_vars['cp']).astype(int)
        # Map discrete variables
        mpvars = self.map_vars
        empresas.loc[:, mpvars] = empresas.loc[:, mpvars].replace(self.map_info).astype(int)
        # Finantial variables

        return empresas

    def reverse_preprocess(self, empresas):

        ## 1. Inverse transformation of locations
        projection_values = self.projection_values
        empresas[loc_vars] = general_projection(empresas, *projection_values)

        ## 2. Inverse mapping
        ##TODO
        return empresas


class Aggregator():
    "Aggregate or read aggregate information."

    def __init__(self, filepath=None, typevars=None, spatial_disc=None):
        if filepath is None:
            typevars = format_typevars(typevars)
            self.typevars = typevars
            self.discretizor = spatial_disc
            self.bool_read_agg = False
        else:
            self.bool_read_agg = True
            typevars = format_typevars(typevars)
            self.typevars = typevars

    def retrieve_aggregation(self, df=None, reindices=None, funct=None):
        "Main function for retrieving aggregation."
        if self.bool_read_agg:
            # TODO: Function to read file
            filepath, typevars = self.filepath, self.typevars
            agglocs, aggfeatures = read_aggregation(filepath, typevars)
        else:
            ## Correct inputs
            #################
            locs = df[self.typevars['loc_vars']].as_matrix()
            feat_arr = df[self.typevars['feat_vars']].as_matrix()
            if self.typevars['agg_var'] is None:
                ## Check if discretized or to do
                if type(self.discretizor) == np.ndarray:
                    agg_arr = self.discretizor
                    agglocs = average_position_by_aggarr(locs, agg_arr)
                else:
                    agg_arr = self.discretizor.map2id(locs)
                    agglocs = self.discretizor.discretize(locs)
                self.typevars['agg_var'] = 'aggvar'
            else:
                agg_arr = df[self.typevars['agg_var']].as_matrix()
                agglocs = average_position_by_aggarr(locs, agg_arr)
            if reindices is None:
                N_t = locs.shape[0]
                reindices = np.array(range(N_t)).reshape((N_t, 1))
            if len(feat_arr.shape) == 1:
                feat_arr = feat_arr.reshape(feat_arr.shape[0], 1)
            ######################################################
            ## Compute agglocs and aggfeatures
            aggfeatures = create_aggregation(agg_arr, feat_arr, reindices,
                                             self.typevars, funct)
        ## Format output
        agglocs = np.array(agglocs)
        ndim, N_t = len(agglocs.shape), agglocs.shape[0]
        agglocs = agglocs if ndim > 1 else agglocs.reshape((N_t, 1))
        return agglocs, aggfeatures

    def retrieve_aggloc(self, point_i):
        point_i = point_i.reshape(1, 2)
        region = self.spatial_disc.map2id(point_i)
        return region

    def retrieve_agglocs(self, locs):
        regions = self.spatial_disc.map2id(locs)
        return regions


def create_aggregation(agg_arr, feat_arr, reindices, typevars=None,
                       funct=None):
    "Create aggregation."

    ## 0. Formatting inputs
    typevars = format_typevars(typevars, feats_dim=feat_arr.shape[1])
    feat_vars, agg_var = typevars['feat_vars'], typevars['agg_var']
    df1 = pd.DataFrame(agg_arr, columns=[agg_var])
    df2 = pd.DataFrame(feat_arr, columns=feat_vars)
    df = pd.concat([df1, df2], axis=1)

    ## 1. Use specific function or default aggregate counts
    if funct is None:
        agg_desc, _ = compute_aggregate_counts(df, agg_var, feat_vars,
                                               reindices)
    else:
        agg_desc = funct(df, agg_var, feat_vars, reindices)
    return agg_desc


def format_typevars(typevars, locs_dim=None, feats_dim=None):
    "Check typevars."
    if typevars is None:
        typevars = {'agg_var': 'agg'}
        if locs_dim is not None:
            loc_vars = [chr(97+i) for i in range(locs_dim)]
            typevars['loc_vars'] = loc_vars
        if feats_dim is not None:
            feat_vars = [str(i) for i in range(feats_dim)]
            typevars['feat_vars'] = feat_vars
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
