

"""
Module oriented to group all classes and functions which function is to
preprocess the data and prepare new data and structuctures to be use in the
processes we want to perform.

TODO:
-----
- Reverse process
- Processor class
"""

import numpy as np
import pandas as pd
from itertools import product

from preprocess_cols import generate_replace, transform_cnae_col

from pySpatialTools.Geo_tools.geo_transformations import general_projection
from pythonUtils.ProcessTools import Processer


class Firms_Preprocessor(Processer):
    "Special class to preprocess firms data."
    projection_values = None
    map_vars = None
    map_indices = None

    ## TODO: map indices

    def __init__(self, typevars, logfile, ):
        self.typevars = typevars
        self.map_vars = ['cnae', 'cp']
        ## General process parameters
        self.proc_name = "Preprocess empresas"
        self.proc_desc = "Preprocessing data in order to be treatable"
        #self.subproc_desc = ["Preprocess manufacturas", "Cleaning servicios"]
        #self.t_expended_subproc = [0, 0]

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
        # generate replacement in discrete vars
        t_vals = {'cnae': sorted(list(empresas['cnae'].unique())),
                  'cp': sorted(list(empresas['cp'].unique()))}
        self.map_info = generate_replace(t_vals)
        # cnae variable
        empresas.loc[:, 'cnae'] = transform_cnae_col(empresas['cnae'], cnae_lvl)
        # Map discrete variables
        mpvars = self.map_vars
        empresas.loc[:, mpvars] = empresas.loc[:, mpvars].replace(self.map_info).astype(int)
        # Finantial variables
        ### TODO
        return empresas

    def reverse_preprocess(self, empresas):

        ## 1. Inverse transformation of locations
        projection_values = self.projection_values
        empresas[loc_vars] = general_projection(empresas, *projection_values)

        ## 2. Inverse mapping
        ##TODO
        return empresas

