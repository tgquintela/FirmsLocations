
"""
Module oriented to group the classes and functions which tasks are to
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

from pythonUtils.ProcessTools import Processer
from preprocess_cols import generate_replace, transform_cnae_col
from pySpatialTools.Geo_tools.geo_transformations import general_projection


class Firms_Preprocessor(Processer):
    "Special class to preprocess firms data."
    projection_values = None
    map_vars = None
    map_indices = None

    ## TODO: map indices

    def __init__(self, typevars, logfile):
        self.typevars = typevars
        self.map_vars = ['cnae', 'cp']
        ## General process parameters
        self.proc_name = "Preprocess empresas"
        self.proc_desc = "Preprocessing data in order to be treatable"
        self.subproc_desc = ["Reindexing", "Locations transformation",
                 "Features transformation"]
        self.t_expended_subproc = [0, 0, 0]
        self.logfile = logfile

    def preprocess(self, empresas, cnae_lvl=2, method_proj='ellipsoidal', radians=False):
        "Function to preprocess firms data."

        ## 0. Set vars
        t00 = self.setting_global_process()
        # Important vars
        finantial_vars = [e for e in self.typevars['feat_vars'] if e != 'cnae']
        loc_vars = self.typevars['loc_vars']
        self.projection_values = [loc_vars, method_proj, True, radians]
        # 1. Indices
        t0 = self.set_subprocess([0])
        self.map_indices = zip(list(empresas.index), range(empresas.shape[0]))
        empresas.index = range(empresas.shape[0])
        self.close_subprocess([0], t0)
        # 2. Location transformation
        t0 = self.set_subprocess([1])
        empresas[loc_vars] = general_projection(empresas, loc_vars,
                                                method=method_proj,
                                                inverse=False,
                                                radians=radians)
        self.close_subprocess([1], t0)
        ## 3. Feature array
        t0 = self.set_subprocess([2])
        # cnae variable
        empresas.loc[:, 'cnae'] = transform_cnae_col(empresas['cnae'], cnae_lvl)
        # generate replacement in discrete vars
        t_vals = {'cnae': sorted(list(empresas['cnae'].unique())),
                  'cp': sorted(list(empresas['cp'].unique()))}
        self.map_info = generate_replace(t_vals)
        # Map discrete variables
        mpvars = self.map_vars
        empresas.loc[:, mpvars] = empresas.loc[:, mpvars].replace(self.map_info).astype(int)
        # Finantial variables
        ### TODO
        self.close_subprocess([2], t0)
        ## Untrack process
        self.close_process(t00)

        return empresas

    def reverse_preprocess(self, empresas):

        ## 1. Inverse transformation of locations
        projection_values = self.projection_values
        empresas[loc_vars] = general_projection(empresas, *projection_values)

        ## 2. Inverse mapping
        ##TODO
        return empresas

