
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
import os
#import pandas as pd
import shelve
#from itertools import product

from ..Preprocess.geo_filters import get_statistics2fill_locations,\
    fill_locations, fill_nulls
from ..Preprocess.aux_standarization_functions import pre_read_servicios,\
    get_sequencial_servicios, read_manufactures

from ..Preprocess.preprocess_cols import create_CA_column,\
    join_and_store_empresas_temporal, join_and_store_empresas_atemporal,\
    join_empresas_atemporal, compute_apertura_cierre, generate_replace,\
    create_sector_columns, clean_colnames_manu, transform_cnae_col,\
    store_empresas_atemporal_years
from ..Preprocess.financial_interpolation import financial_interpolation

from ..IO.standarization_io_utils import write_ca_cp, write_locs_statistics,\
    write_uncorrect_locs, write_ca2code, write_nif2code, write_cp2code,\
    write_nif2names

from pythonUtils.ProcessTools import Processer
from pythonUtils.Logger import Logger
from preprocess_cols import generate_replace, transform_cnae_col,\
    create_sector_columns, clean_colnames_manu
from pySpatialTools.Preprocess.Transformations.Transformation_2d import\
    general_projection

CA_names = ['Andalucia', 'Aragon', 'Asturias', 'Balears', 'Canarias',
            'Cantabria', 'CastillaLeon', 'CastillaMancha', 'Catalunya',
            'Ceuta_Melilla', 'Extremadura', 'Galicia', 'LaRioja_Navarra',
            'Madrid', 'Murcia', 'Pais Vasco', 'Valencia']
loc_vars, reg_var = ['ES-X', 'ES-Y'], 'cp'


class Firms_PrePrecomputations(Processer):
    """Class processer which standarizes and cleans firms data."""

    def _initialization(self):
        self.proc_name = "Main precomputations"
        self.proc_desc = """Main precomputations of features and values."""
        self.subproc_desc = ["Pfeatures computation", "Qvalues computation"]
        self.t_expended_subproc = [0, 0]

    def __init__(self, logfile, pathdata):
        self._initialization()
        self.logfile = Logger(logfile) if type(logfile) == str else logfile
        self.pathdata = pathdata

    def compute(self):
        ## 0. Set vars
        t00 = self.setting_global_process()

        ## 1. Precompute pointfeatures
        t0 = self.set_subprocess([0])
        pass


class Firms_Standarization(Processer):
    """Class processer which standarizes and cleans firms data."""

    def _initialization(self):
        self.proc_name = "Standarization and cleaning empresas data"
        self.proc_desc = """Preprocessing data of empresas in order to be
            treatable and easy retrievable from files."""
        self.subproc_desc = ["Get CA-CP dictionary",
                             "Standarization of manufactures",
                             "Standarization of services and storing"]
        self.t_expended_subproc = [0, 0, 0]

    def __init__(self, logfile, pathdata):
        self._initialization()
        self.logfile = Logger(logfile) if type(logfile) == str else logfile
        self.pathdata = pathdata

    def clean_raw_data(self, pathdata):
        ## 0. Set vars
        t00 = self.setting_global_process()

        ## 1. Pre-read servicios
        t0 = self.set_subprocess([0])
        ca_cp_dict, raw_locs_serv, nifs_serv, cps_serv, names_serv,\
            muni_serv, null_serv_cp, null_serv_muni, null_serv_locs =\
            pre_read_servicios(pathdata)
        self.close_subprocess([0], t0)

        ## 2. Read and process manufactures
        t0 = self.set_subprocess([1])
        # Read
        manufactures, raw_locs_manu, nifs_manu, cps_manu, names_manu,\
            muni_manu, null_manu_cp, null_manu_muni, null_manu_locs =\
            read_manufactures(pathdata)
        # Joining useful data
        nifs = nifs_serv + nifs_manu
        names = names_serv + names_manu
        raw_locs = np.concatenate([raw_locs_serv, raw_locs_manu])
        raw_cps = cps_serv + cps_manu
        raw_muni = muni_serv + muni_manu
        null_cp = np.concatenate(null_serv_cp+[null_manu_cp])
        null_muni = np.concatenate(null_serv_muni+[null_manu_muni])
        null_locs = np.concatenate([np.concatenate(null_serv_locs, axis=0),
                                    null_manu_locs], axis=0)
#        print null_serv_locs, null_manu_locs
#        print null_cp.shape, null_muni.shape, null_locs.shape

#        nulls = np.logical_and(np.logical_not(null_cp),
#                               np.logical_not(null_muni))
        nnulls = np.logical_and(np.logical_not(null_cp),
                                np.logical_not(null_locs))
        new_raw_locs = raw_locs[nnulls]
        new_raw_cps = [raw_cps[i] for i in range(len(raw_cps)) if nnulls[i]]
        new_raw_muni = [raw_muni[i] for i in range(len(raw_muni)) if nnulls[i]]
#        print nnulls.sum(), len(new_raw_locs), len(new_raw_cps), len(raw_cps)
        # Preparing fill locations
        mean_locs, std_locs, u_cps =\
            get_statistics2fill_locations(new_raw_locs, new_raw_cps)
        assert(len(mean_locs) == len(std_locs))
        assert(len(std_locs) == len(u_cps))
        # Chaging manufactures
        manufactures.columns = clean_colnames_manu(manufactures.columns)
        manufactures = create_sector_columns(manufactures, 'manufactures')
        manufactures = compute_apertura_cierre(manufactures)

        manufactures = fill_nulls(manufactures, mean_locs, std_locs, u_cps,
                                  new_raw_muni, new_raw_cps, new_raw_locs,
                                  os.path.join(self.pathdata, 'extra'))
        manufactures = create_CA_column(manufactures, ca_cp_dict)
        assert('ca' in list(manufactures.columns))

#
#        manufactures = fill_locations(manufactures, loc_vars, reg_var,
#                                      mean_locs, std_locs, u_cps)
#
        self.close_subprocess([1], t0)

        ## 3. Standarization and join data
        t0 = self.set_subprocess([2])
        empresas_atemporal = []
        for servicios, ca_name in get_sequencial_servicios(pathdata):
            assert(pd.isnull(servicios['nom']).sum() == 0)
            servicios = fill_nulls(servicios, mean_locs, std_locs, u_cps,
                                   new_raw_muni, new_raw_cps, new_raw_locs,
                                   os.path.join(self.pathdata, 'extra'))
            assert(pd.isnull(servicios['nom']).sum() == 0)
            servicios = compute_apertura_cierre(servicios)
            assert(pd.isnull(servicios['nom']).sum() == 0)
            servicios = create_sector_columns(servicios, 'servicios')
            assert(pd.isnull(servicios['nom']).sum() == 0)
            servicios.loc[:, 'ca'] = ca_name
            join_and_store_empresas_temporal(servicios, manufactures, ca_name,
                                             self.pathdata)
            empresas_atemporal_i =\
                join_empresas_atemporal(servicios, manufactures, ca_name)
            empresas_atemporal.append(empresas_atemporal_i)
            store_empresas_atemporal_years(empresas_atemporal_i, ca_name,
                                           self.pathdata)

        join_and_store_empresas_atemporal(empresas_atemporal, self.pathdata)
        financial_interpolation(self.pathdata)

        # Write extradata
        u_CA = list(set(ca_cp_dict.values()))
#        u_cps = np.unique([e for e in raw_cps if e != float('nan')])
#        u_cps = np.unique([e for e in raw_cps if e != '00nan'])
        write_ca_cp(ca_cp_dict, self.pathdata)
        write_locs_statistics(mean_locs, std_locs, u_cps, self.pathdata)
        write_uncorrect_locs(nifs, raw_locs, self.pathdata)
        write_ca2code(u_CA, self.pathdata)
        write_nif2code(nifs, self.pathdata)
        write_cp2code(u_cps, self.pathdata)
        write_nif2names(nifs, names, self.pathdata)

        self.close_subprocess([2], t0)

        self.close_process(t00)

    def _self_store(self, namefile):
        with shelve.open(namefile) as db:
            db['preprocessor'] = self


class Firms_Preprocessor(Processer):
    "Special class to preprocess firms data."

    def _initialization(self):
        self.projection_values = None
        self.map_vars = None
        self.map_indices = None
        self.map_vars = ['cnae', 'cp']
        ## General process parameters
        self.proc_name = "Preprocess empresas"
        self.proc_desc = "Preprocessing data in order to be treatable"
        self.subproc_desc = ["Reindexing", "Locations transformation",
                             "Features transformation"]
        self.t_expended_subproc = [0, 0, 0]

    def __init__(self, typevars, logfile):
        self._initialization()
        self.typevars = typevars
        self.logfile = logfile

    def preprocess(self, empresas, cnae_lvl=2, method_proj='ellipsoidal', radians=False):
        "Function to preprocess firms data."

        ## 0. Set vars
        t00 = self.setting_global_process()
        # Important vars
        financial_vars = [e for e in self.typevars['feat_vars'] if e != 'cnae']
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
