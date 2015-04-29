

import time
from os.path import join, exists
from os import makedirs
import numpy as np
import pandas as pd

#### PARSE DATA
cleanedfilesdata = 'Data/clean_data/pruebadata'
logfile = 'Data/Outputs/logfile.log'
# Importing modules
from Mscthesis.IO import Servicios_Parser
from os.path import join

## Parse files
servicios_parser = Servicios_Parser(cleaned=False, logfile=logfile)
servicios = servicios_parser.parse(cleanedfilesdata)

#### TRANSFORM COORDINATES
from Mscthesis.Geo_tools.geo_transformations import transf4compdist_global_homo
from Mscthesis.Geo_tools.geo_filters import filter_uncorrect_coord_spain
from Mscthesis.Retrieve.cnae_utils import transform_cnae_col


type_var = 'cnae'
loc_vars = ['ES-X', 'ES-Y']
data = servicios[['cnae', 'ES-X', 'ES-Y']]
del servicios

## Transforming data
data['cnae'] = transform_cnae_col(data['cnae'], 2)
data.index = range(data.shape[0])
data = transf4compdist_global_homo(data, loc_vars)

### Test for a given distance
from Mscthesis.Models.pjensen import Pjensen
neighs_dir = 'Data/Outputs/neighs/neighs_0_5'

pjensen = Pjensen(logfile, neighs_dir)
net, type_vals, N_x = pjensen.built_nets(data, type_var, loc_vars, radius)

