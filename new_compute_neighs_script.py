

import time
from os.path import join, exists
from os import makedirs
import numpy as np
import pandas as pd

#### PARSE DATA
cleanedfilesdata = 'Data/clean_data/pruebadata'
logfile = 'Data/Outputs/Logs/logfile.log'
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

data = servicios[['cnae', 'ES-X', 'ES-Y']]
loc_vars = ['ES-X', 'ES-Y']
del servicios

# Transformation
data = transf4compdist_global_homo(data, loc_vars)

#### GET CNAE index level specified
data['cnae'] = transform_cnae_col(data['cnae'], 2)

#### Compute matrix
from Mscthesis.Geo_tools.geo_retrieve import Compute_self_neighs

radius = [0.1, 0.25, 0.5, 0.75, 1., 1.5, 2., 5., 10.]
type_var='cnae'
basepathf = 'Data/Outputs/neighs/neighs_'
pathfile = [basepathf+str(radius[i]).replace('.', '_') for i in range(len(radius))]
filenames = ['neighs']*len(radius)
lim_rows = [20000000]*len(radius)

comp_neighs = Compute_self_neighs(pathfile, filenames, lim_rows, radius, logfile, lim_neighs=True)
comp_neighs.compute_neighs(data, loc_vars)




