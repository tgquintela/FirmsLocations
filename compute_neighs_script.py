


import time
from os.path import join, exists
from os import makedirs
import numpy as np
import pandas as pd

#### PARSE DATA
texoutput = 'Data/results/tex_exploratory.tex'
fileinstructions = 'Data/about_data/stats_instructions.csv'
cleanedfilesdata = 'Data/clean_data/pruebadata'

statsfiledata = 'Data/Outputs'

# Importing modules
from Mscthesis.IO import Servicios_Parser
from os.path import join
import time

## Parse files
t0 = time.time()
servicios_parser = Servicios_Parser(cleaned=False)
servicios = servicios_parser.parse(cleanedfilesdata)
print 'Data parsed in %f seconds. Starting computing neighs.' % (time.time()-t0)

#### TRANSFORM COORDINATES
from Mscthesis.Geo_tools.geo_transformations import transf4compdist_global_homo
from Mscthesis.Geo_tools.geo_filters import filter_uncorrect_coord_spain
from Mscthesis.Retrieve.cnae_utils import transform_cnae_col

data = servicios[['cnae', 'ES-X', 'ES-Y']]
loc_vars = ['ES-X', 'ES-Y']
del servicios


#data = data.dropna(how='all')
##data = data[data['ES-X'] != 0]
#data = filter_uncorrect_coord_spain(data, loc_vars)
#data.index = range(data.shape[0])

data = transf4compdist_global_homo(data, loc_vars)

#data[['ES-X', 'ES-Y']] = 

#### GET CNAE index level specified
data['cnae'] = transform_cnae_col(data['cnae'], 2)

#### Compute matrix
from Mscthesis.Geo_tools.geo_retrieve import compute_neighs, compute_neighs_and_save
radius = .25
radiuss = [.1, .25, .75, 1]
type_var='cnae'
pathfile = 'Data/Outputs/neighs/neighs'
lim_rows = 100000

for i in range(len(radiuss)):
    t0 = time.time()
    aux_str = str(radiuss[i])
    aux_str = aux_str.replace('.', '_')
    pathfile_i = pathfile+'_' + aux_str
    if not exists(pathfile_i): makedirs(pathfile_i)
    compute_neighs_and_save(data, loc_vars, radiuss[i], pathfile_i, lim_rows)
    print "Neighs with r=%f km computed in %f seconds." % (radiuss[i], time.time()-t0)



