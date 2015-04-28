



import time
from os.path import join
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


data = data.dropna(how='all')
#data = data[data['ES-X'] != 0]
data = filter_uncorrect_coord_spain(data, loc_vars)
data.index = range(data.shape[0])

data = transf4compdist_global_homo(data, loc_vars)

#data[['ES-X', 'ES-Y']] = 

#### GET CNAE index level specified
data['cnae'] = transform_cnae_col(data['cnae'], 2)

##### TEST retrieving
radius = 10.
radius = radius/6371.009

#n = data.shape[0]
n = 100

## kdtree
from scipy.spatial import KDTree
t0 = time.time()
kdtree = KDTree(data[loc_vars].as_matrix(), leafsize=10000)

neighss = []
for i in range(n):
    neighs = kdtree.query_ball_point(data[loc_vars].as_matrix()[i], radius)
    neighss.append(np.array(neighs))
print "KDTree finished in %f seconds." % (time.time()-t0)

## manual retrieve
t0 = time.time()

neighss2 = []
for i in range(n):
    point = data[loc_vars].as_matrix()[i]
    logi = np.ones(data.shape[0]).astype(bool)
    logi = np.logical_and(logi, data[loc_vars].as_matrix()[:,0] <= point[0]+radius)
    logi = np.logical_and(logi, data[loc_vars].as_matrix()[:,0] >= point[0]-radius)
    logi = np.logical_and(logi, data[loc_vars].as_matrix()[:,1] <= point[1]+radius)
    logi = np.logical_and(logi, data[loc_vars].as_matrix()[:,1] >= point[1]-radius)
    neighs = np.where(logi)[0]
    neighss2.append(neighs)
print "Manual retrieve finished in %f seconds." % (time.time()-t0)





