

import numpy as np
from Mscthesis.IO import Municipios_Parser

from pySpatialTools.Retrieve import general_density_assignation
from pySpatialTools.Retrieve import KRetriever
from pySpatialTools.Geo_tools import general_projection

from Mscthesis.Preprocess.comp_complementary_data import population_assignation_f, compute_population_data


mparser = Municipios_Parser(None)
data, typ = mparser.parse('/home/tono/mscthesis/code/Data/municipios_data/municipios-espana_2014_complete.csv')

params = {'f_weights': 'exponential', 'params_w':{'max_r':10.}, 'f_dens': population_assignation_f, 'params_d':{}}
data.loc[:, typ['loc_vars']] = general_projection(data, typ['loc_vars'], method='ellipsoidal', inverse=False, radians=False)

locs = data[typ['loc_vars']]
retriever = KRetriever
info_ret = np.ones(data.shape[0]).astype(int)*3

m = compute_population_data(locs, data, typ, retriever, info_ret, params)

print data.loc[np.where(m > 500000)[0], data.columns[:1]]
