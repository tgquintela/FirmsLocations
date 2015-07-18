
import numpy as np
from pySpatialTools.Geo_tools import general_projection
from pythonUtils.Logger import Logger

#### 1. Prepare empresas

from Mscthesis.IO import Firms_Parser
from Mscthesis.Preprocess import Firms_Preprocessor


parentpath = '/home/tono/mscthesis/code/Data/pruebas_clean'
logfile = '/home/tono/mscthesis/code/Data/Outputs/Logs/log_clean.log'


## Parse
logger = Logger(logfile)
parser = Firms_Parser(logger)
empresas, typevars = parser.parse(parentpath, year=2006)

## Preprocess
preprocess = Firms_Preprocessor(typevars, logger)
empresas = preprocess.preprocess(empresas)


### Prepare municipios

from Mscthesis.IO import Municipios_Parser
from pySpatialTools.Retrieve import general_density_assignation
from pySpatialTools.Retrieve import KRetriever
from Mscthesis.Preprocess.comp_complementary_data import population_assignation_f, compute_population_data

# municipios file
mpiosfile = '/home/tono/mscthesis/code/Data/municipios_data/municipios-espana_2014_complete.csv'

mparser = Municipios_Parser(None)
data, typ = mparser.parse(mpiosfile)

params = {'f_weights': 'exponential', 'params_w':{'max_r':10.}, 'f_dens': population_assignation_f, 'params_d':{}}
data.loc[:, typ['loc_vars']] = general_projection(data, typ['loc_vars'], method='ellipsoidal', inverse=False, radians=False)

locs = data[typ['loc_vars']]
retriever = KRetriever
info_ret = np.ones(data.shape[0]).astype(int)*3

m = compute_population_data(locs, data, typ, retriever, info_ret, params)

empresas['population_idx'] = m
typevars['pop_var'] = 'population_idx'



#### 2. Compute model
from pySpatialTools.IO import create_reindices


## Define aggregator
agg = Aggregator(typevars=typevars)

## Define permuts
reindices = create_reindices(empresas.shape[0], m)

## Define retriever (Neigh has to know typevars)  (TODO: define bool_var)
retriever = CircRetriever(empresas[typevars['loc_vars']].as_matrix())
aggretriever = KRetriever(locs)

Neigh = Neighbourhood(retriever, typevars, empresas, reindices, aggretriever, funct=None)
del locs, retriever

## Define info retriever and conditional aggregator
create_info_ret()

## Define descriptormodel
descriptormodel = Pjensen(empresas, typevars)

## Define process
modelprocess = ModelProcess(logfile, Neigh, descriptormodel, typevars=typevars, lim_rows=100000,
                            proc_name='Test')

corrs = modelprocess.compute_net(empresas, 2., True, reindices)
matrix = modelprocess.compute_matrix(empresas, info_ret=None, cond_agg=None, reindices)







