
"""
Test 2
======
- Separation between Mscthesis and pySpatialTools
"""

## Mscthesis application
##########################
from Mscthesis.IO.clean_module import clean
from Mscthesis.IO import Firms_Parser
from Mscthesis.Preprocess import Firms_Preprocessor

## Define parameters and info needed
logfile = 'Data/Outputs/Logs/logfile_2015_07_04.log'
parentpath = '/home/tono/mscthesis/code/Data/pruebas_clean'
inpath = '/home/tono/mscthesis/code/Data/pruebas_raw/raw1'
typevars = {'loc_vars': ['ES-X', 'ES-Y'], 'feat_vars': ['cnae'],
            'agg_var': 'cp'}

## Cleaning (TODO: Prepare as a process class)
#clean(inpath, parentpath, extension='csv')

## Parse empresas
parser = Firms_Parser(cleaned=True, logfile=logfile)
empresas = parser.parse(parentpath=parentpath, year=2006)

## Preprocess
preprocess = Firms_Preprocessor(typevars)
empresas = preprocess.preprocess(empresas)


## Spatial module application
###############################
#from pySpatialTools.Retrieve.spatialdiscretizer import GridSpatialDisc
from pySpatialTools.Preprocess import Aggregator
from pySpatialTools.Retrieve import Neighbourhood, CircRetriever
from pySpatialTools.Models.pjensen import Pjensen
from Mscthesis.Models import ModelProcess
import numpy as np

## Parameters and info needed
n_permuts = 10


## Define aggregator
agg = Aggregator(typevars=typevars)

## Define permuts
reindices = create_permtutation(n_permuts)
#reindices = np.zeros((empresas.shape[0], 11))
#reindices[:, 0] = np.array(range(empresas.shape[0]))
#for i in range(1, 11):
#    reindices[:, i] = np.random.permutation(np.arange(empresas.shape[0]))

## Define retriever (Neigh has to know typevars)
locs = empresas[typevars['loc_vars']].as_matrix()
retriever = CircRetriever(locs)
Neigh = Neighbourhood(retriever)
Neigh.define_mainretriever(retriever)
Neigh.define_aggretrievers(agg, empresas, reindices)
del locs, retriever


## Define descriptormodel
descriptormodel = Pjensen(empresas, typevars)

## Define process
modelprocess = ModelProcess(logfile, Neigh, descriptormodel, typevars=typevars, lim_rows=100000,
                            proc_name='Test')
modelprocess.compute_net(empresas, 2., True, reindices)


