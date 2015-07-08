

#from Mscthesis.Retrieve.spatialdiscretizer import GridSpatialDisc
from Mscthesis.IO import Firms_Parser
from Mscthesis.Preprocess import Aggregator, Firms_Preprocessor
from Mscthesis.Retrieve import Neighbourhood, CircRetriever
from Mscthesis.Models import ModelProcess
from Mscthesis.Models.pjensen import Pjensen

import numpy as np

from Mscthesis.IO.clean_module import clean

logfile = 'Data/Outputs/Logs/logfile_2015_07_04.log'
parentpath = '/home/tono/mscthesis/code/Data/pruebas_clean'
inpath = '/home/tono/mscthesis/code/Data/pruebas_raw/raw1'

#clean(inpath, parentpath, extension='csv')

##parse empresas
parser = Firms_Parser(cleaned=True, logfile=logfile)
empresas = parser.parse(parentpath=parentpath, year=2006)

## Define aggregator
typevars = {'loc_vars': ['ES-X', 'ES-Y'], 'feat_vars': ['cnae'],
            'agg_var': 'cp'}
agg = Aggregator(typevars=typevars)

## Preprocess
preprocess = Firms_Preprocessor(typevars)
empresas = preprocess.preprocess(empresas)

## Define permuts
reindices = np.zeros((empresas.shape[0], 11))
reindices[:, 0] = np.array(range(empresas.shape[0]))
for i in range(1, 11):
    reindices[:, i] = np.random.permutation(np.arange(empresas.shape[0]))

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


