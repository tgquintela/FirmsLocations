

#from Mscthesis.Retrieve.spatialdiscretizer import GridSpatialDisc
from Mscthesis.IO import Firms_Parser
from Mscthesis.Preprocess import Aggregator
from Mscthesis.Retrieve import Neighbourhood, CircRetriever
from Mscthesis.Models import ModelProcess
from Mscthesis.Models.pjensen import Pjensen

import numpy as np


logfile = 'Data/Outputs/Logs/logfile_2015_07_04.log'
parentpath = '/home/antonio/Desktop/MSc Thesis/code/Data/Outputs/Pruebas'

##parse empresas
parser = Firms_Parser(cleaned=True, logfile=logfile)
empresas = parser.parse(parentpath=parentpath, year=2006)

## Define aggregator
typevars = {'loc_vars': ['ES-X', 'ES-Y'], 'feat_vars': ['cnae'],
            'agg_var': 'cp'}
agg = Aggregator(typevars=typevars)

## Preprocess
preprocess = Firms_Preprocessor(typevars)
empresas = preprocess.apply_preprocess(empresas)

## Define permuts
permuts = np.zeros((10000, 10))
for i in range(10):
    permuts[:, i] = np.random.permutation(np.arange(10000))

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
modelprocess.compute_net(empresas, type_vars, loc_vars, radius, permuts=None,
                         agg_var=None)


