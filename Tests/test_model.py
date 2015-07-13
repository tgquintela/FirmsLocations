

from pySpatialTools.Preprocess import Aggregator
from pySpatialTools.Retrieve import Neighbourhood, CircRetriever
from pySpatialTools.Models import ModelProcess
from pySpatialTools.Models.pjensen import Pjensen


logfile = '/home/tono/mscthesis/code/Data/Outputs/Logs/log_clean.log'
typevars = {'loc_vars': ['ES-X', 'ES-Y'], 'feat_vars': ['cnae'],
            'agg_var': 'cp'}

## Define aggregator
agg = Aggregator(typevars=typevars)

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

