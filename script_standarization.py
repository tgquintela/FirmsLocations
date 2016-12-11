
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import numpy as np
from FirmsLocations.Preprocess.firms_preprocessor import Firms_Standarization
from FirmsLocations.Computers.precomputers import PrecomputerCollection


## Pathpameters
# Set path parameters
execfile('set_pathparameters.py')
# Set precomputation parameters
execfile('set_precomputationparameters.py')

### Data Standarization
firms_standard = Firms_Standarization(logfile, pathdata)
firms_standard.clean_raw_data(pathdata_in)

### Data precomputation
precomps = PrecomputerCollection(logfile, pathfolder)
##precomps.precompute(pars_pop=pars_pop, pars_pfeatures=pars_pfeatures,
##                    pars_qval=pars_qvals)
#precomps.precompute(pars_locs=pars_locs)
precomps.precompute(pars_locs=pars_locs, pars_regs=pars_regs,
                    pars_pfeatures=pars_pfeatures, pars_qval=pars_qvals)
#precomps.precompute(pars_locs=pars_locs, pars_pfeatures=pars_pfeatures,
#                    pars_qval=pars_qvals, pars_pop=pars_pop)


def join_manufactures_cnae_code(pfeatures):
    """

    Parameters
    ----------
    pfeatures: np.ndarray (n, 2)
        1col: cnae, 2col: servicios, both or manufactures

    Returns
    -------
    pfeats: np.ndarray (n, 1)
        the coded categorical features.

    """
    pfeats = -1*np.ones(len(pfeatures)).astype(int)
    for i in range(len(pfeatures)):
        pfeats[i] = int(str(int(pfeatures[i, 0]))+str(int(pfeatures[i, 1])))
    return pfeats
