
from Mscthesis.Computers.precomputers import PrecomputerCollection
from Mscthesis.Computers.computers import Directmodel, LocationGeneralModel,\
    LocationOnlyModel


## Pathpameters
# Set path parameters
execfile('set_pathparameters.py')
# Set precomputation parameters
execfile('set_precomputationparameters.py')
# Set computation paramters
execfile('set_computationparameters.py')

### Data precomputation
precomps = PrecomputerCollection(logfile, pathfolder, old_computed=True)

### Models
#dirmodel = Directmodel(logfile, pathfolder, precomps, num_cores=2)
#dirmodel.compute(pars_directmodel)

locmodel = LocationOnlyModel(logfile, pathfolder, precomps, num_cores=1)
locmodel.compute(pars_loconly_model)
#
#locgeneralmodel = LocationGeneralModel(logfile, pathfolder, precomps)
#locgeneralmodel.compute(pars_loc_model)
