
##### 0. IMPORTS
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

locs = empresas[typevars['loc_vars']]

#### 2. Spatial clustering

## Discretizor

## Spatial relation

## Custering


