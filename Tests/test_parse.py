
"""
Script for test parser data.

TESTED!
"""

from Mscthesis.IO import Firms_Parser
from Mscthesis.Preprocess import Firms_Preprocessor
from pythonUtils.Logger import Logger

parentpath = '/home/tono/mscthesis/code/Data/pruebas_clean'
logfile = '/home/tono/mscthesis/code/Data/Outputs/Logs/log_clean.log'
#typevars = {'loc_vars': ['ES-X', 'ES-Y'], 'feat_vars': ['cnae'],
#            'agg_var': 'cp'}

## Parse
logger = Logger(logfile)
parser = Firms_Parser(logger)
empresas, typevars = parser.parse(parentpath, year=2006)

## Preprocess
preprocess = Firms_Preprocessor(typevars, logger)
empresas = preprocess.preprocess(empresas)
