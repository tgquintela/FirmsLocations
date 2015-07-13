
"""
Script for test parser data.
"""

from Mscthesis.IO import Firms_Parser
from pythonUtils.Logger import Logger

parentpath = '/home/tono/mscthesis/code/Data/pruebas_clean'
logfile = '/home/tono/mscthesis/code/Data/Outputs/Logs/log_clean.log'
typevars = {'loc_vars': ['ES-X', 'ES-Y'], 'feat_vars': ['cnae'],
            'agg_var': 'cp'}

## Parse
logger = Logger(logfile)
parser = Firms_Parser(logger)
empresas = parser.parse(parentpath, year=2006)

## Preprocess
preprocess = Firms_Preprocessor(typevars)
empresas = preprocess.preprocess(empresas)
