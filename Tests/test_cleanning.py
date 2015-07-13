
"""
Script to test cleanning task
"""

from Mscthesis.Clean import CleanProcess
from pythonUtils.Logger import Logger

inpath = '/home/tono/mscthesis/code/Data/pruebas_raw/raw1'
outpath = '/home/tono/mscthesis/code/Data/pruebas_clean'
logfile = '/home/tono/mscthesis/code/Data/Outputs/Logs/log_clean.log'

logger = Logger(logfile)
cleaner = CleanProcess(logger)
cleaner.clean(inpath, outpath)
