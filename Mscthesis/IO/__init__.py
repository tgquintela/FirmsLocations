
"""
Module which groups all the functions relate to parse the data or to write it.

TODO
----
Normalize the columns

"""

import pandas as pd
#from os.path import join
import datetime
from os.path import join
import os
import time
import numpy as np

from Mscthesis.Preprocess.preprocess import filter_servicios, cnae2str, \
    cp2str, filter_servicios_dict
from aux_functions import concat_from_dict, write_dataframe_to_csv, \
    get_index_from_dict
from parse_data import parse_servicios, parse_servicios_columns, \
    parse_manufactures
from aux_functions import parse_xlsx_sheet

from write_log import Logger

########### Global variables needed
##################################################################
message0 = """========================================
Start parsing data:
-------------------
(%s)

"""
message1 = "Parsing data: "
message1a = "Preprocessing, formatting and filtering data: "
message2 = "completed in %f seconds.\n"
message3 = "Total time expended parsing process: %f seconds.\n"
message_close = '----------------------------------------\n'


########### Class for parsing
##################################################################
class Firms_Parser():
    """This class is the one which controls the parsing process of servicios
    information.
    """

    def __init__(self, cleaned=False, indices=None, logfile=None):
        '''Initialization of the parser instance.'''
        # Logfile
        self.logfile = Logger(logfile)
        # Assertion condition
        assert(not (indices is None and cleaned))
        # Parameters to save
        self.cleaned = cleaned
        self.indices = indices
        self.files = {}

    def parse(self, filepath=None, cleaned=None):
        '''Parsing function which considers if we have parsed or not before.'''
        ### 0. Managing inputs
        self.cleaned = self.cleaned if cleaned is None else cleaned
        if cleaned:
            self.files['clean'] = filepath
        else:
            self.files['raw'] = filepath
        # Tracking process with logfile
        t00 = time.time()
        self.logfile.write_log(message0 % (filepath.split('/')[-1]))
        self.logfile.write_log(message1)
        ## 1. Parsing task
        if not self.cleaned:
            ### parse files
            servicios = parse_servicios(join(filepath, 'SERVICIOS'))
            ### filter in parsing
            date = datetime.datetime.strptime('2006-01-01', '%Y-%m-%d')
            loc_vars = ['ES-X', 'ES-Y']
            servicios = filter_servicios_dict(servicios, date, loc_vars)
            ### get indices
            self.indices = get_index_from_dict(servicios)
            ### Concat servicios
            servicios = concat_from_dict(servicios, None)
            ### Parse manufactures
            manufactures = parse_manufactures(filepath)
            ### Concat manufacturas y servicios
            empresas = {'manufacturas': manufactures, 'servicios': servicios}
            empresas = concat_from_dict(empresas, 'type')
        else:
            ### parse cleaned file
            empresas = pd.io.parsers.read_csv(filepath)
            ### get indices
            self.indices = np.array(empresas.index)
        ## Stop to track the parsing
        self.logfile.write_log(message2 % (time.time()-t00))
        ## 2. Tranforming
        # Start tracking process
        t0 = time.time()
        self.logfile.write_log(message1a)
        ## Transformation
        empresas = self.categorize_cols(empresas)
        ## 3. Reindex
        empresas.index = range(empresas.shape[0])
        ## Closing the tracking
        self.logfile.write_log(message2 % (time.time()-t0))
        self.logfile.write_log(message3 % (time.time()-t00))
        self.logfile.write_log(message_close)
        return empresas

    def write_firms(self, empresas, filepath):
        '''Write function in order to save a cleaned dataframe in a file.'''
        #self.files['clean'] = join(path, filename)
        self.files['clean'] = filepath
        write_dataframe_to_csv(empresas, filepath)
        del empresas

    def filter_rows(self, empresas):
        '''Filter to only take into account the active companies in [06-12]'''
        if not self.cleaned:
            date = datetime.datetime.strptime('2006-01-01', '%Y-%m-%d')
            empresas, self.indices = filter_servicios(empresas, date)
            self.cleaned = True
        return empresas

    def categorize_cols(self, empresas):
        '''TO GENERALIZE'''
        empresas = cp2str(empresas)
        empresas = cnae2str(empresas)
        return empresas

    def get_index_from_cleaned(self, infilepath):
        ## 0. Managing inputs
        self.cleaned = False
        ## 1. Parsing
        files = os.listdir(infilepath)
        indices = []
        for f in files:
            empresas = parse_xlsx_sheet(join(infilepath, f))
            indices.append(empresas.index)
        self.indices = (files, indices)
        self.cleaned = True

    def parse_columns(self, filepath=None, columns=None, id_val=None):
        '''Parsing function which considers if we have parsed or not before.'''
        ### 0. Managing inputs
        ## 1. Parsing task
        if columns is None:
            empresas = parse_servicios(filepath)
        else:
            empresas, ids = parse_servicios_columns(filepath, columns, id_val)
        ### Concat servicios
        empresas = concat_from_dict(empresas, None)

        ## 2. Transforming
        empresas = self.categorize_cols(empresas)
        return empresas
