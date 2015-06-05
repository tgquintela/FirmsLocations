
"""
Module which groups all the functions relate to parse the data or to write it.

TODO
----
Normalize the columns

"""

import pandas as pd
#from os.path import join
from os.path import join
import os
import time
import numpy as np

from Mscthesis.Preprocess.preprocess_rows import filter_servicios, \
    filter_servicios_dict
from Mscthesis.Preprocess.preprocess_cols import cnae2str, cp2str
from Mscthesis.Preprocess.preprocess_general import filter_empresas, \
    concat_empresas, filtercols_empresas

from preparation_module import prepare_filterinfo, prepare_concatinfo, \
    prepare_filtercolsinfo

from aux_functions import concat_from_dict, write_dataframe_to_csv, \
    get_index_from_dict
from parse_data import parse_servicios, parse_servicios_columns, \
    parse_manufactures, parse_empresas, parse_finantial_by_year
from aux_functions import parse_xlsx_sheet
from clean_module import check_cleaned, clean

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

    def __init__(self, cleaned=False, indices=None, logfile=None,
                 finantial_bool=False):
        '''Initialization of the parser instance.'''
        # Logfile
        self.logfile = Logger(logfile)
        # Parameters to save
        self.cleaned = cleaned
        self.indices = indices
        self.files = {}
        # Possibility of parse and concat finantial data
        self.finantial_bool = finantial_bool

    def parse(self, parentpath=None, cleaned=None, year=None):
        """Parsing function which considers if we have parsed or not before."""
        ### 0. Managing inputs
        if cleaned is None:
            self.cleaned = check_cleaned(parentpath)
        if self.cleaned:
            self.files['clean'] = parentpath
        else:
            self.files['raw'] = parentpath
        # Tracking process with logfile
        t00 = time.time()
        self.logfile.write_log(message0 % (parentpath.split('/')[-1]))
        self.logfile.write_log(message1)
        ## 1. Parsing task
        if not self.cleaned:
            ## Cleaning
            clean(parentpath, join(parentpath, 'Cleaned'))
            self.files['clean'] = join(parentpath, 'Cleaned')
        ## Parse empresas
        filepath = join(self.files['clean'], 'Main')
        empresas = parse_empresas(filepath)
        ## Filter empresas
        # Prepare filter information
        filterinfo = prepare_filterinfo(year)
        # Filter
        empresas, self.indices = filter_empresas(empresas, filterinfo)
        ## Format data to work
        # Concat info
        concatinfo = prepare_concatinfo()
        # Concat
        empresas = concat_empresas(empresas, **concatinfo)
        # Filter columns
        filtercolsinfo = prepare_filtercolsinfo()
        empresas = filtercols_empresas(empresas, filtercolsinfo)
        ## Stop to track the parsing
        self.logfile.write_log(message2 % (time.time()-t00))

        ## 2. Transforming
        # Start tracking process
        t0 = time.time()
        self.logfile.write_log(message1a)
        # Categorization
        empresas = self.categorize_cols(empresas)
        # Parse and concatenation finantial data
        empresas = self.parse_finantial(empresas, year, self.indices)
        # Reindex
        empresas.index = range(empresas.shape[0])
        ## Closing the tracking
        self.logfile.write_log(message2 % (time.time()-t0))
        self.logfile.write_log(message3 % (time.time()-t00))
        self.logfile.write_log(message_close)
        return empresas

    def reparse(self, parentpath):
        "Reparse file using indices."
        # Tracking process with logfile
        t00 = time.time()
        self.logfile.write_log(message0 % (parentpath.split('/')[-1]))
        self.logfile.write_log(message1)
        ## 1. Parsing task
        ## Parse empresas
        filepath = join(parentpath, 'Main')
        empresas = parse_empresas(filepath)
        ## Filter empresas
        # Filter
        empresas, _ = filter_empresas(empresas, {}, self.indices)
        ## Format data to work
        # Concat info
        concatinfo = prepare_concatinfo()
        # Concat
        empresas = concat_empresas(empresas, concatinfo)
        # Filter columns
        filtercolsinfo = prepare_filtercolsinfo()
        empresas = filtercols_empresas(empresas, filtercolsinfo)
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

    def categorize_cols(self, empresas):
        '''TO GENERALIZE'''
        empresas = cp2str(empresas)
        empresas = cnae2str(empresas)
        return empresas

    def parse_finantial(self, empresas, year, indices):
        "Parse and concat the finantial data."
        if self.finantial_bool:
            finantial = parse_finantial_by_year(self.files['raw'], year)
            # apply filter dict
            empresas, _ = filter_empresas(empresas, {}, self.indices)
            # Concat
            concatinfo = prepare_concatinfo()
            finantial = concat_empresas(finantial, concatinfo)
            # Reindex
            finantial.index = empresas.index
            # Concat horizontally
            empresas = pd.concat([empresas, finantial], axis=1)
        else:
            return empresas
        return empresas
