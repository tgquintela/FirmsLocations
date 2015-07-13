
"""
Module which groups all the functions relate to parse the data or to write it.

TODO
----
Normalize the columns
Firms_Parser as a Processer

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
    get_index_from_dict, parse_xlsx_sheet, check_cleaned
from parse_data import parse_servicios, parse_servicios_columns, \
    parse_manufactures, parse_empresas, parse_finantial_by_year

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
class Firms_Parser(Processer):
    """This class is the one which controls the parsing process of servicios
    information.
    """

    indices = None
    finantial_bool=False
    files = ''

    def __init__(self, logfile, bool_inform=False):
        "Instantiation of the class remembering it is a subclass of Processer."
        self.proc_name = "Firms parser"
        self.proc_desc = "Parser the standarize data from folder"
        self.subproc_desc = ["Parsing data",
                             "Preprocessing, formatting and filtering data"]
        self.t_expended_subproc = [0, 0]
        self.logfile = logfile


    def parse(self, parentpath=None, year=None, finantial_bool=False):
        """Parsing function which considers if we have parsed or not before."""
        ### 0. Managing inputs
        cleaned = check_cleaned(parentpath)
        if not cleaned:
            raise Exception("TODO: control of errors")

        # Tracking process
        globname = parentpath.split('/')
        globname = globname[-2] if globname[-1] == '' else globname[-1]
        self.proc_desc = self.proc_desc+" "+globname
        self.files = parentpath
        t00 = self.setting_process()

        ## 1. Parsing task
        t0 = self.set_subprocess([0])
        ## Parse empresas
        filepath = join(self.files, 'Main')
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
        close_subprocess([0], t0)

        ## 2. Transforming
        # Start tracking process
        t0 = self.set_subprocess([1])
        # Categorization
        empresas = self.categorize_cols(empresas)
        # Parse and concatenation finantial data
        empresas = self.parse_finantial(empresas, year, self.indices,
                                        finantial_bool)
        # Reindex
        empresas.index = range(empresas.shape[0])
        ## Closing the tracking
        close_subprocess([1], t0)
        self.close_process(t00)
        return empresas

    def reparse(self, parentpath):
        "Reparse file using indices."
        # Tracking process with logfile
        t00 = self.setting_process()
        ## 1. Parsing task
        t0 = self.set_subprocess([0])
        ## Parse empresas
        filepath = join(parentpath, 'Main')
        empresas = parse_empresas(filepath)
        ## Filter empresas
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
        close_subprocess([0], t0)

        ## 2. Tranforming
        # Start tracking process
        t0 = self.set_subprocess([1])
        ## Transformation
        empresas = self.categorize_cols(empresas)
        ## 3. Reindex
        empresas.index = range(empresas.shape[0])
        ## Closing the tracking
        close_subprocess([1], t0)
        self.close_process(t00)
        return empresas

    def categorize_cols(self, empresas):
        '''TO GENERALIZE, TODEPRECATE, preprocess function'''
        empresas = cp2str(empresas)
        empresas = cnae2str(empresas)
        return empresas

    def parse_finantial(self, empresas, year, indices, finantial_bool=False):
        "Parse and concat the finantial data. TODO: for standarize data"
        if finantial_bool:
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
