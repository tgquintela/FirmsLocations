
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

from Mscthesis.Preprocess.preprocess import filter_servicios, cnae2str, cp2str
from aux_functions import concat_from_dict, write_dataframe_to_excel,\
    write_dataframe_to_csv
from parse_data import parse_servicios, parse_servicios_columns
from aux_functions import parse_xlsx_sheet


class Servicios_Parser():
    """This class is the one which controls the parsing process of servicios
    information.
    """

    def __init__(self, cleaned=False, indices=None):
        '''Initialization of the parser instance.'''
        # Assertion condition
        assert(not (indices is None and cleaned))
        # Parameters to save
        self.cleaned = cleaned
        self.indices = indices
        self.files = {}

    def parse_and_clean(self, infilepath, outfilepath=None):
        '''Parsing and cleaning process.'''
        ## 0. Managing inputs
        self.cleaned = False
        ## 1. Parsing
        servicios = self.parse(infilepath)
        ## 2. Preprocessing
        servicios = self.filter_rows(servicios)
        ## 3. Output the file
        if outfilepath is not None:
            self.write_servicios(servicios, outfilepath)
        del servicios

    def parse_and_clean_by_file(self, infilepath, outfilepath):
        ## 0. Managing inputs
        self.cleaned = False
        t0 = time.time()
        ## 1. Parsing
        files = os.listdir(infilepath)
        indices = []
        for f in files:
            t1 = time.time()
            servicios = parse_xlsx_sheet(join(infilepath, f))
            servicios = self.filter_rows(servicios)
            indices.append(self.indices)
            print 'End filtering '+f
            write_dataframe_to_excel(servicios, join(outfilepath, f))
            print 'End writting '+f
            s = "The duration of the total process for %s was %f seconds."
            print s % (f, t1-time.time())
            del servicios
        self.indices = (files, indices)
        self.cleaned = True
        print "The process lasted %f seconds." % (time.time()-t0)

    def parse(self, filepath=None, cleaned=None):
        '''Parsing function which considers if we have parsed or not before.'''
        ### 0. Managing inputs
        self.cleaned = self.cleaned if cleaned is None else cleaned
        if cleaned:
            self.files['clean'] = filepath
        else:
            self.files['raw'] = filepath
        ## 1. Parsing task
        if not self.cleaned:
            ### parse files
            servicios = parse_servicios(filepath)
            ### Concat servicios
            servicios = concat_from_dict(servicios, 'Region')
        else:
            ### parse cleaned file
            servicios = pd.io.parsers.read_csv(filepath)
        ## 2. Tranforming
        servicios = self.categorize_cols(servicios)
        return servicios

    def write_servicios(self, servicios, filepath):
        '''Write function in order to save a cleaned dataframe in a file.'''
        #self.files['clean'] = join(path, filename)
        self.files['clean'] = filepath
        write_dataframe_to_csv(servicios, filepath)
        del servicios

    def filter_rows(self, servicios):
        '''Filter to only take into account the active companies in [06-12]'''
        if not self.cleaned:
            date = datetime.datetime.strptime('2006-01-01', '%Y-%m-%d')
            servicios, self.indices = filter_servicios(servicios, date)
            self.cleaned = True
        return servicios

    def categorize_cols(self, servicios):
        '''TO GENERALIZE'''
        servicios = cp2str(servicios)
        servicios = cnae2str(servicios)
        return servicios

    def get_index_from_cleaned(self, infilepath):
        ## 0. Managing inputs
        self.cleaned = False
        ## 1. Parsing
        files = os.listdir(infilepath)
        indices = []
        for f in files:
            servicios = parse_xlsx_sheet(join(infilepath, f))
            indices.append(servicios.index)
        self.indices = (files, indices)
        self.cleaned = True

    def parse_columns(self, filepath=None, columns=None, id_val=None):
        '''Parsing function which considers if we have parsed or not before.'''
        ### 0. Managing inputs
        ## 1. Parsing task
        if columns is None:
            servicios = parse_servicios(filepath)
        else:
            servicios, ids = parse_servicios_columns(filepath, columns, id_val)
        ### Concat servicios
        servicios = concat_from_dict(servicios, None)

        ## 2. Tranforming
        servicios = self.categorize_cols(servicios)
        return servicios
