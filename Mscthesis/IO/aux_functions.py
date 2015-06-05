
"""
Auxiliary functions for parsing and managing input data.
"""

import pandas as pd
import numpy as np
from os.path import join


def parse_xlsx_sheet(f, n=0):
    """Parse a sheet of a xlsx file."""
    xl_file = pd.ExcelFile(f)
    dfs = xl_file.parse(xl_file.sheet_names[n], na_values=['n.a.', 'n.d.'])
    return dfs


def concat_from_dict(d, keyvar):
    """Function to join all the dictionaries of the servicios data."""
    for e in d.keys():
        if keyvar is not None:
            aux = d[e]
            del d[e]
            Reg = pd.DataFrame(e, index=aux.index, columns=[keyvar])
            d[e] = pd.concat([aux, Reg], axis=1)
    d = pd.concat(list(d.values()))
    return d


def write_dataframe(d, name, path, extension):
    if extension in ['xlsx', 'xls']:
        write_dataframe_to_excel(d, name, path)
    elif extension == 'csv':
        write_dataframe_to_csv(d, name, path)


def write_dataframe_to_csv(d, name, path=''):
    """Function to write in csv the dataframes."""
    ns = name.split('.')
    name = name+'.csv' if len(ns) == 1 else ns[0]+'.csv'
    filepath = join(path, name)
    d.to_csv(filepath, encoding='utf-8', sep=';')


def write_dataframe_to_excel(d, name, path=''):
    """Function to write in csv the dataframes."""
    name = name if len(name.split()) == 1 else name
    filepath = join(path, name)
    d.to_excel(filepath)


def get_index_from_dict(d):
    """Function to get the indices from the dataframes."""
    d_ind = {}
    for e in d.keys():
        idxs = np.array(d[e].index)
        d_ind[e] = idxs
    return d_ind


def get_extension(filename):
    return filename.split('.')[-1]
