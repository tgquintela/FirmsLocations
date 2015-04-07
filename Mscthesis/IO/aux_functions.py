
"""
Auxiliary functions for parsing and managing input data.
"""

import pandas as pd
from os.path import join


def parse_xlsx_sheet(f, n=0):
    """Parse a sheet of a xlsx file."""
    xl_file = pd.ExcelFile(f)
    dfs = xl_file.parse(xl_file.sheet_names[n], na_values=['n.a.', 'n.d.'])
    return dfs


def concat_from_dict(d, keyvar):
    """Function to join all the dictionaries of the servicios data."""
    for e in d.keys():
        m = d[e].shape[0]
        if keyvar is not None:
            Reg = pd.DataFrame([e]*m, columns=[keyvar])
            d[e] = pd.concat([d[e], Reg], axis=1)
    d = pd.concat(list(d.values()))
    return d


def write_dataframe_to_csv(d, name, path=''):
    """Function to write in csv the dataframes."""
    name = name+'.csv' if len(name.split()) == 1 else name
    filepath = join(path, name)
    d.to_csv(filepath, encoding='utf-8')


def write_dataframe_to_excel(d, name, path=''):
    """Function to write in csv the dataframes."""
    name = name if len(name.split()) == 1 else name
    filepath = join(path, name)
    d.to_excel(filepath)
