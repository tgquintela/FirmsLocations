
"""
Module which the main functions to parse the data.
"""

from os import listdir
from os.path import isfile, join
from aux_functions import parse_xlsx_sheet, get_extension
import numpy as np
import pandas as pd
import datetime


################################# PARSE DATA ##################################
###############################################################################
def parse_empresas(filepath):
    """Parse function in order to parse empresas giving the parent path."""
    ### Parse servicios
    servicios = parse_servicios(join(filepath, 'Servicios'))
    ### Parse manufactures
    manufactures = parse_manufactures(filepath)
    ## Join in a dict
    empresas = {'manufacturas': manufactures, 'servicios': servicios}
    return empresas


def parse_servicios(mypath):
    "Parse function for servicios."
    ## Prepare inputs and needed variables and functions
    # Useful functions

    check_xlsx = lambda f: f[-5:] == '.xlsx'
    check_csv = lambda f: get_extension(f) == 'csv'
    get_name = lambda f: f.split('.')[0]

    ## Compute
    # Obtain list of names
    onlyfiles = [f for f in listdir(mypath)
                 if isfile(join(mypath, f)) and check_xlsx(f)]
    servicios_l = [(get_name(f), parse_xlsx_sheet(join(mypath, f)))
                   for f in onlyfiles]
    onlyfiles = [f for f in listdir(mypath)
                 if isfile(join(mypath, f)) and check_csv(f)]
    servicios_l = [(get_name(f), pd.read_csv(join(mypath, f), sep=';',
                                             index_col=0))
                   for f in onlyfiles] + servicios_l
    servicios = dict(servicios_l)
    return servicios


def parse_servicios_columns(mypath, columns, id_val):
    # Useful functions
    check_xlsx = lambda f: f[-5:] == '.xlsx'
    get_name = lambda f: f[:-5]
    ## Parse
    onlyfiles = [f for f in listdir(mypath)
                 if isfile(join(mypath, f)) and check_xlsx(f)]
    servicios = []
    ids = []
    names = []
    for f in onlyfiles:
        names.append(get_name(f))
        servicios.append(parse_xlsx_sheet(join(mypath, f))[columns])
        if id_val is None:
            ids.append(np.array(servicios.index))
        else:
            if type(id_val) == list:
                ids.append(parse_xlsx_sheet(join(mypath, f))[id_val])
            else:
                ids.append(parse_xlsx_sheet(join(mypath, f))[[id_val]])

    servicios = dict(names, servicios)
    ids = dict(names, ids)

    return servicios, ids


def parse_manufactures(mypath):
    if isfile(join(mypath, "Manufactures.xlsx")):
        manufactures = parse_xlsx_sheet(join(mypath, "Manufactures.xlsx"))
        f = lambda x: datetime.date(x, 12, 31)
        manufactures.loc[:, 'cierre'] = manufactures['cierre'].apply(f)
    elif isfile(join(mypath, "Manufactures.csv")):
        manufactures = pd.read_csv(join(mypath, "Manufactures.csv"), sep=';',
                                   index_col=0)
    return manufactures


def parse_finantial_by_year(parentpath, year):
    "Parse the finantial data with the years considered."

    filepath = join(parentpath, 'Finantial')
    if year in [2006, 2007, 2008, 2009, 2010, 2011, 2012]:
        year = str(int(year))
    filepath = join(filepath, year)
    finantial = parse_empresas(filepath)

    return finantial


############################# PARSE LEGEND FILES ##############################
###############################################################################
def parse_cnae(mypath=None):
    if mypath is None:
        mypath = 'Data/raw_data/'
    filename = "cnae93rev1_clean.xls"
    legend_sabi_cnae = parse_xlsx_sheet(join(mypath, filename))
    return legend_sabi_cnae


def parse_legend_services(mypath=None):
    if mypath is None:
        mypath = 'Data/raw_data/'
    filename = "Servicios_Leyenda.xlsx"
    legend_services = parse_xlsx_sheet(join(mypath, filename))
    return legend_services


############################# PARSE INSTRUCTIONS ##############################
###############################################################################
def parse_instructions_file(fileinstructions):
    """Parse the file of instructions about variables."""

    ## 0. Needed functions
    def integerization(x):
        try:
            return int(x)
        except:
            return x
    f_vars = lambda x: [e.strip() for e in x.split(',')]
    f_vars_co = lambda x: f_vars(x) if len(f_vars(x)) > 1 else f_vars(x)[0]

    def f_str_nan(x):
        if type(x) == str:
            return x
        else:
            aux = '' if np.isnan(x) else x
            return aux

    ## 1. Read file
    describ_info = pd.read_csv(fileinstructions, ';')
    ## 2. Format columns
    describ_info['variables'] = describ_info['variables'].apply(f_vars_co)
    describ_info['n_bins'] = describ_info['n_bins'].apply(integerization)
    aux = describ_info['variables_name'].apply(f_str_nan)
    describ_info['variables_name'] = aux
    describ_info['agg_time'] = describ_info['agg_time'].apply(f_str_nan)
    describ_info['Description'] = describ_info['Description'].apply(f_str_nan)

    return describ_info


######## TODO
def parse_and_clean_raw():
    "Parse the raw data and transform it to clean data."
    pass
