
"""
Clean module
============
This module contains all the functions required to format properly the data
from SABI dataset to be optimally used by this package.
The data is formated to the structure of folders that the package recognizes.

Folder structure ===========
============================
Parent_folder
    |-- Main
        |-- Servicios
    |-- Finantial
        |-- year
            |-- Servicios
        |-- ...
    |-- Aggregated
        |-- Agg_by_cp
============================

"""

import numpy as np
from itertools import product
import datetime
import pandas as pd

from os.path import exists, join, isfile
import os

from aux_functions import parse_xlsx_sheet, write_dataframe_to_excel


############################## GLOBAL VARIABLES ###############################
###############################################################################
types_legend = ['Activo', 'Activo Circulante', 'Pasivo Fijo', 'Pasivo liquido',
                'Trabajadores', 'Valor agregado', 'Valor ventas']
types = ['Act', 'ActC', 'Pasfijo', 'Pasliq', 'Trab', 'Va', 'Vtas']
years = [2006, 2007, 2008, 2009, 2010, 2011, 2012]
years_key = ['06', '07', '08', '09', '10', '11', '12']

## Column info transformation
main_cols = ['Nom', 'nif', 'cnae', 'cp', 'localidad', 'ES-X', 'ES-Y',
             'apertura', 'cierre']
mapping_manu = {'localitat': 'localidad', 'ESX': 'ES-X', 'ESY': 'ES-Y'}
types_m = ['Act', 'ActC', 'Pasfijo', 'Pasliq', 'Treb', 'Va', 'Vdes']
check_xlsx = lambda f: f[-5:] == '.xlsx'


def aggregate_by_mainvar(parentfolder, agg_var, loc_vars, type_var=None):
    """Function to aggregate variables by the selected variable considering a
    properly structured data.
    """
    ## Parse with class the structure and return the data
    filepath = join(parentfolder, 'Main')
    empresas = parse_empresas(filepath)
    # Concat info
    concatinfo = prepare_concatinfo()
    # Concat
    empresas = concat_empresas(empresas, concatinfo)
    ## Aggregation
    positions = average_position_by_cp(empresas, agg_var, loc_vars)
    if type_var is not None:
        types = counting_type_by_cp(empresas, agg_var, type_var)
        df = pd.concat([positions, types])
        cols = {'types': types.columns, 'positions': positions.columns}
    else:
        df = positions
        cols = {'positions': positions.columns}
    return df, cols


def clean(inpath, outpath):
    """Do the cleaning data from the raw initial data. It formats the data to a
    folder structure in which it is separated the main information of a company
    with the finantial information in order to save memory and read unnecessary
    information for some tasks.
    """
    ## 0. Ensure creation of needed folders
    if not exists(outpath):
        os.mkdir(outpath)
    if not exists(join(outpath, 'Main')):
        os.mkdir(join(outpath, 'Main'))
    if not exists(join(join(outpath, 'Main'), 'Servicios')):
        os.mkdir(join(join(outpath, 'Main'), 'Servicios'))
    if not exists(join(outpath, 'Finantial')):
        os.mkdir(join(outpath, 'Finantial'))
    folders = os.listdir(join(outpath, 'Finantial'))
    folders_years = [str(int(e)) for e in years]
    for f in folders_years:
        if f not in folders:
            os.mkdir(join(join(outpath, 'Finantial'), f))
        os.mkdir(join(join(join(outpath, 'Finantial'), f), 'Servicios'))
    ## Creation of the finantial cols
    aux = []
    for i in range(len(years_key)):
        aux.append([''.join(e) for e in product([years_key[i]], types)])
    finantial_cols = aux

    ## 1. Parse manufactures
    manufacturas = parse_xlsx_sheet(join(inpath, 'Manufactures.xlsx'))
    # Rename columns
    cols = manufacturas.columns
    newcolnames = clean_colnames_manu(cols)
    manufacturas.columns = newcolnames
    # Compute extra variables
    extra = compute_extra_cols(manufacturas)
    manufacturas = pd.concat([manufacturas, extra], axis=1)
    # Correct coordinates
    coords = ['ES-X', 'ES-Y']
    manufacturas[coords] = reformat_coordinates_manu(manufacturas, coords)
    # Separate and save
    name = 'Manufactures.xlsx'
    write_dataframe_to_excel(manufacturas[main_cols], name,
                             join(outpath, 'Main'))
    for i in range(len(finantial_cols)):
        y = folders_years[i]
        write_dataframe_to_excel(manufacturas[finantial_cols[i]], name,
                                 join(join(outpath, 'Finantial'), y))
    del manufacturas

    ## 1. Parse servicios
    onlyfiles = [f for f in os.listdir(inpath)
                 if isfile(join(inpath, f)) and check_xlsx(f)]
    for f in onlyfiles:
        # parse servicios
        servicios = parse_xlsx_sheet(join(join(inpath, 'SERVICIOS'), f))
        # Rename columns
        cols = servicios.columns
        newcolnames = clean_colnames_servi(cols)
        servicios.columns = newcolnames
        # Compute extra variables
        apertura = obtain_open_aperture_date(servicios)
        servicios['apertura'] = apertura
        # Separate and save
        write_dataframe_to_excel(servicios[main_cols], f,
                                 join(join(outpath, 'Main'), 'Servicios'))
        # Write servicios
        path_fin = join(outpath, 'Finantial')
        for i in range(len(finantial_cols)):
            y = folders_years[i]
            write_dataframe_to_excel(servicios[finantial_cols[i]], f,
                                     join(join(path_fin, y), 'Servicios'))
    pass


def clean_colnames_manu(cols):
    "Clean names of the manufactures."
    # Format properly
    cols = [e.strip() for e in cols]
    # Replace the finantial variables
    cols_f = ['y'+''.join(e) for e in product(years_key, types_m)]
    cols_f_g = [''.join(e) for e in product(years_key, types)]
    replace_f = dict(zip(cols_f, cols_f_g))
    cols = replace_colnames(cols, replace_f)
    # Replace the main
    cols = replace_colnames(cols, mapping_manu)
    return cols


def clean_colnames_servi(cols):
    "Clean names of the servicios."
    # Format properly
    cols = [e.strip() for e in cols]
    return cols


def reformat_coordinates_manu(df, coord_var):
    """Divide the coordinates for 10^6 in order to get the correct
    dimensionality."""
    aux = df[coord_var].as_matrix()/10**6
    return aux


def compute_extra_cols(df):
    ## Compute aperture date
    bool_null = np.array(df['constituc'].isnull())
    aux1 = obtain_open_aperture_date(df.loc[bool_null, :])
    f = lambda x: datetime.datetime(int(x), 1, 1)
    aux2 = np.zeros(np.logical_not(bool_null).sum()).astype(datetime.datetime)
    idxs = np.nonzero(np.logical_not(bool_null))[0]
    for i in xrange(idxs.shape[0]):
        aux2[i] = f(int(df.loc[idxs[i], 'constituc']))
    aux = pd.DataFrame(columns=['apertura'], index=df.index)
    aux.loc[bool_null, 'apertura'] = aux1
    aux.loc[np.logical_not(bool_null), 'apertura'] = aux2
    ## Compute close date
    bool_act = np.array(df.loc[:, 'estat'] == 'Activa')
    bool_2012 = np.array(df.loc[:, 'tancament']) == 2012
    cierre = np.zeros(bool_2012.shape)
    cierre = np.array(df.loc[:, 'tancament'])
    cierre[np.logical_and(bool_act, bool_2012)] = 2013
    cierre_aux = np.zeros(cierre.shape[0]).astype(datetime.datetime)
    for i in xrange(cierre.shape[0]):
        cierre_aux[i] = f(cierre[i])
    cierre = pd.DataFrame(cierre_aux, columns=['cierre'], index=aux.index)
    ## Concat
    extras = pd.concat([aux, cierre], axis=1)
    return extras


def replace_colnames(cols, replaces):
    "Replace the names keeping the order in the list of colnames."
    for c in cols:
        if c in replaces.keys():
            cols[cols.index(c)] = replaces[c]
    return cols


def obtain_open_aperture_date(df):
    "Obtain the date of aperture of the each company."

    m_y = len(years)
    ## Obtain bool arrays
    bool_m = np.zeros((df.shape[0], m_y)).astype(bool)
    for i in range(m_y):
        bool_m[:, i] = check_year_open(df, years[i])

    ## Obtain date
    dates = np.zeros(bool_m.shape[0])
    for i in range(m_y):
        logi = bool_m[:, i]
        dates[np.logical_and(dates == 0, logi)] = i+1
    ## Format dates
    dates = dates + years[0]-1
    dates = dates.astype(int)
    aux = np.zeros(dates.shape).astype(datetime.datetime)
    for i in range(aux.shape[0]):
        aux[i] = datetime.datetime(int(dates[i]), 1, 1)
    dates = aux
    return dates


def check_year_open(df, year):
    """Function to check if there is any variables not none to check if there
    was opened the selected year.
    """
    i = years.index(year)
    year_key = [years_key[i]]
    comb = product(year_key, types)
    comb = [''.join(e) for e in comb]

    logis = np.logical_not(df[comb].isnull().as_matrix())
    m = logis.shape[1]

    logi = np.zeros(logis.shape[0]).astype(bool)
    for i in range(m):
        logi = np.logical_or(logi, logis[:, i])
    return logi
