
"""
Auxiliar standarization functions

"""

import pandas as pd
import numpy as np
import os

from ..IO.aux_functions import parse_xlsx_sheet
from ..Preprocess.geo_filters import check_correct_spain_coord

extra_folder = 'extra'
servicios_columns = ['nom', 'nif', 'cp', 'cnae', 'localidad', 'x', 'y', 'es-x',
                     'es-y', 'holding', 'cierre',
                     '06act', '07act', '08act', '09act', '10act', '11act',
                     '12act', '13act', '06actc', '07actc', '08actc', '09actc',
                     '10actc', '11actc', '12actc', '13actc', '06pasfijo',
                     '07pasfijo', '08pasfijo', '09pasfijo', '10pasfijo',
                     '11pasfijo', '12pasfijo', '13pasfijo', '06pasliq',
                     '07pasliq', '08pasliq', '09pasliq', '10pasliq',
                     '11pasliq', '12pasliq', '13pasliq', '06trab', '07trab',
                     '08trab', '09trab', '10trab', '11trab', '12trab',
                     '13trab', '06va', '07va', '08va', '09va', '10va', '11va',
                     '12va', '13va', '06vtas', '07vtas', '08vtas', '09vtas',
                     '10vtas', '11vtas', '12vtas', '13vtas']


def pre_read_servicios(pathdata):
    raw_locs_serv, nifs_serv, cps_serv, names_serv = [], [], [], []
    muni_serv, null_cp, null_muni, null_locs = [], [], [], []
    ca_cp_dict = {}

    for df, ca_name in get_sequencial_servicios(pathdata):
        names_serv += list(df['nom'])
        nifs_serv += list(df['nif'])
        cps_serv += list(df['cp'])
        raw_locs_serv.append(df[['es-x', 'es-y']].as_matrix())

        # Dictionary cp_ca creation
        u_cp = df['cp'][np.logical_not(pd.isnull(df['cp']))].unique()
        u_cp = [e for e in u_cp if e != float('nan')]
        u_cp = [(5-len(str(int(e))))*'0'+str(int(e)) for e in u_cp]
        for cp in u_cp:
            ca_cp_dict[cp] = ca_name
        u_cp = df['cp'][np.logical_not(pd.isnull(df['cp']))].unique()
        u_cp = [((2-len(str(int(e))))*'0'+str(int(e)))[:2] for e in u_cp]
        for cp in u_cp:
            ca_cp_dict[cp] = ca_name
        null_cp.append(pd.isnull(df['cp']).as_matrix())
        muni_serv += list(df['localidad'])
        null_muni.append(pd.isnull(df['localidad']).as_matrix())
        null_loc =\
            np.logical_not(check_correct_spain_coord(df[['es-x', 'es-y']]))
        null_locs.append(null_loc)

    raw_locs_serv = np.concatenate(raw_locs_serv)

    return ca_cp_dict, raw_locs_serv, nifs_serv, cps_serv, names_serv,\
        muni_serv, null_cp, null_muni, null_locs


def read_manufactures(pathdata):
    filepath = os.path.join(pathdata, 'Manufactures.xlsx')
    manufactures = parse_xlsx_sheet(filepath)
    manufactures.columns = [e.lower().strip() for e in manufactures.columns]
    raw_locs_manu = manufactures[['esx', 'esy']].as_matrix()
    nifs_manu = list(manufactures['nif'])
    cps_manu = list(manufactures['cp'])
    names_manu = list(manufactures['nom'])
    muni_manu = list(manufactures['localitat'])
    ## Nulls computation
    null_cp = pd.isnull(manufactures['cp']).as_matrix()
    null_muni = pd.isnull(manufactures['localitat'].as_matrix())
    manufactures[['esx', 'esy']] = manufactures[['esx', 'esy']]/1000000.
    null_locs = check_correct_spain_coord(manufactures[['esx', 'esy']])
    null_locs = np.logical_not(null_locs)
    return manufactures, raw_locs_manu, nifs_manu, cps_manu, names_manu,\
        muni_manu, null_cp, null_muni, null_locs


def get_sequencial_servicios(pathdata):
    listdirs = os.listdir(os.path.join(pathdata, 'SERVICIOS'))
    listxlsx = []
    for file_ in listdirs:
        sp = file_.split('.')
        if len(sp) > 1:
            if sp[-1] == 'xlsx':
                listxlsx.append(file_)
    for file_ in listxlsx:
        filepath = os.path.join(os.path.join(pathdata, 'SERVICIOS'), file_)
        df = parse_xlsx_sheet(filepath)
#        df.columns = [e.lower().strip() for e in df.columns]
        df.columns = servicios_columns
        ca_name = file_.split('.')[0]
        yield df, ca_name


def cp_fillna(df):
    """Hierharchy of imputing information CP.

    1. Closer location imputing by using 'Localitat'
    2. Closer location imputing (null 'Localitat')
    3. Random cp assignation using 'Localitat' (null location)
    4. Discarted (wrong and useless data)

    """
    return 
