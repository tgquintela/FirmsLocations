

"""
Module to preprocess the data and filter unuseful cols or reformat the data in
column-wise way.
"""

import numpy as np
import pandas as pd
import os
import datetime
from itertools import product


############################## GLOBAL VARIABLES ###############################
###############################################################################
types = ['Act', 'ActC', 'Pasfijo', 'Pasliq', 'Trab', 'Va', 'Vtas']
years = [2006, 2007, 2008, 2009, 2010, 2011, 2012]
years_key = ['06', '07', '08', '09', '10', '11', '12']
## Column info transformation
main_cols = ['Nom', 'nif', 'cnae', 'cp', 'localidad', 'ES-X', 'ES-Y',
             'apertura', 'cierre']
mapping_manu = {'localitat': 'localidad', 'ESX': 'ES-X', 'ESY': 'ES-Y',
                'esx': 'es-x', 'esy': 'es-y'}
types_m = ['Act', 'ActC', 'Pasfijo', 'Pasliq', 'Treb', 'Va', 'Vdes']
# Variables to store
vars_atemporal = ['nom', 'nif', 'cp', 'ca', 'es-x', 'es-y', 'sector', 'cnae',
                  'apertura', 'cierre']
#vars_temporal = product(year_key, types)
vars_atemporal2 = ['nif', 'cp', 'ca', 'es-x', 'es-y', 'sector', 'cnae',
                   'apertura', 'cierre']


###############################################################################
############################### JOINING EMPRESAS ##############################
###############################################################################
def join_empresas_atemporal(servicios, manufactures, ca_name):
    """"""
    manu_ca = manufactures[manufactures['ca'].apply(lambda x: x == ca_name)]
    empresas =\
        pd.concat([servicios[vars_atemporal], manu_ca[vars_atemporal]], axis=0)
    return empresas


def join_and_store_empresas_atemporal(empresas, pathdata):
    namefile = 'empresas'
    pathfold = os.path.join(pathdata, 'atemporal')
    pathfile = os.path.join(pathfold, namefile)
    empresas = pd.concat(empresas, axis=0)
    ## Reindices
    empresas.indices = range(len(empresas))
    ## Cp correct
    f_cp = lambda x: (5-len(str(int(x))))*'0'+str(int(x))
    empresas.loc[:, 'cp'] = empresas['cp'].apply(f_cp)
#    ##### DEBUG: temporal
#    try:
#        empresas.loc[:, 'cp'] = empresas['cp'].apply(f_cp)
#    except:
#        for i in range(len(empresas)):
#            f_cp(empresas.loc[i, 'cp'])
#    ################################
    ## Apertura y cierre
    empresas.loc[:, 'apertura'] =\
        empresas['apertura'].apply(lambda x: x.strftime('%F'))
    empresas.loc[:, 'cierre'] =\
        empresas['cierre'].apply(lambda x: x.strftime('%F'))
    ## Store
    empresas[vars_atemporal2].to_csv(pathfile, sep=';')
    empresas[['nom', 'nif']].to_excel(os.path.join(pathfold, 'empresas.xlsx'))


def store_empresas_atemporal_years(empresas, ca_name, pathdata):
    """"""
    pathfold_locs = os.path.join(pathdata, 'locations')
    pathfold_sector = os.path.join(pathdata, 'sector')

    apertura = empresas['apertura'].apply(lambda x: x.strftime('%F'))
    apertura = apertura.apply(lambda x: int(x[:4])).as_matrix()
    cierre = empresas['cierre'].apply(lambda x: x.strftime('%F'))
    cierre = cierre.apply(lambda x: int(x[:4])).as_matrix()

    for i in range(len(years)):
        logi = np.logical_and(years[i] <= cierre, years[i] >= apertura)
        locs_year = empresas[['nif', 'es-x', 'es-y']][logi]
        sector_year = empresas[['nif', 'sector', 'cnae']][logi]
        locs_year.indices = range(len(locs_year))
        sector_year.indices = range(len(sector_year))

        namefile = ca_name+'_'+years_key[i]+'.csv'
        locs_year.to_csv(os.path.join(pathfold_locs, namefile), sep=';')
        sector_year.to_csv(os.path.join(pathfold_sector, namefile), sep=';')


def join_and_store_empresas_temporal(servicios, manufactures, ca_name,
                                     pathdata):
    logi = manufactures['ca'].apply(lambda x: x == ca_name).as_matrix()
    manu_ca = manufactures[logi]
    for year_key in years_key:
        vars_temporal_year = [year_key+type_.lower() for type_ in types]
        servicios_year = servicios[['nif']+vars_temporal_year]
        manu_ca_year = manu_ca[['nif']+vars_temporal_year]
        servicios_year, manu_ca_year, collapsed =\
            filter_unique_nif(servicios_year, manu_ca_year)
        empresas = pd.concat([servicios_year, manu_ca_year, collapsed], axis=0)
        namefile = ca_name+'_'+year_key+'.csv'
        pathfile = os.path.join(os.path.join(pathdata, 'temporal'), namefile)
        logi = check_year_open(empresas, year_key)
        empresas.columns = ['nif'] + types
        empresas[logi].to_csv(pathfile, sep=';')


###############################################################################
############################### DATES FORMATTING ##############################
###############################################################################
def compute_apertura_cierre(df):
    apertura = obtain_open_aperture_date(df)
    cierre = obtain_close_date(df)
    if 'apertura' in df.columns:
        del df['apertura']
    if 'cierre' in df.columns:
        del df['cierre']
    df.index = range(len(df))
    df = pd.concat([df, pd.DataFrame({'apertura': apertura}),
                    pd.DataFrame({'cierre': cierre})], axis=1)
#    df.loc[:, 'apertura'] = apertura
#    df.loc[:, 'cierre'] = cierre
    return df


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
    aux = np.zeros(dates.shape).astype(datetime.date)
    for i in range(aux.shape[0]):
        aux[i] = datetime.date(int(dates[i]), 1, 1)
    dates = aux
    return dates


def obtain_close_date(df):
    "Obtain close date"
    m_y = len(years)
    ## Obtain bool arrays
    bool_m = np.zeros((df.shape[0], m_y)).astype(bool)
    for i in range(m_y):
        bool_m[:, i] = check_year_open(df, years[i])

    ## Obtain date
    dates = np.zeros(bool_m.shape[0])
    for i in range(m_y):
        logi = bool_m[:, i]
        dates[logi] = i

    ## Format dates
    dates = dates + years[0]
    dates = dates.astype(int)
    aux = np.zeros(dates.shape).astype(datetime.date)
    for i in range(aux.shape[0]):
        aux[i] = datetime.date(int(dates[i]), 12, 31)
    dates = aux
    return dates


def check_year_open(df, year):
    """Function to check if there is any variables not none to check if there
    was opened the selected year.
    """
    if type(year) == int:
        i = years.index(year)
    else:
        assert(year in years_key)
        i = years_key.index(year)
    year_key = [years_key[i]]
    comb = product(year_key, types)
    comb = [''.join(e).lower() for e in comb]

    logis = np.logical_not(df[comb].isnull().as_matrix())
    m = logis.shape[1]

    logi = np.zeros(logis.shape[0]).astype(bool)
    for i in range(m):
        logi = np.logical_or(logi, logis[:, i])
    return logi


###############################################################################
############################# CREATE EXTRA COLUMNS ############################
###############################################################################
def create_CA_column(empresas, ca_cp_dict):
    def f(x):
        try:
            return ca_cp_dict[(5-len(str(int(x))))*'0'+str(int(x))]
        except:
            return ca_cp_dict[((2-len(str(int(x))))*'0'+str(int(x)))[:2]]
#    f = lambda x: ca_cp_dict[(5-len(str(int(x))))*'0'+str(int(x))]
    empresas['ca'] = empresas['cp'].apply(f)
    return empresas


def create_sector_columns(empresas, sector):
    sector = sector.lower().strip()
    assert(sector in ['manufactures', 'servicios'])
    empresas.loc[:, 'sector'] = sector
    return empresas


###############################################################################
############################ COLUMNS STANDARIZATION ###########################
###############################################################################
def clean_colnames_manu(cols):
    "Clean names of the manufactures."
    # Format properly
    cols = [e.strip() for e in cols]
    # Replace the Financial variables
    cols_f = ['y'+''.join(e) for e in product(years_key, types_m)]
    cols_f += ['y'+''.join(e).strip().lower()
               for e in product(years_key, types_m)]
    cols_f_g = [''.join(e).lower().strip() for e in product(years_key, types)]
    replace_f = dict(zip(cols_f, 2*cols_f_g))
    cols = replace_colnames(cols, replace_f)
    # Replace the main
    cols = replace_colnames(cols, mapping_manu)
    return cols


def replace_colnames(cols, replaces):
    "Replace the names keeping the order in the list of colnames."
    for c in cols:
        if c in replaces.keys():
            cols[cols.index(c)] = replaces[c]
    return cols


###############################################################################
#################################### OTHERS ###################################
###############################################################################
def filter_unique_nif(servicios, manufactures):
    nif_servicios, nif_manu = list(servicios['nif']), list(manufactures['nif'])
    assert(len(nif_servicios) == len(set(nif_servicios)))
    assert(len(nif_manu) == len(set(nif_manu)))
    cols_serv = [c for c in servicios.columns if c != 'nif']
    cols_manu = [c for c in manufactures.columns if c != 'nif']
    ncols = len(servicios.columns)
    logi_serv, logi_manu, new_rows = [], [], []
    for i in range(len(nif_servicios)):
        if nif_servicios[i] in nif_manu:
            j = nif_manu.index(nif_servicios[i])
            print i, j, cols_serv, cols_manu
            print servicios.iloc[i, range(1, ncols)].as_matrix(), manufactures.iloc[j, range(1, ncols)].as_matrix()
            logi_serv.append(i)
            logi_manu.append(j)
            fin = collapse_finance(servicios.iloc[i,  range(1, ncols)],
                                   manufactures.iloc[j,  range(1, ncols)])
            new_rows.append([nif_servicios[i]]+list(fin))

    new_rows = pd.DataFrame(new_rows, columns=servicios.columns)
    logi_serv = [i not in logi_serv for i in range(len(servicios))]
    logi_manu = [i not in logi_manu for i in range(len(manufactures))]
    servicios = servicios[logi_serv]
    manufactures = manufactures[logi_manu]
    return servicios, manufactures, new_rows


def collapse_finance(servicios, manufactures):
    f_corr = lambda x: np.logical_not(np.logical_or(np.isnan(x), x == 0))
    servicios = np.array(servicios).astype(float)
    manufactures = np.array(manufactures).astype(float)
    print servicios, manufactures, type(servicios)
    corr_serv = f_corr(servicios)
    collapsed = []
    for i in range(len(servicios)):
        if corr_serv[i]:
            collapsed.append(servicios[i])
        else:
            collapsed.append(manufactures[i])
    collapsed = np.array(collapsed)
    return collapsed


def collapse_pfeatures_nif(pfeatures):
    if len(pfeatures.shape) == 1:
        return pfeatures
    f_corr = lambda x: np.logical_not(np.logical_or(np.isnan(x), x == 0))
    correctness = f_corr(pfeatures)
    new_pfeatures = []
    for col in range(pfeatures.shape[1]):
        if correctness[:, col].sum():
            i = np.where(correctness[:, col])[0][0]
        else:
            i = 0
        new_pfeatures.append(pfeatures[i, col])
    new_pfeatures = np.array(new_pfeatures)
    return new_pfeatures



def filtercols_empresas(empresas, filtercolsinfo):
    "TODO:"
    return empresas


def categorize_cols(df):
    df = cp2str(df)
    df = cnae2str(df)
    return df


def generate_replace(type_vals):
    "Generate the replace for use indices and save memory."
    repl = {}
    for v in type_vals.keys():
        repl[v] = dict(zip(type_vals[v], range(len(type_vals[v]))))
    return repl


def transform_cnae_col(cnae_col, lvl):
    """"""
    lvl_n = len(cnae_col[1])
    if lvl >= lvl_n:
        return cnae_col
    else:
        return cnae_col.apply(lambda x: x[:lvl])


############################# Particular columns ##############################
###############################################################################
def cp2str(df):
    """Retransform cp to string."""
    def cp2str_ind(x):
        try:
            x = str(int(x))
            x = (5-len(x))*'0'+x
        except:
            pass
        return x
    if 'cp' in df.columns:
        df.loc[:, 'cp'] = df['cp'].apply(cp2str_ind)
    return df


def cnae2str(df):
    """Transforming cnae code to string."""
    def cnae2str_ind(x):
        try:
            x = str(int(x))
        except:
            pass
        return x
    if 'cnae' in df.columns:
        df.loc[:, 'cnae'] = df['cnae'].apply(cnae2str_ind)
    return df


def to_float(df):
    ## Columns which has to be numbers
    cols = ['06Act', '07Act', '08Act', '09Act', '10Act', '11Act', '12Act',
            '13Act', '06ActC', '07ActC', '08ActC', '09ActC', '10ActC',
            '11ActC', '12ActC', '13ActC', '06Pasfijo', '07Pasfijo',
            '08Pasfijo', '09Pasfijo', '10Pasfijo', '11Pasfijo', '12Pasfijo',
            '13Pasfijo', '06Pasliq', '07Pasliq', '08Pasliq', '09Pasliq',
            '10Pasliq', '11Pasliq', '12Pasliq', '13Pasliq', '06Va', '07Va',
            '08Va', '09Va', '10Va', '11Va', '12Va', '13Va', '06Vtas', '07Vtas',
            '08Vtas', '09Vtas', '10Vtas', '11Vtas', '12Vtas', '13Vtas']
    ## Transformation
    columns = df.columns
    for col in columns:
        if col in cols:
            df.loc[:, col] = df[col]
    return df


def to_int(df):
    cols = ['06Trab', '07Trab', '08Trab', '09Trab', '10Trab', '11Trab',
            '12Trab', '13Trab']
    ## Transformation
    columns = df.columns
    for col in columns:
        if col in cols:
            df.loc[:, col] = df[col].astype(int)
    return df
