
"""
"""

import os
import pandas as pd
import numpy as np

ca_names = ['Andalucia', 'Aragon', 'Asturias', 'Balears', 'Canarias',
            'Cantabria', 'CastillaLeon', 'CastillaMancha', 'Catalunya',
            'Ceuta_Melilla', 'Extremadura', 'Galicia', 'LaRioja_Navarra',
            'Madrid', 'Murcia', 'Pais Vasco', 'Valencia']
## WARNING: Temporal
ca_names = ['Balears', 'Canarias']
years_str = ['06', '07', '08', '09', '10', '11', '12']


def get_financial_data(pathdata):
    """"""
    pathfolder = os.path.join(pathdata, 'Cleaned/FirmsData/cleaned_temporal')
    files = os.listdir(pathfolder)

    finance, nifs, years = [], [], []
    for year in range(2006, 2013):
        year_str = str(year)[2:]
        assert(year_str in years_str)
        files_y = filter_files(files, year_str)

        finance_y = []
        for filename in files_y:
            finance_y.append(pd.read_csv(os.path.join(pathfolder, filename),
                             sep=';', index_col=0))
        finance_y = pd.concat(finance_y, axis=0)
        finance_y.indices = range(len(finance_y))
        nifs += list(finance_y['nif'])
        finance_cols = [c for c in finance_y.columns if c != 'nif']
        finance.append(finance_y[finance_cols].as_matrix())
        years.append(np.ones(len(finance_y)).astype(int)*year)
    finance = np.concatenate(finance, axis=0)
    years = np.concatenate(years)
    assert(len(finance) == len(years))
    assert(len(finance) == len(nifs))
    return finance, years, nifs


def get_sector_data(pathdata):
    """"""
    pathfolder = os.path.join(pathdata, 'Cleaned/FirmsData/sector')
    files = os.listdir(pathfolder)

    typefirms, nifs, years = [], [], []
    for year in range(2006, 2013):
        year_str = str(year)[2:]
        assert(year_str in years_str)
        files_y = filter_files(files, year_str)

        typefirms_y = []
        for filename in files_y:
            typefirms_y.append(pd.read_csv(os.path.join(pathfolder, filename),
                               sep=';', index_col=0))
        typefirms_y = pd.concat(typefirms_y, axis=0)
        typefirms_y.indices = range(len(typefirms_y))
        nifs += list(typefirms_y['nif'])
        type_cols = [c for c in typefirms_y.columns if c != 'nif']
        typefirms.append(typefirms_y[type_cols])
        years.append(np.ones(len(typefirms_y)).astype(int)*year)
    typefirms = pd.concat(typefirms, axis=0)
    years = np.concatenate(years)
    assert(len(typefirms) == len(years))
    assert(len(typefirms) == len(nifs))
    return typefirms, years, nifs


def get_regions(pathdata):
    """Get regions from pathdata."""
    pathfolder = os.path.join(pathdata, 'Cleaned/FirmsData/atemporal')
    filename = os.path.join(pathfolder, 'empresas')
    df = pd.read_csv(filename, sep=';', index_col=0)
    nifs = list(df['nif'])
    regions = df[['cp', 'ca']].as_matrix()
    return regions, nifs


def compute_regions(regions, columns=None):
    if columns is None:
        columns = range(regions.shape[1])
    if type(columns) == int:
        columns = [columns]
    return regions[:, columns]


def get_locations(pathdata):
    locations, years, nif = [], [], []
    for y in range(2006, 2013):
        locations_y = get_locations_by_year(pathdata, y)[0]
        loc_vars = [c for c in locations_y.columns if c != 'nif']
        nif += list(locations_y['nif'])
        years.append(np.ones(len(locations_y)).astype(int)*y)
        locations.append(locations_y[loc_vars].as_matrix())
    locations = np.concatenate(locations, axis=0)
    years = np.concatenate(years, axis=0)
    return locations, years, nif


def get_locations_by_year(pathdata, year=None):
    year = years_str if year is None else year
    year = [year] if type(year) == int else year
    pathfolder = os.path.join(pathdata, 'Cleaned/FirmsData/locations')
    files = os.listdir(pathfolder)

    locations = []
    for i in range(len(year)):
        y = str(year[i])[2:]
        assert(y in years_str)
        files_y = filter_files(files, y)
        locations_y = []
        for filename in files_y:
            locations_y.append(pd.read_csv(os.path.join(pathfolder, filename),
                                           sep=';', index_col=0))
        locations_y = pd.concat(locations_y, axis=0)
        locations.append(locations_y)
    return locations


def get_atemporal_data(pathdata):
    pathdata = os.path.join(pathdata, 'atemporal')
    empresas = pd.read_csv(os.path.join(pathdata, 'empresas'), sep=';',
                           index_col=0)
    return empresas


def get_sequencial_financial_data(pathdata):
    """Get sequencially all the data from the files."""

    for ca_name in ca_names:
        yield get_financial_by_ca(pathdata, ca_name), ca_name


def get_sequencial_financial_cleaneddata(pathdata):
    """Get sequencially all the data from the files."""

    for ca_name in ca_names:
        yield get_cleanedfinancial_by_ca(pathdata, ca_name), ca_name


def get_cleanedfinancial_by_ca(pathdata, ca, year=None):
    pathdata = os.path.join(pathdata, 'cleaned_temporal')
    ca = filter_ca_name(ca)
    year = years_str if year is None else year
    years = year if type(year) == list else [year]
    empresas = []
    for year in years:
        if type(year) == int:
            namefile = ca+'_'+str(year)[2:]+'.csv'
        else:
            namefile = ca+'_'+year+'.csv'
        empresas.append(pd.read_csv(os.path.join(pathdata, namefile), sep=';',
                                    index_col=0))
    return empresas


def get_financial_by_ca(pathdata, ca, year=None):
    pathdata = os.path.join(pathdata, 'temporal')
    ca = filter_ca_name(ca)
    year = years_str if year is None else year
    years = year if type(year) == list else [year]
    empresas = []
    for year in years:
        if type(year) == int:
            namefile = ca+'_'+str(year)[2:]+'.csv'
        else:
            namefile = ca+'_'+year+'.csv'
        empresas.append(pd.read_csv(os.path.join(pathdata, namefile), sep=';',
                                    index_col=0))
    return empresas


def filter_ca_name(ca):
    for ca_n in ca_names:
        logi = ca.strip().lower() in ca_n
        logi = logi or ca.strip() in ca_n
        logi = logi or ca in ca_n.strip()
        logi = logi or ca in ca_n.strip().lower()
        logi = ca_n.strip().lower() in ca
        logi = logi or ca_n.strip() in ca
        logi = logi or ca_n in ca.strip()
        logi = logi or ca_n in ca.strip().lower()
        if logi:
            return ca_n
    raise KeyError("Incorrect CA input.")


def filter_files(files, y):
    return [f for f in files if y in f]


###############################################################################
def write_financial(pathdata, empresas_temp, ca_name):
    """"""
    assert(len(empresas_temp) == len(years_str))
    pathfolder = os.path.join(pathdata, 'cleaned_temporal')
    for i in range(len(years_str)):
        namefile = ca_name+'_'+years_str[i]+'.csv'
        pathfile = os.path.join(pathfolder, namefile)
        empresas_temp[i].to_csv(pathfile, sep=';')
