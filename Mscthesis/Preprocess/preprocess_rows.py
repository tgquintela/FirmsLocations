
"""
De moment el primer sera fer un preproces de les dades, posant-les en un format
amb el que puguem treballar be, eliminant les empreses que no tenen cap
activitat en els anys 2006-2012 (tancament en anys anteriors), etc.
I el segon sera una analisi estadistica per saber quantes empreses tenim de
cada tipus, anys d'activitat, etc.
"""

"""
Module to preprocess the data and filter unuseful rows.
"""


import numpy as np
import datetime
from Mscthesis.Geo_tools.geo_filters import filter_uncorrect_coord_spain,\
    filter_bool_uncorrect_coord_spain


def filter_bool_dict(empresas, bool_arrays):
    "This function filter and dictionary of data."
    ## Loop over the dictionary
    for e1 in empresas.keys():
        if type(empresas[e1]) == dict:
            empresas[e1] = filter_bool_dict(empresas[e1], bool_arrays[e1])
        else:
            empresas[e1] = empresas[e1].loc[bool_arrays[e1], :]

    return empresas


def retrieve_empresas_dict(d_data, coord_filter_nfo, date_filter_nfo):
    """This function for retrieving a dictionary of dataand return the boolean
    array.
    """
    bool_arrays = {}
    bool_arrays['manufacturas'] = retrieve_empresas(d_data['manufacturas'],
                                                    coord_filter_nfo,
                                                    date_filter_nfo)
    bool_arrays['servicios'] = retrieve_empresas(d_data['servicios'],
                                                 coord_filter_nfo,
                                                 date_filter_nfo)
    return bool_arrays


def retrieve_empresas(empresas, coord_filter_nfo, date_filter_nfo):
    """General retrieve function.
    """
    if type(empresas) == dict:
        bool_arrays = {}
        for emp in empresas.keys():
            bool_arrays[emp] = retrieve_empresas_d(empresas[emp],
                                                   date_filter_nfo,
                                                   coord_filter_nfo)
    else:
        bool_arrays = retrieve_empresas_d(empresas, date_filter_nfo,
                                          coord_filter_nfo)

    return bool_arrays


def retrieve_empresas_d(empresas, date_filter_nfo, coord_filter_nfo):
    "Retrieve companies which match the conditions given in the parameters."
    bool_dates = retrieve_by_dates(empresas, **date_filter_nfo)
    bool_coord = retrieve_coordinates(empresas, **coord_filter_nfo)

    bool_arrays = np.logical_and(bool_dates, bool_coord)
    return bool_arrays


def filter_servicios(servicios, date, coord_vars=[]):
    """This function performs the filtering of the companies which do not have
    acitivity the years of the study.
    ----------
    DEPRECATED
    ----------
    """
    #servicios = filter_by_activity(servicios)
    servicios = filter_by_date(servicios, date)
    servicios = filter_by_spaincoordinates(servicios, coord_vars)
    indices = servicios.index
    # Transform special columns
    servicios = cp2str(servicios)
    servicios = cnae2str(servicios)
    return servicios, indices


def filter_servicios_dict(d, date, coord_vars):
    """Filter the servicios.
    """
    for e in d.keys():
        aux = d[e]
        del d[e]
        d[e], _ = filter_servicios(aux, date, coord_vars)
    return d


############################### Filter activity ###############################
###############################################################################
def filter_by_activity(servicios):
    """This function filter the rows by activity.
    ----------
    DEPRECATED
    ----------
    """
    indices_0 = np.array(servicios.index)
    total_activo = ['06Act', '07Act', '08Act', '09Act', '10Act', '11Act',
                    '12Act']
    indices = dropna_rows(servicios, total_activo)
    indices_b = list([ind for ind in indices_0 if ind not in indices])
    #servicios = servicios.loc[indices]
    servicios.drop(indices_b)
    return servicios


################################ Filter dates #################################
###############################################################################
def retrieve_by_dates(empresas, method, date_filter_nfo):
    """Main function to retrieve by dates. It acts as a switcher function to
    all the functions.
    """

    if method == 'by_years':
        bool_dates = retrieve_by_years(empresas, **date_filter_nfo)
    elif method == 'by_interval_years':
        bool_dates = retrieve_by_intervaldates(empresas, **date_filter_nfo)

    return bool_dates


def retrieve_by_year(servicios, year):
    "Retrieve the companies which were opened the year considered."
    format_date = '%Y-%m-%d'
    to_date = lambda x: datetime.datetime.strptime(x, format_date).date()
    to_format = lambda x: x.split(' ')[0]
    apertura = servicios['apertura'].apply(to_date)
    cierre = servicios['cierre'].apply(to_format).apply(to_date)
    bool_0 = apertura <= datetime.date(year, 12, 31)
    bool_1 = cierre >= datetime.date(year, 01, 01)
    bool_year = np.logical_and(bool_0, bool_1)

    return bool_year


def retrieve_by_years(servicios, years, all_y=True):
    "Retrieve the companies which were opened along the years considered."

    ## 0. Format inputs
    if type(years) not in [np.ndarray, list]:
        years = np.array([years])
    else:
        years = np.array(years)
    n = servicios.shape[0]

    ## 1. Obtain bool arrays
    bool_years = np.ones(n) if all_y else np.zeros(n)
    for i in range(years.shape[0]):
        bool_aux = retrieve_by_year(servicios, years[i])
        if all_y:
            bool_years = np.logical_and(bool_years, bool_aux)
        else:
            bool_years = np.logical_or(bool_years, bool_aux)

    return bool_years


def retrieve_by_intervaldates(servicios, years):
    """Retrieve the companies which were opened between the years considered.
    It could be done with the retrieve_by_years function and with all_y=False.
    """

    ## 0. Format inputs
    if type(years) not in [np.ndarray, list]:
        years = np.array([years])
    else:
        years = np.array(years)

    format_date = '%Y-%m-%d'
    to_date = lambda x: datetime.datetime.strptime(x, format_date).date()
    apertura = servicios['apertura'].apply(to_date)
    cierre = servicios['cierre'].apply(to_date)

    ## Creation of the bool arrays
    bool_0 = apertura <= datetime.date(years[-1], 12, 31)
    bool_1 = cierre >= datetime.date(years[0], 01, 01)

    bool_year = np.logical_and(bool_0, bool_1)

    return bool_year


def filter_by_date2(servicios, date0, date1):
    "Retrieve the companies which are open between this to dates."
    indices_0 = np.array(servicios.index)
    fecha_lim = date
    indices = (servicios['cierre'][servicios['cierre'] >= fecha_lim]).index
    indices = np.array(indices)
    indices_b = list([ind for ind in indices_0 if ind not in indices])
    return indices


def filter_by_date(servicios, date):
    """Filter by date.
    ----------
    DEPRECATED
    ----------
    """
    indices_0 = np.array(servicios.index)
    fecha_lim = date
    indices = (servicios['cierre'][servicios['cierre'] >= fecha_lim]).index
    indices = np.array(indices)
    indices_b = list([ind for ind in indices_0 if ind not in indices])
    #servicios = servicios.loc[indices]
    servicios.drop(indices_b)
    return servicios


############################# Filter coordinates ##############################
###############################################################################
def retrieve_coordinates(empresas, coord_vars, method):
    """Return the bool_array which informs to us the correct using the method
    selected.
    """
    if method == 'nullcoordinates':
        bool_coord = np.logical_and(empresas[coord_vars[0]] != 0,
                                    empresas[coord_vars[1]] != 0)

    elif method == 'spaincoordinates':
        bool_coord = filter_bool_uncorrect_coord_spain(empresas, coord_vars)

    return bool_coord


def filter_by_nullcoordinates(servicios, coord_vars):
    """Filter the rows with null values in coordinates.
    """
    idxs = np.logical_and(servicios[coord_vars[0]] != 0,
                          servicios[coord_vars[1]] != 0)
    servicios = servicios[idxs]
    return servicios


def filter_by_spaincoordinates(servicios, coord_vars):
    """Filter the rows with non correct values of coordinates of spain.
    """
    return filter_uncorrect_coord_spain(servicios, coord_vars)


###############################################################################
############################# AUXILIARY FUNCTIONS #############################
###############################################################################
def dropna_rows(df, columns):
    """Delete rows with absolute null values in the observations."""
    indices = np.array(df[columns].dropna(how='all').index)
    return indices
