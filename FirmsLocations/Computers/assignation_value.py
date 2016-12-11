
"""
Assignation_value utils
-----------------------
Module which contains the utilities to assign a quality value to a business
from data information.

"""

import os
import shelve
import numpy as np
import pandas as pd
from ..Preprocess.financial_utils import financial_size_computation


def general_qvalue_assignation(pathfolder, method, pars):
    """Switcher function of all possible qvalues assignation."""

    pathfile_data = 'Cleaned/FirmsData/atemporal/empresas'

    if method == 'dates':
        pathdata = os.path.join(pathfolder, pathfile_data)
        data = pd.read_csv(pathdata, sep=';', index_col=0)
        nif, qvalue, year = qvalues_dates_methods(data, **pars)
    elif method == 'financial':
        nif, qvalue, year = qvalues_finance_methods(pathfolder, **pars)

    elif method == 'dates_financial':
        pathdata = os.path.join(pathfolder, pathfile_data)
        data = pd.read_csv(pathdata, sep=';', index_col=0)
        pathfin = os.path.join(pathfolder, pathfile_fin)

    return nif, qvalue, year


###############################################################################
################################ DATES QVALUES ################################
###############################################################################
def qvalues_dates_methods(data, method, params):
    ## Data treatment
    nif = list(data['nif'])
    apertura = data['apertura'].apply(lambda x: int(x[:4])).as_matrix()
    cierre = data['cierre'].apply(lambda x: int(x[:4])).as_matrix()
    del data
    ## Preparation of inputs
    nif_years, apertura_y, cierre_y, actual_year = [], [], [], []
    for year in range(2006, 2013):
        logi = np.logical_and(year <= cierre, year >= apertura)
        nif_years += [nif[i] for i in range(len(logi)) if logi[i]]
        apertura_y.append(apertura[logi])
        cierre_y.append(cierre[logi])
        actual_year.append(np.ones(logi.sum()).astype(int)*year)
    actual_year = np.concatenate(actual_year)
    apertura_y = np.concatenate(apertura_y)
    cierre_y = np.concatenate(cierre_y)

    ## Find method switcher
    if method == 'proportional':
        f = proportional_assignation
    elif method == 'exponential':
        f = exponential_assignation
    elif method == 'economic':
        f = economic_asssignation

    ## Computation of the value
    qvalue = qvalue_date_assignation(actual_year, apertura_y, cierre_y,
                                     f, params)

    return nif_years, qvalue, actual_year


def qvalue_date_assignation(actual_year, open_year, close_year, f, params):
    """General function for quality value computation from dates of close and
    open.

    Parameters
    ----------
    actual_year: numpy.ndarray or int
        the year it was opened.
    open_year: numpy.ndarray or int
        the year it was closed.
    close_year: numpy.ndarray or int
        the year it was closed.
    f: function
        fucntion which has to be applied to compute quality value. It is needed
        that has input values of years_open, years2close and other parameters
        specified in the params variable.
    params: dict
        parameters.

    Returns
    -------
    q_value: float
        the quality value assigned from dates information.

    """

    ## Compute differential of years
    years_open, years2close = years_descriptors(actual_year, open_year,
                                                close_year)
    ## Apply function for quality computation
    q_value = f(years2close, years_open, **params)

    return q_value


def years_descriptors(actual_year, open_year, close_year):
    """Compute the two important date variables in order to assign a value of
    quality.
    """
    years_open = actual_year - open_year
    years2close = close_year - actual_year
    return years_open, years2close


def proportional_assignation(years2close, years_open, maxyears=0):
    "Assignation a quality value proportional to the years to close."

    ## 0. Control of the maxyears variable
    if maxyears == 0:
        maxyears = np.max(years2close+years_open)
#    if maxyears < np.max(years2close):
#        maxyears = np.max(years2close)

    ## 1. Assignation function
    q_value = (years2close+years_open)/float(maxyears)

    return q_value


def exponential_assignation(years2close, years_open, beta=0.9):
    "Assgination "

    ## 1. Assignation function
    q_value = (1. + beta) ** (years2close - 1)
    q_value[years2close] = 0.

    return q_value


def economic_asssignation(years2close, years_open, cost_f, params_cost,
                          benefit_f, params_ben):
    """Compute the economic value of being in that position in the selected
    year.
    """

    c = cost_f(years2close, years_open, **params_cost)
    b = benefit_f(years2close, years_open, **params_ben)
    q_value = b - c

    return q_value


def special_assignation(years2close, years_open, years2end=-1):
    ## Setting paramters of years2end
    if years2end == -1:
        years2end = np.max(years2close)

    ## Computing measure

    return q_value


###############################################################################
############################## FINANCIAL QVALUES ##############################
###############################################################################
def qvalues_finance_methods(pathfolder, method, params):
    if method == 'diff_magnitude':
        pathfile = os.path.join(pathfolder, 'Cleaned/Precomputed/Pfeatures')
        filenames = os.listdir(pathfile)
        for filename in filenames:
            if params['methodname'] in filename:
                break
        filename = os.path.join(pathfile, filename)
        nif, year, qvalue = diff_magnitude_qvalue(filename)
    return nif, qvalue, year


def raw_finance_qvalues(data):
    fin = data[[c for c in data.columns if c != 'nif']].as_matrix()
    pass


def diff_magnitude_qvalue(filename):
    db = shelve.open(filename)
    nif = db['nif']
    year = db['year']
    pfeatures = db['pfeatures']
    db.close()

    mag, correctable = financial_size_computation(pfeatures)
    reindices = np.where(correctable)[0]

    nif_uniques = set(nif)
    diff_mag, new_years, nifs = [], [], []
    for nif_u in nif_uniques:
        idxs = [i for i in range(len(nif)) if nif_u == nif[i]]
        years_nif = year[idxs]
        if (years_nif.max()-years_nif.min()) != len(years_nif)-1:
            print years_nif
            assert((years_nif.max()-years_nif.min()) == len(years_nif)-1)
        for i in range(len(years_nif)-1):
            idx0 = np.where(years_nif == (years_nif.min()+i))[0][0]
            idx1 = np.where(years_nif == (years_nif.min()+i+1))[0][0]
            if correctable[idxs[idx0]] and correctable[idxs[idx1]]:
                n_idxs1 = np.where(reindices == idxs[idx1])[0][0]
                n_idxs0 = np.where(reindices == idxs[idx0])[0][0]
                mag_y = mag[n_idxs1]-mag[n_idxs0]
                mag_y /= mag[n_idxs0]
                diff_mag.append(mag_y)
                nifs.append(nif_u)
                new_years.append(years_nif.min()+i)
    new_years = np.array(new_years)
    diff_mag = np.array(diff_mag)
    return nifs, new_years, diff_mag
