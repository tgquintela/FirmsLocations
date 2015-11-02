
"""
Assignation_value
-----------------
Module which contains the utilities to assign a quality value to a business
considering only the open and close times.

"""

import numpy as np


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
        maxyears = np.max(years2close)
    if maxyears < np.max(years2close):
        maxyears = np.max(years2close)

    ## 1. Assignation function
    q_value = maxyears - years2close

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
