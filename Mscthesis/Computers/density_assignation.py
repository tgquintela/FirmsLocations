
"""
Density assignation data
------------------------
Module which is oriented to compute the assignation of the density variable
to a point considering a spatial distributions of features in the space.

"""

import numpy as np
from scipy.interpolate import Rbf, bisplrep, bisplev

from pySpatialTools.Retrieve import CircRetriever
from pySpatialTools.api.spatialdesc_utils import\
    create_pst_interpolation

#from pythonUtils.ProcessTools import Processer
#from pySpatialTools.Interpolation.density_assignation import \
#    general_density_assignation


class Value_Interpolator:
    def __init__(self, interpolate_o, x, y, pars_xy, pars_init):
        pars_init[pars_xy[0]] = x
        pars_init[pars_xy[1]] = y
        self.interpolate_f = interpolate_o(**pars_init)

    def compute(self, x_int, pars={}):
        return self.interpolate_f(x_int, **pars)


def general_geo_interpolation(sp_data, value_data, sp_data_int,
                              method='', pars={}):
    """General function to interpolate some points and assign a value to
    other spatial elements.
    """
    ## 0. Preparation of inputs
    method = 'spline' if method == '' else method
    method = method.lower()
    ## 1. Switcher
    if method == 'spline':
        value_data_int =\
            spline_interpolation(sp_data, value_data, sp_data_int, pars)
    elif method == 'rbf':
        value_data_int =\
            rbf_interpolation(sp_data, value_data, sp_data_int, pars)
    elif method == 'pst':
        value_data_int =\
            pst_interpolation(sp_data, value_data, sp_data_int, pars)
    return value_data_int


def spline_interpolation(sp_data, value_data, sp_data_int, pars={}):
    tck = bisplrep(sp_data[:, 0], sp_data[:, 1], value_data, **pars)
    value_data_int = bisplev(sp_data_int[:, 0], sp_data_int[:, 1], tck)
    return value_data_int


def rbf_interpolation(sp_data, value_data, sp_data_int, pars={}):
    rbf = Rbf(sp_data[:, 0], sp_data[:, 1], value_data)
    value_data_int = rbf(sp_data_int[:, 0], sp_data_int[:, 1], **pars)
    return value_data_int


#def pst_interpolation(sp_data, value_data, sp_data_int, pars={}):
#    pars['retriever'] = pars['retriever'](sp_data)
#    pars['values'] = value_data
#    value_data_int = general_density_assignation(sp_data_int, **pars)
#    return value_data_int


def pst_interpolation(sp_data, value_data, sp_data_int, pars={}):
    """Assumption CircRetriever."""
    interpolator = create_pst_interpolation(sp_data, value_data, sp_data_int,
                                            CircRetriever, pars['ret'],
                                            pars['interpolation'])
    value_data_int = interpolator.compute().ravel()
    return value_data_int

#sp_data = np.random.random((100, 2))
#value_data = np.random.random((100))
#sp_data_int = np.random.random((10000, 2))
#pars = {'ret': {'info_ret': 20.},
#        'interpolation': {'f_weight': 'gaussian',
#                          'pars_w': {'max_r': 20.,
#                                     'S': 3.4105868102821946},
#                          'f_dens': 'weighted_avg',
#                          'pars_d': {}}}
