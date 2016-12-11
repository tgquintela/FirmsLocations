

import numpy as np
import copy


###### Functions to filter input data
f_corr = lambda x: np.logical_not(np.logical_or(np.isnan(x), x == 0))
f_log = lambda x: np.log(np.abs(x))*np.sign(x)


def f_filter_finance(nif_ref, year_ref, pfeatures, qvalue):
    """Filter finance data."""
    logi = np.all(f_corr(pfeatures), axis=1)
    nif_ref = [nif_ref[i] for i in range(len(logi)) if logi[i]]
    year_ref = year_ref[logi]
    pfeatures = pfeatures[logi]
    qvalue = qvalue[logi]
    return nif_ref, year_ref, pfeatures, qvalue


def f_filter_logfinance(nif_ref, year_ref, pfeatures, qvalue):
    """Filter finance data."""
    logi = np.all(f_corr(pfeatures), axis=1)
    nif_ref = [nif_ref[i] for i in range(len(logi)) if logi[i]]
    year_ref = year_ref[logi]
    pfeatures = pfeatures[logi]
    qvalue = qvalue[logi]
    pfeatures2 = pfeatures[:]
    rate_act = np.arctan(pfeatures[:, 1]/pfeatures[:, 0])
    mag_act = f_log(pfeatures[:, 0])
    rate_pas = np.arctan(pfeatures[:, 3]/pfeatures[:, 2])
    mag_pas = f_log(pfeatures[:, 2]+pfeatures[:, 3])
    employee_size = f_log(pfeatures[:, 4])
    va = f_log(pfeatures[:, 5])
    vtas = f_log(pfeatures[:, 6])
    pfeatures2[:, 0] = rate_act
    pfeatures2[:, 1] = mag_act
    pfeatures2[:, 2] = rate_pas
    pfeatures2[:, 3] = mag_pas
    pfeatures2[:, 4] = employee_size
    pfeatures2[:, 5] = va
    pfeatures2[:, 6] = vtas
    return nif_ref, year_ref, pfeatures2, qvalue


def f_filter_null(nif_ref, year_ref, pfeatures, qvalue):
    """Null filter data."""
    return nif_ref, year_ref, pfeatures, qvalue


###### Perturbation
def f_pert_null_instantiation(X, locs, classe, pars={}):
    obj = classe(**pars)
    return obj


def f_pert_features_instantiation(X, locs, classe, pars={}):
    obj = classe(X, **pars)
    return obj


def f_pert_partialfeatures_instantiation(X, locs, classe, pars={}):
    obj = classe(len(X), **pars)
    return obj


def f_pert_partialpstemp_instantiation(X, locs, regs, classe, pars={}):
    obj = classe(len(X), **pars)
    return obj


###### Format
def f_null_format(X, y, classe, pars):
    return X, y


def f_null_spatiotemporal_format(X, y, locs, regs, format_obj, format_pars={}):
    """Dummy function to format the data for spatio-temporal models.
    """
    return X, y, locs, regs


###### Functions to convert to the proper element the input information
def dummy_function_conversion(x, p=None):
    return x


def f_null_instantiation(classe, pars=None):
    if pars is not None:
        obj = classe(**pars)
    else:
        obj = classe()
    return obj
