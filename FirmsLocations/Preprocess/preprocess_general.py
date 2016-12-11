
"""
Module which groups useful and general functions for preprocessing the data.
"""

from preprocess_rows import retrieve_empresas_dict, filter_bool_dict
from FirmsLocations.IO.aux_functions import concat_from_dict


def filter_empresas(empresas, filterinfo, indices=None):
    "Filter empresas using filterinfo."
    # Retrieve indices (bool_array)
    if indices is None:
        indices = retrieve_empresas_dict(empresas, **filterinfo)
    # Filter using indices
    empresas = filter_bool_dict(empresas, indices)
    return empresas, indices


def concat_empresas(empresas, serviciosvar, companiesvar):
    """Concatenatio of the empresas dictionary to obtain a complete dataframe
    structure with the correct formatted data.
    """
    empresas['servicios'] = concat_from_dict(empresas['servicios'],
                                             serviciosvar)
    empresas = concat_from_dict(empresas, companiesvar)
    return empresas


def filtercols_empresas(empresas, filtercolsinfo):
    "TODO: improve"
    return empresas
