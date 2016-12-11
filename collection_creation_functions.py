
"""
Collection creation functions
-----------------------------
Collection of functions to support and ease the massive standarized creation
of parameters to be used for the main program.
"""

import numpy as np
from itertools import product
from pySpatialTools.utils.perturbations import\
    PartialPermutationPerturbationGeneration
from collection_auxiliar_functions import dummy_function_conversion,\
    f_pert_partialfeatures_instantiation, f_null_format,\
    f_null_spatiotemporal_format, f_pert_partialpstemp_instantiation


###### Creation perturbations functions
def create_permutation_feature(number, rate_pert=1., name=None):
    """Create permutation for features."""
    n = np.random.randint(0, 100000)
    if name is None:
        name = f_stringer_pert_rate(rate_pert)

    lista_permuts = []
    for i in range(number):
        lista_permuts.append((name, PartialPermutationPerturbationGeneration,
                              {'seed': n+i, 'rate_pert': rate_pert},
                              f_pert_partialfeatures_instantiation))
    return lista_permuts


def create_permutation_sptemp(number, rate_pert=1., name=None,
                              pert_flag=False):
    """Create permutation."""
    n = np.random.randint(0, 100000)
    if name is None:
        name = f_stringer_pert_rate(rate_pert)

    lista_permuts = []
    for i in range(number):
        lista_permuts.append((name, PartialPermutationPerturbationGeneration,
                              {'seed': n+i, 'rate_pert': rate_pert},
                              f_pert_partialpstemp_instantiation, pert_flag))
    return lista_permuts


def f_stringer_pert_rate(rate_pert):
    n_name = str(np.around(rate_pert, 2))+'0'*2
    n_name = n_name[:4]
    n_name = n_name.replace('.', '')
    name = "rate_permut_"+n_name
    return name


###### Creation models functions
def create_null_format_info():
    return [('null_format', None, {}, f_null_format)]


def create_null_sptemp_format_info():
    return [('null_format', None, {}, f_null_spatiotemporal_format)]


###### Creation models functions
def creation_models(classe, name_class, parameters, f_stringer=None):
    """

    Example
    -------
    >>> parameters = [('n_estimators', [10, 25, 50, 75, 100], 'rf_reg_nest'),
                      ('a', [5, 4, 6], 'jhf')]
    >>> name_class = 'rf'
    >>> classe = 'random'
    >>> lista_parameters = creation_models(classe, name_class, parameters)

    """
    comb_pars = [parameters[i][1] for i in range(len(parameters))]
    ranges = [range(len(parameters[i][1])) for i in range(len(parameters))]
    names_pars = [parameters[i][0] for i in range(len(parameters))]
    code_names_pars = [parameters[i][2] for i in range(len(parameters))]

    if len(parameters) == 0:
        return [(name_class, classe, {})]
    lista_parameters = []
    for p in product(*ranges):
        par = tuple([comb_pars[i][p[i]] for i in range(len(p))])
        d = dict(zip(names_pars, par))
        names_val = generate_par_strings(par, f_stringer)
        name_par = '_'.join([''.join((code_names_pars[i], names_val[i]))
                             for i in range(len(names_val))])
        name_par = name_class+'_'+name_par

        lista_parameters.append((name_par, classe, d))
    return lista_parameters


def generate_par_strings(lista_par, f_stringer=None):
    if f_stringer is None:
        str_lista_par = []
        for l in lista_par:
            str_lista_par.append(str(l)[:4])
    else:
        str_lista_par = [f_stringer(l) for l in lista_par]

    return str_lista_par


###### Creation sampling functions
def creation_sampling(classe, name_class, parameters, f_stringer=None):
    """

    Example
    -------
    >>> parameters = [('n_estimators', [10, 25, 50, 75, 100], 'rf_reg_nest'),
                      ('a', [5, 4, 6], 'jhf')]
    >>> name_class = 'rf'
    >>> classe = 'random'
    >>> lista_parameters = creation_sampling(classe, name_class, parameters)

    """
    comb_pars = [parameters[i][1] for i in range(len(parameters))]
    ranges = [range(len(parameters[i][1])) for i in range(len(parameters))]
    names_pars = [parameters[i][0] for i in range(len(parameters))]
    code_names_pars = [parameters[i][2] for i in range(len(parameters))]

    if len(parameters) == 0:
        return [(name_class, classe, {})]
    lista_parameters = []
    for p in product(*ranges):
        par = tuple([comb_pars[i][p[i]] for i in range(len(p))])
        d = dict(zip(names_pars, par))
        names_val = generate_par_strings(par, f_stringer)
        name_par = '_'.join([''.join((code_names_pars[i], names_val[i]))
                             for i in range(len(names_val))])
        name_par = name_class+'_'+name_par

        lista_parameters.append((name_par, classe, d))
    return lista_parameters


def creation_sptemp_sampling(classe, name_class, parameters, f_stringer=None):
    pass

###### Creation models functions
def creation_scorers(classe, name_class, parameters, f_stringer=None,
                     f_trans=None):
    """

    Example
    -------
    >>> parameters = [('n_estimators', [10, 25, 50, 75, 100], 'rf_reg_nest'),
                      ('a', [5, 4, 6], 'jhf')]
    >>> name_class = 'rf'
    >>> classe = 'random'
    >>> lista_parameters = creation_sampling(classe, name_class, parameters)

    """
    if f_trans is None:
        f_trans = dummy_function_conversion

    comb_pars = [parameters[i][1] for i in range(len(parameters))]
    ranges = [range(len(parameters[i][1])) for i in range(len(parameters))]
    names_pars = [parameters[i][0] for i in range(len(parameters))]
    code_names_pars = [parameters[i][2] for i in range(len(parameters))]

    if len(parameters) == 0:
        return [(name_class, classe, {}, f_trans)]
    lista_parameters = []
    for p in product(*ranges):
        par = tuple([comb_pars[i][p[i]] for i in range(len(p))])
        d = dict(zip(names_pars, par))
        names_val = generate_par_strings(par, f_stringer)
        name_par = '_'.join([''.join((code_names_pars[i], names_val[i]))
                             for i in range(len(names_val))])
        name_par = name_class+'_'+name_par

        lista_parameters.append((name_par, classe, d, f_trans))
    return lista_parameters
