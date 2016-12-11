
"""
Perturbation test combinations
------------------------------
Module to deal with possible combinations of parameters in order to compute
perturbation tests.

"""

import numpy as np
import copy

from joblib import Parallel, delayed
import multiprocessing
import time


###############################################################################
############################### Spatial models ################################
###############################################################################


###############################################################################
######################### Spatial-descriptors models ##########################
###############################################################################
def application_pst_sklearn_models(loc_ref, year_ref, pfeatures, qvalue,
                                   parameters):
    """The application of the models after compute spatial-inferred features.

    Parameters
    ----------
    loc_ref: np.ndarray
        the locations reference.
    year_ref: np.ndarray
        the years of each location is active.
    pfeatures: np.ndarray
        the element features for each location.
    qvalue: np.ndarray
        the values we want to predict.
    parameters: tuple
        the parameters which descriptves the model we want to apply.
        That information is summarized in the possible values for:
            * The information for create the pySpatialTools descriptors.
            * The sampling, perturbation and sklearn model information possible
            to test.


    Returns
    -------
    descriptors: np.ndarray
        the descriptors computed by joinning the location-based features and
        the element features.
    qvalue: np.ndarray
        the ordered values we want to predict.

    """
    # Initialization
    parameters_pst, parameters_train = parameters

    ## Split in years and compute descriptors separately
    descriptors, qvals = compute_pst_descriptors_by_year(loc_ref, year_ref,
                                                         pfeatures, qvalue,
                                                         parameters_pst)
    ## Application of the models
    scores, best_pars_info =\
        application_sklearn_models(pfeatures, qvalue, parameters_train)
    return scores, best_pars_info


def compute_pst_descriptors_by_year(loc_ref, year_ref, pfeatures, qvalue,
                                    parameters_pst):
    """Compute descriptors using ``pySpatialTools``.

    Parameters
    ----------
    loc_ref: np.ndarray
        the locations reference.
    year_ref: np.ndarray
        the years of each location is active.
    pfeatures: np.ndarray
        the element features for each location.
    qvalue: np.ndarray
        the values we want to predict.
    parameters_pst: tuple
        the parameters of the description creation.

    Returns
    -------
    descriptors: np.ndarray
        the descriptors computed by joinning the location-based features and
        the element features.
    qvalue: np.ndarray
        the ordered values we want to predict.

    """
    years_u = np.unique(year_ref)
    descriptors, new_qvals = [], []
    for y_u in years_u:
        logi_y = year_ref == y_u
        desc_i =\
            compute_pst_descriptors(loc_ref[logi_y], pfeatures[logi_y],
                                    parameters_pst)
        descriptors.append(desc_i)
        new_qvals.append(qvalue[logi_y])
    descriptors = np.concatenate(descriptors, axis=0)
    new_qvals = np.concatenate(new_qvals, axis=0)

    return descriptors, new_qvals


###############################################################################
############################# Auxiliar functions ##############################
###############################################################################
def names_parameters_computation(parameters):
    """Names of each combination computation. Extraction of names
    information from parameters information."""
    perturbations_info, format_info, models_info = parameters[:3]
    samplings_info, scorer_info = parameters[3:]
    pert_names, format_names, model_names = [], [], []
    sampling_names, scorer_names = [], []
    for i in range(len(perturbations_info)):
        pert_names.append(perturbations_info[i][0])
    for i in range(len(format_info)):
        format_names.append(format_info[i][0])
    for i in range(len(models_info)):
        model_names.append(models_info[i][0])
    for i in range(len(samplings_info)):
        sampling_names.append(samplings_info[i][0])
    for i in range(len(scorer_info)):
        scorer_names.append(scorer_info[i][0])
    return pert_names, format_names, model_names, sampling_names, scorer_names
