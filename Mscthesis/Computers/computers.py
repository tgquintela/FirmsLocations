
"""
Computers
---------
Main computers collections which gets all the precomputers and use that data
to compute.

"""

import os
import numpy as np

import copy

from joblib import Parallel, delayed
import multiprocessing

from pythonUtils.Logger import Logger
from pythonUtils.ProcessTools import Processer
from sklearn.grid_search import BaseSearchCV
from pythonUtils.perturbation_tests.sklearn_models import \
    Sklearn_permutation_test
from pythonUtils.perturbation_tests.pst_models import \
    Pst_permutation_test

#from Mscthesis.Preprocess.financial_utils import f_corr
from Mscthesis.IO.io_process_computations import store_model
from precomputers import PrecomputerCollection
from computer_processers import application_pst_sklearn_models
from computer_processers import names_parameters_computation
from computer_utils import get_references_intersection, join_loaded_features,\
    get_ordered_locations, get_ordered_regions, separate_by_times


class GeneralComputer(Processer):
    """Factorization of computer manager class for specific model selected.
    That class is used as a container of all the interaction functions with the
    ROM-stored data.
    """

    def _initialization_spec(self):
        self.files = []
        self.subproc_desc = [""]
        self.t_expended_subproc = [0]
        self.proc_name = self._name
        self.proc_desc = "Computation of models"
        ## Spec
        self._computer = []

    def __init__(self, logfile, pathfolder, precomputers=None, num_cores=None):
        self._initialization()
        self._initialization_spec()
        self.pathfolder = pathfolder
        self.pathdata = os.path.join(pathfolder, 'Cleaned/Results')

        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        elif num_cores == 0:
            self.num_cores = 1
        else:
            self.num_cores = num_cores

        self.logfile = Logger(logfile) if type(logfile) == str else logfile
        if precomputers is not None:
            self.precomputers = precomputers
        else:
            self.precomputers = PrecomputerCollection(logfile, pathfolder,
                                                      old_computed=True)

    ############################### COMPUTATION ###############################
    ###########################################################################
    def compute(self, listparams):
        """Main function to compute the models for the given parameters."""
        ## 0. Set vars
        t00 = self.setting_global_process()
        # Prepare subprocesses
        n_cases = len(listparams)
        subprocesses = ["Read data of %s", "Compute %s", "Storing %s"]
        self.t_expended_subproc = [[0, 0, 0] for i in range(n_cases)]
        self.subproc_desc = [copy.copy(subprocesses) for i in range(n_cases)]
        self._create_subprocess_hierharchy(self.subproc_desc)

        ## 1. Computations and storing
        for i in range(len(listparams)):
            ## Parameters
            files_def, parameters, name_compute = listparams[i]
            names_comb = names_parameters_computation(parameters)
            self.subprocesses_description_setting(i, name_compute)

            ## Read files and get data
            t0 = self.set_subprocess([i, 0])
            data = self.get_data(files_def)
            self.close_subprocess([i, 0], t0)
            import time
            print 'Get data:', time.time()-t0
            ## Compute
            t0 = self.set_subprocess([i, 1])
            scores, best_pars_info = self._compute_i(parameters, *data)
            self.close_subprocess([i, 1], t0)
            ## Storing
            t0 = self.set_subprocess([i, 2])
            self._store_i(scores, best_pars_info, names_comb, listparams[i])
            self.close_subprocess([i, 2], t0)
#            self._compute_i(*listparams[i])

        self.files = os.listdir(os.path.join(self.pathdata, 'Scores'))
        self.listparams = listparams
        assert(len(self.files) == len(self.listparams))
        ## Untrack process
        self.close_process(t00)

#   TODEPRECATE
#   ###########
#    def compute_i(self, files_def, parameters, name_compute):
#        """Compute each needed computation testing all the required
#        combinations of parameters.
#        """
#        ## Computation of the model
##        for i in range(len(files_def)):
#        files_feats, files_qvals, f_filter = self._get_pathfiles(files_def)
#        ## Computation of each case
#        scores, best_pars_info, names_comb =\
#            self._compute_ij(files_feats, files_qvals, f_filter, parameters)
#        ## Store the results of each case
#        self._store_ij(scores, best_pars_info, names_comb,
#                       files_feats, files_qvals, parameters, name_compute)

    ######################### READING DATA PRECOMPUTED ########################
    ###########################################################################
    def _get_data_xy_ij(self, file_feats, file_qvals, f_filter):
        """Get data for the file i of feats and the file j of qvals."""
        ## Get data
        # Get the features for element i
        hashes_feats, nif_feats, year_feats, pfeatures, methodvalues_feats =\
            self._get_features(file_feats)
        # Get the labels for element j
        hashes_qvals, nif_qvals, year_qvals, qvalue, methodvalues_qvals =\
            self._get_qvals(file_qvals)

        ## Align the features and the labels
        nif_ref, year_ref, pfeatures, qvalue =\
            get_references_intersection(nif_feats, year_feats,
                                        pfeatures, nif_qvals,
                                        year_qvals, qvalue)
        nif_ref, year_ref, pfeatures, qvalue =\
            f_filter(nif_ref, year_ref, pfeatures, qvalue)
        ## Format properly
        if len(pfeatures.shape) == 1:
            pfeatures = pfeatures.reshape((len(pfeatures), 1))
        return nif_ref, year_ref, pfeatures, qvalue

    def _get_aligned_data_locations(self, nif_ref):
        locations, years, nifs = self._get_locations()
        locs = np.zeros((len(nif_ref), 2))
        for i in range(len(nif_ref)):
            locs[i] = locations[nifs.index(nif_ref[i])]
        return locs

    def _get_features(self, files_feats):
        """Interaction with the data features stored."""
        if type(files_feats) == list:
            hash_feats, nif_feats, year_feats, pfeatures, methodvalues_feats =\
                [], [], [], [], []
            for k in range(len(files_feats)):
                hashes_k, nif_k, year_k, pfeatures_k, methodvalues_k =\
                    self.precomputers.precomputer_pfeatures.\
                    _retrieve_i(files_feats[k])
                hash_feats.append(hashes_k)
                nif_feats.append(nif_k)
                year_feats.append(year_k)
                pfeatures.append(pfeatures_k)
                methodvalues_feats.append(methodvalues_k)
            # Join features
            hashes, nif_feats, year_feats, pfeatures, methodvalues_feats =\
                join_loaded_features(hash_feats, nif_feats, year_feats,
                                     pfeatures, methodvalues_feats)
        else:
            print files_feats
            hashes, nif_feats, year_feats, pfeatures, methodvalues_feats =\
                self.precomputers.precomputer_pfeatures.\
                _retrieve_i(files_feats)
        return hashes, nif_feats, year_feats, pfeatures, methodvalues_feats

    def _get_qvals(self, files_qvals):
        """Interaction with the data qvals stored."""
        hashes, nif_qvals, year_qvals, qvalue, methodvalues_qvals =\
            self.precomputers.precomputer_qvalues.\
            _retrieve_i(files_qvals)
        return hashes, nif_qvals, year_qvals, qvalue, methodvalues_qvals

    def _get_locations(self):
        pathfolder = self.precomputers.precomputer_locations.pathfolder
        namefile = os.listdir(pathfolder)[0]
        namepath = os.path.join(pathfolder, namefile)
        hashes, nifs, years, locations, _ =\
            self.precomputers.precomputer_locations._retrieve_i(namepath)
#        locations, years, nifs = get_locations(namepath)
        return hashes, nifs, years, locations

    def _get_region_data(self):
        pathfolder = self.precomputers.precomputer_regions.pathfolder
        namefile = os.listdir(pathfolder)[0]
        namepath = os.path.join(pathfolder, namefile)
        nif_reg, code_reg, reg_pars =\
            self.precomputers.precomputer_regions._retrieve_i(namepath)
        return nif_reg, code_reg, reg_pars

    def _retrieve(self):
        listfiles = os.listdir(self.pathfolder)
        precomputed = []
        for namefile in listfiles:
            precomputed.append(self._retrieve_i(namefile))
        return precomputed

    def _get_pathfiles(self, files_def):
        files_feats, files_qvals, f_filter = files_def
        pfeatures_path = 'Cleaned/Precomputed/Pfeatures'
        qvalues_path = 'Cleaned/Precomputed/Qvalues'
        if len(files_feats.split('/')) == 1:
            files_feats = os.path.join(os.path.join(self.pathfolder,
                                                    pfeatures_path),
                                       files_feats)
        if len(files_qvals.split('/')) == 1:
            files_qvals = os.path.join(os.path.join(self.pathfolder,
                                                    qvalues_path),
                                       files_qvals)
        return files_feats, files_qvals, f_filter

    ##################### AUXILIAR ADMINISTRATIVE FUNCTIONS ###################
    ###########################################################################
    def subprocesses_description_setting(self, i, name_compute):
        "Set the descriptions of the subprocesses."
        self.subproc_desc[i][0] = self.subproc_desc[i][0] % name_compute
        self.subproc_desc[i][1] = self.subproc_desc[i][1] % name_compute
        self.subproc_desc[i][2] = self.subproc_desc[i][2] % name_compute

    def _export_administrative_information_i(self):
        "Export information."
        i = len(self._computer)-1
        administrative_info = {}
        administrative_info['subproc_desc'] = self.subproc_desc[i]
        administrative_info['t_expended_subproc'] = self.t_expended_subproc[i]
        administrative_info['t_perturb_tests'] =\
            self._computer[i]._times_processes
        administrative_info['num_cores'] = self.num_cores
        return administrative_info


###############################################################################
############################### Specific models ###############################
###############################################################################
################################ Direct Model #################################
###############################################################################
class Directmodel(GeneralComputer):
    """Based to apply directly a sklearn model using the features given by
    the financial firms information.
    """
    _name = 'Directmodel'

    def get_data(self, files_def):
        """Get the needed data to apply the model."""
        files_feats, files_qvals, f_filter = self._get_pathfiles(files_def)
        nif_ref, year_ref, pfeatures, qvalue =\
            self._get_data_xy_ij(files_feats, files_qvals, f_filter)
        return nif_ref, year_ref, pfeatures, qvalue

    def _compute_i(self, parameters, nif_ref, year_ref, pfeatures, qvalue):
        """Computation for a given data."""
        ## Application of the models
#        scores, best_pars_info =\
#            application_sklearn_models_paral(pfeatures, qvalue, parameters,
#                                             self.num_cores)
        computer = Sklearn_permutation_test(self.num_cores)
        scores, best_pars_info =\
            computer.compute(pfeatures, qvalue, parameters)
        self._computer.append(computer)
        return scores, best_pars_info

    def _store_i(self, scores, best_pars_info, names_comb, listparams):
        """Store the results of the computations."""
        files_def, parameters, name_compute = listparams
        files_feats, files_qvals, _ = self._get_pathfiles(files_def)
        precomp_files = ('pfeatures', files_feats), ('qvalues', files_qvals)
        scores_folder = os.path.join(self.pathdata, 'Scores')
        administrative_info = self._export_administrative_information_i()
        store_model(scores_folder, scores, best_pars_info, names_comb,
                    precomp_files, parameters, name_compute,
                    administrative_info)


############################ Location only Model ##############################
###############################################################################
class LocationOnlyModel(GeneralComputer):
    """Based on applying a location-based model using the features given the
    type of each location.
    """
    _name = "LocationOnlyModel"

    def get_data(self, files_def):
        """Get the needed data to apply the model."""
        files_feats, files_qvals, f_filter = self._get_pathfiles(files_def)
        nif_ref, year_ref, pfeatures, qvalue =\
            self._get_data_xy_ij(files_feats, files_qvals, f_filter)
        ## Locations management
        hashes, nifs, years, locations = self._get_locations()
        locations = get_ordered_locations(locations, years, nifs,
                                          year_ref, nif_ref)
        reg_nifs, reg_data, _ = self._get_region_data()
        reg_data = get_ordered_regions(reg_data, reg_nifs, nif_ref)
        return nif_ref, year_ref, pfeatures, qvalue, locations, reg_data

    def _compute_i(self, parameters, nif_ref, year_ref, pfeatures, qvalue,
                   locations, reg_data):
        """Compute main function."""
        ## Prepare inputs
        pfeatures = separate_by_times(pfeatures, year_ref)
        qvalue = separate_by_times(qvalue, year_ref)
        locations = separate_by_times(locations, year_ref)
        reg_data = separate_by_times(reg_data, year_ref)
        nif_ref = separate_by_times(nif_ref, year_ref)

        ## Apply permutations
        computer = Pst_permutation_test(self.num_cores)
        scores, best_pars_info =\
            computer.compute(pfeatures, qvalue, locations, reg_data, nif_ref,
                             parameters)
        self._computer.append(computer)
        return scores, best_pars_info


########################### Location General Model ############################
###############################################################################
class LocationGeneralModel(GeneralComputer):
    """Based on applying a location-based inferring of features to complement
    the element features we have on the data.
    """
    _name = "LocationGeneralModel"

    def get_data(self, files_def):
        """Get the needed data to apply the model."""
        files_feats, files_qvals, f_filter = self._get_pathfiles(files_def)
        nif_ref, year_ref, pfeatures, qvalue =\
            self._get_data_xy_ij(files_feats, files_qvals, f_filter)
        ## Locations management
        locations, years, nifs = self._get_locations()
        locations = get_ordered_locations(locations, years, nifs, nif_ref,
                                          year_ref)
        return nif_ref, year_ref, pfeatures, qvalue, locations

    def _compute_i(self, parameters, nif_ref, year_ref, pfeatures, qvalue,
                   locations):
        """Compute main function."""
        computer = PstSklearn_permutation_test(self.num_cores)
        scores, best_pars_info =\
            computer.compute(pfeatures, qvalue, parameters)
        self._computer.append(computer)
        return scores, best_pars_info
