
"""
Computers
---------
Main computers collections which gets all the precomputers and use that data
to compute .

"""

import os
import shelve
import numpy as np
from pythonUtils.Logger import Logger
from pythonUtils.ProcessTools import Processer

from sklearn import cross_validation, grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from Mscthesis.Preprocess.preprocess_rows import align_firms_data
from Mscthesis.Preprocess.financial_utils import f_corr


class GeneralComputer(Processer):
    """Factorization of computer manager class for specific model selected."""

    def _initialization(self):
        self.files = []

        self.subproc_desc = [""]
        self.t_expended_subproc = [0]

    def __init__(self, logfile, pathfolder, precomputers):
        self._initialization()
        self.pathfolder = pathfolder
        self.logfile = Logger(logfile) if logfile == str else logfile
        self.precomputers = precomputers

    def compute(self, listparams):
        ## 0. Set vars
        t00 = self.setting_global_process()
        # Prepare subprocesses
        n_cases = len(listparams)
        self.t_expended_subproc = [0 for e in range(n_cases)]
        self.subproc_desc = [self._name+"_"+str(e) for e in range(n_cases)]

        ## Computations and storing
        for i in range(len(listparams)):
            t0 = self.set_subprocess([0])
            self._store(*self.compute_i(*listparams[i]))
            self.close_subprocess([0], t0)
        self.files = os.listdir(self.pathfolder)
        self.listparams = listparams
        assert(len(self.files) == len(self.listparams))
        ## Untrack process
        self.close_process(t00)

    def _retrieve(self):
        listfiles = os.listdir(self.pathfolder)
        precomputed = []
        for namefile in listfiles:
            precomputed.append(self._retrieve_i(namefile))
        return precomputed


class Directmodel(GeneralComputer):
    """Based to apply directly a RandomForest model using the features given by
    the financial firms information.
    """

    def compute_i(self, model, parameters_grid, f_datafin, perturbations):
        ## Get point features
        n_pos_feats = len(self.precomputers.precomputer_pfeatures)
        n_pos_qvalues = len(self.precomputers.precomputer_qvalues)
        for i in range(n_pos_feats):
            for j in range(n_pos_qvalues):
                nif_feats, year_feats, pfeatures, methodvalues_feats =\
                    self.precomputers.precomputer_pfeatures[i]
                nif_qvals, qvalue, year_qvals, methodvalues_qvals =\
                    self.precomputers.precomputer_qvalues[j]
                reindices_qvals = join_matrix_nif_years(nif_feats, nif_qvals,
                                                        year_feats, year_qvals)
                nif_qvals, qvalue, year_qvals = nif_qvals[reindices_qvals],\
                    qvalue[reindices_qvals], year_qvals[reindices_qvals]

                ## Compute
                model, score = apply_sklearn_model(pfeatures, qvalue,
                                                   model, parameters_grid)
                self._store_i(model, score, i, j)

    def _store_i(self, model, score, i, j):
        modelname = 'RandomForestRegressor'
        precomputers_id = [('pfeatures', i), ('qvalues', j)]
        store_model(self.pathfolder, model, score, modelname, precomputers_id)


###############################################################################
############################ Individual functions #############################
###############################################################################
############################ RandomForest Finance #############################
###############################################################################
def rf_model_computations(pathpfeats, pathqvals, pars_model, f_datafin,
                          perturbations, samplings=None):
    ## Prepare data
    pfeats, qvals, year, nif = prepare_data_direct_model(pathpfeats, pathqvals)
    ## Apply model
    results = []
    for i in range(len(f_datafin)):
        results.append(loop4ftrans(f_datafin[i], pfeats, qvals))
    return results


def loop4ftrans(f_trans_i, pfeats, qvals):
    ## Prepare data
    pfeats_tr, qvals_tr = f_trans_i(pfeats, qvals)
    ## Apply model
    model, best_estimator_, best_score_ =\
        apply_rf(pfeats_tr, qvals_tr, pars_model)
    scores_pert = []
    for p in perturbations:
        pfeats_tr, qvals_tr = f_trans_i(p.apply2finance(pfeats), qvals)
        score = apply_scorer(model, pfeats_tr, qvals_tr)
        scores_pert.append(score)
    results = {'model': model, 'best_pars': best_estimator_}
    results['best_score'] = best_score_
    results['scores'] = scores_pert
    return results


def apply_rf(x, y, parameters_grid, samplings=None):
    rf_reg = RandomForestRegressor(oob_score=True, n_jobs=-1)
    clf_model = GridSearchCV(rf_reg, parameters_grid)
    clf_model.fit(x, y)
    return clf_model, clf_model.best_estimator_, clf_model.best_score_


def apply_scorer(model, pfeats_tr, qvals_tr):
    score = model.score(pfeats_tr, qvals_tr)
    return score


def f_datafin_zeros(pfeats, qvals):
    pfeats_tr = pfeats.copy()
    pfeats_tr[f_corr(pfeats)] = 0.
    return pfeats_tr, qvals


def f_datafin_nans(pfeats, qvals):
    pfeats_tr = pfeats.copy()
    qvals_tr = qvals.copy()
    logi = np.all(f_corr(pfeats), axis=1)
    pfeats_tr = pfeats_tr[logi]
    qvals_tr = qvals_tr[logi]
    return pfeats_tr, qvals_tr


def prepare_data_direct_model(pathpfeats, pathqvals):
    db = shelve.open(pathpfeats)
    pfeats, year_pf, nif_pf = db['pfeatures'], db['year'], db['nif']
    db.close()
    db = shelve.open(pathqvals)
    qvals, year_qv, nif_qv = db['qvalue'], db['year'], db['nif']
    db.close()

    iss_pf, iss_qv = align_firms_data(nif_pf, year_pf, nif_qv, year_qv)
    pfeats, qvals = pfeats[iss_pf], qvals[iss_qv]
    year = year_pf[iss_pf]
    nif = [nif_pf[i] for i in range(len(nif_pf)) if i in iss_pf]

    return pfeats, qvals, year, nif


def prepare_data_spatial_model(pathlocs, pathpfeats, pathqvals):
    db = shelve.open(pathlocs)
    locs, year_loc, nif_loc = db['locations'], db['year'], db['nif']
    db.close()
    db = shelve.open(pathpfeats)
    pfeats, year_pf, nif_pf = db['pfeatures'], db['year'], db['nif']
    db.close()
    db = shelve.open(pathqvals)
    qvals, year_qv, nif_qv = db['qvalue'], db['year'], db['nif']
    db.close()

    iss_pf, iss_qv = align_firms_data(nif_pf, year_pf, nif_qv, year_qv)
    pfeats, qvals = pfeats[iss_pf], qvals[iss_qv]
    year = year_pf[iss_pf]
    nif = [nif_pf[i] for i in range(len(nif_pf)) if i in iss_pf]
    iss, iss_loc = align_firms_data(nif, year, nif_loc, year_loc)
    year, nif = year[iss], [nif[i] for i in range(len(nif)) if i in iss]
    locs = locs[iss_loc]
    return locs, pfeats, qvals, year, nif


def spatialcounts_model_computations(pathfeats, pathqvals, pars_model,
                                     f_data, perturbations, samplings=None):
    ## Asserts inputs
    assert(type(f_data) == list)
    assert(type(perturbations) == list)

    locs, pfeats, qvals, year, nif =\
        prepare_data_spatial_model(pathplocs, pathpfeats, pathqvals)
    locs, pfeats = locs

    create_counter_types_matrix(locs, pfeats, retriever_o, pars_ret)

    models = []


def errors_finance_computation(pathdata, pars_get_data, model, perturb_info):
    ## Pars
    X, y = get_data_finance(pathdata, **pars_get_data)
    model = train_model(model, X, y)
    for i in range(len(perturb_info)):
        scores = apply_perturbs_finance(model, perturb_info[i], y)
    return scores


def train_model(model, X, y):
    correctness = f_all_corr(X)
    model = model.fit(X, y)
    return model


def get_data_finance(pathdata, method, pars):
    pass


def apply_perturbs_finance(model, perturbs, y):
    pass

