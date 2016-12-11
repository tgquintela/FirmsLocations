
"""
Computer utils
--------------
Auxiliar functions for main computer functions and objects.

"""

import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV

from sklearn import cross_validation, grid_search
from sklearn.ensemble import RandomForestRegressor

from Mscthesis.IO.precomputers_io import generate_yearnif_hash
from Mscthesis.Preprocess.preprocess_rows import align_firms_data


###############################################################################
################################## Examples ###################################
###############################################################################
#pathdata = "/home/antonio/Desktop/MSc Thesis/code/Data/Data/"
#pathfeats = "/Cleaned/Precomputed/Pfeatures/"
#parameters_grid = {'n_estimators': [10, 20], 'oob_score': [True]}
#
#filefeats0 = "2016-09-12-raw_type_firms_cnae2_-6727949354442861019"
#filefeats1 = "2016-09-12-tono_mag_536426093523753772"
#files_feats = [[pathdata+pathfeats+filefeats0, pathdata+pathfeats+filefeats1]]


###############################################################################
###################### Main auxiliar computer functions #######################
###############################################################################
def get_references_intersection(nif_feats, year_feats, pfeatures,
                                nif_qvals, year_qvals, qvalue):
    """Compute the intersection to apply the model."""
    # Get reindices
    reindices_feats, reindices_qvals =\
        join_matrix_nif_years(nif_feats, nif_qvals,
                              year_feats, year_qvals)
    # Reindices
    nif_ref = reindices_list(nif_feats, reindices_feats)
    year_ref = reindices_array(year_feats, reindices_feats)
    pfeatures = reindices_array(pfeatures, reindices_feats)
    if len(pfeatures.shape) == 1:
        pfeatures = pfeatures.reshape(-1, 1)
    qvalue = reindices_array(qvalue, reindices_qvals)
    return nif_ref, year_ref, pfeatures, qvalue


def join_loaded_features(hash_feats, nif_feats, year_feats, pfeatures,
                         methodvalues_feats):
    assert(len(nif_feats) == len(hash_feats))
    assert(len(nif_feats) == len(year_feats))
    assert(len(nif_feats) == len(pfeatures))
    assert(len(nif_feats) == len(methodvalues_feats))

    nif_ref, year_ref, indices = get_uniques_nif_year(nif_feats, year_feats)
    for k in range(len(nif_feats)):
        reindices_refs, reindices_feats =\
            join_matrix_nif_years(nif_ref, nif_feats[k],
                                  year_ref, year_feats[k])
        for k1 in range(k+1):
            pfeatures[k-k1] = reindices_array(pfeatures[k-k1], reindices_feats)
        nif_ref = reindices_list(nif_ref, reindices_refs)
        year_ref = reindices_array(year_ref, reindices_refs)
#        pfeatures[k] = pfeatures[k][indices[k]]
        if len(pfeatures[k].shape) == 1:
            pfeatures[k] = pfeatures[k].reshape(-1, 1)
    ## Concatenate
    pfeatures = np.concatenate(pfeatures, axis=1)
    return nif_ref, year_ref, pfeatures, methodvalues_feats


def join_matrix_nif_years(nif_feats, nif_qvals, year_feats, year_qvals):
    """Join the indices of the nif and years for different features and labels.
    """
    hashes_feats = generate_yearnif_hash(year_feats, nif_feats)
    hashes_qvals = generate_yearnif_hash(year_qvals, nif_qvals)

    ## Reindices for feats and qvals
    reindices_feats, reindices_qvals = [], []
    for i in range(len(hashes_feats)):
        try:
            j = hashes_qvals.index(hashes_feats[i])
            reindices_feats.append(i)
            reindices_qvals.append(j)
        except:
            pass

    return reindices_feats, reindices_qvals


def get_uniques_nif_year(nif_lists, year_lists):
    """Get only uniques combinations of nif and years."""
    nif_feats, year_feats, hashes_feats, indices = [], [], [], []
    for i in range(len(nif_lists)):
        indices.append([])
        hashes = generate_yearnif_hash(year_lists[i], nif_lists[i])
        for j in xrange(len(nif_lists[i])):
            if hashes[j] not in hashes_feats:
                hashes_feats.append(hashes[j])
                nif_feats.append(nif_lists[i][j])
                year_feats.append(year_lists[i][j])
                indices[i].append(j)
    return nif_feats, year_feats, indices


def separate_by_times(x, times):
    t = np.unique(times)
    if type(x) == list:
        x_t = []
        for t in times:
            logi = times == t
            x_ti = []
            for i in range(len(logi)):
                if logi[i]:
                    x_ti.append(x[i])
            x_t.append(x_ti)
    else:
        x_t = []
        for t in times:
            logi = x == t
            x_t.append(x[logi])
    return x_t


def get_ordered_locations(locations, years, nifs, year_ref, nif_ref):
    """Get ordered locations."""
    hash_locs = generate_yearnif_hash(years, nifs)
    hash_feat = generate_yearnif_hash(year_ref, nif_ref)
    new_indices = []
    for i in range(len(hash_feat)):
        new_indices.append(hash_locs.index(hash_feat[i]))
    locations_ordered = locations[new_indices]

    return locations_ordered


def get_ordered_regions(reg_data, nif_reg, nif_ref):
    """"""
    new_reg_data = []
    for i in range(len(nif_ref)):
        idx = nif_reg.index(nif_ref[i])
        new_reg_data.append(reg_data[idx])
    reg_data = np.concatenate(new_reg_data)
    return reg_data


###############################################################################
######################## Auxiliar nif-year management #########################
###############################################################################
# Move to preprocess_rows
def order_by(nif_feats, year_feats, by='nif'):
    """Order nifs and years by one of them."""
    df = pd.concat([pd.DataFrame(nif_feats), pd.DataFrame(year_feats)], axis=1)
    df.columns = [0, 1]
    if by == 'nif':
        ordering_hierharchy = [0, 1]
    else:
        ordering_hierharchy = [1, 0]
    df = df.sort(columns=ordering_hierharchy, ascending=[True, True])

    reindices = np.array(df.index)
    nif_feats = list(df[0])
    year_feats = np.array(df[1])

    return reindices, nif_feats, year_feats


def generate_yearnif_hash(years, nifs):
    """Hash for the years and nifs to ease and speed up the task of manage
    both arrays.
    """
    hashes = [hash(nifs[i]+str(years[i])) for i in range(len(years))]
    return hashes


def files_data_generation(candidates, namesmeasures):
    """Select all possible files that are related with the given measures."""
    selected = []
    for name in namesmeasures:
        sele_name = []
        for i in range(len(candidates)):
            if name in candidates[i]:
                sele_name.append(candidates[i])
        selected.append(sele_name)
    return selected


###############################################################################
############################ Individual functions #############################
###############################################################################
def apply_sklearn_model(x, y, model, parameters_grid, samplings=None):
    model_instance = model()
    clf_model = GridSearchCV(model_instance, parameters_grid)
    clf_model.fit(x, y)
    return clf_model, clf_model.best_estimator_, clf_model.best_score_


def reindices_array(array, reindices):
    assert(max(reindices) < len(array))
    return np.array(array)[reindices]


def reindices_list(lista, reindices):
    assert(max(reindices) < len(lista))
    return [lista[rei] for rei in reindices]


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


################################# In process ##################################
###############################################################################
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
