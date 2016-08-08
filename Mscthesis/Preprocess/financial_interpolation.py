
"""
financial_interpolation
-----------------------
Financial cleaning and interpolation imputing.

"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from ..IO.io_standarized import get_atemporal_data,\
    get_sequencial_financial_data, write_financial


def financial_interpolation(pathdata_in):
    """"""
    empresas_atemporal = get_atemporal_data(pathdata_in)
    for empresas_ca, ca_name in get_sequencial_financial_data(pathdata_in):
        new_empresas_temp_ca =\
            financial_value_interpolation(empresas_atemporal, empresas_ca,
                                          ca_name)
        write_financial(pathdata_in, new_empresas_temp_ca, ca_name)


def financial_value_interpolation(empresas_atemporal, empresas_temp_ca,
                                  ca_name):
    def f_int(x):
        try:
            return int(x)
        except:
            return x
    empresas_ca = empresas_atemporal[empresas_atemporal['ca'] == ca_name]
    nifs_atemp = list(empresas_ca['nif'])

    y_open = open_dates_info(empresas_ca, empresas_temp_ca)
    y_correct = correct_financial_info(empresas_ca, empresas_temp_ca)
    features = assignation_features(empresas_ca, empresas_temp_ca)

    ## Concurrent imputation by temporal interpolation and knn regression
    ## Hardcoded stages: 4
    features = temporal_interpolation_imputation(features, y_open, y_correct)
    for t in range(4):
        features = knn_interpolation_imputation(features, y_open)
        features = temporal_interpolation_imputation(features, y_open)

    new_empresas_temp_ca = []
    for year in range(y_open.shape[1]):
        y_finc_year = np.logical_and(y_open[:, year],
                                     np.all(y_correct[:, :, year], axis=1))
        ## Temporal!!!:
        y_finc_year = y_open[:, year]
        nifs_year = [nifs_atemp[i] for i in range(len(nifs_atemp))
                     if y_finc_year[i]]
        emp = pd.concat([pd.DataFrame(nifs_year),
                         pd.DataFrame(features[y_finc_year, :, year])],
                        axis=1)
        emp.columns = empresas_temp_ca[year].columns
        emp.loc[:, emp.columns[5]] = emp[emp.columns[5]].apply(f_int)
        new_empresas_temp_ca.append(emp)

    return new_empresas_temp_ca


def temporal_interpolation_imputation(features, y_open, y_correct=None):
    if y_correct is None:
        y_correct = np.logical_not(np.logical_and(np.isnan(features),
                                                  features == 0))

    for i in range(len(features)):
        for j in range(len(y_correct[0])):
            logi = np.logical_and(np.logical_not(y_correct[i, j, :]),
                                  y_open[i, :])
            if np.any(logi):
                idxs = np.where(logi)[0]
                correct = y_correct[i, j, :]
                idxs_correct = np.where(correct)[0]
                for ixs in idxs:
                    weights = 1./np.abs(idxs_correct-ixs)
                    if weights.sum() == 0:
                        continue
                    values = features[i, j, correct]
                    new_val = np.dot(weights, values)/float(weights.sum())
                    features[i, j, ixs] = new_val
    return features


def knn_interpolation_imputation(features, y_open):
    pool_knn = np.vstack([features[:, :, i] for i in range(features.shape[2])])
    complete = np.sum(np.isnan(pool_knn), axis=1) == 0
    complete = np.logical_and(complete, np.sum(pool_knn == 0, axis=1) == 0)
    pool_knn = pool_knn[complete]
    means, stds = np.mean(pool_knn, axis=0), np.std(pool_knn, axis=0)

    ## Uni
    for i in range(features.shape[1]):
        missings = np.logical_or(np.isnan(features), features == 0)
        tr_f = [j for j in range(features.shape[1]) if j != i]
        reg = KNeighborsRegressor(5, 'distance')
        train = (pool_knn[:, tr_f]-means[tr_f])/stds[tr_f]
        test = (pool_knn[:, i]-means[i])/stds[i]
        reg = reg.fit(train, test)
        for k in range(features.shape[2]):
            trainable = np.all(np.logical_not(missings[:, tr_f, k]), axis=1)
            trainable = np.logical_and(trainable, missings[:, i, k])
            if trainable.sum() == 0:
                continue
            train_f = (features[trainable][:, tr_f, k]-means[tr_f])/stds[tr_f]
            predict = reg.predict(train_f)
            features[trainable, i, k] = predict*stds[i]+means[i]
    return features


def pos_computations(logis):
    pos = []
    for i in range(logis[:, i]):
        np.unique(logis[:, i])


def assignation_features(empresas_ca, empresas_temp_ca):
    var_list = ['act', 'actc', 'pasfijo', 'pasliq', 'trab', 'va', 'vtas']
    nifs = list(empresas_ca['nif'])
    feats = np.zeros((len(nifs), len(var_list), len(empresas_temp_ca)))
    for k in range(len(empresas_temp_ca)):
        columns = list(empresas_temp_ca[k].columns)
        nifs_temp = list(empresas_temp_ca[k]['nif'])
        nifs_idxs = [nifs.index(e) for e in nifs_temp]
        for j in range(len(var_list)):
            for i in range(len(columns)):
                if var_list[j] == columns[i].lower():
                    break
            feats[nifs_idxs, j, k] =\
                empresas_temp_ca[k][columns[i]].as_matrix()
    return feats


def open_dates_info(empresas_ca, empresas_temp_ca):
    y_open = np.zeros((len(empresas_ca), len(empresas_temp_ca))).astype(bool)
    for i in range(len(empresas_temp_ca)):
        gt_f = lambda x: 2006+i >= int(x[:4])
        lt_f = lambda x: 2006+i <= int(x[:4])
        y_open[:, i] = np.logical_and(empresas_ca['apertura'].apply(gt_f),
                                      empresas_ca['cierre'].apply(lt_f))
    return y_open


def correct_financial_info(empresas_ca, empresas_temp_ca):
    var_list = ['act', 'actc', 'pasfijo', 'pasliq', 'trab', 'va', 'vtas']
    nifs = list(empresas_ca['nif'])

    y_correct = np.zeros((len(nifs), len(var_list), len(empresas_temp_ca)))
    y_correct = y_correct.astype(bool)

    for k in range(len(empresas_temp_ca)):
        columns = list(empresas_temp_ca[k].columns)
        nifs_temp = list(empresas_temp_ca[k]['nif'])
        nifs_idxs = [nifs.index(e) for e in nifs_temp]
        for j in range(len(var_list)):
            for i in range(len(columns)):
                if var_list[j] == columns[i].lower():
                    break
            if var_list[j] == 'trab':
                logi =\
                    np.logical_or(pd.isnull(empresas_temp_ca[k][columns[i]]),
                                  empresas_temp_ca[k][columns[i]] == 0)
            else:
                logi = np.array(pd.isnull(empresas_temp_ca[k][columns[i]]))
            y_correct[nifs_idxs, j, k] = np.logical_not(logi)

    return y_correct
