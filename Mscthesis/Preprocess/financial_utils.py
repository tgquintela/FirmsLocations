
"""
Financial utils
---------------
Functions to tackle financial firms data.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


f_log = lambda x: np.log(np.abs(x))*np.sign(x)
f_corr = lambda x: np.logical_not(np.logical_or(np.isnan(x), x == 0))


def financial_size_computation(pfeatures):
    """Creation of the size magnitude from financial data."""
    correct_logi, _, correctable = obtain_correctness(pfeatures)
    mag_exact = create_correct_mag(pfeatures[correct_logi])
    _, models_i = magnitude_creation_model(pfeatures[correct_logi], mag_exact)
    mag = create_magnitude(pfeatures, models_i)
    return mag, correctable


def magnitude_creation_model(pfeatures, mag):
    models_i = []
    for i in range(pfeatures.shape[1]):
        lin = LinearRegression()
        lin.fit(f_log(pfeatures[:, [i]]), mag)
        models_i.append(lin)
    model = LinearRegression()
    model.fit(f_log(pfeatures[:, range(pfeatures.shape[1])]), mag)
    return model, models_i


def create_magnitude(pfeatures, models_i):
    correct_logi, pcorrect_logi, correctable = obtain_correctness(pfeatures)

    mag = np.zeros(len(pfeatures))
    mag[correct_logi] = create_correct_mag(pfeatures[correct_logi])
    mag[pcorrect_logi] = create_pcorrect_mag(pfeatures[pcorrect_logi],
                                             models_i)
    mag = mag[correctable]
    return mag


def obtain_correctness(pfeatures):
    nfeats = pfeatures.shape[1]
    correct_logi = np.all(f_corr(pfeatures[:, range(nfeats)]), axis=1)
    pcorrect_logi = np.any(f_corr(pfeatures[:, range(nfeats)]), axis=1)
    pcorrect_logi = np.logical_and(np.logical_not(correct_logi), pcorrect_logi)
    correctable = np.logical_or(correct_logi, pcorrect_logi)
    return correct_logi, pcorrect_logi, correctable


def create_correct_mag(pfeatures):
    mag_act = f_log(pfeatures[:, 0])
    mag_pas = f_log(pfeatures[:, 2]+pfeatures[:, 3])
    mag = (mag_pas+mag_act)/2.
    ## Adding a quantification of default loses by closing (mean cutoff)
    # That makes all possitives and closing is always losing (bad)
    mag = mag+mag.mean()
    return mag


def create_pcorrect_mag(pfeatures, models_i):
    mag = np.zeros((len(pfeatures), pfeatures.shape[1]))
    for i in range(pfeatures.shape[1]):
        logi = f_corr(pfeatures[:, i])
        mag[logi, i] = models_i[i].predict(f_log(pfeatures[logi][:, [i]]))

    mag = mag.sum(1)/(mag != 0).sum(1)
    return mag
