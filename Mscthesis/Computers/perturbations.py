
"""
Perturbations
-------------

"""

import numpy as np
import os
import shelve


class Perturbation:

    def _store(self, pathdata, pars, namefile):
        os.path.join(pathdata, namefile)
        db = shelve.open(pathdata)
        db['perturbation'] = self.perturbation
        db['type'] = self._type
        db['pars'] = pars
        db.close()


class Permutation(Perturbation):
    _type = 'permutation'

    def __init__(self, info):
        if type(info) == int:
            self.perturbation = np.random.permutation(info)
        elif type(info) == list:
            info = np.array(info)
        if type(info) == np.ndarray:
            uniques = np.unique(info)
            permutation = -1*np.ones(len(info))
            for u in uniques:
                logi = u == info
                p = np.random.permutation(logi.sum())
                permutation[logi] = np.where(logi)[0][p]
            assert((permutation == -1).sum() == 0)
            self.perturbation = permutation.astype(int)

    def apply2finance(self, finance):
        return finance[self.perturbation]

    def apply2locations(self, locations):
        return locations[self.perturbation]


class FinancialJittering(Perturbation):
    _type = 'financialjittering'

    def _init__(self, finance):
        stds = np.nanstd(finance, axis=0)
        self.perturbation = np.random.random(finance.shape)*stds

    def apply2finance(self, finance):
        return finance + self.perturbation

    def apply2locations(self, locations):
        return locations


class LocationJittering(Perturbation):
    _type = 'locationjittering'

    def __init__(self, locs_stds):
        self.perturbation = locs_stds*np.random.random((len(locs_stds), 2))

    def apply2finance(self, finance):
        return finance

    def apply2locations(self, locations):
        assert(locations.shape == self.perturbation.shape)
        return locations + self.perturbation
