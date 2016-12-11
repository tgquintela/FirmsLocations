
"""
Standarization io utils
-----------------------
"""

import os
import pandas as pd
import numpy as np


###############################################################################
############################## Writers functions ##############################
###############################################################################
def write_nif2names(nifs, names, pathdata):
    df = {}
    df['nif'] = nifs
    df['name'] = names

    df = pd.DataFrame(df)
    pathfile = os.path.join(os.path.join(pathdata, 'extra'), 'nif2names.xlsx')
#    df.to_csv(pathfile, sep=';')
    df.to_excel(pathfile)


def write_ca2code(u_CA, pathdata):
    u_CA = list(set(u_CA))
    sorted_CA = sorted(u_CA)
    df = {}
    df['CA'] = sorted_CA
    df['code'] = range(len(u_CA))

    df = pd.DataFrame(df)
    pathfile = os.path.join(os.path.join(pathdata, 'extra'), 'ca_codes')
    df.to_csv(pathfile, sep=';')


def write_nif2code(nifs, pathdata):
    sorted_nifs = sorted(nifs)

    df = {}
    df['nif'] = sorted_nifs
    df['code'] = range(len(nifs))

    df = pd.DataFrame(df)
    pathfile = os.path.join(os.path.join(pathdata, 'extra'), 'nif_codes')
    df.to_csv(pathfile, sep=';')


def write_cp2code(u_cps, pathdata):
    standarizor_cp = lambda cp: (5-len(str(int(cp))))*'0'+str(int(cp))
    u_cps = list(np.sort(u_cps))
    u_cps = sorted([standarizor_cp(cp) for cp in u_cps])

    df = {}
    df['cp'] = u_cps
    df['code'] = range(len(u_cps))

    df = pd.DataFrame(df)
    pathfile = os.path.join(os.path.join(pathdata, 'extra'), 'cp_codes')
    df.to_csv(pathfile, sep=';')


def write_ca_cp(ca_cp_dict, pathdata):
    df = {}
    df['CA'] = ca_cp_dict.values()
    df['CP'] = ca_cp_dict.keys()
    df = pd.DataFrame(df)
    pathfile = os.path.join(os.path.join(pathdata, 'extra'), 'cp-ca')
    df.to_csv(pathfile, sep=';')


def write_locs_statistics(mean_locs, std_locs, u_cps, pathdata):
    df = {}
    mean_locs = np.array(mean_locs)
    std_locs = np.array(std_locs)
    df['cp'] = [(5-len(str(int(e))))*'0'+str(int(e)) for e in u_cps]
    df['mean_latitude'] = mean_locs[:, 0]
    df['mean_longitud'] = mean_locs[:, 1]
    df['std_latitude'] = std_locs[:, 0]
    df['std_longitud'] = std_locs[:, 1]
    df = pd.DataFrame(df)
    pathfile = os.path.join(os.path.join(pathdata, 'extra'),
                            'locs-regions_statistics')
    df.to_csv(pathfile, sep=';')


def write_uncorrect_locs(nifs, raw_locs, pathdata):
    df = {}
    df['nif'] = nifs
    df['latitude'] = raw_locs[:, 0]
    df['longitud'] = raw_locs[:, 1]
    df = pd.DataFrame(df)
    pathfile = os.path.join(os.path.join(pathdata, 'extra'), 'missing_locs')
    df.to_csv(pathfile, sep=';')
