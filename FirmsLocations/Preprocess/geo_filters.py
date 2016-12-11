
"""
Geofilters
----------
Filters coded oriented to filter and detect uncorrect data.

"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KDTree
from pySpatialTools.Preprocess.Transformations.Transformation_2d.geo_filters\
    import check_in_square_area


def check_correct_spain_coord(coord, radians=False):
    "Check if the coordinates given are in Spain or not."
    coord = np.array(coord)
    lim_points = np.array([[-18.25, 4.5], [27.75, 44]])
    if radians:
        lim_points = np.pi/180*lim_points
    logi = check_in_square_area(coord, lim_points)
    return logi


def filter_uncorrect_coord_spain(data, coord_vars, radians=False):
    "Filter not corrrect spain coordinates."
    coord = data[coord_vars].as_matrix()
    logi = check_correct_spain_coord(coord, radians)
    return data[logi]


def filter_bool_uncorrect_coord_spain(data, coord_vars, radians=False):
    "Filter data from pandas dataframe structure."
    coord = data[coord_vars].as_matrix()
    logi = check_correct_spain_coord(coord, radians)
    return logi


def fill_locations_by_region_jittering(locations, uncorrect, regions):
    """Creation random locations for uncorrect locations."""
    ## 0. Preparing computations
    new_locations = locations[:]
    u_regs = np.unique(regions)
    regs_mean_locs = []
    regs_std_locs = []

    ## 1. Computing statistical correct locations
    for reg in u_regs:
        logi = np.logical_and(regions == reg, np.logical_not(uncorrect))
        reg_mean_locs = np.mean(locations[logi], axis=0)
        reg_std_locs = np.std(locations[logi], axis=0)

        regs_mean_locs.append(reg_mean_locs)
        regs_std_locs.append(reg_std_locs)

    ## 2. Computing new locations for uncorrect
    idxs = np.where(uncorrect)[0]
    new_locs = []
    for i in idxs:
        reg = regions[i]
        i_reg = np.where(u_regs == reg)[0][0]
        # Random creation
        loc = np.random.random(2)*regs_std_locs[i_reg] + regs_mean_locs[i_reg]
        new_locs.append(loc)

    ## 3. Replace
    new_locations[uncorrect] = np.array(new_locs)
    return new_locations


def get_statistics2fill_locations(locations, regions):
    ## 0. Preparing computations
    correct = check_correct_spain_coord(locations)
    regions = [e for e in regions if e != float('nan') and e != np.nan]
    u_regs = np.unique(regions)
    regs_mean_locs = []
    regs_std_locs = []

    ## 1. Computing statistical correct locations
    for reg in u_regs:
        logi = np.logical_and(regions == reg, correct)
        reg_mean_locs = np.mean(locations[logi], axis=0)
        reg_std_locs = np.std(locations[logi], axis=0)

        regs_mean_locs.append(reg_mean_locs)
        regs_std_locs.append(reg_std_locs)
    return regs_mean_locs, regs_std_locs, u_regs


def fill_locations(df, loc_vars, reg_var, mean_locs, std_locs, u_regs):
    ## 0. Preparation computations
    locs = df[loc_vars].as_matrix()
    regions = df[reg_var].as_matrix()
    correct = check_correct_spain_coord(locs)
    idxs = np.where(np.logical_not(correct))[0]

    ## 1. Compute new locations
    new_locs = []
    for i in idxs:
        reg = regions[i]
        i_reg = np.where(u_regs == reg)[0][0]
        # Random creation
        loc = np.random.random(2)*std_locs[i_reg] + mean_locs[i_reg]
        new_locs.append(loc)

    df[loc_vars][np.logical_not(correct)] = np.array(new_locs)
    return df


###############################################################################
############################ Auxiliar to cleanning ############################
###############################################################################
def fill_nulls(df, mean_locs, std_locs, u_cps, raw_muni, raw_cps, raw_locs,
               pathdata):
    loc_vars, reg_var = ['es-x', 'es-y'], 'cp'
    locs = df[loc_vars].as_matrix()

    null_locs = np.logical_not(check_correct_spain_coord(locs))
    null_cps = pd.isnull(df[reg_var]).as_matrix()
    null_possible = np.array([e in u_cps for e in list(df['cp'])]).astype(bool)
    null_imp = np.logical_and(np.logical_not(null_possible), null_locs)
    null_both = np.logical_or(np.logical_and(null_locs, null_cps), null_imp)
    null_neither = np.logical_and(np.logical_not(null_locs),
                                  np.logical_not(null_cps))
#    print null_locs.sum(), null_cps.sum(), null_both.sum()
    null_cps2locs = np.logical_and(null_locs, np.logical_not(null_cps))
    null_cps2locs = np.logical_and(null_cps2locs, null_possible)
    null_locs2cps = np.logical_and(null_cps, np.logical_not(null_locs))
#    print null_both.sum(), null_cps2locs.sum(), null_locs2cps.sum()
#    print null_locs.sum(), null_cps.sum(), null_imp.sum()

    ## Inputing locations from cp
    if null_cps2locs.sum():
        new_locs = create_cp2locs(mean_locs, std_locs, u_cps, null_cps2locs,
                                  list(df['cp']))
        df_null_locs = pd.DataFrame({'nif': list(df['nif'][null_cps2locs]),
                                     'es-x': new_locs[:, 0],
                                     'es-y': new_locs[:, 1]})
        df['es-x'][null_cps2locs] = new_locs[:, 0]
        df['es-y'][null_cps2locs] = new_locs[:, 1]
    else:
        df_null_locs = pd.DataFrame({'nif': [], 'es-x': [], 'es-y': []})
    df_null_locs.to_csv(os.path.join(pathdata, 'cps2locs'), sep=';')

    ## Inputing cp from locations
    if null_locs2cps.sum():
        new_cps = create_locs2cp(locs, null_locs2cps, raw_locs, raw_cps)
        df_null_cps = pd.DataFrame({'nif': list(df['nif'][null_locs2cps]),
                                    'cp': list(new_cps)})
        df['cp'][null_locs2cps] = new_cps
    else:
        df_null_cps = pd.DataFrame({'nif': [], 'cp': []})
    df_null_cps.to_csv(os.path.join(pathdata, 'locs2cps'), sep=';')

    ## Inputing cp and locations from municipio
#    localidades = list(df['localidad'][null_both])
#    localidades_known = list(df['localidad'][np.logical_not(null_both)])
#    cp
#    new2_cps, new2_locs = create_locsandcp()
    localidades = [e.strip().lower() for e in list(df['localidad'][null_both])]

    df_null_both = pd.DataFrame({'nif': list(df['nif'][null_both]),
#                                 'localidad': localidades,
                                 'cp': list(df['cp'][null_both]),
                                 'es-x': df['es-x'][null_both],
                                 'es-y': df['es-y'][null_both]})
#                                 'cp': list(new2_cps),
#                                 'es-x': new2_locs[:, 0],
#                                 'es-y': new2_locs[:, 1]})
    df_null_both.to_csv(os.path.join(pathdata, 'nulllocsandcps'), sep=';')

#    df['cp'][null_both] = new2_cps
#    df['es-x'][null_both] = new2_locs[:, 0]
#    df['es-y'][null_both] = new2_locs[:, 1]
#    print df.shape, null_neither.sum()
    df = df[null_neither]

    return df


def create_cp2locs(mean_locs, std_locs, u_regs, uncorrect, regions):
    idxs = np.where(uncorrect)[0]
    new_locs = []
    for i in idxs:
        reg = regions[i]
        i_reg = np.where(u_regs == reg)[0][0]
        # Random creation
        loc = np.random.random(2)*std_locs[i_reg] + mean_locs[i_reg]
        new_locs.append(loc)
    new_locs = np.array(new_locs)
    return new_locs


def create_locs2cp(locs, null_locs2cps, raw_locs, raw_cps):
    locs_cp = locs[null_locs2cps]
    new_cps = retrieve_7major_cp(locs_cp, raw_locs, raw_cps)
    return new_cps


def retrieve_7major_cp(locs, raw_locs, raw_cps):
    raw_cps = np.array(raw_cps).astype(int)
    ret = KDTree(raw_locs)
    new_cps = []
    for i in range(len(locs)):
        neighs = ret.query(locs[[i]], 7)[1].ravel()
        c = Counter([raw_cps[nei] for nei in neighs])
        new_cps.append(c.keys()[np.argmax(c.values())])
    return new_cps


def create_locsandcp():
    pass
