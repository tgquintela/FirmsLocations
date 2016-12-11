
"""
Precomputers functions
----------------------
The functions used by the precomputation step.
"""

import numpy as np

from ..Retrieve.cnae_utils import transform_cnae_col, transform_sector_col
#from ..IO.precomputers_io import get_finance  # , get_whole_data_firms
from ..IO.io_standarized import get_sequencial_financial_data,\
    get_sector_data, get_financial_data
from ..Preprocess.financial_utils import financial_size_computation

raw_features = []
type_features = ['sector', 'cnae']
financial_features = []


###############################################################################
############################ Pfeatures computation ############################
###############################################################################
### Collection of possible functions to compute the possible features
def general_pfeatures_computation(pathdata, method, pars):
    """General pfeatures computation function switcher."""
    ## 0. Parse and prepare data
    method = 'raw_finance' if method == '' else method
    ## 1. Switcher of functions
    if method == 'raw_finance':
        finance_data, year, nif = get_financial_data(pathdata)
        pfeatures, year, nif =\
            raw_finance_pfeatures_computation(finance_data, year, nif)
    elif method == 'raw_type_firms':
        pfeatures, year, nif = raw_type_pfeatures_computation(pathdata, **pars)
    elif method == 'financial_magnitude':
        finance_data, year, nif = get_financial_data(pathdata)
        pfeatures, year, nif =\
            financial_magnitude_pfeatures_computation(finance_data, year, nif)
#    elif method == 'gibrat_value':
#        finance_data, year, nif = get_financial_data(pathdata)
#        pfeatures = apply_gibrat_trans(finance_data, year, **pars)
    elif method == 'diff_financial':
        pfeatures = diff_financial_pfeatures_computation(pathdata, **pars)
    return nif, year, pfeatures


def raw_type_pfeatures_computation(pathdata, lvl):
    pfeatures, year, nif = get_sector_data(pathdata)
    cnae = transform_cnae_col(pfeatures['cnae'], lvl).as_matrix()
    sector = transform_sector_col(pfeatures['sector']).as_matrix()
    pfeatures = np.stack([cnae, sector]).T
    pfeatures, year, nif = join_types(pfeatures, nif, year)
    return pfeatures, year, nif


def join_types(pfeats, nif, year):
    nif_uniques = set(nif)
    pfeatures, nif_clean, year_clean = [], [], []
    for nif_u in nif_uniques:
        indices0 = [i for i in range(len(nif)) if nif_u == nif[i]]
        year_i0 = year[indices0]
        year_uniques = np.unique(year_i0)
        for y in year_uniques:
            idxs = np.where(year_i0 == y)[0]
            is1 = [indices0[i] for i in idxs]
#            print pfeats[is1], nif_u, year_i0
            ######## WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ### Collapse the cnae to the servicios better (more descriptive)
#            assert(len(np.unique(pfeats[is1, 0])) == 1)
#            assert(len(is1) == 1)
            s_ind = is1[0] if len(is1) > 1 else is1[0]
            aux = np.array([pfeats[s_ind, 0],
                            round(np.mean(pfeats[is1, 1]), 1)])
            pfeatures.append(aux)
            year_clean.append(y)
            nif_clean.append(nif_u)
    pfeatures = np.array(pfeatures)
    year_clean = np.array(year_clean)
    return pfeatures, year_clean, nif_clean


def financial_magnitude_pfeatures_computation(finance_data, year, nif):
    ## Get raw financial data unique
    pfeatures, year, nif =\
        raw_finance_pfeatures_computation(finance_data, year, nif)

    mag, correctable = financial_size_computation(pfeatures)

    year = year[correctable]
    nif = [nif[i] for i in range(len(nif)) if correctable[i]]
    return mag, year, nif


def raw_finance_pfeatures_computation(finance_data, year, nif):
    years_u = np.unique(year)
    pfeatures, new_year, new_nif = [], [], []
    for year_i in years_u:
        year_filter = year == year_i
        pfeatures_year = finance_data[year_filter]
        nif_year = [nif[i] for i in range(len(nif)) if year_filter[i]]
        nifs_u = set(nif_year)
        for nif_y in nifs_u:
            idxs = [i for i in range(len(nif_year)) if nif_y == nif_year[i]]
            pfeatures_year_nif = collapse_pfeatures_nif(pfeatures_year[idxs])
            pfeatures.append(pfeatures_year_nif)
            new_nif.append(nif_y)
            new_year.append(year_i)
    new_year = np.array(new_year)
    pfeatures = np.array(pfeatures)
    return pfeatures, new_year, new_nif


def collapse_pfeatures_nif(pfeatures):
    if len(pfeatures.shape) == 1:
        return pfeatures
    f_corr = lambda x: np.logical_not(np.logical_or(np.isnan(x), x == 0))
    correctness = f_corr(pfeatures)
    new_pfeatures = []
    for col in range(pfeatures.shape[1]):
        if correctness[:, col].sum():
            i = np.where(correctness[:, col])[0][0]
        else:
            i = 0
        new_pfeatures.append(pfeatures[i, col])
    new_pfeatures = np.array(new_pfeatures)
    return new_pfeatures


def diff_financial_pfeatures_computation(pathdata, method, pars):
    ## Function transformation
    if method == 'raw':
        f = lambda x, p={}: x
    elif method == 'gibrat':
        pass

    ## Computation
    nifs, finance_data = [], []
    for finance, ca_name in get_sequencial_financial_cleaneddata(pathdata):
        finance_cols = [c for c in finance[0].columns if c != 'nif']
        for year_i in range(len(finance)-1):
            ## NIFS
            nifs += list(finance[year_i]['nif'])
            nif_y = list(finance[year_i]['nif'])
            nif_y1 = list(finance[year_i+1]['nif'])
            ## Finance data
            fin_y = f_tr(finance[year_i][finance_cols].as_matrix(), **pars)
            fin_y1 = f_tr(finance[year_i+1][finance_cols].as_matrix(), **pars)
            ## Computation
            diff_m = np.zeros(fin_y.shape)
            for i in range(len(fin_y)):
                if nif_y[i] in nif_y1:
                    j = nif_y1.index(nif_y[i])
                    diff_m[i] = fin_y1[j]-fin_y[i]
                else:
                    diff_m[i] = diff_m[i]-fin_y[i]
            finance_data.append(diff_m)
    finance_data = np.concatenate(finance_data)
    return finance_data


###############################################################################
##################### Population interpolation computation ####################
###############################################################################
def general_population_interpolation(pathdata, method, pars):
    muni_parser = Municipios_Parser(logfile)
    muni_data, muni_vars = muni_parser.parse(muni_file)
    muni_locs = muni_data[muni_vars['loc_vars']].as_matrix()
    muni_feat = muni_data[muni_vars['feat_vars']].as_matrix()

    general_geo_interpolation(muni_locs, muni_feat, sp_data_int)
