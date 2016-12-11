
"""
set computation parameters
--------------------------
Computation parameters which depends now in the path of files.

"""

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold

from pySpatialTools.Retrieve import KRetriever, CircRetriever
from pySpatialTools.utils.perturbations import NonePerturbation,\
    PermutationPerturbationGeneration
from pythonUtils.sklearn_tools.cross_validation import KFold_list

from collection_auxiliar_functions import *
from collection_creation_functions import *


## Collection of possible files
_pfeatures_files = ['2016-11-01-raw_finance_-7893940389519280411',
                    '2016-11-01-raw_type_firms_cnae2_-6727949354442861019',
                    ]
_qvalues_files = ['2016-11-01-tono_mag_diff_-7344402660780134763',
                  ]

## Collection of possible parameters
#models_direct_info =\
#    [('rf_reg_nest10', RandomForestRegressor, {"n_estimators": 10}),
#     ('rf_reg_nest25', RandomForestRegressor, {"n_estimators": 25}),
##     ('rf_reg_nest50', RandomForestRegressor, {"n_estimators": 50}),
##     ('rf_reg_nest75', RandomForestRegressor, {"n_estimators": 75})
#     ]
#scorer_info = [('r2_score', r2_score, {}, dummy_function_conversion), ]
#perts_info = [('none_perturb1', NonePerturbation, {}, f_pert_null_instantiation),
#              ('globalpermut', PermutationPerturbationGeneration,
#               {'seed': 0}, f_pert_features_instantiation),
#              ('')
#              ]
#samplings_info = [('kfold10', KFold, {"n_folds": 10})]

perts_info =\
    create_permutation_feature(2, rate_pert=1.0, name=None) +\
    create_permutation_feature(2, rate_pert=0.8, name=None)
format_info =\
    create_null_format_info()
models_direct_info =\
    creation_models(RandomForestRegressor, 'rf_reg',
                    [("n_estimators", [10], 'nest')])
samplings_info =\
    creation_sampling(KFold, 'kfold', [("n_folds", [10], '')], f_stringer=None)
scorer_info =\
    creation_scorers(r2_score, 'r2_score', [])


## Collection of possible list of parameters
pars_dir_model0 = (perts_info, format_info, models_direct_info, samplings_info,
                   scorer_info)
print pars_dir_model0

## Final parameter list collection
#pars_directmodel =\
#    [((_pfeatures_files[0], _qvalues_files[0], f_filter_finance),
#      pars_dir_model0,
#      'finance-mag-DirectModel-None_perturb-None_filter-rf_reg-Kfold-r2_score'),
#     ((_pfeatures_files[0], _qvalues_files[0], f_filter_logfinance),
#      pars_dir_model0,
#      'financefilt-mag-DirectModel-None_perturb-None_filter-rf_reg-Kfold-r2_score')
#     ]


perts_sptemp_info =\
    create_permutation_sptemp(2, rate_pert=1.0, name=None) +\
    create_permutation_sptemp(2, rate_pert=0.8, name=None)
format_sptemp_info =\
    create_null_sptemp_format_info()
models_sptemp_info = [('None_model')]
samplings_sptemp_info =\
    creation_sampling(KFold_list, 'Kfold_sptemp', [("n_folds", [3], '')],
                      f_stringer=None)
scorer_sptemp_info =\
    creation_scorers(r2_score, 'r2_score', [])


pars_loc_model0 = (perts_sptemp_info, format_sptemp_info, models_sptemp_info,
                   samplings_sptemp_info, scorer_sptemp_info)

pars_loconly_model =\
    [((_pfeatures_files[0], _qvalues_files[0], f_filter_finance),
      pars_loc_model0,
      'finance-mag-LocOnlyModel-None_perturb-None_filter-rf_reg-Kfold-r2_score'),
     ]
