
"""
set precomputation parameters
-----------------------------
Parameters to set the precomputations in order to facilitate the computation
model tasks

"""

max_r0 = 20.
max_r1 = 50.
S0 = 3.4105868102821946
S1 = 8.770781287761192
#### Parameters lists
pars_qvals = [{'codename': 'porportionaldates',
               'pars': {'method': 'dates',
                        'pars': {'method': 'proportional', 'params': {}}}},
              {'codename': 'tono_mag_diff',
               'pars': {'method': 'financial',
                        'pars': {'method': 'diff_magnitude',
                                 'params': {'methodname': 'raw_finance'}}}}
              ]
pars_pfeatures = [{'codename': 'raw_finance',
                   'pars': {'method': 'raw_finance', 'pars': {}}},
                  {'codename': 'raw_type_firms_cnae2',
                   'pars': {'method': 'raw_type_firms', 'pars': {'lvl': 2}}},
                  {'codename': 'tono_mag',
                   'pars': {'method': 'financial_magnitude', 'pars': {}}}
                  ]
pars_pop = [{'codename': 'pst_gaus20',
             'pars': {'method': 'pst',
                      'pars': {'ret': {'info_ret': max_r0},
                               'interpolation': {'f_weight': 'gaussian',
                                                 'pars_w': {'max_r': max_r0,
                                                            'S': S0},
                                                 'f_dens': 'weighted_avg',
                                                 'pars_d': {}}}}},
            {'codename': 'pst_gaus50',
             'pars': {'method': 'pst',
                      'pars': {'ret': {'info_ret': max_r1},
                               'interpolation': {'f_weight': 'gaussian',
                                                 'pars_w': {'max_r': max_r1,
                                                            'S': S1},
                                                 'f_dens': 'weighted_avg',
                                                 'pars_d': {}}}}}
            ]
pars_locs = [{'codename': 'ellipsoidal_proj',
              'pars': {'method': 'ellipsoidal', 'radians': False}}]
pars_regs = [{'codename': 'regions', 'pars': {'columns': [0]}}]
