
"""
Script to clean data from SABI dataset using clean_module

"""


from Mscthesis.IO.clean_module import clean, creation_aggregate

inpath = 
outpath = 
agg_var = 'cp'
loc_vars = ['ES-X', 'ES-Y']


clean(inpath, outpath)
aggregate_by_mainvar(outpath, agg_var, loc_vars)
