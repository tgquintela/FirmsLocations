


import numpy as np
import pandas as pd

from Mscthesis.IO import Firms_Parser
from Mscthesis.Retrieve.cnae_utils import transform_cnae_col
from Mscthesis.Preprocess.comp_complementary_data import aggregate_by_var,\
    average_position_by_cp, counting_type_by_cp

from Mscthesis.Geo_tools.geo_transformations import general_projection

from Mscthesis.Models.pjensen import Pjensen

outpath = '/home/antonio/Desktop/MSc Thesis/code/Data/Outputs/Pruebas'
logfile='/home/antonio/Desktop/MSc Thesis/code/Data/Outputs/Pruebas/log.log'


agg_var = 'cp'
loc_vars = ['ES-X', 'ES-Y']
type_vars = ['cnae']
radius= 0.005

## Parsing empresas
parser = Firms_Parser(cleaned=True, logfile=logfile)
empresas = parser.parse(outpath, year=2006)

## Transformation
empresas[loc_vars] = general_projection(empresas, loc_vars, method='ellipsoidal', inverse=False, radians=False)
empresas['cnae'] = transform_cnae_col(empresas['cnae'], 2)

## Correlation computation
pjensen = Pjensen(logfile=logfile,lim_rows=10000, proc_name='Prueba0')
out = pjensen.compute_net(empresas, type_vars, loc_vars, radius, permuts=2, agg_var=agg_var)




