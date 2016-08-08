
"""
IO population data
------------------
IO population data.
"""

import pandas as pd
import numpy as np
import os


def parse_munipios_data(pathdata):
    pathfile = 'Cleaned/MunicipiosData/municipios-espana_2014_complete.csv'
    pathfile = os.path.join(pathdata, pathfile)
    municipios = pd.read_csv(pathfile, sep=';')
    pop_data = municipios['Poblacion'].as_matrix()
    pop_locs = municipios[['longitud', 'latitud']].as_matrix()
    return pop_data, pop_locs
