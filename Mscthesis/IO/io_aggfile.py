
"""
Module which groups all the aggregated precomputed information in order to
save computational power.
"""

import pandas as pd
from Mscthesis.Preprocess.preprocess_cols import cp2str

def read_agg(filepath):
    "Read file of aggregated info."
    table = pd.read_csv(filepath, sep=';')
    table = cp2str(table)
    return table
