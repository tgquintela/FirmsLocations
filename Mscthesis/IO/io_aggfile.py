
"""
Module which groups all the aggregated precomputed information in order to
save computational power.
"""

import pandas as pd


def read_agg(filepath):
    "Read file of aggregated info."
    table = pd.read_csv(filepath, sep=';')
    return table
