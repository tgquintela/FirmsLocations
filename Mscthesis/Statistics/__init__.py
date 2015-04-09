
"""
Module oriented to compute the desired statistical description of each
selected variables or bunch of variables with the selected method.
"""

import pandas as pd
import numpy as np
from stats_functions import compute_stats
from Mscthesis.IO.parse_data import parse_instructions_file
from Mscthesis.IO.output_to_latex import describe2latex
import time


class Statistics():
    """The object which performs the computation of the statistics.

    TODO
    ----
    Check if the variables in the info are in the dataframe and act in
    consequence.

    """

    def __init__(self, fileinstructions, study_info={}):
        '''Initialization of the stats computation.'''
        describ_info = parse_instructions_file(fileinstructions)
        self.info = describ_info
        self.stats = None
        self.study_info = study_info

    def compute_stats(self, dataframe, info=None):
        '''Function to compute the statistics for all the columns.'''
        ## 0. Prepare inputs
        self.info = self.info if info is None else info
        t0 = time.time()
        ## 1. Compute stats
        stats = []
        for i in self.info.index:
            t1 = time.time()
            info_var = dict(self.info.iloc[i])
            s = "The stats of variable %s has been computed in %f seconds."
            stats.append(compute_stats(dataframe, info_var))
            print s % (info_var['variables'], time.time()-t1)
        ## 2. Save and return
        self.stats = stats
        aux = pd.DataFrame(np.sum(dataframe.notnull()), columns=['non-null'])
        self.study_info['global_stats'] = aux
        print "Stats computed in %f seconds." % (time.time()-t0)
        return stats

    def to_latex(self, filepath=None):
        ## 1. Compute transformation
        doc = describe2latex(self.study_info, self.stats)
        ## 2. Write output
        if filepath is None:
            return doc
        else:
            #Write doc
            with open(filepath, 'w') as myfile:
                myfile.write(doc)
            myfile.close()
