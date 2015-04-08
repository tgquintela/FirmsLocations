
"""
Script for computing the stats.
"""

## Setting files
texoutput = 'Data/results/tex_exploratory.tex'
fileinstructions = 'Data/about_data/stats_instructions.csv'
cleanedfilesdata = 'Data/clean_data/prunedcleaned_data'

statsfiledata = 'Data/Outputs'

# Importing modules
from Mscthesis.IO import Servicios_Parser
from Mscthesis.Statistics import Statistics
from os.path import join
import os
import shelve
import time

from Mscthesis.IO.output_to_latex import describe2latex


## Parse files
t0 = time.time()
servicios_parser = Servicios_Parser(cleaned=False)
servicios = servicios_parser.parse(cleanedfilesdata)
print 'Data parsed in %f seconds. Starting computing stats.' % (time.time()-t0)

## Stats computation
stats_container = Statistics(fileinstructions, study_info=None)
stats = stats_container.compute_stats(servicios)

## Save
## Save data of the process
statsobj = "stats_object.dat"
statsobj = join(statsfiledata, statsobj)
database = shelve.open(statsobj)
#database['stats_container'] = stats_container
stats_container = database['stats_container']

#stats_container.to_latex(texoutput)
stats_container.study_info = {'author': 'Antonio G.', 'date': '', 'title': 'Study', 'path': join(os.getcwd(), statsfiledata)}
doc = describe2latex(stats_container.study_info, stats_container.stats)
#Write doc
with open(texoutput, 'r+') as myfile:
    myfile.write(doc)
myfile.close()
