
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

#from Mscthesis.IO.aux_functions import parse_xlsx_sheet
#servicios = parse_xlsx_sheet(join(cleanedfilesdata, 'Ceuta_Melilla.xlsx'))

## Stats computation
study_info = {'author': 'Antonio G.', 'date': '', 'title': 'Study', 'path': join(os.getcwd(), statsfiledata)}
stats_container = Statistics(fileinstructions, study_info)
stats = stats_container.compute_stats(servicios)

#stats_container.to_latex(texoutput)
stats_container.to_latex(join(statsfiledata, 'report.tex'))
#doc = describe2latex(stats_container.study_info, stats_container.stats)

## Save
## Save data of the process
statsobj = "stats_object.dat"
statsobj = join(statsfiledata, statsobj)
database = shelve.open(statsobj)
database['info'] = stats_container.info
database['study_info'] = stats_container.study_info
database['stats'] = stats_container.stats

#database['stats_container'] = stats_container
#stats_container = database['stats_container']

