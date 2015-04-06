
"""
Script for computing the stats.
"""

## Setting files
texoutput = 'Data/results/tex_exploratory.tex'
fileinstructions = 'Data/about_data/stats_instructions.csv'
cleanedfilesdata = 'Data/clean_data/cleaned_data'

# Importing modules
from Mscthesis.IO import Servicios_Parser
from Mscthesis.Statistics import Statistics
from os.path import join
import shelve


## Parse files
servicios_parser = Servicios_Parser(cleaned=False)
servicios = servicios_parser.parse(cleanedfilesdata)
print 'Data parsed. Starting computing stats.'

## Stats computation
stats_container = Statistics(fileinstructions, study_info=None)
stats = stats_container.compute_stats(servicios)
stats_container.to_latex(texoutput)
