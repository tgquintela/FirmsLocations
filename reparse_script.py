
"""
Script for parsing the data after being filtered with excel.
"""

#####
## Import data


## Setting files
texoutput = 'Data/results/tex_exploratory.tex'
fileinstructions = 'Data/about_data/stats_instructions.csv'
infilepath = 'Data/raw_data/SERVICIOS2'
outfilepath = 'Data/clean_data/SERVICIOS2'
parserfiledata = 'Data/parse_process_data/'
# Importing modules
from Mscthesis.IO import Servicios_Parser
from os.path import join
import shelve


## Parsing task
servicios_parser = Servicios_Parser(cleaned=False)
servicios_parser.get_index_from_cleaned(infilepath)
servicios_parser.parse_and_clean_by_file(infilepath, outfilepath)

## Save data of the process
parserobj = "parser_object.dat"
parserobj = join(parserfiledata, parserobj)
database = shelve.open(parserobj)
database['parser'] = servicios_parser
