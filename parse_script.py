
"""
Script for parsing the data.
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
servicios_parser.parse_and_clean_by_file(infilepath, outfilepath)
#servicios = parse(outfilepath, cleaned=True)
#from Mscthesis.IO.aux_functions import parse_xlsx_sheet
#servicios = parse_xlsx_sheet('Data/raw_data/SERVICIOS/Ceuta_Melilla.xlsx')

## Save data of the process
parserobj = "parser_object.dat"
parserobj = join(parserfiledata, parserobj)
database = shelve.open(parserobj)
database['parser'] = servicios_parser

