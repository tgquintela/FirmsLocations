
"""
Script.
"""

#####
## Import data
## Transform data to some useful (Filter rows included)
## Get the new columns needed
## Divide dataset in location, and finantial data
## Divide dataset

## Setting files
texoutput = 'Data/results/tex_exploratory.tex'
fileinstructions = 'Data/about_data/stats_instructions.csv'
infilepath = 'Data/raw_data/SERVICIOS2'
outfilepath = 'Data/clean_data/SERVICIOS2'

from Mscthesis.IO import Servicios_Parser
from Mscthesis.Statistics import Statistics

## Parsing task
servicios_parser = Servicios_Parser(cleaned=False)
servicios_parser.parse_and_clean_by_file(infilepath, outfilepath)
#servicios = parse(outfilepath, cleaned=True)
#from Mscthesis.IO.aux_functions import parse_xlsx_sheet
#servicios = parse_xlsx_sheet('Data/raw_data/SERVICIOS/Ceuta_Melilla.xlsx')

parserobj = "parser_andalucia.dat"
import shelve
database = shelve.open(parserobj)
database['parser'] = servicios_parser


## Stats computation
#stats_container = Statistics(fileinstructions, study_info=None)
#stats = stats_container.compute_stats(servicios)
#stats_container.to_latex(texoutput)

