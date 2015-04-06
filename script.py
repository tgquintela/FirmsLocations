
"""
Script.
"""

#####
## Import data
## Transform data to some useful (Filter rows included)
## Get the new columns needed
## Divide dataset in location, and finantial data
## Divide dataset


from IO import *
from preprocess import *


###############################################################################
############################## 0. PREPROCESS DATA #############################
###############################################################################

###### PREPROCESS OF SERVICIOS
##############################
### parse files
servicios = parse_servicios()
### Concat servicios
servicios = concat_from_dict(servicios, 'Region')
### Filter not interesting rows
# servicios = parse_xlsx_sheet('Data/SERVICIOS/Ceuta_Melilla.xlsx')
servicios, indices = filter_servicios(servicios)
### Save changes
write_dataframe_to_csv('servicios.csv', 'Data/clean_data')
# Delete
del servicios

###### PREPROCESS OF MANUFACTURERS
##################################
# # parse manufacterers
# manufacturers = parse_manufacturers()
# ### Save changes
# write_dataframe_to_csv('manufacturers.csv', 'Data/clean_data')
# # Delete
# del manufacturers

###### PREPROCESS OF CNAE
#########################
cnae_legend = parse_cnae()


###### PREPROCESS OF sERVICES LEGEND
####################################
legend_services = parse_legend_services()


###############################################################################
###############################################################################
###############################################################################
### Reload data
servicios = parse_xlsx_sheet('Data/clean_data/servicios.csv')


### Separate by columns
main_id = ['nif']
total_activo = ['06Act', '07Act', '08Act', '09Act', '10Act', '11Act', '12Act']
cols = main_id+total_activo
pos_geo = ['ES-X', 'ES-Y']
