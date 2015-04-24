
import time
from os.path import join
import numpy as np
import pandas as pd

#### PARSE DATA
#from Mscthesis.IO.aux_functions import parse_xlsx_sheet
#import pandas as pd
#import numpy as np
#from Mscthesis.Preprocess.preprocess import cnae2str

#municipios = parse_xlsx_sheet('Data/municipios-espana_2014_complete.xls')
#servicios2 = parse_xlsx_sheet('Data/clean_data/prunedcleaned_data/#Extremadura.xlsx')
#region = pd.DataFrame('Extremadura', index=servicios.index, columns=['Region'])
#servicios = pd.concat([servicios, region], axis=1)
#servicios = cnae2str(servicios)

texoutput = 'Data/results/tex_exploratory.tex'
fileinstructions = 'Data/about_data/stats_instructions.csv'
cleanedfilesdata = 'Data/clean_data/pruebadata'

statsfiledata = 'Data/Outputs'

# Importing modules
from Mscthesis.IO import Servicios_Parser
from os.path import join
import time

## Parse files
t0 = time.time()
servicios_parser = Servicios_Parser(cleaned=False)
servicios = servicios_parser.parse(cleanedfilesdata)
print 'Data parsed in %f seconds. Starting computing stats.' % (time.time()-t0)

#### TRANSFORM COORDINATES
from Mscthesis.Geo_tools.geo_transformations import transf4compdist_global_homo
from Mscthesis.Geo_tools.geo_filters import filter_uncorrect_coord_spain
from Mscthesis.Retrieve.cnae_utils import transform_cnae_col

types_cc = {'canarias': ['Canarias'], 'ceutamelilla': ['Ceuta_Melilla']}
var_cc = ['Region']
types_cc['peninsula'] = [e for e in servicios[var_cc[0]].unique() if e not in types_cc['canarias']+types_cc['ceutamelilla']]
region = pd.DataFrame('', index=servicios.index, columns=['Lat_sector'])
for e in types_cc:
    logi = np.zeros(region.shape[0])
    for e2 in types_cc[e]:
	logi = np.logical_or(logi, servicios[var_cc[0]] == e2)
    region[logi] = e
data = pd.concat([servicios[['cnae', 'ES-X', 'ES-Y']], region], axis=1)
loc_vars = ['ES-X', 'ES-Y']
loc_zone_var = ['Lat_sector']
del servicios


data = data.dropna(how='all')
#data = data[data['ES-X'] != 0]
data = filter_uncorrect_coord_spain(data, loc_vars)

data.index = range(data.shape[0])
data = transf4compdist_global_homo(data, loc_vars)

#data[['ES-X', 'ES-Y']] = 

#### GET CNAE index level specified
data['cnae'] = transform_cnae_col(data['cnae'], 2)

#### Compute matrix
from Mscthesis.Models.pjensen import built_network

radius = 5.
type_var='cnae'

t0 = time.time()

net, sectors, N_x, retrieve_t, compute_t = built_network(data, loc_vars, type_var, radius)

#### SAVING
import shelve
netfiledata = 'Data/Outputs'
netobj = "net_object.dat"
netobj = join(netfiledata, netobj)
database = shelve.open(netobj)
database['net'] = net
database['sectors'] = sectors
database['N_x'] = N_x
database['retrieve_t'] = retrieve_t
database['compute_t'] = compute_t
database['description'] = 'All data'

print 'Net computed in %f seconds.' % (time.time()-t0)


#### Plot
from Mscthesis.Plotting.net_plotting import plot_net_distribution, plot_heat_net

fig1 = plot_net_distribution(net, 50)
fig2 = plot_heat_net(net, sectors)
#data = transf4compdist_global_homo(data, loc_vars, True)



#n_x, n_y = 1000, 1000
#n_levs = 10
#loc_vars = ['ES-X', 'ES-Y']
#coordinates = servicios[loc_vars]
#coordinates = clean_coordinates(coordinates)
#longs, lats = servicios[loc_vars[0]], servicios[loc_vars[1]]
#longs, lats = longs.as_matrix(), lats.as_matrix()

#m = compute_spatial_density(longs, lats, n_x, n_y)
#plotted_map = plot_geo_heatmap(pd.DataFrame(coordinates), n_levs, n_x, n_y)
