
"""
Module which groups the geographical map plots.
"""


from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import numpy as np

#from Mscthesis.Statistics.stats_functions import compute_spatial_density


def clean_coordinates(coordinates):
    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]
    return coordinates


def plot_in_map(coordinates, resolution='f', color_cont=None, marker_size=1):
    """Plot the coordinates in points in the map.
    """

    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]

    # compute coordinates
    longs = coordinates[:, 0]
    lats = coordinates[:, 1]

    lat_0 = np.mean(lats)
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))

    # Set map
    fig = plt.figure()
    mapa = Basemap(projection='merc', lat_0=lat_0, llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, resolution=resolution)
    mapa.drawcoastlines()
    mapa.drawcountries()
    if color_cont is not None:
        mapa.fillcontinents(color=color_cont)
    mapa.drawmapboundary()

    mapa.scatter(longs, lats, marker_size, marker='o', color='r', latlon=True)

    return fig


def plot_geo_heatmap(coordinates, n_levs, n_x, n_y):
    """Plot the coordinates in points in the map.
    """

    ## 00. Preprocess of the data
    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]

    ## 0. Preparing needed variables
    # compute coordinates
    longs = coordinates[:, 0]
    lats = coordinates[:, 1]
    # Preparing corners
    lat_0 = np.mean(lats)
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))

    ## 1. Set map
    fig = plt.figure()
    mapa = Basemap(projection='merc', lat_0=lat_0, llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, resolution='h')
    mapa.drawcoastlines()
    mapa.drawcountries()
    mapa.fillcontinents(color='gray')

    # Draw water
    #mapa.drawmapboundary(fill_color='aqua')
    #mapa.fillcontinents(color='coral')
    mapa.drawlsmask(ocean_color='aqua', lakes=False)

    # mapa.scatter(longs, lats, 10, marker='o', color='k', latlon=True)

    ## 2. Preparing heat map
    density, l_x, l_y = compute_spatial_density(longs, lats, n_x+1, n_y+1)
    clevs = np.linspace(density.min(), density.max(), n_levs+1)
    l_x, l_y = mapa(l_x, l_y)

    ## 3. Computing heatmap
    cs = mapa.contourf(l_x, l_y, density, clevs, cmap=cm.s3pcpn)
    #cs = plt.contourf(l_x, l_y, density, clevs)
    # add colorbar.
    cbar = mapa.colorbar(cs, location='bottom', pad="5%")
    cbar.set_label('density')

    ## 4. Fix details

    # add title
    plt.title('Heat map of companies density')

    return fig


###############################################################################
###############################################################################
###############################################################################
def compute_spatial_density(longs, lats, n_x, n_y):
    """Computation of the spatial density given the latitutes and logitudes of
    the points we want to count.

    TODO
    ----
    Smoothing function

    """
    ## 0. Setting initial variables
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))
    l_x = np.linspace(llcrnrlon, urcrnrlon, n_x+1)
    l_y = np.linspace(llcrnrlat, urcrnrlat, n_y+1)

    ## 1. Computing density
    density, _, _ = np.histogram2d(longs, lats, [l_x, l_y])
    density = density.T

    ## 2. Smothing density

    ## 3. Output
    l_x = np.mean(np.vstack([l_x[:-1], l_x[1:]]), axis=0)
    l_y = np.mean(np.vstack([l_y[:-1], l_y[1:]]), axis=0)
    l_x, l_y = np.meshgrid(l_x, l_y)

    return density, l_x, l_y
