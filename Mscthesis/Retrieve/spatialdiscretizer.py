
"""
Spatial Discretizor utilities.

TODO
----
- Fit irregular discretizer
"""


import numpy as np
from spatial_utils import create_grid, apply_grid
from scipy.spatial.distance import cdist


class SpatialDiscretizor:
    """Spatial Discretizor object. This object performs a discretization of the
    spatial domain and it is able to do:
    - Assign a static predefined regions to each point.
    - Retrieve neighbourhood defined by static regions.
    """

    borders = None
    regionlocs = None
    regions_id = None

    def retrieve_regions(self, point_i, info_i, ifdistance=False):
        ######TODO
        pass

    def retrieve_neighs(self, point_i, locs):
        """Retrieve the neighs given a point. Could be an internal retrieve if
        point_i is an index or an external retrieve if point_i it is not a
        point in locs (point_i is a coordinates).
        """
        regions = self.map_loc2regionid(locs)
        if type(point_i) == int:
            region = regions[point_i]
        else:
            region = self.map_loc2regionid(point_i)
        logi = self.check_neighbours(region, regions)
        return logi

    def discretize(self, locs):
        disc_locs = self.discretize_spec(locs)
        return disc_locs

    def map2id(self, locs):
        regions = self.map_loc2regionid(locs)
        return regions


class GridSpatialDisc(SpatialDiscretizor):
    "Grid spatial discretization. The regions are rectangular with equal size."

    def __init__(self, grid_size, xlim=(None, None), ylim=(None, None)):
        "Main function to map a group of points in a 2d to a grid."
        self.create_grid(grid_size, xlim=xlim, ylim=ylim)

    ##################### Definition of particularities ######################
    ##########################################################################
    def create_grid(self, grid_size, xlim=(None, None), ylim=(None, None)):
        self.borders = create_grid(grid_size=grid_size, xlim=xlim, ylim=ylim)

    def apply_grid(self, locs):
        locs_grid = apply_grid(locs, self.borders[0], self.borders[1])
        return locs_grid

    def discretize_spec(self, locs):
        return self.apply_grid(locs)

    ################################ Functions ###############################
    ##########################################################################
    def map_loc2regionid(self, locs):
        locs_grid = apply_grid(locs, self.borders[0], self.borders[1])
        grid_size = (self.borders[0].shape[0]-1, self.borders[0].shape[0]-1)
        regions_id = map_gridloc2regionid(locs_grid, grid_size)
        return regions_id

    def check_neighbours(self, region, regions):
        logi = regions == region
        return logi


def map_gridloc2regionid(locs_grid, grid_size):
    return locs_grid[:, 0]*grid_size[0]+locs_grid[:, 1]


class CircularSpatialDisc(SpatialDiscretizor):
    """Circular spatial discretization. The regions are circles with different
    sizes."""

    ## TODO: map loc_grid to a id region: map_gridloc2regionid
    def __init__(self, centerlocs, radios):
        "Main information to built the regions."
        if type(radios) in [float, int]:
            radios = np.ones(centerlocs.shape[0])*radios
        self.borders = radios
        self.regionlocs = centerlocs

    ################################ Functions ###############################
    ##########################################################################
    def map_loc2regionid(self, locs):
        return map_circloc2regionid(locs, self.regionlocs, self.borders)

    def check_neighbours(self, region, regions):
        N_r = len(regions)
        logi = np.zeros(N_r).astype(bool)
        for i in xrange(N_r):
            logi[i] = region in regions[i]
        return logi

    def discretize_spec(self, locs):
        ## TODO: See which is their correspondent circle.
        pass


def map_circloc2regionid(locs, centerlocs, radis):
    "Map the each point to the correspondent circular region."
    idxs_dis = distribute_tasks(locs.shape[0], 50000)
    regions_id = [[] for i in range(locs.shape[0])]
    for k in range(len(idxs_dis)):
        logi = cdist(locs[idxs_dis[k][0]:idxs_dis[k][1]], centerlocs) < radis
        aux = np.where(logi)
        for j in range(len(aux[0])):
            regions_id[aux[0][j]].append(aux[1][j])
    return regions_id


def distribute_tasks(n, memlim):
    """Util function for distribute tasks in matrix computation in order to
    save memory or to parallelize computations."""
    lims = []
    inflim = 0
    while True:
        if inflim + memlim > n:
            lims.append([inflim, n])
            break
        else:
            lims.append([inflim, inflim+memlim])
            inflim = inflim+memlim
    return lims


class IrregularSpatialDisc(SpatialDiscretizor):
    "Grid spatial discretization."

    ## TODO: map loc_grid to a id region: map_gridloc2regionid
    def __init__(self, borders=None):
        "Main information to built the regions."
        self.borders = borders

    ##################### Definition of particularities ######################
    ##########################################################################
    def fit_spatialmodel(self, data):
        self.regionlocs, self.borders = somefunction(data)

    ################################ Functions ###############################
    ##########################################################################
    def map_loc2regionid(self, locs):
        return somefunction(locs, self.regionlocs, self.borders)

    def check_neighbours(self, region, regions):
        pass