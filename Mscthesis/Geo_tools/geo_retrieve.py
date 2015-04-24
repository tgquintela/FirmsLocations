
"""
Module which groups all the data oriented to retrieve spatial neighbours.
"""
import numpy as np
from scipy.spatial import KDTree


def compute_neighs(df, loc_vars, radius):
    """
    radius: expressed in kms.
    """

    ## Set radius in which search neighbors
    if type(radius) != list:
        radius = [radius]
    radius = np.array(radius)/6371.009
    ## kdtree to retrieve neighbours
    kdtree = KDTree(df[loc_vars].as_matrix(), leafsize=10000)
    N = df.shape[0]

    neighs = []
    for j in range(radius.shape[0]):
        neighs_j = []
        for i in range(N):
            point = df[loc_vars].as_matrix()[i]
            neighs_j.append(kdtree.query_ball_point(point, radius[j]))
        neighs.append(neighs_j)

    return neighs


def compute_cross_neighs(df1, df2, radius):
    """"""
    pass


def compute_simple_density_pop(points, pob):
    pass
