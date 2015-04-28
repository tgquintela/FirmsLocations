
"""
"""

import numpy as np


def retrieve(kdobject, Coord, i, r):
    """    """
    results = kdobject.query_ball_point(Coord[i, :], r)
    results.remove(i)
    return results


def filter_with_random_nets(net, random_nets, p_thr):
    ## 0. Needed variables
    n = random_nets.shape[2]
    ## 1. Compute bool_net
    bool_net = np.zeros(net.shape)
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            bool_net[i, j] = np.sum(net[i, j] < random_nets[i, j, :])/n < p_thr
    net = net[bool_net]
    return net
