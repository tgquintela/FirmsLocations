
"""
Module which groups all the functions related with the computation of the
spatial correlation using Jensen model.
"""

import numpy as np
import pandas as pd
#import networkx as nx
from scipy.spatial import KDTree

import multiprocessing as mp
import time
import os
from os.path import join

from Mscthesis.IO.write_log import Logger
from model_utils import filter_with_random_nets


########### Global variables needed
##################################################################
message0 = """========================================
Start inferring net:
--------------------
(%s)

"""
message1 = "Processing %s:"
message2 = "completed in %f seconds.\n"
message2a = "Bunch of %s rows completed in %f seconds.\n"
message3 = "Total time expended computing net: %f seconds.\n"
message_close = '----------------------------------------\n'

m_debug1 = "Retrieving neighs in %f seconds."
m_debug2 = "Computing M-index in %f seconds."
m_debug3 = "%f"
m_debug4 = "Computing M-index for k=%s in %f seconds."


########### Class for computing
##################################################################
class Pjensen():
    """
    Model of spatial correlation inference.
    """

    def __init__(self, logfile=None, neighs_dir=None, lim_rows=None,
                 n_procs=None):
        self.logfile = Logger(logfile)
        if neighs_dir is not None:
            self.neighs_dir = neighs_dir
            neighs_files = os.listdir(neighs_dir)
            self.neighs_files = [join(neighs_dir, f) for f in neighs_files]
            self.lim_rows = lim_rows
            self.n_procs = n_procs

    def built_nets(self, df, type_var, loc_vars, radius, permuts=None):
        """Main unction for building the network using M-index.
        TODO: Support to arrays in radius.
        """
        ## 0. Setting needed variables
        self.logfile.write_log(message0 % self.neighs_dir)
        t00 = time.time()
        # Preparing needed vars
        aux = preparing_net_computation(df, type_var, self.lim_rows, permuts)
        cnae_arr, type_vals, n_vals, N_t, N_x = aux[:5]
        reindices, n_calc, bool_inform = aux[5:]
        # KDTree retrieve object instantiation
        locs = df[loc_vars].as_matrix()
        kdtree = KDTree(locs, leafsize=10000)
        radius = radius/6371.009
        if type(radius) == float:
            r = radius
        elif type(radius) == str:
            radius = np.array(df[radius])

        ## 1. Computation of the local spatial correlation with M-index
        corr_loc = np.zeros((n_vals, n_vals, n_calc))
        counts = np.zeros((n_vals, n_vals, n_calc))
        indices = np.array(df.index)
        ## Begin to track the process
        t0 = time.time()
        for i in xrange(N_t):
            # Check radius
            if type(radius) == np.ndarray:
                r = radius[i]
            ## Obtaining neighs of a given point
            point_i = locs[indices[i], :]
            neighs = kdtree.query_ball_point(point_i, r)
            ## Loop over the possible reindices
            for k in range(n_calc):
                #val_i = df.loc[reindices[i, k], type_var]
                val_i = cnae_arr[reindices[i, k]]
                neighs_k = reindices[neighs, k]
                vals = cnae_arr[neighs_k]
                ## Count the number of companies of each type
                corr_loc_i, counts_i = computation_of_counts([vals, val_i,
                                                              n_vals])
                ## Aggregate to local correlation
                corr_loc[val_i, :, k] += corr_loc_i
                counts[val_i, :, k] += counts_i
            ## Finish to track this process
            if bool_inform and (i % self.lim_rows) == 0 and i != 0:
                t_sp = time.time()-t0
                self.logfile.write_log(message2a % (self.lim_rows, t_sp))
                t0 = time.time()
        ## 2. Building a net
        C = global_constants_jensen(n_vals, N_t, N_x)
        # Computing the nets
        net = np.zeros((n_vals, n_vals, n_calc))
        for i in range(n_calc):
            net[:, :, i] = np.log10(np.multiply(C, corr_loc[:, :, i]))
        # Averaging counts
        counts = counts/float(N_t)
        ## Closing process
        self.logfile.write_log(message3 % (time.time()-t00))
        self.logfile.write_log(message_close)
        return C, corr_loc, counts, type_vals, N_x

    def built_network_from_neighs(self, df, type_var, permuts=None):
        """Main function to perform spatial correlation computation."""
        ## 0. Setting needed variables
        self.logfile.write_log(message0 % self.neighs_dir)
        t00 = time.time()
        # Preparing needed vars
        aux = preparing_net_computation(df, type_var, self.lim_rows, permuts)
        cnae_arr, type_vals, n_vals, N_t, N_x = aux[:5]
        reindices, n_calc, bool_inform = aux[5:]

        ## 1. Computation of local spatial correlations
        corr_loc = np.zeros((n_vals, n_vals, n_calc))
        counts = np.zeros((n_vals, n_vals, n_calc))
        for f in self.neighs_files:
            ## Begin to track the process
            self.logfile.write_log(message1 % (f.split('/')[-1]))
            t0 = time.time()
            ## Read the file of neighs
            neighs = pd.read_csv(f, sep=';', index_col=0)
            ## Compute corr with these neighs
            indices = np.array(neighs.index)
            for j in xrange(indices.shape[0]):
                ## Retrieve neighs from neighs dataframe
                neighs_j = neighs.loc[indices[j], 'neighs'].split(',')
                neighs_j = [int(e) for e in neighs_j]
                corr_loc_f, counts_f = self.local_jensen_corr(cnae_arr,
                                                              reindices, j,
                                                              neighs_j, n_vals)
            corr_loc += corr_loc_f
            counts += counts_f
            ## Finish to track this process
            self.logfile.write_log(message2 % (time.time()-t0))
        ## 2. Building a net
        C = global_constants_jensen(n_vals, N_t, N_x)
        # Computing the nets
        net = np.zeros((n_vals, n_vals, n_calc))
        for i in range(n_calc):
            net[:, :, i] = np.log10(np.multiply(C, corr_loc[:, :, i]))
        # Averaging counts
        counts = counts/float(N_t)
        ## Closing process
        self.logfile.write_log(message3 % (time.time()-t00))
        self.logfile.write_log(message_close)
        return C, corr_loc, counts, type_vals, N_x

    def local_jensen_corr(self, cnae_arr, reindices, i, neighs, n_vals):
        """Function wich acts as a switcher between computing M index in
        sequential or in parallel.
        """
        if self.n_procs is not None:
            corrs, counts = compute_M_indexs_parallel(cnae_arr, reindices, i,
                                                      neighs, n_vals,
                                                      self.n_procs)
        else:
            corrs, counts = compute_M_indexs_sequential(cnae_arr, reindices, i,
                                                        neighs, n_vals)
        return corrs, counts

    def build_random_nets(self, df, type_var, n):
        """Montecarlo creation of random nets by permutation."""
        n_vals = len(list(df[type_var].unique()))
        random_nets = np.zeros((n_vals, n_vals, n))
        for i in range(n):
            reindex = np.random.permutation(np.array(df.index))
            random_nets[:, :, i] = self.built_network_from_neighs(df, type_var,
                                                                  reindex)
        return random_nets

#    def quality_point(df, points=None, radius=None, ):
#        if points is None:
#        if radius is None:
#        for i in range(N_t):


###############################################################################
############################### Counts and corr ###############################
###############################################################################
def compute_M_indexs_parallel(cnae_arr, reindices, i, neighs, n_vals, n_procs):
    """Computation of the M index in parallel."""
    ## Loop over the possible reindices
    n_calc = reindices.shape[1]
    args = []
    vals_i = np.zeros(n_calc)
    for k in range(n_calc):
        val_i = cnae_arr[reindices[i, k]]
        neighs_k = reindices[neighs, k]
        vals = cnae_arr[neighs_k]
        args.append([vals, val_i, n_vals])
        vals_i[k] = val_i

    ## Computation of counts
    pool = mp.Pool(n_procs)
    corrs = pool.map(computation_of_counts, args)
    corr_loc = np.zeros((n_vals, n_vals, n_calc))
    counts_i = np.zeros((n_vals, n_vals, n_calc))
    ## Aggregate to local correlation
    for k in range(n_calc):
        corr_loc[vals_i[k], :, k] += corrs[k][0]
        counts_i[vals_i[k], :, k] += corrs[k][1]
    return corr_loc, counts_i


def compute_M_indexs_sequential(cnae_arr, reindices, i, neighs, n_vals):
    """Computation of M index in sequential."""
    ## Loop over the possible reindices
    n_calc = reindices.shape[1]
    #vals_i = np.zeros(n_calc)
    corr_loc = np.zeros((n_vals, n_vals, n_calc))
    counts_i = np.zeros((n_vals, n_vals, n_calc))
    for k in range(n_calc):
        val_i = cnae_arr[reindices[i, k]]
        neighs_k = reindices[neighs, k]
        vals = cnae_arr[neighs_k]
        ## Computation of counts
        corr, counts = computation_of_counts([vals, val_i, n_vals])
        ## Aggregate to local correlation
        corr_loc[val_i, :, k] += corr
        counts_i[val_i, :, k] += counts
    return corr_loc, counts_i


def computation_of_counts(args):
    """Individual function of computation of local counts."""
    vals, idx, n_vals = tuple(args)
    ## Count the number of companies of each type
    counts_i = count_in_neighborhood(vals, n_vals)
    ## Compute the correlation contribution
    corr_loc_i = compute_loc_M_index(counts_i, idx, n_vals)
    return corr_loc_i, counts_i


def count_in_neighborhood(vals, n_vals):
    "Counting neighbours in the neighbourhood."
    counts_i = [np.count_nonzero(np.equal(vals, v)) for v in range(n_vals)]
    counts_i = np.array(counts_i)
    return counts_i


def compute_loc_M_index(counts_i, idx, n_vals):
    "Computing the M index."
    ## Compute the correlation contribution
    counts_i[idx] -= 1
    tot = counts_i.sum()
    if tot == 0:
        corr_loc_i = np.zeros(n_vals)
    elif counts_i[idx] == tot:
        corr_loc_i = np.zeros(n_vals)
        corr_loc_i[idx] = counts_i[idx]/float(tot)
    else:
        corr_loc_i = counts_i/float(tot-counts_i[idx])
        corr_loc_i[idx] = counts_i[idx]/float(tot)
    # Avoid nan values
    corr_loc_i[np.isnan(corr_loc_i)] = 0.
    #### debug
    if np.any(corr_loc_i) < 0:
        print corr_loc_i
    ##########
    return corr_loc_i


###############################################################################
############################### Quality measure ###############################
###############################################################################
def quality_measure_w_search(kdtree, points, cnae_arr, cnae_p, radius):
    ""

    if type(radius) == np.ndarray:
        r = radius

    for i in range(points.shape[0]):
        if type(radius) == np.ndarray:
            r = radius[i]
        ## Obtaining neighs of a given point
        point_i = points[i, :]
        neighs = kdtree.query_ball_point(point_i, r)
        ## Retrieve val
        val_i = cnae_p[i]
        vals = cnae_arr[neighs]





###############################################################################
###############################################################################
###############################################################################
def preparing_net_computation(df, type_var, lim_rows, permuts):
    # Inform
    bool_inform = True if lim_rows is not None else False
    # Values
    type_vals = list(df[type_var].unique())
    n_vals = len(type_vals)
    repl = dict(zip(type_vals, range(n_vals)))
    cnae_arr = np.array(df[type_var].replace(repl))
    # Global stats
    N_t = df.shape[0]
    N_x = [np.sum(df[type_var] == type_v) for type_v in type_vals]
    N_x = np.array(N_x)
    # Preparing reindices
    reindex = np.array(df.index)
    reindex = reindex.reshape((N_t, 1))
    if permuts is not None:
        if type(permuts) == int:
            permuts = [np.random.permutation(N_t) for i in range(permuts)]
            permuts = np.vstack(permuts).T
            bool_ch = len(permuts.shape) == 1
            permuts = permuts.reshape((N_t, 1)) if bool_ch else permuts
        n_per = permuts.shape[1]
        permuts = [reindex[permuts[:, i]] for i in range(n_per)]
        permuts = np.hstack(permuts)
    reindex = [reindex] if permuts is None else [reindex, permuts]
    reindices = np.hstack(reindex)
    n_calc = reindices.shape[1]

    output = (cnae_arr, type_vals, n_vals, N_t, N_x, reindices,
              n_calc, bool_inform)
    return output


def global_constants_jensen(n_vals, N_t, N_x):
    ## Building the normalizing constants
    C = np.zeros((n_vals, n_vals))
    for i in range(n_vals):
        for j in range(n_vals):
            if i == j:
                if N_x[i] <= 1:
                    C[i, j] = 0.
                else:
                    C[i, j] = (N_t-1)/float(N_x[i]*(N_x[i]-1))
            else:
                if N_x[i] == 0 or N_x[j] == 0:
                    C[i, j] = 0.
                else:
                    C[i, j] = (N_t-N_x[i])/float(N_x[i]*N_x[j])
    return C


####### COMPLETE FUNCTIONS
###############################################################################
def built_network(df, loc_vars, type_var, radius):
    """Function for building the network from the locations."""

    type_vals = list(df[type_var].unique())
    n_vals = len(type_vals)

    N_t = df.shape[0]
    N_x = np.array([np.sum(df[type_var] == type_v) for type_v in type_vals])

    retrieve_t, compute_t = 0, 0

    net = np.zeros((n_vals, n_vals))
    for i in range(n_vals):
        ##########
        t0 = time.time()
        ##########
        elements_i = np.where(df[type_var] == type_vals[i])[0]
        N_i = elements_i.shape[0]
        counts_i = compute_neigh_count(df, i, type_vals, loc_vars,
                                       type_var, radius)
        ##########
        retrieve_t += time.time()-t0
        t1 = time.time()
        ##########
        aux = compute_unorm_corrs(counts_i, i)
        ## Normalization
        cte2 = np.log10(np.divide(float(N_t-1), (N_i*(N_i-1))))
        cte2 = 0 if N_x[i] == 1 else cte2
        cte = np.log10(np.divide(float(N_t-N_i), (N_i*N_x)))
        cte[np.where(cte == -np.inf)] = 0
        cte[i] = cte2
        #net[i, :] = np.multiply(cte, aux)
        aux = cte + np.log10(aux)
        aux[np.where(aux == -np.inf)] = 0
        net[i, :] = aux

        ##########
        print "Finished %s in %f seconds." % (type_vals[i], time.time()-t0)
        compute_t += time.time()-t1
        ##########

    return net, type_vals, N_x, retrieve_t, compute_t


def compute_unorm_corrs(counts_i, i):
    """"""

    Nts = np.sum(counts_i, 1)
    unnorm_corrs = np.zeros((counts_i.shape[1]))

    for j in range(counts_i.shape[1]):
        if i == j:
            aux = np.divide(counts_i[:, i].astype(float)-1, Nts)
            unnorm_corrs[i] = np.sum(aux)

        else:
            aux = np.divide(counts_i[:, j].astype(float),
                            Nts-(counts_i[:, i]-1))
            unnorm_corrs[j] = np.sum(aux)

    return unnorm_corrs


def compute_neigh_count(df, j, type_vals, loc_vars, type_var, radius):
    """
    radius: expressed in kms.
    """

    kdtree = KDTree(df[loc_vars].as_matrix(), leafsize=10000)
    elements_j = np.where(df[type_var] == type_vals[j])[0]
    N_j = elements_j.shape[0]
    radius = radius/6371.009

    counts = np.zeros((N_j, len(type_vals)))
    for i in range(N_j):
        k = elements_j[i]
        neighs = kdtree.query_ball_point(df[loc_vars].as_matrix()[k], radius)
        vals = df[type_var][neighs]
        counts[i, :] = np.array([np.sum(vals == val) for val in type_vals])

    counts = counts.astype(int)
    return counts


def jensen_net_from_neighs(df, type_var, neighs_dir):
    """Function for building the network from neighbours."""

    type_vals = list(df[type_var].unique())
    n_vals = len(type_vals)

    N_t = df.shape[0]
    N_x = np.array([np.sum(df[type_var] == type_v) for type_v in type_vals])

    ## See the files in a list
    files_dir = os.listdir(neighs_dir)
    files_dir = [join(neighs_dir, f) for f in files_dir]

    ## Building the sum of local correlations
    corr_loc = np.zeros((n_vals, n_vals))
    for f in files_dir:
        neighs = pd.read_csv(f, sep=';', index_col=0)
        indices = np.array(neighs.index)
        for j in indices:
            ## Retrieve neighs from neighs dataframe
            neighs_j = neighs.loc[j, 'neighs'].split(',')
            neighs_j = [int(e) for e in neighs_j]
            vals = df.loc[neighs_j, type_var]
            ## Count the number of companies of each type
            counts_j = np.array([np.sum(vals == val) for val in type_vals])
            cnae_val = df.loc[j, type_var]
            idx = type_vals.index(cnae_val)
            ## Compute the correlation contribution
            counts_j[idx] -= 1
            if counts_j[idx] == counts_j.sum():
                corr_loc_j = np.zeros(n_vals)
                corr_loc_j[idx] = counts_j[idx]/counts_j.sum()
            else:
                corr_loc_j = counts_j/(counts_j.sum()-counts_j[idx])
                corr_loc_j[idx] = counts_j[idx]/counts_j.sum()
            ## Aggregate to local correlation
            corr_loc[idx, :] += corr_loc

    ## Building the normalizing constants
    C = np.zeros((n_vals, n_vals))
    for i in range(n_vals):
        for j in range(n_vals):
            if i == j:
                C[i, j] = (N_t-1)/(N_x[i]*(N_x[i]-1))
            else:
                C[i, j] = (N_t-N_x[i])/(N_x[i]*N_x[j])

    ## Building a net
    net = np.log10(np.multiply(C, corr_loc))
    return net, type_vals, N_x


def jensen_net(df, type_var, loc_vars, radius, permutations=None):
    """Function for building the network from neighbours."""

    ## 0. Set needed values
    # Values
    type_vals = list(df[type_var].unique())
    n_vals = len(type_vals)
    # Global stats
    N_t = df.shape[0]
    N_x = np.array([np.sum(df[type_var] == type_v) for type_v in type_vals])
    # KDTree retrieve object instantiation
    kdtree = KDTree(df[loc_vars].as_matrix(), leafsize=10000)
    # Preparing reindices
    reindex = np.array(df.index())
    reindex = reindex.reshape((reindex.shape[0], 1))
    if permutations is not None:
        n_per = permutations.shape[1]
        permutations = [reindex[permutations[:, i]] for i in range(n_per)]
        permutations = np.array(permutations).T
    reindex = [reindex] if permutations is None else [reindex, permutations]
    reindices = np.hstack(reindex)
    n_calc = reindices.shape[1]

    ## 1. Computation of the local spatial correlation with M-index
    corr_loc = np.zeros((n_vals, n_vals, n_calc))
    indices = np.array(df.index())
    for i in range(N_t):
        ## Obtaining neighs of a given point
        point_i = df.loc[indices[i], loc_vars].as_matrix()
        neighs = kdtree.query_ball_point(point_i, radius)
        ## Loop over the possible reindices
        for k in range(n_calc):
            val_i = df.loc[reindices[i, k], type_var]
            neighs_k = reindices[neighs, k]
            vals = df.loc[neighs_k, type_var]
            ## Count the number of companies of each type
            counts_i = np.array([np.sum(vals == val) for val in type_vals])
            idx = type_vals.index(val_i)
            ## Compute the correlation contribution
            counts_i[idx] -= 1
            if counts_i[idx] == counts_i.sum():
                corr_loc_i = np.zeros(n_vals)
                corr_loc_i[idx] = counts_i[idx]/counts_i.sum()
            else:
                corr_loc_i = counts_i/(counts_i.sum()-counts_i[idx])
                corr_loc_i[idx] = counts_i[idx]/counts_i.sum()
            ## Aggregate to local correlation
            corr_loc[idx, :, k] += corr_loc_i

    ## 2. Building the normalizing constants
    C = np.zeros((n_vals, n_vals))
    for i in range(n_vals):
        for j in range(n_vals):
            if i == j:
                C[i, j] = (N_t-1)/(N_x[i]*(N_x[i]-1))
            else:
                C[i, j] = (N_t-N_x[i])/(N_x[i]*N_x[j])

    ## 3. Building the nets
    net = np.zeros((n_vals, n_vals, n_calc))
    for i in range(n_calc):
        net[:, :, i] = np.log10(np.multiply(C, corr_loc[:, :, i]))
    return net, type_vals, N_x
