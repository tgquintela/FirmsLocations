
"""
Module which groups all the functions related with the computation of the
spatial correlation using Jensen model.
"""

import numpy as np
#import networkx as nx
from scipy.spatial import KDTree

import time
import os
from os.path import join

from Mscthesis.IO.write_log import Logger


########### Global variables needed
##################################################################
message0 = """========================================
Start inferring net:
--------------------
(%s)

"""
message1 = "Processing %s:"
message2 = "completed in %f seconds.\n"
message3 = "Total time expended computing net: %f seconds.\n"
message_close = '----------------------------------------\n'


########### Class for computing
##################################################################
class Pjensen():
    """
    Model of spatial correlation inference.
    """

    def __init__(self, logfile=None, neighs_dir=None):
        self.logfile = Logger(logfile)
        if neighs_dir is not None:
            self.neighs_dir = neighs_dir
            self.neighs_files = os.listdir(neighs_dir)
            self.neighs_files = [join(neighs_dir, f) for f in self.neighs_files]

    def built_network_from_neighs(self, df, type_var):
        """Main function to perform spatial correlation computation."""
        ## 0. Setting needed variables
        self.logfile.write_log(message0 % self.neighs_dir)
        t00 = time.time()
        # The possible type values
        type_vals = list(df[type_var].unique())
        n_vals = len(type_vals)
        # General counts
        N_t = df.shape[0]
        N_x = np.array([np.sum(df[type_var] == type_v) for type_v in type_vals])
        ## 1. Computation of local spatial correlations
        corr_loc = np.zeros((n_vals, n_vals))
        for f in self.neighs_files:
            ## Begin to track the process
            self.logfile(message1 % (f.split('/')[-1]))
            t0 = time.time()
            ## Read the file of neighs
            neighs = pd.read_csv(f, sep=';', index_col=0)
            ## Compute corr with these neighs
            corr_loc_f = local_jensen_corr_from_neighs(df, type_var,
                                                       neighs, type_vals)
            corr_loc += corr_loc_f
            ## Finish to track this process
            self.logfile(message2 % (time.time()-t0))
        ## 2. Building a net
        C = global_constants_jensen(n_vals, Nt, Nx)
        net = np.log10(np.multiply(C, corr_loc))
        ## Closing process
        self.logfile(message3 % (time.time()-t00))
        self.logfile(message_close)
        return net, type_vals, N_x


###############################################################################
###############################################################################
###############################################################################
def local_jensen_corr_from_neighs(df, type_var, neighs, type_vals):
    """"""
    ## Global variables
    n_vals = len(type_vals)
    indices = np.array(neighs.index)
    corr_loc = np.zeros((n_vals, n_vals))
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
    return corr_loc


def global_constants_jensen(n_vals, Nt, Nx):
    ## Building the normalizing constants
    C = np.zeros((n_vals, n_vals))
    for i in range(n_vals):
        for j in range(n_vals):
            if i == j:
                C[i, j] = (N_t-1)/(Nx[i]*(Nx[i]-1))
            else:
                C[i, j] = (N_t-Nx[i])/(Nx[i]*Nx[j])
    return C


####### COMPLETE FUNCTIONS
#####################################################################################
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
        print "Finished %s in %f seconds." %(type_vals[i], time.time()-t0)
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

    retrieve_t, compute_t = 0, 0

    ## See the files in a list
    files_dir = os.listdir(neighs_dir)
    files_dir = [join(neighs_dir, f) for f in files_dir]

    ## Building the sum of local correlations
    corr_loc = np.zeros((n_vals, n_vals))
    for f in files_list:
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
                C[i, j] = (N_t-1)/(Nx[i]*(Nx[i]-1))
            else:
                C[i, j] = (N_t-Nx[i])/(Nx[i]*Nx[j])

    ## Building a net
    net = np.log10(np.multiply(C, corr_loc))
    return net, type_vals, N_x
