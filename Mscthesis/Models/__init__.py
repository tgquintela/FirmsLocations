
"""
Module which contains the abstract classes used in the
Supermodule Models and the process to apply model used to a particular data.
"""

from model_utils import filter_with_random_nets
from Mscthesis.IO.model_report import create_model_report
from os.path import join
import shelve

import networkx as nx
import numpy as np

import multiprocessing as mp
import time

from Mscthesis.IO.write_log import Logger


###############################################################################
########### Global variables needed for this module
###############################################################################
message0 = """========================================
Start inferring %s:
--------------------
(%s)

"""
message1 = "Processing %s:"
message2 = "completed in %f seconds.\n"
message2a = " %s bunch of %s rows completed in %f seconds.\n"
message3 = "Total time expended computing net: %f seconds.\n"
message_close = '----------------------------------------\n'

m_debug1 = "Retrieving neighs in %f seconds."
m_debug2 = "Computing M-index in %f seconds."
m_debug3 = "%f"
m_debug4 = "Computing M-index for k=%s in %f seconds."

###############################################################################
###############################################################################


###############################################################################
################################## MAIN CLASS #################################
###############################################################################
class ModelProcess():
    """Abstract class for all the models with common utilities.
    ===============================
    Requirements of a class object:
    - Function compute_local_measure
    - Function compute_complete_measure
    - Function compute_local_descriptors

    ===============================
    Functionalities:
    - Compute net from data (parallel/sequential)
    - Compute net from precomputed neighs (parallel/sequential)
    - Compute net from agg points (parallel/sequential)
    - Compute matrix for trainning approach (parallel/sequential)

    ================================
    Problems:
    - Retrieve neighs (online/file)
    - Mask neighs
    - Get and compute descriptors (online/file)
    - Aggregate descriptors (measure dependant)

    ================================
    TODO:
    - lim_rows matrix computation (auxiliary folder to save)
    - ...

    """

    ### Class parameters
    ## Process descriptors
    time_expended = 0.  # Time expended along the process
    n_procs = 0  # Number of cpu used in parallelization (0 no parallel)
    proc_name = None  # Name of the process
    ## Logger info
    lim_rows = 0  # Lim of rows done in a bunch. For matrix comp or information
    logfile = None  # Log file
#    ## Extra information from files
#    neighs_dir = None  # Neighs director if precomputed neighs
#    agg_filepath = None  # aggregate filepath
#    locs_var_agg = None  # locs vars of aggregate file
#    types_vars_agg = None  # descriptors vars of the aggregate file
    ## Bool options
    #bool_agg = False  # Exists aggregate file
    bool_inform = False  # Give information of the process
    bool_r_array = False  # radius as an array
    bool_matrix = False  # compute matrix

    def __init__(self, logfile, aggregator, retriever, descriptormodel,
                 lim_rows=None, n_procs=None, proc_name=None):
        # Logfile
        self.logfile = Logger(logfile)
        ## Aggregator
        self.aggregator = aggregator
        ##############
        ## Precomputed aggregated descriptors
#        if agg_file_info is not None:
#            self.agg_filepath = agg_file_info['filepath']
#            self.locs_var_agg = agg_file_info['locs_vars']
#            self.types_vars_agg = agg_file_info['type_vars']
#            self.bool_agg = True
        ##############
        ## Retriever
        self.retriever = retriever
        ###############
        ## Retriever
#        if k_neig != 0:
#            self.kneigh = True
#        else:
#            self.kneigh = False
        ## Precomputed neighs
#        if neighs_dir is not None:
#            self.neighs_dir = neighs_dir
#            neighs_files = os.listdir(neighs_dir)
#            self.neighs_files = [join(neighs_dir, f) for f in neighs_files]
        ###############
        ## Descriptor model
        self.descriptormodel = descriptormodel
        # Other paramters
        self.lim_rows = lim_rows
        self.n_procs = n_procs
        self.proc_name = proc_name

    ###########################################################################
    ######################## Measure computations #############################
    ###########################################################################
    def compute_net(self, df, type_vars, loc_vars, radius, permuts=None,
                    agg_var=None):
        """Main function for the computation of the matrix. It acts as a
        wrapper over the compute_measure_all function.
        """
        self.bool_matrix = False
        net = self.compute_measure_all(df, type_vars, loc_vars, radius,
                                       permuts, agg_var)
        return net

    def compute_matrix(self, df, type_vars, loc_vars, radius, permuts=None,
                       agg_var=None):
        """Main function for the computation of the matrix. It acts as a
        wrapper over the compute_measure_all function.
        """
        self.bool_matrix = True
        matrix = self.compute_measure_all(df, type_vars, loc_vars, radius,
                                          permuts, agg_var)
        return matrix

    def compute_measure_all(self, df, type_vars, loc_vars, radius,
                            permuts=None, agg_var=None):
        """Main function for building the index of the selected model. This
        function acts as swicher between the different possibilities:
        - Parallel from data/neighs/(agg/preagg)
        - Sequential from data/neighs/(agg/preagg)
        """
        ## 0. Setting needed variables
        m_aux0 = "Training matrix" if self.bool_matrix else "Net"
        m_aux1 = "Trial0" if self.proc_name is None else self.proc_name
        self.logfile.write_log(message0 % (m_aux0, m_aux1))
        t00 = time.time()
        # Preparing needed vars
        aux = init_measure_compute(df, type_vars, loc_vars, radius, permuts)
        del permuts
        df, type_vals, n_vals, N_t, N_x = aux[:5]
        reindices, n_calc, indices = aux[5:]
        # Type vars as parameters of class
        self.var_types = {'loc_vars': loc_vars, 'type_vars': type_vars}
        self.var_types['agg_var'] = agg_var
        # Reduction of dataframe
        useful_vars = loc_vars + type_vars
        if agg_var is not None:
            useful_vars.append(agg_var)
        df = df[useful_vars]

        ## Bool options
        self.bool_r_array = type(radius) == np.ndarray
        self.bool_inform = True if self.lim_rows is not None else False
#        self.bool_agg = True if self.agg_filepath is not None else False
#        self.bool_agg = True if agg_var is not None else False

        ## 1. Computation of the measure
        corr_loc = self.compute_mea_sequ_generic(df, indices, n_vals, N_x,
                                                 radius, reindices)

        ## 2. Building a net (ifs)
        corr_loc = self.descriptormodel.to_complete_measure(corr_loc, n_vals,
                                                            N_t, N_x)

        ## Closing process
        t_expended = time.time()-t00
        self.logfile.write_log(message3 % t_expended)
        self.logfile.write_log(message_close)
        self.time_expended = t_expended

        return corr_loc, type_vals, N_x

    def compute_mea_sequ_generic(self, df, indices, n_vals, N_x,
                                 radius, reindices):
        """Main function to perform spatial correlation computation in a
        sequential mode using aggregated information given by a '''''file'''''.
        """

        ## 0. Intialization of needed variables
        N_t = reindices.shape[0]
        n_calc = reindices.shape[1]
        loc_vars = self.var_types['loc_vars']
        type_vars = self.var_types['type_vars']

        self.retriever.define_mainretriever(df, loc_vars)
        self.retriever.define_aggretriever(self.aggregator, df, reindices)

        locs = df[loc_vars]
        # type_arr
        type_arr = df[type_vars].as_matrix().astype(int)
        ndim = len(type_arr.shape)
        type_arr = type_arr if ndim > 1 else type_arr.reshape((N_t, 1))
        # clean unnecessary
        del df

        ## 1. Computation of local spatial correlations
        if self.bool_matrix:
            corr_loc = []
        else:
            n_vals0, n_vals1 = self.descriptormodel.model_dim
            corr_loc = np.zeros((n_vals0, n_vals1, n_calc))
        global_nfo_desc =\
            self.descriptormodel.compute_global_info_descriptor(n_vals, N_t,
                                                                N_x)
        ## Begin to track the process
        t0 = time.time()
        bun = 0
        for i in xrange(N_t):
            ## Obtaining neighs of a given point
            point_i = locs[indices[i], :]
            ## Loop over the possible reindices
            for k in range(n_calc):
                # Retrieve local characterizers
                val_i, vals =\
                    self.descriptormodel.get_characterizers(i, k, type_arr,
                                                            point_i, reindices,
                                                            self.retriever)
                # Computation of the local measure
                corr_loc_i =\
                    self.descriptormodel.compute_descriptors(vals, val_i,
                                                             n_vals,
                                                             **global_nfo_desc)
                # Aggregation
                if self.bool_matrix:
                    corr_loc.append(corr_loc_i)
                else:
                    corr_loc[val_i, :, k] += corr_loc_i
            ## Finish to track this process
            if self.bool_inform and (i % self.lim_rows) == 0 and i != 0:
                t_sp = time.time()-t0
                bun += 1
                self.logfile.write_log(message2a % (bun, self.lim_rows, t_sp))
                t0 = time.time()
        return corr_loc

    ###########################################################################
    ######################### Statistic significance ##########################
    ###########################################################################
    def filter_with_random_nets(self, nets, p_thr):
        "Filter non-significant weiths."
        net, random_nets = nets[:, :, 0], nets[:, :, 1:]
        net = filter_with_random_nets(net, random_nets, p_thr)
        return net

    ###########################################################################
    ############################# Outputs #####################################
    ###########################################################################
    def to_report(self, net, sectors, dirname, reportname):
        "Generate a folder in which save the report data exported."
        fig1, fig2 = create_model_report(net, sectors, dirname, reportname)
        return fig1, fig2

    def to_pajek(self, net, sectors, netfiledata, filenamenet):
        "Export net to pajek format *.net"
        net_out = nx.from_numpy_matrix(net)
        n_sects = len(sectors)
        net_out = nx.relabel_nodes(net_out, dict(zip(range(n_sects), sectors)))
        nx.write_pajek(net_out, join(netfiledata, filenamenet))

    def save_net_to_file(self, net, sectors, N_t, N_x, outputfile):
        "Save interesting quantities in a external file."
        database = shelve.open(outputfile)
        database['net'] = net
        database['sectors'] = sectors
        database['time'] = self.time_expended
        database['N_t'] = N_t
        database['N_x'] = N_x
        database.close()


###########################################################################
############################ Auxiliar classes #############################
###########################################################################
class DescriptorModel:
    "General class for descriptor models."
    ## TODO: size of descriptor array
    model_dim = (0, 0)  # check compute_descriptors

    def get_characterizers(self, i, k, type_arr, point_i, reindices,
                           retriever):
        """Retrieve local characterizers for i element and k permutation. It
        returns the column index in the output matrix correlation (val_i) and
        trivial descriptors of the neighbourhood (vals). These values are used
        for the specific model function compute_descriptors.
        """
        ## TODO: val_i computed apart
        neighs, type_n = self.retriever.retrieve_neigh(point_i)
        if type_n == 'aggregate':
            val_i = self.compute_value_i(i, k, type_arr, reindices)
            desc = retriever.aggretriever[neighs, :, k]
        else:
            val_i = self.compute_value_i(i, k, type_arr, reindices)
            desc = type_arr[reindices[neighs, k], :]
        return val_i, desc
