
"""
Module which groups all the data oriented to retrieve spatial neighbours.
"""
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
from os.path import join, exists
from os import makedirs
import time

from Mscthesis.IO.write_log import Logger


########### Global variables needed
##################################################################
message00 = """========================================
========================================
Start computing neighs:
-----------------------

"""
message0 = """========================================
Start computing neighs:
-----------------------
(%s)

"""
message0a = "\n"
message1 = "Computing and saving %s:"
message2 = "completed in %f seconds.\n"
message3 = "Total time expended computing net: %f seconds.\n"
message_close = '----------------------------------------\n'
message_close00 = '========================================\n'


########### Class for computing neighs
##################################################################
class Compute_self_neighs():
    """Possibilities:
    - One dirname
    - Filenames: str or list of strs
    - Radius: float, list of floats, np.array, list of arrays (STR)
    - lim_rows: int, list of ints
    """
    def __init__(self, dirnames, filenames, lim_rows, radius, logfile=None):
        # Transformation to lists
        aux = homogenize_comp_neighs(filenames, radius, lim_rows, dirnames)
        filenames, radius, lim_rows, dirnames = aux
        n_proc = len(filenames)
        # Creation of the directory
        for i in range(len(dirnames)):
            if not exists(dirnames[i]): makedirs(dirnames[i])
        # Logfile
        self.logfile = Logger(logfile)
        # Parameters
        self.dirnames = dirnames
        self.radius = radius
        self.lim_rows = lim_rows
        self.filenames = filenames
        self.i_r = 0

    def compute_neighs(self, df, loc_vars):
        """"""
        ## Check lists
        for i in range(len(self.radius)):
            self.compute_neighs_r(df, loc_vars, i)

    def compute_neighs_r(self, df, loc_vars, i):
        """Computation of the neighborhood using an individual radius."""
        ## Start variables used
        r, dirname,  = self.radius[i], self.dirnames[i]
        filename, lim_rows = self.filenames[i], self.lim_rows[i]
        ## Start tracking the process
        t00 = time.time()
        m1, m2 = 'Homogeneous R=%f km', 'Heterogeneous R'
        m = m1 % r if type(r) == float else m2
        self.logfile.write_log(message0 % m)
        ## Compute neighs
        indices = np.array(df.index)
        max_len = len(str(indices[-1]))
        indices_sep = allocate_in_files(indices, lim_rows)
        col = ['neighs']
        for i in range(len(indices_sep)):
            ## Start tracking the process
            t0 = time.time()
            num1, num2 = str(indices_sep[i][0]), str(indices_sep[i][-1])
            m = (max_len-len(num1))*'0'+num1+'-'+(max_len-len(num2))*'0'+num2
            self.logfile.write_log(message1 % m)
            ## Computation of the neighs
            neighs = compute_self_neighs(df.loc[indices_sep[i], :],
                                         loc_vars, r)
            neighs = [str(neighs[j])[1:-1] for j in range(len(neighs))]
            neighs = pd.DataFrame(neighs, index=indices_sep[i], columns=col)
            ## Save file % Non considered ((4-len(str(i)))*'0'+str(i))
            namefile = join(dirname, filename+'_'+m)
            neighs.to_csv(namefile, sep=';')
            ## Stop tracking process
            self.logfile.write_log(message2 % (time.time()-t0))
        ## Stop tracking the process
        self.logfile.write_log(message3 % (time.time()-t00))
        self.logfile.write_log(message_close)


########### Complete functions
##################################################################
def compute_self_neighs(df, loc_vars, radius):
    """
    radius: expressed in kms.
    """
    ## kdtree to retrieve neighbours
    kdtree = KDTree(df[loc_vars].as_matrix(), leafsize=10000)
    N = df.shape[0]

    ## Set radius in which search neighbors
    neighs = []
    if type(radius) == str:
        for i in range(N):
            point = df[loc_vars].as_matrix()[i]
            r = df.loc[i, radius]/6371.009
            neighs.append(kdtree.query_ball_point(point, r))
    elif type(radius) == float:
        r = radius/6371.009
        for i in range(N):
            point = df[loc_vars].as_matrix()[i]
            neighs.append(kdtree.query_ball_point(point, r))
    elif type(radius) == np.ndarray:
        for i in range(N):
            point = df[loc_vars].as_matrix()[i]
            r = radius[i]/6371.009
            neighs.append(kdtree.query_ball_point(point, r))

    return neighs


def compute_neighs_and_save(df, loc_vars, radius, pathfile, lim_rows):
    """"""
    ## 1. Allocating rows for the different files
    indices = np.array(df.index)
    n_files = indices.shape[0]/lim_rows
    indices_sep = [indices[i*lim_rows:(i+1)*lim_rows] for i in range(n_files)]
    indices_sep.append(indices[n_files*lim_rows:])
    ## 2. Computing and saving neighs
    for i in range(len(indices_sep)):
        neighs = compute_neighs(df.loc[indices_sep[i], :], loc_vars, radius)
        neighs = [str(neighs[0][j])[1:-1] for j in range(len(neighs[0]))]
        aux = pd.DataFrame(neighs, index=indices_sep[i], columns=['neighs'])
        namefile = join(pathfile, 'neighs_'+(4-len(str(i)))*'0'+str(i))
        aux.to_csv(namefile, sep=';')


def compute_cross_neighs(df1, df2, radius):
    """"""
    pass


def compute_simple_density_pop(points, pob):
    pass


########### AUXILIAR FUNCTIONS
##################################################################
def homogenize_comp_neighs(filenames, radius, lim_rows, dirnames):
    """Auxiliar function to homogenize the inputs of the class with the
    information of how to compute neighs and where to save them."""
    # Transformation to lists
    if type(filenames) == str:
        filenames = [filenames]
    elif type(filenames) == list:
        assert(np.all(np.array([type(e) for e in filenames]) == str))
    if type(lim_rows) == int:
        lim_rows = [lim_rows]
    elif type(lim_rows) == list:
        assert(np.all(np.array([type(e) for e in lim_rows]) == int))
    if type(radius) == float:
        radius = [radius]
    elif type(radius) == list:
        if type(radius[0]) == float:
            assert(np.all(np.array([type(e) for e in radius]) == float))
        elif type(radius[0]) == str:
            assert(np.all(np.array([type(e) for e in radius]) == str))
    elif type(radius) == np.ndarray:
        if len(radius.shape) == 2:
            radius = [radius[:, i] for i in range(radius.shape[1])]
    # Assertions
    assert(len(filenames) == len(radius))
    assert(len(filenames) == len(lim_rows))
    return filenames, radius, lim_rows, dirnames


def allocate_in_files(indices, lim_rows):
    ## 1. Allocating rows for the different files
    n_files = indices.shape[0]/lim_rows
    indices_sep = [indices[i*lim_rows:(i+1)*lim_rows] for i in range(n_files)]
    aux = indices[n_files*lim_rows:]
    if aux != []:
        indices_sep.append(indices[n_files*lim_rows:])
    return indices_sep