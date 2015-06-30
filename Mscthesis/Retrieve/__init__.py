
"""
Module oriented to 
"""


class Neighbourhood():
    """
    Retrieve neighs.
    """
    aggretriever = None
    retriever = None
    aggfeatures = None

    def define_aggretriever(self, aggregator, df, reindices):
        "Define the aggregation and its retriever."
        ##### TODO: Reindices
        self.aggfeatures, agglocs = aggregator.retrieve_aggregation(df)
        agglocs = np.array(agglocs)
        ndim, N_t = len(agglocs.shape), agglocs.shape[0]
        agglocs = agglocs if ndim == 2 else agglocs.reshape((N_t, 1))
        self.aggretriever = KDTree(agglocs, leafsize=100)

    def define_mainretriever(self, df, loc_vars):
        "Define the main retriever."
        locs = df[loc_vars].as_matrix().astype(float)
        ndim = len(locs.shape)
        locs = locs if ndim == 2 else locs.reshape((N_t, 1))
        self.retriever = KDTree(locs, leafsize=10000)

    def retrieve_neigh(self, point_i):
        """Retrieve the neighs information and the type of retrieving.
        Type of retrieving:
        - aggfeatures
        - indices of neighs
        """
        return neighbourhood, typereturn

    ###########################################################################
    ############################## Aggregation ################################
    ###########################################################################
    ## TODEPRECATE
    def ifcompute_aggregate(self, r):
        "Function to inform about retrieving aggregation values."
        # self.agg_info
        return self.bool_agg and r >= 2


class GridNeigh(Neighbourhood):
    "Neighbourhood grid-based object."

    def __init__(self):
        "Main function to map a group of points in a 2d to a grid."
        pass

    def create_grid(self, locs, grid_size, xlim=None, ylim=None):
        self.x, self.y = create_grid(locs, grid_size, xlim, ylim)

    def apply_grid(self, locs):
        locs_grid = apply_grid(locs, self.x, self.y)


class CircularNeigh(Neighbourhood):
    """General Neighbourhood for circular considerations. It could have
    variable radius.
    """
    def __init__(self, radius):
        pass

    def (self,):

        # KDTree retrieve object instantiation
        locs = df[loc_vars].as_matrix().astype(float)
        ndim = len(locs.shape)
        locs = locs if ndim == 2 else locs.reshape((N_t, 1))
        kdtree1 = KDTree(locs, leafsize=10000)
        agg_desc = None
        if self.bool_agg:
            agg_var = self.var_types['agg_var']
            ## TODO: Compute tables
            agg_desc, axis, locs2 = compute_aggregate_counts(df, agg_var,
                                                             loc_vars,
                                                             type_vars,
                                                             reindices)
            kdtree2 = KDTree(locs2, leafsize=100)

        # type_arr
        type_arr = df[type_vars].as_matrix().astype(int)
        ndim = len(type_arr.shape)
        type_arr = type_arr if ndim == 2 else type_arr.reshape((N_t, 1))
        # clean unnecessary
        del df


class KNeigh(Neighbourhood):
    "General neighbourhood composed by a fixed number of neighbours."

    def __init__(self, k):
        self.k = k

    def 


