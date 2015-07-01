
"""
Module oriented to compute neighbourhood.
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
        agglocs = agglocs if ndim > 1 else agglocs.reshape((N_t, 1))
        self.aggretriever = KDTree(agglocs, leafsize=100)

    def define_mainretriever(self, df, loc_vars):
        "Define the main retriever."
        locs = df[loc_vars].as_matrix().astype(float)
        ndim = len(locs.shape)
        locs = locs if ndim > 1 else locs.reshape((N_t, 1))

        self.retriever = KDTree(locs, leafsize=10000)

    def retrieve_neigh(self, point_i):
        """Retrieve the neighs information and the type of retrieving.
        Type of retrieving:
        - aggfeatures: aggregate
        - indices of neighs: neighs_i
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
    """Neighbourhood grid-based object. It is an static retriever. This means
    that all the computations queries can be retrieved the aggregate values.
    """

    def __init__(self, locs, grid_size, xlim=(None, None), ylim=(None, None)):
        "Main function to map a group of points in a 2d to a grid."
        ## TODO: locs not needed.
        self.create_grid(locs, grid_size, xlim, ylim)

    def create_grid(self, locs, grid_size, xlim=(None, None),
                    ylim=(None, None)):
        self.x, self.y = create_grid(locs, grid_size, xlim, ylim)

    def apply_grid(self, locs):
        locs_grid = apply_grid(locs, self.x, self.y)
        return locs_grid

    def retrieve_neighs_sclass(self, point_i):
        ## Transform point_i in grid coords
        ## Retrieve the coordinates with same grid
        pass

    def define_aggretriever_spec(self, aggregator, df, reindices):
        ""
        _, aggfeatures = aggregator.retrieve_aggregation(locs_grid, None,
                                                         feat_arr, reindices)
        self.aggretriever, self.aggfeatures = locs_grid, aggfeatures


###############################################################################
############################### Grid functions ################################
###############################################################################
def aggretrieve_grid(point_i, agglocs, aggfeatures):
    neighs = retrieve_aggneighs(point_i, agglocs)
    feats = aggretrieve_vals(neighs, aggfeatures)
    return feats


def retrieve_aggneighs(point_i, agglocs):
    "Retrieve aggregate neighs."
    logi = agglocs == point_i
    return logi


def aggretrieve_vals(neighs, aggfeatures):
    # agglocs are unique locations
    if logi.sum() == 0:
        feats = np.zeros(aggfeatures.shape[1])
    else:
        feats = aggfeatures[logi, :]
    return feats



        self.aggfeatures, agglocs = aggregator.retrieve_aggregation(df)
        agglocs = np.array(agglocs)
        ndim, N_t = len(agglocs.shape), agglocs.shape[0]
        agglocs = agglocs if ndim > 1 else agglocs.reshape((N_t, 1))
        self.aggretriever = KDTree(agglocs, leafsize=100)


class CircularNeigh(Neighbourhood):
    """General Neighbourhood for circular considerations. It could have
    variable radius.
    """
    def __init__(self, radius):
        self.radius = radius
        self.retriever = None

    def define_mainretriever(self, locs):
        "Define the main retriever."
        ndim = len(locs.shape)
        locs = locs if ndim > 1 else locs.reshape((N_t, 1))
        self.retriever = KDTree(locs, leafsize=10000)

    def define_aggretriever(self, aggregator, df, reindices):
        "Define the aggregation and its retriever."
        ##### TODO: Reindices
        self.aggfeatures, agglocs = aggregator.retrieve_aggregation(df)
        agglocs = np.array(agglocs)
        ndim, N_t = len(agglocs.shape), agglocs.shape[0]
        agglocs = agglocs if ndim > 1 else agglocs.reshape((N_t, 1))
        self.aggretriever = KDTree(agglocs, leafsize=100)

    def create_aggretriever(self, locs):
        "Creation of the aggretriever."

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


###############################################################################
############################# Circular functions ##############################
###############################################################################
def define_aggretriever_circ(aggregator, locs_grid, type_arr, reindices):
    "Define the aggretriever information."
    # locs_grid, reindices, type_arr
    agglocs, aggfeatures = aggregator.retrieve_aggregation(locs_grid, None,
                                                           type_arr, reindices)
    aggretriever = KDTree(agglocs, leafsize=100)
    return aggretriever, aggfeatures


def aggretrieve_grid(point_i, agglocs, aggfeatures):
    neighs = retrieve_aggneighs(point_i, agglocs)
    feats = aggretrieve_vals(neighs, aggfeatures)
    return feats


def retrieve_aggneighs(point_i, agglocs):
    "Retrieve aggregate neighs."
    aggneighs = agglocs.query_ball(point_i, self.r)
    return aggneighs


def aggretrieve_vals(neighs, aggfeatures):
    # agglocs are unique locations
    if logi.sum() == 0:
        feats = np.zeros(aggfeatures.shape[1])
    else:
        feats = aggfeatures[logi, :]
    return feats




class KNeigh(Neighbourhood):
    "General neighbourhood composed by a fixed number of neighbours."

    def __init__(self, k):
        self.k = k
