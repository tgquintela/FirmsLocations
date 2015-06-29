
"""
Module which contain the functions and code structure to study correlation in
spatial data.
"""


class Aggregator():
    "Aggregate or read aggregate information."

    def __init__(self, typevars, filepath=None, reindices=None):
        self.typevars = typevars
        if filepath is None:
            self.bool_read_agg = False
            self.reindices = reindices
        else:
            self.bool_read_agg = True

    def retrieve_aggregation(self, df=None):
        "Main function for retrieving aggregation."
        if self.bool_read_agg:
            typevars, filepath = self.typevars, self.filepath
            aggfeatures, agglocs = read_aggregation(typevars, filepath)
        else:
            typevars, reindices = self.typevars, self.reindices
            aggfeatures, agglocs = create_aggregation(df, typevars, reindices)
        return aggfeatures, agglocs


def create_aggregation(df, typevars, reindices):
    agg_var, loc_vars = typevars[feat_vars], typevars[loc_vars]
    agg_desc, axis, locs2 = compute_aggregate_counts(df, agg_var, loc_vars,
                                                     type_vars, reindices)
    return agg_desc, locs2


def read_aggregation(typevars, filepath):
    aggtable = read_agg(filepath)
    aggfeatures = aggtable[typevars[feat_vars]]
    agglocs = aggtable[typevars[loc_vars]]
    return aggfeatures, agglocs


class Neighbourhood():
    """
    Aggregate files
    """

    def define_aggretriever(self, aggregator, df):
        "Define the aggregation and its retriever."
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

    def retrieve_neigh(self):
        """Retrieve the neighs information and the type of retrieving.
        Type of retrieving:
        - aggfeatures
        - indices of neighs
        """
        return neighbourhood, typereturn


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

    def 

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
