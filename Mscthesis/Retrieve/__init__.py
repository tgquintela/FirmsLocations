
"""
Module oriented to compute neighbourhood.
"""


from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


class Neighbourhood():
    """
    Retrieve neighs.
    """
    aggretriever = None
    retriever = None
    aggfeatures = None
    agglocs = None

    cond_funct = lambda x, y, z: False

    def __init__(self, retriever):
        self.define_mainretriever(retriever)

    def define_mainretriever(self, retriever):
        self.retriever = retriever

    def define_aggretrievers(self, aggregators, df, reindices, funct=None):
        if type(aggregators) != list:
            aggregators = [aggregators]
        n_agg = len(aggregators)
        agglocs, aggfeatures, aggretriever = [], [], []
        for i in range(n_agg):
            agg = aggregators[i]
            auxlocs, auxfeatures = agg.retrieve_aggregation(df, reindices,
                                                            funct)
            agglocs.append(auxlocs)
            aggfeatures.append(auxfeatures)
            aggretriever.append(agg.discretizor)
        self.agglocs = agglocs
        self.aggfeatures = aggfeatures
        self.aggretriever = aggretriever

    def retrieve_neigh(self, point_i, cond_i, info_i):
        """Retrieve the neighs information and the type of retrieving.
        Type of retrieving:
        - aggfeatures: aggregate
        - indices of neighs: neighs_i
        """
        point_i = point_i.reshape(1, point_i.shape[0])
        typereturn = self.get_type_return(point_i, cond_i)
        if typereturn:
            neighbourhood = self.retrieve_neighs_agg(point_i, info_i)
        else:
            neighbourhood = self.retrieve_neighs_i(point_i, info_i)
        typereturn = 'aggregate' if typereturn else 'individual'
        return neighbourhood, typereturn

    def retrieve_neighs_agg(self, point_i, info_i):
        "Retrieve the correspondent regions."
        out = []
        for i in range(len(self.aggretriever)):
            out.append(self.aggretriever.map2id(point_i))
        return out

    def retrieve_neighs_i(self, point_i, info_i):
        "Retrieve the neighs."
        return self.retriever.retrieve_neighs(point_i, info_i)

    ###########################################################################
    ########################## Condition aggregation ##########################
    ###########################################################################
    def set_aggcondition(self, f):
        "Setting condition function for aggregate data retrieval."
        self.cond_funct = f

    def get_type_return(self, point_i, cond_i):
        "Apply condition setted."
        ## TODO: Add the possibility to not be in aggregate and return False
        return self.cond_funct(point_i, cond_i)


###############################################################################
################################# Retrievers ##################################
###############################################################################
class Retriever():
    "Class which contains the retriever of points."

    def __init__(self, locs):
        self.retriever = self.define_retriever(locs)

    def retrieve_neighs(self, point_i, info_i, ifdistance=False):
        self.retrieve_neighs_spec(point_i, info_i, ifdistance)


class KRetriever(Retriever):
    "Class which contains a retriever of K neighbours."

    def define_retriever(self, locs):
        leafsize = locs.shape[0]
        leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        return KDTree(locs, leafsize=leafsize)

    def retrieve_neighs_spec(self, point_i, info_i, ifdistance=False):
        res = self.retriever.query(point_i, info_i)
        if not ifdistance:
            res = res[1]
        return res


class CircRetriever(Retriever):
    "Circular retriever."
    def define_retriever(self, locs):
        leafsize = locs.shape[0]
        leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        return KDTree(locs, leafsize=leafsize)

    def retrieve_neighs_spec(self, point_i, info_i, ifdistance=False):
        res = self.retriever.query_ball_point(point_i, info_i)
        if ifdistance:
            aux = cdist(point_i, self.retriever.data[res, :])
            res = aux, res
        return res
