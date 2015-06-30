
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


