
"""
"""


def retrieve(kdobject, Coord, i, r):
    """
    """

    results = kdobject.query_ball_point(Coord[i, :], r)
    results.remove(i)
    return results
