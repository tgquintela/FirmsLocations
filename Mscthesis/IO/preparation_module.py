
"""
Auxiliar module in order to group the preparation functions.

"""


def prepare_filterinfo(year):
    "Preparation of the filter info."
    # Default needed variables
    loc_vars = ['ES-X', 'ES-Y']
    if year is None:
        years = [2006, 2007, 2008, 2009, 2010, 2011, 2012]
        all_y = False
    else:
        years = [year]
        all_y = True
    # Creation dicts
    date_filter_nfo = {'method': 'by_years',
                       'date_filter_nfo': {'years': years, 'all_y': all_y}}
    coord_filter_nfo = {'coord_vars': loc_vars, 'method': 'spaincoordinates'}
    filterinfo = {'coord_filter_nfo': coord_filter_nfo,
                  'date_filter_nfo': date_filter_nfo}
    return filterinfo


def prepare_concatinfo():
    "Preparation of the concat info in a dictionary structure."
    concatinfo = {'serviciosvar': None, 'companiesvar': None}
    return concatinfo


def prepare_filtercolsinfo():
    return {}
