
"""
Module oriented to retrieve columns related with cnae index.
"""


def retrieve_cols(df, val, cols):
    lvl = len(val)
    res = df[cols][df['cnae'].apply(lambda x: x[:lvl]) == val]
    return res


def transform_cnae_col(cnae_col, lvl):
    """"""
    lvl_n = len(cnae_col[1])
    if lvl >= lvl_n:
        return cnae_col
    else:
        return cnae_col.apply(lambda x: x[:lvl])
