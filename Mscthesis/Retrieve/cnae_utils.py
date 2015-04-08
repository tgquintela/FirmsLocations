
"""
Module oriented to retrieve columns related with cnae index.
"""


def retrieve_cols(df, val, cols):
    lvl = len(val)
    res = df[cols][df['cnae'].apply(lambda x: x[:lvl]) == val]
    return res
