

"""
Module to preprocess the data and filter unuseful cols or reformat the data in
column-wise way.
"""


def filtercols_empresas(empresas, filtercolsinfo):
    "TODO:"
    return empresas


def categorize_cols(df):
    df = cp2str(df)
    df = cnae2str(df)
    return df


def generate_replace(type_vals):
    "Generate the replace for use indices and save memory."
    repl = {}
    for v in type_vals.keys():
        repl[v] = dict(zip(type_vals[v], range(len(type_vals[v]))))
    return repl


############################# Particular columns ##############################
###############################################################################
def cp2str(df):
    """Retransform cp to string."""
    def cp2str_ind(x):
        try:
            x = str(int(x))
            x = (5-len(x))*'0'+x
        except:
            pass
        return x
    if 'cp' in df.columns:
        df.loc[:, 'cp'] = df['cp'].apply(cp2str_ind)
    return df


def cnae2str(df):
    """Transforming cnae code to string."""
    def cnae2str_ind(x):
        try:
            x = str(int(x))
        except:
            pass
        return x
    if 'cnae' in df.columns:
        df.loc[:, 'cnae'] = df['cnae'].apply(cnae2str_ind)
    return df


def to_float(df):
    ## Columns which has to be numbers
    cols = ['06Act', '07Act', '08Act', '09Act', '10Act', '11Act', '12Act',
            '13Act', '06ActC', '07ActC', '08ActC', '09ActC', '10ActC',
            '11ActC', '12ActC', '13ActC', '06Pasfijo', '07Pasfijo',
            '08Pasfijo', '09Pasfijo', '10Pasfijo', '11Pasfijo', '12Pasfijo',
            '13Pasfijo', '06Pasliq', '07Pasliq', '08Pasliq', '09Pasliq',
            '10Pasliq', '11Pasliq', '12Pasliq', '13Pasliq', '06Va', '07Va',
            '08Va', '09Va', '10Va', '11Va', '12Va', '13Va', '06Vtas', '07Vtas',
            '08Vtas', '09Vtas', '10Vtas', '11Vtas', '12Vtas', '13Vtas']
    ## Transformation
    columns = df.columns
    for col in columns:
        if col in cols:
            df.loc[:, col] = df[col]
    return df


def to_int(df):
    cols = ['06Trab', '07Trab', '08Trab', '09Trab', '10Trab', '11Trab',
            '12Trab', '13Trab']
    ## Transformation
    columns = df.columns
    for col in columns:
        if col in cols:
            df.loc[:, col] = df[col].astype(int)
    return df
