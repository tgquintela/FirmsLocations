
"""
De moment el primer sera fer un preproces de les dades, posant-les en un format
amb el que puguem treballar be, eliminant les empreses que no tenen cap
activitat en els anys 2006-2012 (tancament en anys anteriors), etc.
I el segon sera una analisi estadistica per saber quantes empreses tenim de
cada tipus, anys d'activitat, etc.
"""

import numpy as np


def filter_servicios(servicios, date):
    """This function performs the filtering of the companies which do not have
    acitivity the years of the study.
    """
    #servicios = filter_by_activity(servicios)
    servicios = filter_by_date(servicios, date)
    indices = servicios.index
    # Transform special columns
    servicios = cp2str(servicios)
    servicios = cnae2str(servicios)
    return servicios, indices


def filter_by_activity(servicios):
    """This function filter the rows by activity.
    """
    indices_0 = np.array(servicios.index)
    total_activo = ['06Act', '07Act', '08Act', '09Act', '10Act', '11Act',
                    '12Act']
    indices = dropna_rows(servicios, total_activo)
    indices_b = list([ind for ind in indices_0 if ind not in indices])
    #servicios = servicios.loc[indices]
    servicios.drop(indices_b)
    return servicios


def filter_by_date(servicios, date):
    indices_0 = np.array(servicios.index)
    fecha_lim = date
    indices = (servicios['cierre'][servicios['cierre'] >= fecha_lim]).index
    indices = np.array(indices)
    indices_b = list([ind for ind in indices_0 if ind not in indices])
    #servicios = servicios.loc[indices]
    servicios.drop(indices_b)
    return servicios


###############################################################################
############################# AUXILIARY FUNCTIONS #############################
###############################################################################
def dropna_rows(df, columns):
    """Delete rows with absolute null values in the observations."""
    indices = np.array(df[columns].dropna(how='all').index)
    return indices


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
        df['cp'] = df['cp'].apply(cp2str_ind)
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
        df['cnae'] = df['cnae'].apply(cnae2str_ind)
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
            df[col] = df[col]
    return df


def to_int(df):
    cols = ['06Trab', '07Trab', '08Trab', '09Trab', '10Trab', '11Trab',
            '12Trab', '13Trab']
    ## Transformation
    columns = df.columns
    for col in columns:
        if col in cols:
            df[col] = df[col].astype(int)
    return df
