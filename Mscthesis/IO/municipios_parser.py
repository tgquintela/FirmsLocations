

"""
municipios_parser
-----------------
Module which contains the process of parsing data of municipios.

TODO
----

"""

import pandas as pd

from pythonUtils.ProcessTools import Processer


class Municipios_Parser(Processer):
    """This class is the one which controls the parsing process of municipios
    information.
    """

    indices = None
    files = ''

    def __init__(self, logfile, bool_inform=False):
        "Instantiation of the class remembering it is a subclass of Processer."
        self.proc_name = "Municipios parser"
        self.proc_desc = "Parser the standarize data from file"
        self.subproc_desc = []
        self.t_expended_subproc = []
        self.logfile = logfile

    def parse(self, filepath):
        "Parse the data from the file given."
        data = pd.read_csv(filepath, sep=';', index_col=0)
        typevars = {}
        typevars['pop_vars'] = ['Poblacion', "Superficie", "Densidad"]
        typevars['loc_vars'] = ["longitud", "latitud"]
        return data, typevars
