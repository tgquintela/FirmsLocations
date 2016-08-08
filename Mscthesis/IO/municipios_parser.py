

"""
municipios_parser
-----------------
Module which contains the process of parsing data of municipios.

TODO
----
Municipios per year

"""

import pandas as pd

from pythonUtils.ProcessTools import Processer
from pythonUtils.Logger import Logger


class Municipios_Parser(Processer):
    """This class is the one which controls the parsing process of municipios
    information.
    """

    def _initialization(self):
        self.indices = None
        self.files = []
        self.proc_name = "Municipios parser"
        self.proc_desc = "Parser the standarize data from file"
        self.subproc_desc = []
        self.t_expended_subproc = []

    def __init__(self, logfile, bool_inform=False):
        "Instantiation of the class remembering it is a subclass of Processer."
        self._initialization()
        self.logfile = Logger(logfile) if type(logfile) == str else logfile

    def parse(self, filepath):
        "Parse the data from the file given."
        data, typevars = parse_municipios(filepath)
        return data, typevars


def parse_municipios(filepath):
    data = pd.read_csv(filepath, sep=';', index_col=0)
    typevars = {}
    typevars['feat_vars'] = ['Poblacion', "Superficie", "Densidad"]
    typevars['loc_vars'] = ["longitud", "latitud"]
    return data, typevars
