
"""
Cleanning module
----------------
This module is oriented to standarize the data to a more proper use of space.
The process is defined to format properly the data
from SABI dataset to be optimally used by this package.
The data is formated to the structure of folders that the package recognizes.

Folder structure ===========
============================
Parent_folder
    |-- Main
        |-- Servicios
    |-- Finantial
        |-- year
            |-- Servicios
        |-- ...
    |-- Aggregated
        |-- Agg_by_cp
============================

TODO
====
- Change imports

"""

from pythonUtils.ProcessTools import Processer

from clean_module import folder_structure, get_financial_cols,\
    parse_write_manufactures, parse_write_manufactures


def CleanProcess(Processer):
    "Process defined to clean data from raw data."

    def __init__(self, logfile, bool_inform=False):
        "Instantiation of the class remembering it is a subclass of Processer."
        self.logfile = logfile
        self.bool_inform = bool_inform
        self.proc_name = "Clean process"
        self.proc_desc = "Cleaning raw data to standarize it."
        self.subproc_desc = ["Cleaning manufacturas", "Cleaning servicios"]
        self.t_expended_subproc = [0, 0]

    def clean(inpath, outpath, extension='csv'):
        "Main function of the class."
        ## 0. Track process
        t0 = self.setting_process()
        ## 1. Ensure creation of needed folders
        folder_structure(outpath)
        ## 2. Set financial columns
        finantial_cols = get_financial_cols()
        ## 3. Parse and write manufactures
        t1 = self.set_subprocess([0])
        parse_write_manufactures(inpath, outpath, extension, finantial_cols)
        close_subprocess([0], t1)
        ## 4. Parse and write servicios
        t1 = self.set_subprocess([1])
        parse_write_servicios(inpath, outpath, extension, finantial_cols)
        close_subprocess([1], t1)
        ## 5. Stop tracking process
        self.close_process(t0)
