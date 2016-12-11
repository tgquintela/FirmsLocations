
"""
Precomputer utils
-----------------
Precompute utils, to manage precomputations when there it is needed.
* Precompute population assignation to firms locations.
* Precompute population assignation to CP locations.
* Precompute network CP
* Precompute quality values.
* Precompute neighnet.

"""

import os
from pythonUtils.Logger import Logger
from pythonUtils.ProcessTools import Processer
from pySpatialTools.Preprocess.Transformations.Transformation_2d.\
    geo_transformations import general_projection

from ..IO.precomputers_io import read_population_assignation,\
    write_population_assignation, read_population_assignation_cp,\
    write_population_assignation_cp, read_net_cp, write_net_cp,\
    read_qvalues, write_qvalues, read_neighnet, write_neighnet,\
    read_pfeatures, write_pfeatures, write_locations, read_locations,\
    write_regions, read_regions
from ..IO.io_population_data import parse_munipios_data
from ..IO.io_standarized import get_locations, get_regions, compute_regions

from precomputers_functions import general_pfeatures_computation
from assignation_value import general_qvalue_assignation
from density_assignation import general_geo_interpolation


class PrecomputerCollection(Processer):
    """Collection of precomputers and manager of whole precomputer processing.
    """
    def _initialization(self):
        pass

    def __init__(self, logfile, pathdata, old_computed=False):
        self.precomputer_locations =\
            Precomputer_locations(logfile, pathdata, old_computed)
        self.precomputer_regions =\
            Precomputer_regions(logfile, pathdata, old_computed)
        self.precomputer_population =\
            Precomputer_population(logfile, pathdata, old_computed)
        self.precomputer_population_cp =\
            Precomputer_population_CP(logfile, pathdata, old_computed)
        self.precomputer_network_cp =\
            Precomputer_network_cp(logfile, pathdata, old_computed)
        self.precomputer_qvalues =\
            Precomputer_qvalues(logfile, pathdata, old_computed)
        self.precomputer_neighnet =\
            Precomputer_neighnet(logfile, pathdata, old_computed)
        self.precomputer_pfeatures =\
            Precomputer_pfeatures(logfile, pathdata, old_computed)

    def precompute(self, pars_locs=[], pars_regs=[], pars_pop=[],
                   pars_pop_cp=[], pars_net_cp=[], pars_neighnet=[],
                   pars_pfeatures=[], pars_qval=[]):
        self.precomputer_locations.precompute(pars_locs)
        self.precomputer_regions.precompute(pars_regs)
        self.precomputer_population.precompute(pars_pop)
        self.precomputer_population_cp.precompute(pars_pop_cp)
        self.precomputer_network_cp.precompute(pars_net_cp)
        self.precomputer_neighnet.precompute(pars_neighnet)
        self.precomputer_pfeatures.precompute(pars_pfeatures)
        self.precomputer_qvalues.precompute(pars_qval)

    def reload(self):
        self.precomputer_locations.reload()
        self.precomputer_regions.reload()
        self.precomputer_population.reload()
        self.precomputer_population_cp.reload()
        self.precomputer_network_cp.reload()
        self.precomputer_neighnet.reload()
        self.precomputer_pfeatures.reload()
        self.precomputer_qvalues.reload()


class Precomputer_specific(Processer):
    """Factorization of a precomputer manager class for specific tasks."""

    def _initialization_spec(self):
        self.files = []
        self.listparams = []
        self.subproc_desc = [""]
        self.t_expended_subproc = [0]

    def __init__(self, logfile, pathdata, old_computed=False):
        self._initialization()
        self._initialization_spec()
        pathfolder = os.path.join(pathdata, 'Cleaned/Precomputed')
        pathf = os.path.join(pathfolder, self._pathfolder)
        if not os.path.isdir(pathf):
            os.makedirs(pathf)
        else:
            if old_computed is False:
                filelist = os.listdir(pathf)
                for f in filelist:
                    os.remove(os.path.join(pathf, f))
        self.pathdata = pathdata
        self.pathfolder = os.path.join(pathfolder, self._pathfolder)
        self.logfile = Logger(logfile) if type(logfile) == str else logfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if type(i) == int:
            precomputed = self._retrieve_i(self.files[i])
        else:
            i = list(i)
            files = [self.files[i] for j in i]
            precomputed = self._retrieve(files)
        return precomputed

    def precompute(self, listparams):
        ## 0. Set vars
        t00 = self.setting_global_process()
        # Prepare subprocesses
        n_cases = len(listparams)
        self.t_expended_subproc = [0 for e in range(n_cases)]
        self.subproc_desc = [self._name+"_"+str(e) for e in range(n_cases)]

        ## Computations and storing
        for i in range(len(listparams)):
            t0 = self.set_subprocess([0])
            self._store(*self.compute(listparams[i]))
            self.close_subprocess([0], t0)
        self.files = os.listdir(self.pathfolder)
        self.listparams = listparams
        assert(len(self.files) == len(self.listparams))
        ## Untrack process
        self.close_process(t00)

    def _retrieve(self, listfiles=[]):
        if not len(listfiles):
            listfiles = os.listdir(self.pathfolder)
        precomputed = []
        for namefile in listfiles:
            precomputed.append(self._retrieve_i(namefile))
        return precomputed

    def reload(self, listparams=None):
        self.files = os.listdir(self.pathfolder)
        if listparams is not None:
            self.listparams = listparams
            assert(len(self.files) == len(self.listparams))


class Precomputer_locations(Precomputer_specific):
    """Precomputer class in which the population is assigned to firms.
    """
    _name = "Precomputer_locations"
    proc_name = "Precomputer locations"
    proc_desc = "Precomputer locations"
    _pathfolder = "Locations"

    def compute(self, methodvalues):
        locations, years, key_firms = get_locations(self.pathdata)
        new_locs = general_projection(locations[:, [1, 0]],
                                      **methodvalues['pars'])
        return key_firms, years, new_locs, methodvalues

    def _store(self, key_firms, years, new_locs, methodvalues):
        write_locations(self.pathfolder, key_firms, years, new_locs,
                        methodvalues)

    def _retrieve_i(self, namefile):
        hashes, key_firms, years, new_locs, methodvalues =\
            read_locations(namefile)
        return hashes, key_firms, years, new_locs, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            hashes, key_firms, years, new_locs, methodvalues =\
                read_locations(namefile)
            yield hashes, key_firms, years, new_locs, methodvalues


class Precomputer_regions(Precomputer_specific):
    """Precomputer class in which the regions is assigned to firms.
    """
    _name = "Precomputer_regions"
    proc_name = "Precomputer regions"
    proc_desc = "Precomputer regions"
    _pathfolder = "Regions"

    def compute(self, methodvalues):
        regions_data, key_firms = get_regions(self.pathdata)
        regions_data = compute_regions(regions_data, **methodvalues['pars'])
        return key_firms, regions_data, methodvalues

    def _store(self, key_firms, regions_data, methodvalues):
        write_regions(self.pathfolder, key_firms, regions_data, methodvalues)

    def _retrieve_i(self, namefile):
        key_firms, regions_data, methodvalues = read_regions(namefile)
        return key_firms, regions_data, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            key_firms, regions_data, methodvalues = read_regions(namefile)
            yield key_firms, regions_data, methodvalues


class Precomputer_population(Precomputer_specific):
    """Precomputer class in which the population is assigned to firms.
    """
    _name = "Precomputer_population"
    proc_name = "Precomputer population assignation"
    proc_desc = "Precomputer population assignation and storing"
    _pathfolder = "Population"

    def compute(self, methodvalues):
        pop_data, pop_locs = parse_munipios_data(self.pathdata)
        new_pop_locs = general_projection(pop_locs[:, [1, 0]])
        locations, years, key_firms = get_locations(self.pathdata)
        new_locs = general_projection(locations[:, [1, 0]])
        population_value =\
            general_geo_interpolation(new_pop_locs, pop_data,
                                      new_locs, **methodvalues['pars'])
        return key_firms, years, population_value, methodvalues

    def _store(self, key_firms, years, population_value, methodvalues):
        write_population_assignation(self.pathfolder, key_firms, years,
                                     population_value, methodvalues)

    def _retrieve_i(self, namefile):
        hashes, key_firms, years, population_value, methodvalues =\
            read_population_assignation(namefile)
        return hashes, key_firms, population_value, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            hashes, key_firms, years, population_value, methodvalues =\
                read_population_assignation(namefile)
            yield hashes, key_firms, years, population_value, methodvalues


class Precomputer_population_CP(Precomputer_specific):
    """Precomputer class in which the population is assigned to firms.
    """
    _name = "Precomputer_population_CP"
    proc_name = "Precomputer population assignation to CP"
    proc_desc = "Precomputer population assignation to CP and storing"
    _pathfolder = "Population_CP"

    def compute(self, pars):
        ## TODO
        cps, population_value, methodvalues = [], [], {}
        return cps, population_value, methodvalues

    def _store(self, cps, population_value, methodvalues):
        write_population_assignation_cp(self.pathfolder, cps, population_value,
                                        methodvalues)

    def _retrieve_i(self, namefile):
        cps, population_value, methodvalues =\
            read_population_assignation_cp(namefile)
        return cps, population_value, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            cps, population_value, methodvalues =\
                read_population_assignation_cp(namefile)
            yield cps, population_value, methodvalues


class Precomputer_network_cp(Precomputer_specific):
    """Precomputer the network of cp using the locations of firms and its CP.
    """
    _name = "Precomputer_network_cp"
    proc_name = "Precomputer network of cp"
    proc_desc = "Precomputer network of cp and storing"
    _pathfolder = "Network_CP"

    def compute(self, pars):
        ## TODO
        cps, net, methodvalues = [], [], {}
        return cps, net, methodvalues

    def _store(self, cps, net, methodvalues):
        write_net_cp(self.pathfolder, cps, net, methodvalues)

    def _retrieve_i(self, namefile):
        cps, net, methodvalues = read_net_cp(namefile)
        return cps, net, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            cps, net, methodvalues = read_net_cp(namefile)
            yield cps, net, methodvalues


class Precomputer_financial(Precomputer_specific):
    """Precompute all financial information in order to interpolate
    unkown data and have all data available.
    """
    _name = "Precomputer_financial"
    proc_name = "Precomputer financial values"
    proc_desc = "Precomputer financial values and storing"
    _pathfolder = "Financial"

    def compute(self, pars):
        ## TODO
        ()
        return nif, qvalue, year, methodvalues

    def _store(self, nif, qvalue, year, methodvalues):
        write_qvalues(self.pathfolder, nif, qvalue, year, methodvalues)

    def _retrieve_i(self, namefile):
        nif, qvalue, year, methodvalues = read_qvalues(namefile)
        return nif, qvalue, year, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            nif, qvalue, year, methodvalues = read_qvalues(namefile)
            yield nif, qvalue, year, methodvalues


class Precomputer_neighnet(Precomputer_specific):
    """Precomputer the spatial neighborhood-based network of firms locations.
    """
    _name = "Precomputer_neighnet"
    proc_name = "Precomputer neighborhood-based network"
    proc_desc = "Precomputer neighborhood-based network and storing"
    _pathfolder = "Network_Net"

    def compute(self, pars):
        ## TODO
        ()
        return nif, neighnet, methodvalues

    def _store(self, nif, neighnet, methodvalues):
        write_neighnet(self.pathfolder, nif, neighnet, methodvalues)

    def _retrieve_i(self, namefile):
        nif, neighnet, methodvalues = read_neighnet(namefile)
        return nif, neighnet, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            nif, neighnet, methodvalues = read_neighnet(namefile)
            yield nif, neighnet, methodvalues


class Precomputer_pfeatures(Precomputer_specific):
    """Precomputer the points features based on the firms finantial
    information.
    """
    _name = "Precomputer_pfeatures"
    proc_name = "Precomputer points features"
    proc_desc = "Precomputer points features matrix and storing"
    _pathfolder = "Pfeatures"

    def compute(self, methodvalues):
        nif, year, pfeatures =\
            general_pfeatures_computation(self.pathdata,
                                          **methodvalues['pars'])
        return nif, year, pfeatures, methodvalues

    def _store(self, nif, year, pfeatures, methodvalues):
        write_pfeatures(self.pathfolder, nif, year, pfeatures, methodvalues)

    def _retrieve_i(self, namefile):
        hashes, nif, year, pfeatures, methodvalues = read_pfeatures(namefile)
        return hashes, nif, year, pfeatures, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            hashes, nif, year, pfeatures, methodvalues =\
                read_pfeatures(namefile)
            yield hashes, nif, year, pfeatures, methodvalues


class Precomputer_qvalues(Precomputer_specific):
    """Precomputer the values of quality.
    """
    _name = "Precomputer_qvalues"
    proc_name = "Precomputer quality values"
    proc_desc = "Precomputer quality values and storing"
    _pathfolder = "Qvalues"

    def compute(self, methodvalues):
        nif, qvalue, year = general_qvalue_assignation(self.pathdata,
                                                       **methodvalues['pars'])
        return nif, qvalue, year, methodvalues

    def _store(self, nif, qvalue, year, methodvalues):
        write_qvalues(self.pathfolder, nif, qvalue, year, methodvalues)

    def _retrieve_i(self, namefile):
        hashes, nif, year, qvalue, methodvalues = read_qvalues(namefile)
        return hashes, nif, year, qvalue, methodvalues

    def __iter__(self):
        listfiles = os.listdir(self.pathfolder)
        for namefile in listfiles:
            hashes, nif, year, qvalue, methodvalues = read_qvalues(namefile)
            yield hashes, nif, year, qvalue, methodvalues
