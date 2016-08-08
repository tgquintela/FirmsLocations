
"""
Precomputers IO
---------------
Module which groups all the functions related with input-output precomputation
of useful data from raw, or just clean data.

"""

import os
import shelve
import datetime


def generate_namefile(pathfolder, methodvalues):
    """Generate a namefile from other information."""
    datestr = datetime.datetime.now().date().strftime('%F')
    paramsstr = str(hash(str(methodvalues)))
    namefile = datestr + '-' + methodvalues['codename'] + '_' + paramsstr
    namefile = os.path.join(pathfolder, namefile)
    return namefile


###############################################################################
################################## Locations ##################################
###############################################################################
def write_locations(pathfolder, key_firms, years, locs, methodvalues):
    ## Generate namefile
    namefile = generate_namefile(pathfolder, methodvalues)

    ## Writting
    db = shelve.open(namefile)
    db['nif'] = key_firms
    db['year'] = years
    db['locations'] = locs
    db['methodvalues'] = methodvalues
    db.close()


def read_locations(namefile):
    """Read pre-computed population assignation."""
    db = shelve.open(namefile)
    key_firms = db['nif']
    year = db['year']
    locs = db['population']
    methodvalues = db['methodvalues']
    db.close()
    return key_firms, year, locs, methodvalues


###############################################################################
########################### Population assignation ############################
###############################################################################
def write_population_assignation(pathfolder, key_firms, year, population_value,
                                 methodvalues):
    """Write pre-computed population assignation."""
    ## Generate namefile
    namefile = generate_namefile(pathfolder, methodvalues)

    ## Writting
    db = shelve.open(namefile)
    db['nif'] = key_firms
    db['year'] = year
    db['population'] = population_value
    db['methodvalues'] = methodvalues
    db.close()


def read_population_assignation(namefile):
    """Read pre-computed population assignation."""
    db = shelve.open(namefile)
    key_firms = db['nif']
    year = db['year']
    population_value = db['population']
    methodvalues = db['methodvalues']
    db.close()
    return key_firms, year, population_value, methodvalues


###############################################################################
############################ Neighnet computation #############################
###############################################################################
def write_population_assignation_cp(pathfolder, cps, population_value,
                                    methodvalues):
    """Write pre-computed population assignation."""
    ## Generate namefile
    namefile = generate_namefile(pathfolder, methodvalues)

    ## Writting
    db = shelve.open(namefile)
    db['cps'] = cps
    db['population_value'] = population_value
    db['methodvalues'] = methodvalues
    db.close()


def read_population_assignation_cp(namefile):
    """Read pre-computed population assignation."""
    db = shelve.open(namefile)
    cps = db['cps']
    population_value = db['population_value']
    methodvalues = db['methodvalues']
    db.close()
    return cps, population_value, methodvalues


###############################################################################
############################ Neighnet computation #############################
###############################################################################
def write_net_cp(pathfolder, cps, net, methodvalues):
    """Write pre-computed network cp assignation."""
    ## Generate namefile
    namefile = generate_namefile(pathfolder, methodvalues)

    ## Writting
    db = shelve.open(namefile)
    db['cps'] = cps
    db['net'] = net
    db['methodvalues'] = methodvalues
    db.close()


def read_net_cp(namefile):
    """Read pre-computed network cp assignation."""
    db = shelve.open(namefile)
    cps = db['cps']
    net = db['net']
    methodvalues = db['methodvalues']
    db.close()
    return cps, net, methodvalues


###############################################################################
######################### Quality values computation ##########################
###############################################################################
def write_qvalues(pathfolder, nif, qvalue, year, methodvalues):
    """Write pre-computed quality values assignation."""
    ## Generate namefile
    namefile = generate_namefile(pathfolder, methodvalues)

    ## Writting
    db = shelve.open(namefile)
    db['nif'] = nif
    db['qvalue'] = qvalue
    db['year'] = year
    db['methodvalues'] = methodvalues
    db.close()


def read_qvalues(namefile):
    """Read pre-computed quality values assignation."""
    db = shelve.open(namefile)
    nif = db['nif']
    qvalue = db['qvalue']
    year = db['year']
    methodvalues = db['methodvalues']
    db.close()
    return nif, qvalue, year, methodvalues


###############################################################################
############################ Neighnet computation #############################
###############################################################################
def write_neighnet(pathfolder, nif, neighnet, methodvalues):
    """Write pre-computed neighnet computation."""
    ## Generate namefile
    namefile = generate_namefile(pathfolder, methodvalues)

    ## Writting
    db = shelve.open(namefile)
    db['nif'] = nif
    db['neighnet'] = neighnet
    db['methodvalues'] = methodvalues
    db.close()


def read_neighnet(namefile):
    """Read pre-computed neighnet computation."""
    db = shelve.open(namefile)
    nif = db['nif']
    neighnet = db['neighnet']
    methodvalues = db['methodvalues']
    db.close()
    return nif, neighnet, methodvalues


###############################################################################
######################### Point features computation ##########################
###############################################################################
def write_pfeatures(pathfolder, nif, year, pfeatures, methodvalues):
    """Write pre-computed neighnet computation."""
    ## Generate namefile
    namefile = generate_namefile(pathfolder, methodvalues)

    ## Writting
    db = shelve.open(namefile)
    db['nif'] = nif
    db['year'] = year
    db['pfeatures'] = pfeatures
    db['methodvalues'] = methodvalues
    db.close()


def read_pfeatures(namefile):
    """Read pre-computed neighnet computation."""
    db = shelve.open(namefile)
    nif = db['nif']
    year = db['year']
    pfeatures = db['pfeatures']
    methodvalues = db['methodvalues']
    db.close()
    return nif, year, pfeatures, methodvalues
