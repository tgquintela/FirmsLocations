
"""
io and process computation tools
--------------------------------
The collection of functions to help to interact with files and process
information from them in the computations part.

"""

import shelve
import os


def store_model(pathfolder, scores, best_pars_info, names_comb, precomp_files,
                parameters, name_compute, administrative_info):
    """Main storing function."""
    ## Generate namefiles
    namefile = os.path.join(pathfolder, name_compute)
    ## Store
    db = shelve.open(namefile)
    db['scores'] = scores
    db['parameters'] = best_pars_info
    db['names'] = names_comb
    db['files_read'] = precomp_files
    db['parameters_input'] = parameter_directmodel_extraction(parameters)
    db['administrative_info'] = administrative_info
    db.close()


def read_store_model_descriptions(db):
    from itertools import product
    lengths = tuple([range(len(e)) for e in db['names']])
    names = [p for p in product(*db['names'])]
    indices = [p for p in product(*lengths)]
    return names, indices


def parameter_directmodel_extraction(pars):
    """Extraction of parameters from directmodel parameters."""
    perts_info = [(e[0], e[2]) for e in pars[0]]
    format_pars = [(e[0], e[2]) for e in pars[1]]
    models_pars = [(e[0], e[2]) for e in pars[2]]
    sampling_pars = [(e[0], e[2]) for e in pars[3]]
    scorer_pars = [(e[0], e[2]) for e in pars[4]]
    pars_extracted =\
        perts_info, format_pars, models_pars, sampling_pars, scorer_pars
    return pars_extracted
