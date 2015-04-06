
"""
TODO
----
More than one plot
"""

from string import Template
from os.path import join
import os
import pandas as pd
import numpy as np


def describe2latex(study_info, stats):
    """Function to translate the descriptions of the variables to latex.

    TODO
    ----
    - crete a plot folder
    - get paths to save figures

    """
    ## 0. Needed variables
    os.mkdir(join(study_info['path'], 'Plots'))
    header = built_header()
    title = built_title(study_info)
    content = built_content(study_info, stats)

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/document_template.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(header=header, title=title,
                                                  content=content)

    file_ = open(join(study_info['path'], 'report.tex'), "w")
    file_.write(filetext)

    return filetext


###############################################################################
############################## LEVEL 1 functions ##############################
###############################################################################
def built_content(study_info, stats):
    pages = []
    for st in stats:
        pages.append(page_builder(st, study_info))

    content = '\newpage\n'.join(pages)
    return content


def built_title(study_info):
    ## 0. Needed variables
    title = study_info['title']
    #summary = study_info['summary']
    author = study_info['author']
    #date = study_info['date']

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/portada.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(title=title, author=author,
                                                  date='')
    return filetext


def built_header():
    file_ = open('../data/templates/tex/header.txt', "r")
    filecode = file_.read()
    return filecode


###############################################################################
############################## LEVEL 2 functions ##############################
###############################################################################
def page_builder(info, study_info):
    ## 0. Needed variables
    varname = info['variables_name']
    variables = info['variables']
    vardesc = info['Description']

    typevar = info['type'].lower()
    if typevar in ['discrete', 'categorical']:
        tables = cat_tables(info)
        plots = cat_plots(info, study_info)
    elif typevar == 'continuous':
        tables = cont_tables(info)
        plots = cont_plots(info, study_info)
    elif typevar == 'coordinates':
        tables = coord_tables(info)
        plots = coord_plots(info, study_info)
    elif typevar in ['time', 'temporal']:
        tables = coord_tables(info)
        plots = coord_plots(info, study_info)
    elif typevar == 'tmpdist':
        tables = coord_tables(info)
        plots = coord_plots(info, study_info)
    else:
        print typevar, info['variables']

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/page_cat.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(varname=varname,
                                                  variables=variables,
                                                  vardesc=vardesc,
                                                  tables=tables,
                                                  plots=plots)
    return filetext


def cat_tables(info, max_rows=15):
    # 0. Needed variables
    if info['count_table'].shape[0] > max_rows:
        table = info['count_table'][:max_rows]
    else:
        table = info['count_table']

    table = pd.DataFrame(table, columns=[info['variables']])
    tabular = table.to_latex()
    caption = 'Counts of the most common values of %s.'
    caption = caption % info['variables_name']
    tablelabel = info['variables_name']+'_01'

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/table.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(tabular=tabular,
                                                  caption=caption,
                                                  tablelabel=tablelabel)

    return filetext


def cat_plots(info, study_info):
    # 0. Needed variables
    fname = '%s.png' % (info['variables_name']+'_01')
    fig = info['plots']
    fig.savefig(join(study_info['path']+'/Plots/', fname))
    graphics = 'Plots/'+fname
    caption = 'Plot of the distribution of %s' % info['variables_name']
    imagelabel = info['variables_name']+'_01'

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/image.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(graphics=graphics,
                                                  caption=caption,
                                                  imagelabel=imagelabel)
    return filetext


def cont_tables(info, max_rows=15):
    ## TODO: Number of nulls
    # 0. Needed variables
#    if info['hist_table'][0].shape[0] > max_rows:
#        table = info['hist_table'][0][:max_rows]
#    else:
#        table = info['hist_table']

    # 1
    tabular = "\begin{tabular}{lr}\n\toprule\n\midrule\n\nmean"
    tabular = tabular + " &  %f \\\n\bottomrule\n\end{tabular}\n"
    tabular = tabular % info['mean']
    caption = ''
    tablelabel = info['variables_name']+'_mean'
    # 2
    aux = np.vstack([info['ranges'], info['quantiles']])
    table = pd.DataFrame(aux.T, columns=['ranges', 'quantiles']).transpose()
    tabular2 = table.to_latex()
    caption2 = 'Comparative between quantiles and proportional segments of %s'
    caption2 = caption2 % info['variables_name']
    tablelabel2 = info['variables_name']+'_01'
    # TODO counts

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/table.txt', "r")
    filecode = file_.read()
    filetext1 = Template(filecode).safe_substitute(tabular=tabular,
                                                   caption=caption,
                                                   tablelabel=tablelabel)
    file_ = open('../data/templates/tex/table.txt', "r")
    filecode = file_.read()
    filetext2 = Template(filecode).safe_substitute(tabular=tabular2,
                                                   caption=caption2,
                                                   tablelabel=tablelabel2)
    filetext = '\n\n'.join([filetext1, filetext2])
    return filetext


def cont_plots(info, study_info):
    # 0. Needed variables
    fname = '%s.png' % (info['variables_name']+'_01')
    fig = info['plots']
    fig.savefig(join(study_info['path']+'/Plots/', fname))
    graphics = 'Plots/'+fname
    caption = 'Plot of the distribution of %s' % info['variables_name']
    imagelabel = info['variables_name']+'_01'

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/image.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(graphics=graphics,
                                                  caption=caption,
                                                  imagelabel=imagelabel)
    return filetext


def coord_plots(info, study_info):
    # 0. Needed variables
    fname = '%s.png' % (info['variables_name']+'_01')
    fig = info['plots']
    fig.savefig(join(study_info['path']+'/Plots/', fname))
    graphics = 'Plots/'+fname
    caption = 'Plot spatial distribution of companies'
    imagelabel = info['variables_name']+'_01'

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/image.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(graphics=graphics,
                                                  caption=caption,
                                                  imagelabel=imagelabel)
    return filetext


def coord_tables(info):
    return ''


def temp_plots(info, study_info):
    # 0. Needed variables
    fname = '%s.png' % (info['variables_name']+'_01')
    fig = info['plots']
    fig.savefig(join(study_info['path']+'/Plots/', fname))
    graphics = 'Plots/'+fname
    caption = 'Plot of the distribution of %s' % info['variables_name']
    imagelabel = info['variables_name']+'_01'

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/image.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(graphics=graphics,
                                                  caption=caption,
                                                  imagelabel=imagelabel)
    return filetext


def temp_tables(info):
    # 0. Needed variables
    pre_post = info['pre_post']
    cols = ['pre', 'through', 'post']
    table = np.array([pre_post[e] for e in cols]).reshape((1, 3))
    tabular = table.to_latex()
    caption = 'Counts of the most common values of %s.'
    caption = caption % info['variables_name']
    tablelabel = info['variables_name']+'_01'

    ## 1. Applying to the template
    file_ = open('Mscthesis/data/templates/tex/table.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(tabular=tabular,
                                                  caption=caption,
                                                  tablelabel=tablelabel)
    return filetext


def tmpdist_plots(info, study_info):
    # 0. Needed variables
    fname = '%s.png' % (info['variables_name']+'_01')
    fig = info['plots']
    fig.savefig(join(study_info['path']+'/Plots/', fname))
    graphics = 'Plots/'+fname
    caption = 'Plot of the distribution of %s' % info['variables_name']
    imagelabel = info['variables_name']+'_01'

    ## 1. Applying to the template
    file_ = open('../data/templates/tex/image.txt', "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(graphics=graphics,
                                                  caption=caption,
                                                  imagelabel=imagelabel)
    return filetext


def tmpdist_tables(info):
    return ''
