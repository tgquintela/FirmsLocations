
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
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    if not os.path.exists(join(study_info['path'], 'Plots')):
        os.mkdir(join(study_info['path'], 'Plots'))
    header = built_header()
    title = built_title(study_info)
    content = built_content(study_info, stats)

    ## 1. Applying to the template
    templ_fl = join(this_dir, '../data/templates/tex/document_template.txt')
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(header=header, title=title,
                                                  content=content)
    filetext = filetext.encode('utf-8')
    return filetext
    #file_ = open(join(study_info['path'], 'report.tex'), "w")
    #file_.write(filetext)

    #return filetext


###############################################################################
############################## LEVEL 1 functions ##############################
###############################################################################
def built_content(study_info, stats):
    intro = build_intro(study_info)
    pages = []
    for st in stats:
        pages.append(page_builder(st, study_info))

    content = '\\newpage\n'.join(pages)
    content = '\\newpage\n'.join([intro, content])
    content = content.decode('utf-8')
    return content


def built_title(study_info):
    ## 0. Needed variables
    title = study_info['title']
    #summary = study_info['summary']
    author = study_info['author']
    #date = study_info['date']

    ## 1. Applying to the template
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    templ_fl = join(this_dir, '../data/templates/tex/portada.txt')
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(title=title, author=author,
                                                  date='')
    filetext = filetext.decode('utf-8')
    return filetext


def built_header():
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    templ_fl = join(this_dir, '../data/templates/tex/header.txt')
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filecode = filecode.decode('utf-8')
    return filecode


###############################################################################
############################## LEVEL 2 functions ##############################
###############################################################################
def build_intro(study_info):
    global_stats = study_info['global_stats']
    text = '\section{Variables}\n'+global_stats.to_latex()
    text = text.decode('utf-8')
    return text


def page_builder(info, study_info):
    ## 0. Needed variables
    varname = info['variables_name'].decode('utf-8').encode('utf-8')
    variables = info['variables']
    vardesc = info['Description'].decode('utf-8').encode('utf-8')

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
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    templ_fl = join(this_dir, '../data/templates/tex/page.txt')
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(varname=varname,
                                                  variables=variables,
                                                  vardesc=vardesc,
                                                  tables=tables,
                                                  plots=plots,
                                                  comments='',
                                                  artificialcomments='')
    filetext = filetext.decode('utf-8')
    return filetext


###############################################################################
############################## LEVEL 3 functions ##############################
###############################################################################
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
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    templ_fl = join(this_dir, '../data/templates/tex/table.txt')
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(tabular=tabular,
                                                  caption=caption,
                                                  tablelabel=tablelabel)

    return filetext


def cat_plots(info, study_info):
    # 0. Needed variables
    if not 'plots' in info.keys():
        return ''
    # Save plots to computer
    graphics, caption, imagelabel = plot_saving(info, study_info)

    ## 1. Applying to the template
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    filename = get_filename_template(len(graphics))
    templ_fl = join(this_dir, filename)
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(caption=caption,
                                                  imagelabel=imagelabel)
    filetext = substitute_plots(graphics, filetext)

    return filetext


def cont_tables(info, max_rows=15, nmax_cols=7):
    ## TODO: Number of nulls
    # 0. Needed variables
#    if info['hist_table'][0].shape[0] > max_rows:
#        table = info['hist_table'][0][:max_rows]
#    else:
#        table = info['hist_table']

    # 1
    tabular = "\\begin{tabular}{lr}\n\\toprule\n\midrule\n\nmean"
    tabular = tabular + " &  %f \\\\\n\\bottomrule\n\end{tabular}\n"
    tabular = tabular % info['mean']
    caption = ''
    tablelabel = info['variables_name']+'_mean'
    # 2
    aux = np.vstack([info['ranges'], info['quantiles']])
    table = pd.DataFrame(aux.T, columns=['ranges', 'quantiles'])
    # Formatting table to fit into a page
    ni, na = table.index[0], table.shape[0]
    nmax_cols = nmax_cols if na >= nmax_cols else na
    idxs = np.linspace(ni, na-1, nmax_cols).round().astype(int)
    table = table[['ranges', 'quantiles']]
    table = table.transpose()
    table = table[idxs]
    tabular2 = table.to_latex(float_format=lambda x: '%.2f' % x)
    caption2 = 'Comparative between quantiles and proportional segments of %s'
    caption2 = caption2 % info['variables_name']
    tablelabel2 = info['variables_name']+'_01'
    # TODO counts

    ## 1. Applying to the template
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    templ_fl = join(this_dir, '../data/templates/tex/table.txt')
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext1 = Template(filecode).safe_substitute(tabular=tabular,
                                                   caption=caption,
                                                   tablelabel=tablelabel)

    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    templ_fl = join(this_dir, '../data/templates/tex/table.txt')
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext2 = Template(filecode).safe_substitute(tabular=tabular2,
                                                   caption=caption2,
                                                   tablelabel=tablelabel2)
    filetext = '\n\n'.join([filetext1, filetext2])
    return filetext


def cont_plots(info, study_info):
    # 0. Needed variables
    if not 'plots' in info.keys():
        return ''
    # Save plots to computer
    graphics, caption, imagelabel = plot_saving(info, study_info)

    ## 1. Applying to the template
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    filename = get_filename_template(len(graphics))
    templ_fl = join(this_dir, filename)
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(caption=caption,
                                                  imagelabel=imagelabel)
    filetext = substitute_plots(graphics, filetext)

    return filetext


def coord_plots(info, study_info):
    # 0. Needed variables
    if not 'plots' in info.keys():
        return ''
    # Save plots to computer
    graphics, caption, imagelabel = plot_saving(info, study_info)

    ## 1. Applying to the template
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    filename = get_filename_template(len(graphics))
    templ_fl = join(this_dir, filename)
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(caption=caption,
                                                  imagelabel=imagelabel)
    filetext = substitute_plots(graphics, filetext)

    return filetext


def coord_tables(info):
    return ''


def temp_plots(info, study_info):
    # 0. Needed variables
    if not 'plots' in info.keys():
        return ''
    # Save plots to computer
    graphics, caption, imagelabel = plot_saving(info, study_info)

    ## 1. Applying to the template
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    filename = get_filename_template(len(graphics))
    templ_fl = join(this_dir, filename)
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(caption=caption,
                                                  imagelabel=imagelabel)
    filetext = substitute_plots(graphics, filetext)

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
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    templ_fl = join(this_dir, '../data/templates/tex/table.txt')
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(tabular=tabular,
                                                  caption=caption,
                                                  tablelabel=tablelabel)
    return filetext


def tmpdist_plots(info, study_info):
    # 0. Needed variables
    if not 'plots' in info.keys():
        return ''
    # Save plots to computer
    graphics, caption, imagelabel = plot_saving(info, study_info)

    ## 1. Applying to the template
    this_dir, this_filename = os.path.split(os.path.abspath(__file__))
    filename = get_filename_template(len(graphics))
    templ_fl = join(this_dir, filename)
    file_ = open(templ_fl, "r")
    filecode = file_.read()
    filetext = Template(filecode).safe_substitute(caption=caption,
                                                  imagelabel=imagelabel)
    filetext = substitute_plots(graphics, filetext)

    return filetext


def tmpdist_tables(info):
    return ''


###############################################################################
############################## Auxiliar functions #############################
###############################################################################
def get_filename_template(n):
    """Return the template file for plotting."""
    if n == 1:
        filename = '../data/templates/tex/image.txt'
    elif n == 2:
        filename = '../data/templates/tex/image2.txt'
    elif n == 4:
        filename = '../data/templates/tex/image4.txt'
    return filename


def substitute_plots(graphics, filetext):
    """Substitute the plot directions into the latex text."""
    filetext = Template(filetext).safe_substitute(graphics1=graphics[0])
    if len(graphics) > 1:
        filetext = Template(filetext).safe_substitute(graphics2=graphics[1])
    elif len(graphics) > 2:
        filetext = Template(filetext).safe_substitute(graphics3=graphics[2])
        filetext = Template(filetext).safe_substitute(graphics4=graphics[3])
    return filetext


def plot_saving(info, study_info):
    """This function save the plots stored in the stats info and return its
    path directions and additional information stored in stats info.
    """
    varname = info['variables_name'].replace(".", "_")
    varname = varname.replace("-", "_")
    fig = info['plots']
    if type(fig) != list:
        fname = '%s.png' % (varname+'_01')
        fname = fname.replace(" ", "")
        fig.savefig(join(study_info['path']+'/Plots/', fname))
        graphics = ['Plots/'+fname]
    elif type(fig) == list:
        graphics = []
        for i in range(len(fig)):
            fname = '%s.png' % (varname+'_0'+str(i+1))
            fname = fname.replace(" ", "")
            fig[i].savefig(join(study_info['path']+'/Plots/', fname))
            graphics.append('Plots/'+fname)
    caption = 'Plot of the distribution of %s' % info['variables_name']
    imagelabel = varname+'_01'

    return graphics, caption, imagelabel
