
"""
Module which contains mains functions and abstract classes used in the
Supermodule Models.
"""

from model_utils import filter_with_random_nets
from Mscthesis.IO.model_report import model_report


class Model():
    """Abstract class for all the models with common utilities."""
    def filter_with_random_nets(self, nets, p_thr):
        "Filter non-significant weiths."
        net, random_nets = nets[:, :, 0], nets[:, :, 1:]
        net = filter_with_random_nets(net, random_nets, p_thr)
        return net

    def to_report(self, net, sectors, dirname, reportname):
        "Generate a folder in which save the report data exported."
        fig1, fig2 = model_report(net, sectors, dirname, reportname)
        return fig1, fig2

