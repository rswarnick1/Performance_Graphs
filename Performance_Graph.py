import numpy as np
import scipy.stats as stats
import scipy as sp
import matplotlib.pyplot as plt

from pybnesian import DirectedGraph
from TaxonomyNodeClass import TaxonomyNode


class PerformanceGraph(DirectedGraph):
    
    def __init__(self, root_node: TaxonomyNode, nodes: list[TaxonomyNode] = None, arcs = None):
        super().__init__()