__version__ = '2.3.dev0'

from .base import PBinaryVariable, PMultiVariable
from .factor_graph import PFactorGraph
from .simple_inference import simple_grid, general_graph
from .simple_constrained_inference import general_constrained_graph
