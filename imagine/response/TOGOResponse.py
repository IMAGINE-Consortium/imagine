# %% IMPORTS
# Built-in imports
import abc

# Package imports
import numpy as np

# IMAGINE imports
from imagine.observables import Measurements, Simulations
from imagine.tools import BaseClass, req_attr

from imagine.fields.TOGOModel import Model

# All declaration
__all__ = ['Response']


# %% CLASS DEFINITIONS
class Response(Model):

    def __init__(self, grid,  parameter_def_dict, output_space, call_by_method=False):
        # Call super constructor
        input_param_space = grid if parameter_def_dict is None else (grid, parameter_def_dict)
        super().__init__(input_param_space, output_space, call_by_method)

        self._parameter_names = parameter_def_dict
        self._unit = grid.box[0].unit

    def unit_adaptor(self, other_unit):
        return self.unit*other_unit
