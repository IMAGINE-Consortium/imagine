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

    def __init__(self, grid, parameter_def_dict, unit, output_shape, call_by_method=False):
        # Call super constructor
        input_param_space = grid if parameter_def_dict is None else (grid, parameter_def_dict)
        super().__init__(input_param_space, output_shape)

        self._input_grid = grid
        self._data_space = output_shape
        self._parameter_names = parameter_def_dict
