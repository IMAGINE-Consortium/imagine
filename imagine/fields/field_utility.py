r"""
This module contains utlity classes that can be used to combine fields
in IMAGINE.

A brief summary of the module:

* :py:class:FieldAdder` — to add fields with the same units and the same grid`

"""

# %% IMPORTS
# Built-in imports
import abc

# Package imports
import astropy.units as u

# IMAGINE imports
from imagine.fields import Field
from imagine.tools import req_attr

# All declaration
__all__ = ['FieldAdder']


class FieldAdder(Field):
    def __init__(self, grid, summand_1, summand_2, parameters={}, ensemble_size=None,
                 ensemble_seeds=None, dependencies={}):

        if summand_1.grid != summand_2.grid:
            raise ValueError('Fields can only be added if defined on the same grid')
        if summand_1.TYPE != summand_2.TYPE:
            raise ValueError('Fields can only be added if having the same type')

        for pn in summand_1.parameter_names:
            if pn in summand_2.parameter_names:
                raise KeyError('The two summands may not have the same parameter names')

        super().__init__(self, grid, parameters={}, ensemble_size=None,
                         ensemble_seeds=None, dependencies={})

        self.summand_1 = summand_1
        self.summand_2 = summand_2
        self.TYPE = summand_1.TYPE
        self.UNITS = summand_1.UNITS
        self.NAME = summand_1.NAME + '_plus_' + summand_2.NAME
        self.PARAMETER_NAMES = summand_1.parameter_names + summand_2.parameter_names
    #

    @property
    def data_description(self):
        return self.summand_1.data_description()

    @property
    def data_shape(self):
        return self.summand_1.data_shape()

    def compute_field(self, seed):
        return self.summand_1.compute_field(seed) + self.summand_2.compute_field(seed)


class ArrayAdder(Field):
    def __init__(self, grid, field, array, parameters={}, ensemble_size=None,
                 ensemble_seeds=None, dependencies={}):

        super().__init__(self, grid, parameters={}, ensemble_size=None,
                         ensemble_seeds=None, dependencies={})
        if field.grid.shape != array.shape:
            raise ValueError('Array can only be added if having the same shape as the grid of the field')
        self.field = field
        self.array = array

    def compute_field(self, seed):
        return self.summand_1.compute_field(seed) + self.array
