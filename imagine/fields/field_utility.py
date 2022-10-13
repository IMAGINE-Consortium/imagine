"""
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
            raise ValueError('Fields can only be added if having the same units')
        for pn in summand_1.parameter_names:
            if pn in summand_2.parameter_names:
                raise KeyError('The two summands may not have the same parameter names')
        self.summand_1 = summand_1
        self.summand_2 = summand_2
        self.UNITS = summand_1.UNITS
        self.TYPE  = summand_1.TYPE
        self.PARAMETER_NAMES = summand_1.parameter_names+summand_2.parameter_names
        self.NAME = summand_1.NAME + '_plus_' + summand_2.NAME

        super().__init__(grid,  parameters={}, ensemble_size=None,
                    ensemble_seeds=None, dependencies={})

    @property
    def data_description(self):
        return(self.summand_1.data_description())

    @property
    def data_shape(self):
        return(self.summand_1.data_shape)

    def compute_field(self, seed):
        return self.summand_1.compute_field(seed) + self.summand_2.compute_field(seed)


class ArrayField(Field):



    def __init__(self, field, parameters={}, ensemble_size=None,
                 ensemble_seeds=None, dependencies={}):

        if not isinstance(field, Field):
            raise TypeError('Field needs to be an instance of an IMAGINE field class')
        self.array = field.get_data()
        self.UNITS = field.UNITS
        self.TYPE  = field.TYPE
        self.NAME  = field.NAME+'_array'
        self.PARAMETER_NAMES = [field.NAME+'_scale']
        self._data_description = field.data_description

        super().__init__(field.grid, parameters={}, ensemble_size=None,
                         ensemble_seeds=None, dependencies={})

    @property
    def data_description(self):
        return(self._data_description())

    @property
    def data_shape(self):
        return(self.array.shape())

    def compute_field(self, seed):
        return self.parameters[self.PARAMETER_NAMES[0]]*self.array

