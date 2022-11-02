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
from imagine.fields import MagneticField
from imagine.tools import req_attr

# All declaration
__all__ = ['MagneticFieldAdder', 'ArrayMagneticField']


class MagneticFieldAdder(MagneticField):
    UNITS = MagneticField.UNITS
    TYPE = MagneticField.TYPE
    NAME = "MagneticFieldAdder"

    def __init__(self, grid, field_1, field_2, parameters={}, ensemble_size=None,
                 ensemble_seeds=None, dependencies={}):


        # Unpack and assign parameters
        param_field1 = {key:parameters[key] for key in field_1.PARAMETER_NAMES}
        param_field2 = {key:parameters[key] for key in field_2.PARAMETER_NAMES}
        summand_1 = field_1(grid=grid, parameters=param_field1)
        summand_2 = field_2(grid=grid, parameters=param_field2)

        if summand_1.grid != summand_2.grid:
            raise ValueError('Fields can only be added if defined on the same grid')
        if summand_1.TYPE != summand_2.TYPE:
            raise ValueError('Fields can only be added if having the same units')
        for pn in summand_1.parameter_names:
            if pn in summand_2.parameter_names:
                raise KeyError('The two summands may not have the same parameter names')
        self.summand_1 = summand_1
        self.summand_2 = summand_2

        self.PARAMETER_NAMES = summand_1.parameter_names+summand_2.parameter_names
        
        #print("Initializing MagneticFieldAdder")
        #print("Changing: "+ self.NAME)
        self.NAME = summand_1.NAME + '_plus_' + summand_2.NAME
        #print("New name: "+ self.NAME)

        super().__init__(grid,  parameters=parameters, ensemble_size=ensemble_size,
                         ensemble_seeds=ensemble_seeds, dependencies=dependencies)

    @property
    def data_description(self):
        return(self.summand_1.data_description())

    @property
    def data_shape(self):
        return(self.summand_1.data_shape)

    def compute_field(self, seed):
        return self.summand_1.compute_field(seed) + self.summand_2.compute_field(seed)


"""
class MagneticFieldAdder(MagneticField):
    UNITS = MagneticField.UNITS
    TYPE = MagneticField.TYPE
    NAME = "MagneticFieldAdder"

    def __init__(self, grid, summand_1, summand_2, parameters={}, ensemble_size=None,
                 ensemble_seeds=None, dependencies={}):

        print("MF Adder__init__: ", parameters)
        if summand_1.grid != summand_2.grid:
            raise ValueError('Fields can only be added if defined on the same grid')
        if summand_1.TYPE != summand_2.TYPE:
            raise ValueError('Fields can only be added if having the same units')
        for pn in summand_1.parameter_names:
            if pn in summand_2.parameter_names:
                raise KeyError('The two summands may not have the same parameter names')
        self.summand_1 = summand_1
        self.summand_2 = summand_2

        self.PARAMETER_NAMES = summand_1.parameter_names+summand_2.parameter_names
        
        #print("Initializing MagneticFieldAdder")
        #print("Changing: "+ self.NAME)
        self.NAME = summand_1.NAME + '_plus_' + summand_2.NAME
        #print("New name: "+ self.NAME)

        super().__init__(grid,  parameters={}, ensemble_size=None,
                         ensemble_seeds=None, dependencies={})

    @property
    def data_description(self):
        return(self.summand_1.data_description())

    @property
    def data_shape(self):
        return(self.summand_1.data_shape)

    def compute_field(self, seed):
        print("MagneticField is computed! b1, b2: ", self.summand_1.parameters['b_arm_1'],self.summand_1.parameters['b_arm_2'])
        return self.summand_1.compute_field(seed) + self.summand_2.compute_field(seed)
"""

class ArrayMagneticField(MagneticField):
    UNITS = MagneticField.UNITS
    TYPE = MagneticField.TYPE
    NAME = "ArrayMagneticField"
    PARAMETER_NAMES = ['array_field','array_field_amplitude']

    @property
    def data_shape(self):
        return(self.array_field.shape)

    def compute_field(self, seed):
        return self.parameters['array_field_amplitude']*self.parameters['array_field']


"""

class ArrayField(Field):

    def __init__(self, array_field, config={}, parameters={}, ensemble_size=None,
                 ensemble_seeds=None, dependencies={}):

	# Possible checks:
		# grid must be used the generate the array
		# config must contain all field attributes
		# type must be a valid imagine base_field type
        self.array_field = array_field
        self.UNITS = config['units']
        self.TYPE  = config['type']
        self.NAME  = config['field_name']
        self.PARAMETER_NAMES = [config['field_name']+'_scale']

        # problem here is that we need the method data_discription from base_fields
        # and we only know what type the field is upon init so need ugly tests
        self._data_description = field.data_description

        super().__init__(config['grid'], parameters={}, ensemble_size=None,
                         ensemble_seeds=None, dependencies={})

    @property
    def data_description(self):
        return(self._data_description())

    @property
    def data_shape(self):
        return(self.array.shape())

    def compute_field(self, seed):
        return self.parameters[self.PARAMETER_NAMES[0]]*self.array_field


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

"""
