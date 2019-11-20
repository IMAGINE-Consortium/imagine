"""
GeneralFieldFactory is designed for generating
ensemble of field configuration DIRECTLY
and/or
handle of field to be conducted by simulators

through member function .generate,
factory object take a given set of variable values
(can be at any chain point in bayesian analysis)
and translate it into physical parameter value and
return a field object with current parameter set

attributes:
    
.name
    factory name, useful as factory id
    
.type
    specify what field the factory produce, 'scalar', 'spinor', 'vector', 'tensor' ...
    
.boxsize
    physical size of simulation box, by default the box is 3D cartesian
    
.resolution
    how many bins on each direction of simulation box

.default_parameters
    dictionary storing parameter name as entry, default parameter value as content
    
.default_variables
    return a dict of (logic) variables wrt default_parameters
    
.active_parameters
    tuple of parameter names which can vary, not necessary to cover all default parameters

.parameter_ranges
    dictionary storing varying range of all default parameters
    
._map_variables_to_parameters
    given variable dict, translate to parameter dict with parameter value mapped from variable value

.generate
    takes active variable dict, ensemble size and random seed value defined in Pipeline
    it translates active variable value to parameter value and update a copy of default parameter dict
    and send it to field class, which will hand in to simulator
"""

import numpy as np
from copy import deepcopy
import logging as log
from imagine.fields.field import GeneralField
from imagine.tools.carrier_mapper import unity_mapper
from imagine.tools.icy_decorator import icy


@icy
class GeneralFieldFactory(object):

    def __init__(self, boxsize=None, resolution=None):
        """
        GeneralFieldFactory class initialization function
        
        boxsize
            list/tuple of float, physical size of simulation box (3D Cartesian frame)
            
        resolution
            list/tuple of int, discretization size in corresponding dimension
        """
        log.debug('@ field_factory::__init__')
        self.field_type = 'scalar'
        self.name = 'general'
        self.field_class = GeneralField
        self.boxsize = boxsize
        self.resolution = resolution
        self.default_parameters = dict()
        # the following two must after .default_parameters initialisation
        self.active_parameters = tuple()
        self.parameter_ranges = dict()
        
    @property
    def field_type(self):
        return self._field_type

    @field_type.setter
    def field_type(self, field_type):
        assert isinstance(field_type, str)
        self._field_type = field_type

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str)
        self._name = name
    
    @property
    def field_class(self):
        return self._field_class

    @field_class.setter
    def field_class(self, field_class):
        self._field_class = field_class
    
    @property
    def boxsize(self):
        return self._boxsize

    @boxsize.setter
    def boxsize(self, boxsize):
        if boxsize is None:
            self._boxsize = None
        else:
            assert isinstance(boxsize, (list, tuple))
            assert (len(boxsize) == 3)
            # force size in float
            self._boxsize = tuple(np.array(boxsize, dtype=np.float))

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        if resolution is None:
            self._resolution = None
        else:
            assert isinstance(resolution, (list, tuple))
            assert (len(resolution) == 3)
            # force resolutioin in int
            self._resolution = tuple(np.array(resolution, dtype=np.int))

    @property
    def default_parameters(self):
        return self._default_parameters

    @default_parameters.setter
    def default_parameters(self, new_defaults):
        assert isinstance(new_defaults, dict)
        try:
            self._default_parameters.update(new_defaults)
            log.debug('update default parameters %s' % str(new_defaults))
        except AttributeError:
            self._default_parameters = new_defaults
            log.debug('set default parameters %s' % str(new_defaults))

    @property
    def active_parameters(self):
        return self._active_parameters

    @active_parameters.setter
    def active_parameters(self, active_parameters):
        assert isinstance(active_parameters, (list, tuple))
        # check if input is inside factory's parameter pool
        for av in active_parameters:
            assert (av in self.default_parameters)
        self._active_parameters = tuple(active_parameters)
        log.debug('set active parameters %s' % str(active_parameters))

    @property
    def parameter_ranges(self):
        return self._parameter_ranges

    @parameter_ranges.setter
    def parameter_ranges(self, new_ranges):
        """
        parameters
        ----------
        
        new_ranges
            python dict in form {'parameter-name': (min, max)}
        """
        assert isinstance(new_ranges, dict)
        for k, v in new_ranges.items():
            # check if k is inside default
            assert (k in self.default_parameters.keys())
            assert isinstance(v, (list, tuple))
            assert (len(v) == 2)
        try:
            self._parameter_ranges.update(new_ranges)
            log.debug('update parameter ranges %s' % str(new_ranges))
        except AttributeError:
            self._parameter_ranges = new_ranges
            log.debug('set parameter ranges %s' % str(new_ranges))

    @property
    def default_variables(self):
        """
        translate default parameter into default (logic) variable
        notice that all variables range is always fixed as [0,1]
        
        return
        ------
        default variable dict
        """
        log.debug('@ field_factory::default_variables')
        tmp = dict()
        for par, def_val in self.default_parameters.items():
            low, high = self.parameter_ranges[par]
            tmp[par] = float(def_val - low)/float(high - low)
        return tmp

    def _map_variables_to_parameters(self, variables):
        """
        convert Bayesian sampling variables into model parameters
        
        parameters
        ----------
        
        variables
            python dict in form {'parameter-name', logic-value}
        
        return
        ------
        python dict in form {'parameter-name', physical-value}
        """
        log.debug('@ field_factory::_map_variables_to_parameters')
        assert isinstance(variables, dict)
        parameter_dict = dict()
        for variable_name in variables:
            # variable_name must have been registered in .default_parameters
            # and, also being active
            assert (variable_name in self.default_parameters and variable_name in self.active_parameters)
            low, high = self.parameter_ranges[variable_name]
            # unity_mapper defined in imainge.tools.carrier_mapper
            mapped_variable = unity_mapper(variables[variable_name], low, high)
            parameter_dict[variable_name] = mapped_variable
        return parameter_dict

    def generate(self, variables=dict(), ensemble_size=1, ensemble_seeds=None):
        """
        generate GeneralFIeld object hosting the given set of variables
        
        parameters
        ----------
        
        variables
            a dict of variables with name and value
        
        ensemble_size
            number of instances in a field ensemble
        
        ensemble_seeds
            seeds for generating random numbers 
            in realising instances in field ensemble
            if ensemble_seeds is None, 
            field_class initialization will take all seed as 0
            
        return
        ------
        a GeneralField object
        """
        log.debug('@ field_factory::generate')
        # map variable value to parameter value
        # in mapping, variable name will be checked in default_parameters
        mapped_variables = self._map_variables_to_parameters(variables)
        # copy default parameters and update wrt argument
        work_parameters = deepcopy(self.default_parameters)
        # update is safe
        work_parameters.update(mapped_variables)
        # generate fields
        result_field = self.field_class(parameters=work_parameters,
                                        ensemble_size=ensemble_size,
                                        ensemble_seeds=ensemble_seeds)
        log.debug('generated field with work-parameters %s' % work_parameters)
        return result_field

    @staticmethod
    def _interval(mean, sigma, n):
        return float(mean - n * sigma), float(mean + n * sigma)

    @staticmethod
    def _positive_interval(mean, sigma, n):
        return max(float(0), float(mean - n * sigma)), float(mean + n * sigma)
