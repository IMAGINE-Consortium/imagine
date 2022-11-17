# %% IMPORTS
# Built-in imports
import abc
import logging as log

# Package imports
import numpy as np
import nifty8 as ift

# IMAGINE imports
from imagine.tools import BaseClass
from ..grid.TOGOGrid import ParameterSpace

# All declaration
__all__ = ['Model']


# %% CLASS DEFINITIONS
# Define abstract base class for creating models in IMAGINE
class Model(BaseClass, metaclass=abc.ABCMeta):
    """
    This is the base class which can be used to include a completely new model
    in the IMAGINE pipeline. Base classes for specific physical quantites
    (e.g. magnetic models) are already available in the module
    :mod:`imagine.models.basic_models`.
    Thus, before subclassing `Model`, check whether a more specialized
    subclass is available.

    For more details check the :ref:`components:Models` section in the
    documentation.

    Parameters
    ----------
    grid : imagine.models.grid.BaseGrid
        Instance of :py:class:`imagine.models.grid.BaseGrid` containing a
        n-dimensional grid where the model is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    """

    def __init__(self, input_param_space, output_param_space, call_by_method=False):
        log.debug('@ model::__init__')

        if not isinstance(input_param_space, ParameterSpace):
            raise TypeError
        self._input_param_space = input_param_space

        if not isinstance(output_param_space, ParameterSpace):
            raise TypeError
        self._output_param_space = output_param_space

        self._unit = None

        if not call_by_method:
            raise KeyError('This is is an abstract class, which should not be instantiated by the user')

        # Call super constructor
        super().__init__()

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, other_unit):
        if self.unit is not None:
            raise ValueError('Imagine.Model.unit already set')

        self._unit = other_unit

    def unit_adaptor(self, other_unit):
        return self.unit

    @property
    def input_param_space(self):
        return self._input_param_space

    @property
    def output_param_space(self):
        return self._output_param_space

    def __add__(self, ModelToAdd):
        raise NotImplementedError

    def __mul__(self, ModelToMultiply):
        raise NotImplementedError

    def __matmul__(self, ModelToConnect):
        if not isinstance(ModelToConnect, Model):
            raise TypeError()
        if not isinstance(ModelToConnect._output_param_space,  type(self._input_param_space)):
            raise TypeError('Imagine.Model: Only Models with fitting output and input can be connected, you tried {} (input) and {} (output)'.format(type(self._input_param_space), type(ModelToConnect._output_param_space)))

        m = Model(ModelToConnect._input_param_space, self._output_param_space, call_by_method=True)

        m.unit = self.unit_adaptor(ModelToConnect.unit)

        def _new_compute_model(parameters):
            return self.compute_model(ModelToConnect.compute_model(parameters))
        setattr(m, 'compute_model', _new_compute_model)
        return m

    def __call__(self, parameters):
        return self.compute_model(parameters)

    def compute_model(self, parameters):
        """
        This should be overridden with a derived class. It must return an array
        with dimensions compatible with the associated `model_type`.
        See :doc:`documentation <components>`.

        Should not be used directly (use :py:meth:`__call___` instead).

        Parameters
        ----------
        parameters : dictionary
        """
        raise NotImplementedError


    @staticmethod
    def return_first_common_base(o1, o2):
        if not isinstance(o1, Model):
            raise TypeError('Model.return_first_common_base: object o1 is not a (derived) Model instance')
        if not isinstance(o2, Model):
            raise TypeError('Model.return_first_common_base: object o2 is not a (derived) Model instance')
        t1 = type(o1)
        t2 = type(o2)

        base_1 = list(type(o1).__mro__)
        base_2 = list(type(o2).__mro__)
        if len(base_1) <= 2:
            raise TypeError('Model.return_first_common_base: object o1 must be derived Model instance')
        if len(base_2) <= 2:
            raise TypeError('Model.return_first_common_base: object o2 must be derived Model instance')

        if t1 == t2:
            return t1
        base_1.reverse()
        base_2.reverse()
        if type(o1) in base_2:
            return type(o1)
        if type(o2) in base_1:
            return type(o2)
        tt = Model
        for i, bo1 in enumerate(base_1):
            if base_2[i] != bo1:
                return tt
            else:
                tt = bo1
