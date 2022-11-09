# %% IMPORTS
# Built-in imports
import abc
import logging as log

# Package imports
import numpy as np
import nifty8 as ift

# IMAGINE imports
from imagine.tools import BaseClass, req_attr

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

    def __init__(self, grid, parameter_names, unit, internal_shape, call_by_method=False):
        log.debug('@ model::__init__')

        if not call_by_method:
            raise KeyError('This is is an abstract class, which should not be instantiated by the user')

        # Call super constructor
        super().__init__()

        if internal_shape is not None and type(internal_shape) != tuple:
            raise TypeError('Internal shape must be tuple of ints')
        self.internal_shape = internal_shape
        if type(parameter_names) != list:
            raise TypeError('parameter_names must be list of strings')
        self._parameter_names = parameter_names
        # missing unit check
        self._unit = unit
        self.grid = grid

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

    def __add__(self, ModelToAdd):
        # replace this with a more elaborate type checker which preserves the type of the models?
        if not self.unit == ModelToAdd.unit:
            raise TypeError()
        if not self.data_shape == ModelToAdd.data_shape:
            raise TypeError()

        if type(self) == type(ModelToAdd):
            m = type(self)(self.grid)
        else:
            parameter_names = ModelToAdd.parameter_names + [p for p in self.parameter_names if p not in ModelToAdd.parameter_names]
            CommonBase = self.return_first_common_base(self, ModelToAdd)
            print('__add__ ', CommonBase)
            m = CommonBase(self.grid, parameter_names)

        def _new_compute_model(parameters):
            return self.compute_model(parameters) + ModelToAdd.compute_model(parameters)

        setattr(m, 'compute_model', _new_compute_model)
        return m

    def __mul__(self, ModelToMultiply):
        # replace this with a more elaborate type checker which preserves the type of the models?
        if not self.data_shape == ModelToMultiply.data_shape:
            raise TypeError()
        parameter_names = ModelToMultiply.parameter_names + [p for p in self.parameter_names if p not in ModelToMultiply.parameter_names]
        m = Model(self.grid,  parameter_names, self.unit*ModelToMultiply.unit, self.internal_shape, True)

        def _new_compute_model(parameters):
            return self.compute_model(parameters)*ModelToMultiply.compute_model(parameters)

        setattr(m, 'compute_model', _new_compute_model)
        return m

    @property
    def data_shape(self):
        if self.internal_shape is None:
            return (*self.grid.shape, )
        else:
            return(*self.grid.shape, *self.internal_shape)

    @property
    def parameter_names(self):
        """Parameters of the model"""
        return self._parameter_names

    @property
    def unit(self):
        """Physical unit of the model"""
        return(self._unit)

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


class NiftyModel(ift.Operator):
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

    def __init__(self, grid, parameter_names, unit, internal_shape, call_by_method=False):
        log.debug('@ model::__init__')

        if not call_by_method:
            raise KeyError('This is is an abstract class, which should not be instantiated by the user')

        # Call super constructor
        super().__init__()

        if internal_shape is not None and type(internal_shape) != tuple:
            raise TypeError('Internal shape must be tuple of ints')
        self.internal_shape = internal_shape
        if type(parameter_names) != list:
            raise TypeError('parameter_names must be list of strings')
        self._parameter_names = parameter_names
        # missing unit check
        self._unit = unit
        self.grid = grid

    @property
    def data_shape(self):
        if self.internal_shape is None:
            return (*self.grid.shape, )
        else:
            return(*self.grid.shape, *self.internal_shape)

    @property
    def parameter_names(self):
        """Parameters of the model"""
        return self._parameter_names

    @property
    def unit(self):
        """Physical unit of the model"""
        return(self._unit)

    def compute_model(self, parameters):
        if not isinstance(parameters, ift.MultiField):
            parameters = ift.MultiField.from_dict(parameters)

        raise NotImplementedError
