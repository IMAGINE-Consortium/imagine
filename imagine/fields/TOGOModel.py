# %% IMPORTS
# Built-in imports
import abc
import logging as log

# Package imports
import numpy as np

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

    def __init__(self, grid, parameter_names, unit, internal_shape, call_by_subclass=False):
        log.debug('@ model::__init__')

        if not call_by_subclass:
            raise KeyError('This is is an abstract class, wuch should not be instantiated by the user')

        # Call super constructor
        super().__init__()

        if type(internal_shape) != tuple:
            raise TypeError('Internal shape must be tuple of ints')
        self.internal_shape = internal_shape
        if type(parameter_names) != tuple:
            raise TypeError('parameter_names must be lsit of strings')
        self.parameter_names = parameter_names
        # missing unit check
        self.unit = unit
        self.grid = grid

    def __add__(self, ModelToAdd):
        if not self.unit == ModelToAdd.unit:
            raise TypeError()
        if not self.data_shape == ModelToAdd.data_shape:
            raise TypeError()
        m = Model(self.grid, self.units, self.parameter_names, call_by_subclass=True)

        def _new_compute_field(m, parameters):
            return self.compute_model(parameters) + ModelToAdd.compute_model(parameters)

        setattr(m, 'compute_model', _new_compute_field)
        return m

    @property
    def data_shape(self):
        return(*self.grid.shape, *self.internal_shape)

    @property
    def parameter_names(self):
        """Parameters of the model"""
        return self.parameter_names

    @property
    def units(self):
        """Physical units of the model"""
        return(self.units)

    @abc.abstractmethod
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
