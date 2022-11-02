# %% IMPORTS
# Built-in imports
import abc
import logging as log

# Package imports
import numpy as np

# IMAGINE imports
from imagine.tools import BaseClass, req_attr

# All declaration
__all__ = ['Field']


# %% CLASS DEFINITIONS
# Define abstract base class for creating fields in IMAGINE
class Field(BaseClass, metaclass=abc.ABCMeta):
    """
    This is the base class which can be used to include a completely new field
    in the IMAGINE pipeline. Base classes for specific physical quantites
    (e.g. magnetic fields) are already available in the module
    :mod:`imagine.fields.basic_fields`.
    Thus, before subclassing `GeneralField`, check whether a more specialized
    subclass is not available.

    For more details check the :ref:`components:Fields` section in the
    documentation.

    Parameters
    ----------
    grid : imagine.fields.grid.BaseGrid
        Instance of :py:class:`imagine.fields.grid.BaseGrid` containing a
        3D grid where the field is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    """

    UNITS = None
    PARAMETER_NAMES = []

    def __init__(self, grid):
        log.debug('@ field::__init__')

        # Call super constructor
        super().__init__()

        self.grid = grid

    def __add__(self, FieldToAdd):
        if not self.UNITS == FieldToAdd.UNITS:
            raise
        return


    @property
    @req_attr
    def parameter_names(self):
        """Parameters of the field"""
        return self.PARAMETER_NAMES

    @property
    @req_attr
    def units(self):
        """Physical units of the field"""
        return(self.UNITS)

    @abc.abstractproperty
    def data_shape(self):
        """Shape of the field data array"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_field(self, parameters):
        """
        This should be overridden with a derived class. It must return an array
        with dimensions compatible with the associated `field_type`.
        See :doc:`documentation <components>`.

        Should not be used directly (use :py:meth:`__call___` instead).

        Parameters
        ----------
        parameters : dictionary
        """
        raise NotImplementedError
