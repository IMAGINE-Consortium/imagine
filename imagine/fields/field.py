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
    ensemble_size : int
        Number of realisations in field ensemble
    ensemble_seeds : list
        Random seeds for generating random field realisations
    """

    def __init__(self, grid, *, parameters={}, ensemble_size=None,
                 ensemble_seeds=None, dependencies={}):
        log.debug('@ field::__init__')

        # Call super constructor
        super().__init__()

        self.grid = grid
        self._parameters = {}
        self.parameters = parameters
        # For convenience, when ensemble info is not available,
        # assumes an ensemble of 1
        if ensemble_size is None and ensemble_seeds is None:
            ensemble_size = 1
        self.ensemble_size = ensemble_size
        self.ensemble_seeds = ensemble_seeds
        # Placeholders
        self._deterministic_data = None
        self.dependencies = dependencies

        assert self.type not in self.dependencies_list, 'Field cannot depend on its own field type'

    @property
    @req_attr
    def type(self):
        """Type of the field"""
        return(self.TYPE)

    @property
    @req_attr
    def parameters_list(self):
        """Parameters of the field"""
        return self.PARAMETERS_LIST

    @property
    def dependencies_list(self):
        """Dependencies on other fields"""
        return(getattr(self, 'DEPENDENCIES_LIST', []))

    @property
    @req_attr
    def name(self):
        """
        Name of the field
        """
        return(self.NAME)

    @property
    def stochastic_field(self):
        """
        *True* if the field is stochastic or *False* if the field is
        deterministic (i.e. the output depends only on the parameter values and
        not on the seed value).
        Default value is *False* if subclass does not override
        :attr:`~STOCHASTIC_FIELD`.

        """
        return(getattr(self, 'STOCHASTIC_FIELD', False))

    @property
    @req_attr
    def units(self):
        """Physical units of the field"""
        return(self.UNITS)

    @abc.abstractproperty
    def data_description(self):
        """Summary of what is in each axis of the data array"""
        raise NotImplementedError

    @abc.abstractproperty
    def data_shape(self):
        """Shape of the field data array"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_field(self, seed):
        """
        This should be overridden with a derived class. It must return an array
        with dimensions compatible with the associated `field_type`.
        See :doc:`documentation <components>`.

        Should not be used directly (use :py:meth:`get_data` instead).

        Parameters
        ----------
        seed : int
            If the field is stochastic, this argument allows setting the random
            number generator seed accordingly.
        """
        raise NotImplementedError

    def get_data(self, i_realization=0, dependencies={}):
        """
        Evaluates the physical field defined by this class.

        Parameters
        ----------
        i_realization : int
            If the field is stochastic, this indexes the realization generated.
            Default value: 0 (i.e. the first realization).
        dependencies : dict
            If the :py:data:`dependencies_list` is non-empty, a dictionary containing
            the requested dependencies must be provided.

        Returns
        -------
        field : astropy.units.quantity.Quantity
            Array of shape :py:data:`data_shape` whose contents are described by
            :py:data:`data_description` in units :py:data:`field_units`.
        """
        if self.stochastic_field:
            assert i_realization < self.ensemble_size
            # Checks and updates dependencies
            self._update_dependencies(dependencies)
            # Computes stochastic field
            seed = self.ensemble_seeds[i_realization]
            field = self.compute_field(seed)
            self._check_realisation(field)
        elif self._deterministic_data is None:
            # Checks and updates dependencies
            self._update_dependencies(dependencies)
            # Computes and caches deterministic field
            field = self.compute_field(None)
            self._check_realisation(field)
            self._deterministic_data = field
        else:
            # Uses the cache of deterministic field
            field = self._deterministic_data

        return field

    def _update_dependencies(self, dependencies):
        for dep in self.dependencies_list:
            if dep not in dependencies:
                raise KeyError('Missing field dependency {}'.format(dep))
        self.dependencies = dependencies

    def _check_realisation(self, field):
        # Checks the units
        assert self.units.is_equivalent(field.unit), 'Field units should be '+self.units
        # Checks the shape
        try:
            assert field.shape == self.data_shape
        except AssertionError:
            print('Incorrect shape, it should be:', self.data_shape)
            print('It is instead:', field.shape)
            print('Description:', self.data_description)
            raise

    @property
    def ensemble_seeds(self):
        return self._ensemble_seeds

    @ensemble_seeds.setter
    def ensemble_seeds(self, ensemble_seeds):
        if ensemble_seeds is None:  # in case no seeds given, choose something
            self._ensemble_seeds = np.random.randint(0, 2**31-1,
                                                     self.ensemble_size)
        else:
            if self.ensemble_size is None:
                self.ensemble_size = len(ensemble_seeds)
            assert len(ensemble_seeds) == self.ensemble_size
            self._ensemble_seeds = ensemble_seeds

    @property
    def parameters(self):
        """
        Dictionary containing parameters used for this field."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        for k in parameters:
            assert (k in self.parameters_list)
        self._parameters.update(parameters)
        log.debug('update full-set parameters %s' % (parameters))
