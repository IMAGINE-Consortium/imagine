import logging as log
from imagine.tools.icy_decorator import icy
import numpy as np

@icy
class GeneralField(object):
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
    grid : imagine.fields.grid.BaseGrid or None
        Instance of :py:class:`imagine.fields.grid.BaseGrid` containing a
        3D grid where the field is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    ensemble_size : int
        Number of realisations in field ensemble
    ensemble_seeds : list
        Random seeds for generating random field realisations
    """
    def __init__(self, grid=None, parameters=dict(), ensemble_size=None,
                 ensemble_seeds=None, dependencies={}):
        log.debug('@ field::__init__')
        
        self.grid = grid
        self.parameters = parameters
        # For convenience, when ensemble info is not available,
        # assumes an ensemble of 1
        if (ensemble_size is None) and (ensemble_seeds is None):
            ensemble_size = 1
        self.ensemble_size = ensemble_size
        self.ensemble_seeds = ensemble_seeds
        # Placeholders
        self._data = None
        self._deterministic_data = None
        self.dependencies = dependencies

    @property
    def field_type(self):
        """Description the field"""
        raise NotImplementedError

    @property
    def dependencies_list(self):
        """Dependencies on other fields"""
        return []
        
    @property
    def field_name(self):
        """
        Should be overriden with the name of the field
        """
        raise NotImplementedError

    @property
    def stochastic_field(self):
        """
        Should be overriden with value True if the field is stochastic
        or False if the field is deterministic (i.e. the output depends only
        on the parameter values and not on the seed value).
        """
        raise NotImplementedError

    @property
    def field_units(self):
        """Physical units of the field"""
        raise NotImplementedError

    @property
    def data_description(self):
        """Summary of what is in each axis of the data array"""
        raise NotImplementedError

    @property
    def data_shape(self):
        """Shape of the field data array"""
        raise NotImplementedError

    def get_field(self, seed):
        """
        This should be overridden with a derived class. It must return an array
        with dimensions compatible with the associated `field_type`.
        See :doc:`documentation <components>`.

        Parameters
        ----------
        seed : int
            If the field is stochastic, this argument allows setting the random
            number generator seed accordingly.
        """
        raise NotImplementedError

    def _get_data(self, iseed):
        # Equivalent to the user-facing data property, but does not evaluate
        # and cache the whole ensemble at once

        if self.stochastic_field:
            # Computes stochastic field
            seed = self.ensemble_seeds[iseed]
            field = self.get_field(seed)
            self._check_realisation(field)
        elif self._deterministic_data is None:
            # Computes and caches deterministic field
            field = self.get_field(None)
            self._check_realisation(field)
            self._deterministic_data = field
        else:
            # Uses the cache of deterministic field
            field = self._deterministic_data

        return field

    @property
    def data(self):
        """
        Field data computed by this class with dimensions compatible with
        the associated `field_type`.

        The data is returned as a list, where each element corresponding to
        a single realisation. If the field is *not* stochastic, all the list
        elements are references to the same object.

        If the field_type is 'dummy', dictionaries containing parameters are
        returned instead of data arrays

        """
        if self._data is None:
            self._check_dependencies()
            
            self._data = [self._get_data(i)
                          for i in range(self.ensemble_size)]

        return self._data

    def _check_dependencies(self):
        for dep in self.dependencies_list:
            if dep not in self.dependencies:
                raise KeyError('Missing field dependency {}'.format(dep))
        
    
    def _check_realisation(self, field):
        # Checks the units
        assert self.field_units.is_equivalent(field.unit), 'Field units should be '+self.field_units
        # Checks the shape
        try:
            assert field.shape == self.data_shape
        except AssertionError:
            print('Incorrect shape, it should be:', self.data_shape)
            print('It is instead:', self._data.shape)
            print('Description:', self.data_description)
            raise

    @property
    def field_checklist(self):
        """Dictionary with all parameter names as keys"""
        return dict()

    @property
    def ensemble_seeds(self):
        return self._ensemble_seeds

    @ensemble_seeds.setter
    def ensemble_seeds(self, ensemble_seeds):
        if ensemble_seeds is None:  # in case no seeds given, choose something
            self._ensemble_seeds = np.random.randint(0, 2**32,
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
            assert (k in self.field_checklist.keys())
        try:
            self._parameters.update(parameters)
            log.debug('update full-set parameters %s' % str(parameters))
        except AttributeError:
            self._parameters = parameters
            log.debug('set full-set parameters %s' % str(parameters))
