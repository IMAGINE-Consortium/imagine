import logging as log
from imagine.tools.icy_decorator import icy

@icy
class GeneralField(object):
    """
    This is the base class which can be used to include a completely new field
    in the IMAGINE pipeline. Base classes for specific physical quantites
    (e.g. magnetic fields) are already available in the module
    :mod:`imagine.fields.basic_fields`.
    Thus, before subclassing `GeneralField`, check whether a more specialized
    subclass is not available.

    For more details check the :ref:`design_components:Fields` section in the
    documentation.

    Parameters
    ----------
    grid : imagine.fields.BaseGrid or None
        Instance of `imagine.fields.BaseGrid` containing a 3D grid where the field
        is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    ensemble_size : int
        Number of realisations in field ensemble
    ensemble_seeds
        Random seed(s) for generating random field realisations

    Attributes
    ----------
    field_name : str
        Name of the physical field (class attribute)
    """
    field_name = 'unset' # This is actually a class attribute
    def __init__(self, grid=None, parameters=dict(), ensemble_size=1,
                 ensemble_seeds=None):
        log.debug('@ field::__init__')
        self.grid = grid
        self.parameters = parameters
        self.ensemble_size = ensemble_size
        self.ensemble_seeds = ensemble_seeds
        # Placeholders
        self._data = None

    @property
    def field_type(self):
        raise NotImplemented
    @property
    def field_units(self):
        raise NotImplemented
    @property
    def data_description(self):
        raise NotImplemented
    @property
    def data_shape(self):
        raise NotImplemented

    def get_field(self):
        """
        This should be overridden with a derived class. It must return an array
        with dimensions compatible with the associated `field_type`.
        See :doc:`documentation <design_components>`.
        """
        raise NotImplementedError

    @property
    def data(self):
        """
        Field data computed by this class with dimensions compatible with
        the associated `field_type`. See :doc:`documentation <design_components>`.
        """
        if self._data is None:
            self._data = self.get_field()
            assert self.field_units.is_equivalent(self._data.unit), 'Field units should be '+self.field_units
        
        try:
            assert self._data.shape == self.data_shape
        except AssertionError:
            print('Incorrect shape, it should be:', self.data_shape)
            print('It is instead:', self._data.shape)
            print('Description:', self.data_description)
            raise

        return self._data

    @property
    def field_checklist(self):
        return dict()

    @property
    def ensemble_size(self):
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, ensemble_size):
        assert (ensemble_size > 0)
        self._ensemble_size = round(ensemble_size)

    @property
    def ensemble_seeds(self):
        return self._ensemble_seeds

    @ensemble_seeds.setter
    def ensemble_seeds(self, ensemble_seeds):
        if ensemble_seeds is None:  # in case no seeds given, all 0
            self._ensemble_seeds = [int(0)]*self._ensemble_size
        else:
            assert (len(ensemble_seeds) == self._ensemble_size)
            self._ensemble_seeds = ensemble_seeds

    @property
    def parameters(self):
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

    def report_parameters(self, realization_id=int(0)):
        """
        return parameters with random seed associated to realization id
        """
        log.debug('@ field::report_parameters')
        # if checklist has 'random_seed' entry
        if 'random_seed' in self.field_checklist.keys():
            self._parameters.update({'random_seed': self._ensemble_seeds[realization_id]})
        return self._parameters
