# %% IMPORTS
# Built-in imports
import abc
import logging as log

# Package imports
import astropy.units as u

# IMAGINE imports
from imagine.fields.grid import BaseGrid, UniformGrid
from imagine.priors import Prior
from imagine.tools import BaseClass, unity_mapper, req_attr

# All declaration
__all__ = ['FieldFactory','CustomFieldFactory']


# %% CLASS DEFINITIONS
class FieldFactory(BaseClass, metaclass=abc.ABCMeta):
    """
    FieldFactory is designed for generating
    ensemble of field configuration DIRECTLY and/or
    handle of field to be conducted by simulators

    Through calling the factory, the factory object takes
    a given set of variable values
    (can be at any chain point in bayesian analysis)
    and translates it into physical parameter values,
    returning a field object with current parameter set.


    Parameters
    ----------
    boxsize : list/tuple of floats
        The physical size of simulation box (i.e. edges of the box).
    resolution : list/tuple of ints
        The discretization size in corresponding dimension
    grid : imagine.fields.BaseGrid or None
        If present, the supplied instance of `imagine.fields.BaseGrid` is
        used and the arguments `boxsize` and `resolution` are ignored
    field_kwargs : dict
        Any extra keyword arguments that should be used in the field instantiation
    """
    def __init__(self, *, grid=None, boxsize=None, resolution=None,
                 active_parameters=(), field_kwargs={}):
        log.debug('@ field_factory::__init__')

        # Call super constructor
        super().__init__()

        if self.field_type == 'dummy':
            # In dummy fields, we do not use a grid
            self._grid = None
            self._boxsize = None
            self._resolution = None
        else:
            # Uses user defined grid if `grid` is present
            if grid is not None:
                assert isinstance(grid, BaseGrid)
                self._grid = grid
            # Otherwise, assumes a regular Cartesian grid
            # Which is generated when the property is first called
            else:
                self._grid = None
                self._boxsize = boxsize
                self._resolution = resolution
        self.field_kwargs = field_kwargs

        # Placeholders
        self.default_parameters = self.DEFAULT_PARAMETERS
        self.active_parameters = active_parameters
        self.parameter_ranges = {}
        self.priors = self.PRIORS

    def __call__(self, *, variables={}, ensemble_size=None,
                 ensemble_seeds=None):
        """
        Takes an active variable dictionary, an ensemble size and a random
        seed value, translates the active variables to parameter values
        (updating the default parameter dictionary accordingly) and send this
        to an instance of the field class.

        Parameters
        ----------
        variables : dict
            Dictionary of variables with name and value
        ensemble_size : int
            Number of instances in a field ensemble
        ensemble_seeds
            seeds for generating random numbers
            in realising instances in field ensemble
            if ensemble_seeds is None,
            field_class initialization will take all seed as 0

        Returns
        -------
        result_field : imagine.fields.field.Field
            a Field object
        """
        log.debug('@ field_factory::generate')
        # map variable value to parameter value
        # copy default parameters and update wrt argument
        work_parameters = dict(self.default_parameters)
        # update is safe
        work_parameters.update(variables)
        # generate fields
        result_field = self.field_class(grid=self.grid,
                                        parameters=work_parameters,
                                        ensemble_size=ensemble_size,
                                        ensemble_seeds=ensemble_seeds,
                                        **self.field_kwargs)
        log.debug('generated field with work-parameters %s' % work_parameters)
        return result_field

    @property
    @req_attr
    def field_class(self):
        """Python class whose instances are produced by the present factory"""
        return self.FIELD_CLASS

    @property
    def field_name(self):
        """Name of the physical field"""
        return self.field_class.NAME

    @property
    def name(self):
        # For backwards-compatibility only
        return self.field_name

    @property
    def field_type(self):
        """Type of physical field."""
        return self.field_class.TYPE

    @property
    def field_units(self):
        """Units of physical field."""
        return self.field_class.UNITS

    @property
    def grid(self):
        """
        Instance of `imagine.fields.BaseGrid` containing a 3D grid where the
        field is/was evaluated
        """
        if self._grid is None:
            if (self._boxsize is not None) and (self._resolution is not None):
                self._grid = UniformGrid(box=self._boxsize,
                                         resolution=self._resolution)
            elif self.field_type != 'dummy':
                raise ValueError('Non-dummy fields must be initialized with'
                                 'either a valid Grid object or its properties'
                                 '(box and resolution).')
        return self._grid

    @property
    def resolution(self):
        """
        How many bins on each direction of simulation box
        """
        return self._resolution

    @property
    @req_attr
    def default_parameters(self):
        """
        Dictionary storing parameter name as entry, default parameter value
        as content
        """
        return self._default_parameters

    @default_parameters.setter
    def default_parameters(self, new_defaults):
        assert isinstance(new_defaults, dict)
        new_defaults = {key: u.Quantity(value)
                        for (key, value) in new_defaults.items()}
        try:
            self._default_parameters.update(new_defaults)
            log.debug('update default parameters %s' % str(new_defaults))
        except AttributeError:
            self._default_parameters = new_defaults
            log.debug('set default parameters %s' % str(new_defaults))

    @property
    @req_attr
    def priors(self):
        """
        A dictionary containing the priors associated with each parameter.
        Each prior is represented by an instance of
        :py:class:`imagine.priors.prior.Prior`.

        To set new priors one can update the priors dictionary using
        attribution (any missing values will be set to
        :py:class:`imagine.priors.basic_priors.FlatPrior`).
        """
        return self._priors

    @priors.setter
    def priors(self, new_prior_dict):
        if not hasattr(self, '_priors'):
            self._priors = {}
        parameter_ranges = {}

        # Uses previous information
        prior_dict = self._priors.copy()
        prior_dict.update(new_prior_dict)

        for name in self.active_parameters:
            assert (name in prior_dict), 'Missing Prior for '+name

        for name in prior_dict:
            prior = prior_dict[name]
            assert isinstance(prior, Prior), 'Prior must be an instance of :py:class:`imagine.priors.prior.GeneralPrior`.'
            self._priors[name] = prior
            parameter_ranges[name] = prior.range

        self.parameter_ranges = parameter_ranges

    @property
    def parameter_ranges(self):
        """
        Dictionary storing varying range of all default parameters in
        the form {'parameter-name': (min, max)}
        """
        return self._parameter_ranges

    @parameter_ranges.setter
    def parameter_ranges(self, new_ranges):
        assert isinstance(new_ranges, dict)
        for k, v in new_ranges.items():
            assert (len(v) == 2)
        try:
            self._parameter_ranges.update(new_ranges)
            log.debug('update parameter ranges %s' % str(new_ranges))
        except AttributeError:
            self._parameter_ranges = new_ranges
            log.debug('set parameter ranges %s' % str(new_ranges))

    @staticmethod
    def _interval(mean, sigma, n):
        return mean-n*sigma, mean+n*sigma

    @staticmethod
    def _positive_interval(mean, sigma, n):
        return max(0, mean-n*sigma), mean+n*sigma


class CustomFieldFactory(FieldFactory):
    """
    Generates a FieldFactory without subclassing

    Parameters
    ----------
    field_class : imagine.fields.Field
        The field class based on which one wants to create a field factory
    active_parameters : tuple
        List of parameter to be varied/constrained
    default_parameters : dict
        Dictionary containing the default values of the inactive parameters
    priors : dict
        Dictionary containing parameter names as keys and
        :py:obj:`Prior <imagine.priors.prior.Prior>` objects as values, for
        all the active parameters.
    grid : imagine.fields.grid.Grid
        :py:obj:`Grid <imagine.fields.grid.Grid>` object that will be used
        by the fields generated by this field factory
    field_kwargs : dict
        Initialization keyword arguments to be passed to the fields produced
        by this factory.
    """
    # Class attributes (unused in this case)
    FIELD_CLASS = None
    DEFAULT_PARAMETERS = {}
    PRIORS = {}

    def __init__(self, field_class, active_parameters=(),
                 default_parameters={}, priors={}, grid=None, boxsize=None,
                 resolution=None, field_kwargs={}):

        self._field_class = field_class
        self.active_parameters = active_parameters
        self.default_parameters = default_parameters
        self.priors = priors

        super().__init__(grid=grid, boxsize=boxsize, resolution=resolution,
                         active_parameters=active_parameters,
                         field_kwargs=field_kwargs)

    @property
    def field_class(self):
        """Python class whose instances are produced by the present factory"""
        return self._field_class
