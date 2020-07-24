# %% IMPORTS
# Built-in imports
import abc
import logging as log

# Package imports
import astropy.units as u

# IMAGINE imports
from imagine.fields.grid import BaseGrid, UniformGrid
from imagine.priors import GeneralPrior
from imagine.tools import BaseClass, unity_mapper, req_attr

# All declaration
__all__ = ['FieldFactory']


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

    Example
    -------
    To include a new Field_Factory, one needs to create a derived class
    with customized initialization. Below we show an example which is
    compatible with the :py:class:`xConstantField` showed in the
    :ref:`components:Fields` section of the documentation::

        @icy
        class xConstantField_Factory(GeneralFieldFactory):
            def __init__(self, grid=None, boxsize=None, resolution=None):
                super().__init__(grid, boxsize, resolution)
                self.field_class = xConstantField
                self.default_parameters = {'constantA': 5.0}
                self.parameter_ranges = {'constantA': [-10., 10.]}

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
        self.parameter_ranges = {}
        self.active_parameters = active_parameters
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
        # in mapping, variable name will be checked in default_parameters
        mapped_variables = self._map_variables_to_parameters(variables)
        # copy default parameters and update wrt argument
        work_parameters = dict(self.default_parameters)
        # update is safe
        work_parameters.update(mapped_variables)
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
        return(self.FIELD_CLASS)

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
        try:
            self._default_parameters.update(new_defaults)
            log.debug('update default parameters %s' % str(new_defaults))
        except AttributeError:
            self._default_parameters = new_defaults
            log.debug('set default parameters %s' % str(new_defaults))

    @property
    def active_parameters(self):
        """
        Tuple of parameter names which can vary, not necessary to cover all
        default parameters
        """
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
    @req_attr
    def priors(self):
        """
        A dictionary containing the priors associated with each parameter.
        Each prior is represented by an instance of
        :py:class:`imagine.priors.prior.GeneralPrior`.

        To set new priors one can update the priors dictionary using
        attribution (any missing values will be set to
        :py:class:`imagine.priors.basic_priors.FlatPrior`).
        """
        return self._priors

    def _prior_correlator(self, prior_a, prior_b, corr_coefficient):
        raise NotImplementedError

    @priors.setter
    def priors(self, new_prior_dict):
        if not hasattr(self, '_priors'):
            self._priors = {}

        parameter_ranges = {}

        # Uses previous information
        prior_dict = self._priors.copy()
        prior_dict.update(new_prior_dict)

        for name in self.default_parameters:
            assert (name in prior_dict), 'Missing Prior for '+name
            prior = prior_dict[name]
            assert isinstance(prior, GeneralPrior), 'Prior must be an instance of :py:class:`imagine.priors.prior.GeneralPrior`.'
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
            # check if k is inside default
            assert (k in self.default_parameters.keys())
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
        A dictionary containing default parameter values converted into
        default normalized variables (i.e with values scaled to be in the
        range [0,1]).
        """
        log.debug('@ field_factory::default_variables')
        tmp = dict()
        for par, def_val in self.default_parameters.items():
            low, high = self.parameter_ranges[par]
            tmp[par] = float(def_val - low)/float(high - low)
        return tmp

    def _map_variables_to_parameters(self, variables):
        """
        Converts Bayesian sampling variables into model parameters

        Parameters
        ----------
        variables : dict
            Python dictionary in form {'parameter-name', logic-value}

        Returns
        -------
        parameter_dict : dict
            Python dictionary in form {'parameter-name', physical-value}
        """
        log.debug('@ field_factory::_map_variables_to_parameters')
        assert isinstance(variables, dict)
        parameter_dict = {}
        for variable_name in variables:
            # variable_name must have been registered in .default_parameters
            # and, also being active
            assert (variable_name in self.default_parameters and
                    variable_name in self.active_parameters)
            low, high = self.parameter_ranges[variable_name]
            # Ensures consistent physical units, if needed
            if isinstance(low, u.Quantity):
                units = low.unit;
                low = low.value
                high = high.to(units).value
            else:
                units = 1
            # unity_mapper defined in imainge.tools.carrier_mapper
            mapped_variable = unity_mapper(variables[variable_name], low, high)
            parameter_dict[variable_name] = mapped_variable * units
        return parameter_dict

    @staticmethod
    def _interval(mean, sigma, n):
        return(mean-n*sigma, mean+n*sigma)

    @staticmethod
    def _positive_interval(mean, sigma, n):
        return(max(0, mean-n*sigma), mean+n*sigma)
