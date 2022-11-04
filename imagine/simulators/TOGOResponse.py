# %% IMPORTS
# Built-in imports
import abc

# Package imports
import numpy as np

# IMAGINE imports
from imagine.observables import Measurements, Simulations
from imagine.tools import BaseClass, req_attr

# All declaration
__all__ = ['Response']


# %% CLASS DEFINITIONS
class Response(BaseClass, metaclass=abc.ABCMeta):
    """
    Response base class

    New Responses must be introduced by sub-classing the present class.
    Overriding the method :py:meth:`simulate` to convert a list of fields
    into simulated observables. For more details see
    :ref:`components:Responses` section of the documentation.

    Parameters
    ----------
    measurements : imagine.Measurements
        An observables dictionary containing the set of measurements that will be
        used to prepare the mock observables

    Attributes
    ----------
    grid : imagine.Basegrid
        Grid object where the fields were evaluated (NB if a common grid is not being
        used, this is set to None
    grids: imagine.Basegrid
        Grid objects for each individual field None if common grid is being used)
    fields : dict
        Dictionary containing field types as keys and the sum of evaluated fields
        as values
    observables : list
        List of Observable keys
    output_units : astropy.units.Unit
        Output units used in the simulator
    """

    def __init__(self, input_grid_s, output_grid_s):
        # Call super constructor
        super().__init__()

        self._input_grid = input_grid_s
        self._output_grid = output_grid_s
        self.required_fields = None

    @property
    @req_attr
    def allowed_grid_types(self):
        """
        Must be overriden with a list or set of allowed grid types that work with this Response.
        Example: ['cartesian']
        """
        return self.ALLOWED_GRID_TYPES

    @property
    def use_common_grid(self):
        """
        Must be overriden with a list or set of allowed grid types that work with this Response.
        Example: ['cartesian']
        """
        return getattr(self, 'USE_COMMON_GRID', True)

    @abc.abstractmethod
    def simulate(self):
        """

        """
        raise NotImplementedError

    def __call__(self, field_list):
        """
        Runs the simulator over a Fields list

        Parameters
        ----------
        field_list : list
            List of imagine.Field object which must include all the `required_field_types`

        Returns
        -------
        sims : imagine.Simulations
            A Simulations object containing all the specified mock data
        """

        return self.simulate()
