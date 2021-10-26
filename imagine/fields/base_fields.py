r"""
This module contains basic base classes that can be used to include new fields
in IMAGINE. The classes found here here correspond to the physical fields most
commonly found by members of the IMAGINE community and may be improved in the
future.

A brief summary of the module:

* :py:class:`MagneticField` — for models of the galactic/Galactic Magnetic Field, :math:`\mathbf{B}(\mathbf{r})`
* :py:class:`ThermalElectronDensityField` — for models of the density of thermal electrons, :math:`n_e(\mathbf{r})`
* :py:class:`CosmicRayElectronDensityField`— for models of the density/flux of cosmic ray electrons, :math:`n_{\rm cr}(\mathbf{r})`
* :py:class:`DummyField` — allows passing parameters to a :py:obj:`Simulator <imagine.simulators.simulator.Simulator>` without having to evaluate anything on a :py:obj:`Grid <imagine.fields.grid.Grid>`


See also :doc:`IMAGINE Components <components>` section of the docs.
"""

# %% IMPORTS
# Built-in imports
import abc

# Package imports
import astropy.units as u

# IMAGINE imports
from imagine.fields import Field
from imagine.tools import req_attr

# All declaration
__all__ = ['MagneticField', 'ThermalElectronDensityField', 'DummyField']


# %% CLASS DEFINITIONS
class MagneticField(Field):
    """
    Base class for the inclusion of new models for magnetic fields.
    It should be subclassed following the template provided.

    For more details, check the :ref:`components:Magnetic Fields` Section
    of the documentation.

    Parameters
    ----------

    grid : imagine.fields.grid.BaseGrid
        Instance of :py:class:`imagine.fields.grid.BaseGrid` containing a 3D
        grid where the field is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    ensemble_size : int
        Number of realisations in field ensemble
    ensemble_seeds
        Random seed(s) for generating random field realisations
    """

    # Class attributes
    TYPE = 'magnetic_field'
    UNITS = u.microgauss

    @property
    def data_description(self):
        return(['grid_x', 'grid_y', 'grid_z', 'component (x,y,z)'])

    @property
    def data_shape(self):
        return(*self.grid.shape, 3)


class ThermalElectronDensityField(Field):
    """
    Base class for the inclusion of models for spatial distribution of thermal
    electrons. It should be subclassed following the template provided.

    For more details, check the :ref:`components:Thermal electrons`
    Section of the documentation.

    Parameters
    ----------
    grid : imagine.fields.grid.BaseGrid
        Instance of :py:class:`imagine.fields.grid.BaseGrid` containing a 3D
        grid where the field is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    ensemble_size : int
        Number of realisations in field ensemble
    ensemble_seeds
        Random seed(s) for generating random field realisations

    """

    # Class attributes
    TYPE = 'thermal_electron_density'
    UNITS = u.cm**(-3)

    @property
    def data_description(self):
        return(['grid_x', 'grid_y', 'grid_z'])

    @property
    def data_shape(self):
        return tuple(self.grid.shape)


class CosmicRayElectronDensityField(Field):
    """
    First attempt at implementing CRE field
    - 3d scalar field


    What do we need for eventual synchrotron emissivity?
    - number density
    - cr spectrum as a function of position or effective energy function as function of posisiton and observing frequency
    """

    # Class attributes
    TYPE = 'cosmic_ray_electron_density'
    UNITS = u.cm**(-3)

    @property
    def data_description(self):
        return(['grid_x', 'grid_y', 'grid_z'])

    @property
    def data_shape(self):
        return tuple(self.grid.shape)


class DummyField(Field, metaclass=abc.ABCMeta):
    """
    Base class for a dummy Field used for sending parameters and settings to
    specific Simulators rather than computing and storing a physical field.
    """

    # Class attributes
    TYPE = 'dummy'
    UNITS = None
    PARAMETER_NAMES = None

    def __init__(self, *args, **kwargs):
        kwargs['grid'] = None
        super().__init__(**kwargs)

    @property
    def data_description(self):
        return([])

    @property
    def data_shape(self):
        return(None)

    @property
    def parameter_names(self):
        """Parameters of the field"""
        return list(self.field_checklist)

    @property
    @req_attr
    def field_checklist(self):
        """Parameters of the dummy field"""
        return self.FIELD_CHECKLIST

    @property
    @req_attr
    def simulator_controllist(self):
        """
        Dictionary containing fixed Simulator settings
        """
        return self.SIMULATOR_CONTROLLIST

    def compute_field(self, *args, **kwargs):
        pass

    def get_data(self,  i_realization=0, dependencies={}):
        """
        Mock evaluation of the dummy field defined by this class.

        Parameters
        ----------
        i_realization : int
            Index of the current realization
        dependencies : dict
            If the :py:data:`dependencies_list` is non-empty, a dictionary containing
            the requested dependencies must be provided.

        Returns
        -------
        parameters : dict
            Dictionary of containing a copy of the Field parameters including
            an extra entry with the random seed that should be used with the
            present realization (under the key: 'random_seed')
        """
        self._update_dependencies(dependencies)
        parameters = self._parameters.copy()
        parameters['random_seed'] = self.ensemble_seeds[i_realization]

        return parameters
