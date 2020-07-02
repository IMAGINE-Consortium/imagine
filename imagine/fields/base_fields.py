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
import astropy.units as u
from imagine.fields.field import Field


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
    NAME = 'magnetic_field_base'
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
    NAME = 'thermal_electrons_base'
    TYPE = 'thermal_electron_density'
    UNITS = u.cm**(-3)
    data_description = ['grid_x', 'grid_y', 'grid_z']

    @property
    def data_shape(self):
        return tuple(self.grid.shape)


class CosmicRayElectronDensityField(Field):
    """
    Not yet implemented
    """

    # Class attributes
    NAME = 'cosmic_ray_electrons_base'
    TYPE = 'cosmic_ray_electron_density'

    def __init__():
        raise NotImplementedError


class DummyField(Field):
    """
    Base class for a dummy Field used for sending parameters and settings to
    specific Simulators rather than computing and storing a physical field.
    """

    # Class attributes
    NAME = 'dummy_base'
    TYPE = 'dummy'
    UNITS = None

    @property
    def data_description(self):
        return([])

    @property
    def data_shape(self):
        return(self.grid.shape)

    @property
    def simulator_controllist(self):
        """
        Dictionary containing fixed Simulator settings
        """
        return dict()

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
