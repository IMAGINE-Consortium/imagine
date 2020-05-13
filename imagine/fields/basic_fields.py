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
import numpy as np
import astropy.units as u
from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy



@icy
class MagneticField(GeneralField):
    """
    Base class for the inclusion of new models for magnetic fields.
    It should be subclassed following the template provided.

    For more details, check the :ref:`components:Magnetic Fields` Section
    of the documentation.

    Parameters
    ----------

    grid : imagine.fields.grid.BaseGrid or None
        Instance of :py:class:`imagine.fields.grid.BaseGrid` containing a 3D
        grid where the field is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    ensemble_size : int
        Number of realisations in field ensemble
    ensemble_seeds
        Random seed(s) for generating random field realisations

    """
    field_name = 'magnetic_field_base'

    @property
    def field_type(self):
        return 'magnetic_field'
    @property
    def field_units(self):
        return u.microgauss
    @property
    def data_description(self):
        return  ['grid_x','grid_y','grid_z','component (x,y,z)']
    @property
    def data_shape(self):
        return tuple([i for i in self.grid.shape] + [3])

@icy
class ThermalElectronDensityField(GeneralField):
    """
    Base class for the inclusion of models for spatial distribution of thermal
    electrons. It should be subclassed following the template provided.

    For more details, check the :ref:`components:Thermal electrons`
    Section of the documentation.

    Parameters
    ----------
    grid : imagine.fields.grid.BaseGrid or None
        Instance of :py:class:`imagine.fields.grid.BaseGrid` containing a 3D
        grid where the field is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    ensemble_size : int
        Number of realisations in field ensemble
    ensemble_seeds
        Random seed(s) for generating random field realisations

    """
    field_name = 'thermal_electrons_base'

    @property
    def field_type(self):
        return 'thermal_electron_density'
    @property
    def field_units(self):
        return u.cm**(-3)
    @property
    def data_description(self):
        return  ['grid_x','grid_y','grid_z']
    @property
    def data_shape(self):
        return tuple(self.grid.shape)


@icy
class CosmicRayElectronDensityField(GeneralField):
    """
    Not yet implemented
    """
    def field_type(self):
        return 'cosmic_ray_electron_density'
    
    def __init__():
        raise NotImplementedError


@icy
class DummyField(GeneralField):
    """
    Base class for a dummy Field used for sending parameters and settings to
    specific Simulators rather than computing and storing a physical field.
    """
    field_name = 'dummy_base'
    @property
    def field_type(self):
        return 'dummy'
    @property
    def field_units(self):
        raise None
        
    def _get_data(self, iseed):
        # In the case of a dummy field, a dictionary is returned instead of 
        # of a data array, updated with the present value of the random seed
        parameters = self._parameters.copy()
        parameters['random_seed'] = self.ensemble_seeds[iseed]
        return parameters
        
        
