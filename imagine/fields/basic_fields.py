import numpy as np
import astropy.units as u
from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy

@icy
class DummyField(GeneralField):
    """
    Base class for a dummy Field used for sending parameters and settings to
    specific Simulators rather than computing and storing a physical field.
    """
    field_name = 'dummy_base'
    field_type = 'dummy'
    @property
    def data(self):
        # A dummy field should never have data called!
        raise NotImplementedError('A dummy field should have no data!')
    @property
    def field_type(self):
        return None
    @property
    def field_units(self):
        raise None

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

