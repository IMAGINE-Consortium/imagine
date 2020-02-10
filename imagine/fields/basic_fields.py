import numpy as np
from imagine import GeneralField

class DummyField(GeneralField):
    """
    Base class for a dummy Field used for sending parameters and settings to
    specific Simulators rather than computing and storing a physical field

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

class MagneticField(GeneralField):
    """
    Base class for magnetic fields within IMAGINE
    """
    field_name = 'magnetic_field_base'

    @property
    def field_type(self):
        return 'magnetic_field'
    @property
    def data_description(self):
        return  ['grid_x','grid_y','grid_z','component (x,y,z)']
    @property
    def data_shape(self):
        return tuple([i for i in self.grid.shape] + [3])


class ThermalElectronDensityField(GeneralField):
    """
    Base class for cosmic ray thermal electron density
    """
    field_name = 'thermal_electrons_base'
    
    @property
    def field_type(self):
        return 'cosmic_ray_electron_density'
    @property
    def data_description(self):
        return  ['grid_x','grid_y','grid_z']
    @property
    def data_shape(self):
        return tuple(self.grid.shape)
    
