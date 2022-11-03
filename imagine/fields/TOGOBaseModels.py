from .TOGOModel import Model
import astropy.units as apu


class VectorField(Model):

    def __init__(self, grid, parameter_names, unit, dimension, call_by_subclass=False):
        super().__init__(self, grid, parameter_names, unit, (dimension,), call_by_subclass)

    def vec_abs(self):
        m = ScalarField(self.grid, self.parameter_names, self.unit, call_by_subclass=True)
        return m


class ScalarField(Model):

    def __init__(self, grid, parameter_names, unit, call_by_subclass=False):
        super().__init__(self, grid, parameter_names, unit, (1,), call_by_subclass)


class MagneticField(VectorField):

    def __init__(self, grid, parameter_names, call_by_subclass=False):
        super().__init__(self, grid, parameter_names, apu.microgauss, (3,), call_by_subclass)


class ThermalElectronDensityField(ScalarField):

    def __init__(self, grid, parameter_names, call_by_subclass=False):
        super().__init__(self, grid, parameter_names, apu.cm**(-3),  call_by_subclass)
