from .TOGOModel import Model
import astropy.units as apu
import numpy as np


class VectorField(Model):

    def __init__(self, grid, parameter_names, unit, dimension, call_by_method):
        if not isinstance(dimension, int):
            raise TypeError('Dimension of VectorField must have integer type')
        super().__init__(grid, parameter_names, unit, (dimension,), call_by_method)

    def vec_abs(self):
        m = ScalarField(self.grid, self.parameter_names, self.unit, True)

        def _new_compute_model(parameters):
            return np.sqrt(np.sum(self.compute_model(parameters)**2, axis=-1))

        setattr(m, 'compute_model', _new_compute_model)
        return m


class ScalarField(Model):

    def __init__(self, grid, parameter_names, unit, call_by_method):
        super().__init__(grid, parameter_names, unit, None, call_by_method)


class MagneticField(VectorField):

    def __init__(self, grid, parameter_names, call_by_method):
        super().__init__(grid, parameter_names, apu.microgauss, 3, call_by_method)


class ThermalElectronDensityField(ScalarField):

    def __init__(self, grid, parameter_names, call_by_method):
        super().__init__(grid, parameter_names, apu.cm**(-3), call_by_method)
