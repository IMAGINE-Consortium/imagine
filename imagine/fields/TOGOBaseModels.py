import numpy as np

from .TOGOModel import Model
from .TOGOGrid import ParameterSpace, ParameterSpaceDict, ScalarSpace


class Field(Model):

    def __init__(self, grid, internal_shape, parameter_def_dict, unit, call_by_method=False):
        if internal_shape is not None and type(internal_shape) != tuple:
            raise TypeError('Internal shape must be tuple of ints')
        self.internal_shape = internal_shape
        if not isinstance(parameter_def_dict, dict):
            raise TypeError('parameter_names must be dictionary of units')
        input_space = ParameterSpaceDict(parameter_def_dict, generate_from_param_dict=True)

        # missing unit check
        self._unit = unit
        self.grid = grid
        super().__init__(input_param_space=input_space, output_param_space=grid, call_by_method=call_by_method)

    @property
    def data_shape(self):
        if self.internal_shape is None:
            return (*self.grid.shape, )
        else:
            return(*self.grid.shape, *self.internal_shape)

    @property
    def unit(self):
        return self._unit

    @property
    def parameter_units(self):
        return self._input_param_space


class VectorField(Field):

    def __init__(self, grid, parameter_def_dict, unit, dimension, call_by_method):
        if not isinstance(dimension, int):
            raise TypeError('Dimension of VectorField must have integer type')
        super().__init__(grid, (dimension,), parameter_def_dict, unit, call_by_method)

    def __add__(self, ModelToAdd):
        if not isinstance(ModelToAdd, VectorField):
            raise TypeError('Imagine.VectorField: Only Vectorfields can be added')
        if not self.unit == ModelToAdd.unit:
            raise TypeError('Imagine.VectorField: Only Vectorfields with the same units can be added')
        if not self.grid == ModelToAdd.grid:
            raise TypeError('Imagine.VectorField: Only Vectorfields on the same grid can be added')
        if not self.internal_shape == ModelToAdd.internal_shape:
            raise TypeError('Imagine.VectorField: Only Vectorfields with the same internal dimensions can be added')

        parameter_def = dict(ModelToAdd.parameter_units, **self.parameter_units)
        m = VectorField(self.grid, parameter_def, self.unit, self.internal_shape[0], True)

        def _new_compute_model(parameters):
            return self.compute_model(parameters) + ModelToAdd.compute_model(parameters)
        setattr(m, 'compute_model', _new_compute_model)
        return m

    def __mul__(self, ModelToMultiply):
        if not isinstance(ModelToMultiply, VectorField):
            raise TypeError('Imagine.VectorField: Only Vectorfields can be multiplied')
        if not self.grid == ModelToMultiply.grid:
            raise TypeError('Imagine.VectorField: Only Vectorfields on the same grid can be multiplied')
        if not self.dimension == ModelToMultiply.dimension:
            raise TypeError('Imagine.VectorField: Only Vectorfields with the same internal dimesnions can be multiplied')

        parameter_def = dict(ModelToMultiply.parameter_units, **self.parameter_units)
        m = VectorField(self.grid, parameter_def, self.unit*ModelToMultiply.unit, self.dimension, True)

        def _new_compute_model(parameters):
            return self.compute_model(parameters)*ModelToMultiply.compute_model(parameters)
        setattr(m, 'compute_model', _new_compute_model)
        return m

    def vec_abs(self):
        m = ScalarField(self.grid, self.parameter_units, self.unit, True)

        def _new_compute_model(parameters):
            return np.sqrt(np.sum(self.compute_model(parameters)**2, axis=-1))

        setattr(m, 'compute_model', _new_compute_model)
        return m

    @staticmethod
    def transform_coeff(grid, component=None):
        cos_theta = grid.z / grid.r_spherical
        sin_theta = np.sqrt(1 - cos_theta**2)  #  TODO: Test vs np.sin(np.arcos(cos_theta))

        tan_phi = grid.y/grid.z
        tmp_phi = np.sqrt(1 + tan_phi**2)
        # phi = np.arctan2(pos[1], pos[0])

        cos_phi = 1/tmp_phi  # TODO: Test vs np.cos(np.arctan2(pos[1], pos[0]))
        sin_phi = tan_phi*tmp_phi  # TODO: Test vs np.sin(np.arctan2(pos[1], pos[0]))

        if component is None:  # return only the coefficients is cheaper in this case
            return sin_theta, cos_theta, sin_phi, sin_theta
        if component == 'parallel':
            return np.swapaxes(np.array([sin_theta * cos_phi, cos_theta * cos_phi, -sin_phi]), 0, -1)
        if component == 'theta':
            return np.swapaxes(np.array([sin_theta * sin_phi, cos_theta * sin_phi,  cos_phi]), 0, -1)
        if component == 'phi':
            return np.swapaxes(np.array([cos_theta,          -sin_theta,            0.]), 0, -1)

    def spherical_project(self):
        raise NotImplementedError()

    def radial_component(self):
        m = ScalarField(self.grid, self.parameter_units, self.unit, True)

        def _new_compute_model(parameters):
            return np.sum(self.transform_coeff(self.grid, 'parallel')*self.compute_model(parameters), axis=-1)

        setattr(m, 'compute_model', _new_compute_model)
        return m


class ScalarField(Field):

    def __init__(self, grid, parameter_names, unit, call_by_method):
        super().__init__(grid, None, parameter_names, unit, call_by_method)

    def __add__(self, ModelToAdd):
        if not isinstance(ModelToAdd, ScalarField):
            raise TypeError('Imagine.ScalarField: Only ScalarFields can be added')
        if not self.unit == ModelToAdd.unit:
            raise TypeError('Imagine.ScalarField: Only ScalarFields with the same units can be added')
        if not self.grid == ModelToAdd.grid:
            raise TypeError('Imagine.ScalarField: Only ScalarFields on the same grid can be added')

        parameter_def = dict(ModelToAdd.parameter_units, **self.parameter_units)
        m = ScalarField(self.grid, parameter_def, self.unit,  True)

        def _new_compute_model(parameters):
            return self.compute_model(parameters) + ModelToAdd.compute_model(parameters)
        setattr(m, 'compute_model', _new_compute_model)
        return m

    def __mul__(self, ModelToMultiply):
        if not isinstance(ModelToMultiply, ScalarField):
            raise TypeError('Imagine.ScalarField: Only ScalarFields can be multiplied')
        if not self.grid == ModelToMultiply.grid:
            raise TypeError('Imagine.ScalarField: Only ScalarFields on the same grid can be multiplied')

        parameter_def = dict(ModelToMultiply.parameter_units, **self.parameter_units)
        m = ScalarField(self.grid, parameter_def, self.unit*ModelToMultiply.unit, True)

        def _new_compute_model(parameters):
            return self.compute_model(parameters)*ModelToMultiply.compute_model(parameters)
        setattr(m, 'compute_model', _new_compute_model)
        return m
