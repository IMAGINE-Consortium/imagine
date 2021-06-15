import nifty7 as ift
import astropy.units as u
import numpy as np
import itertools

import imagine as img
from imagine.simulators import Simulator

__all__ = ['NiftyLOS']


class NiftyLOS(Simulator):
    """
    wrapper around the nifty los response as an alternative to hammurabi
    """

    # Class attributes
    SIMULATED_QUANTITIES = ['fd', 'dm', 'sync']
    REQUIRED_FIELD_TYPES = []
    OPTIONAL_FIELD_TYPES = ['dummy','magnetic_field',
                            'thermal_electron_density',
                            'cosmic_ray_electron_density']
    ALLOWED_GRID_TYPES = ['cartesian']

    def __init__(self, measurements, grid, rtype, **kwargs):
        assert rtype in ['full_box', 'random_pos']
        # Send the Measurements to the parent class
        super().__init__(measurements)
        self.grid = grid
        distances = tuple([b/r for b,r in zip(grid.box, grid.resolution)])
        self.domain = ift.makeDomain(ift.RGSpace(grid.resolution, distances))
        if 'ends' not in kwargs:
            ends = [0.5*b for b in grid.box]
        else:
            ends = kwargs['ends']
        if rtype == 'full_box':
            import healpy as hp
            nside = kwargs['nside']
            n_data = 12*nside**2
            pix = np.arange(0, n_data)
            x, y, z = hp.pix2vec(ipix=pix, nside=nside)
            r = np.sqrt(np.sum(np.asarray([0.25*b**2 for b in grid.box])))
            response = self.get_response(self.domain, [r*x, r*y, r*z], ends, n_data)
        else:
            response = self.get_response(self.domain, kwargs['starts'], ends, len(kwargs['starts'][0]))
        self.response = response
        keys = [t[0] for t in measurements.keys()]
        if 'fd' in keys or 'sync' in keys:
            self.vector_trafo = VectorTransform(self.domain, ('x', 'y', 'z'), ('theta', 'phi', 'radial'))
        else:
            self.vector_trafo = None
        self.unit = u.Unit('')
        self.temp_mag = {}
        # Stores class-specific attributes

    @staticmethod
    def get_response(domain, starts, end, n_los):
        ends = list()
        ends.append(np.full(shape=(n_los,), fill_value=end[0]))
        ends.append(np.full(shape=(n_los,), fill_value=end[1]))
        ends.append(np.full(shape=(n_los,), fill_value=end[2]))
        return ift.LOSResponse(domain, starts, ends)

    def simulate(self, key, coords_dict, realization_id, output_units):
        # Accesses fields and grid
        key = key[0]
        if key == 'fd':
            assert 'magnetic_field' in self.fields and 'thermal_electron_density' in self.fields

            B = ift.MultiField.from_dict({k: ift.makeField(self.domain, self.fields['magnetic_field'].value[:, :, :, i])
                                          for i, k in enumerate(self.vector_trafo._rg_keys)
                                          })
            B = self.vector_trafo(B)

            ne = ift.Field(self.domain, self.fields['thermal_electron_density'])
            out = self.response(B['radial']*ne).val_rw() * output_units
        elif key == 'sync':
            assert 'magnetic_field' in self.fields and 'cosmic_ray_electron_density' in self.fields
            B = self.vector_trafo(self.fields['magnetic_field'].value)
            nr = self.fields['cosmic_ray_electron_density']
            out = self.response(ift.Field(self.domain, (B['phi']**2 + B['theta']**2) * nr)).val_rw() * output_units
        elif key == 'dm':
            assert 'thermal_electron_density' in self.fields
            ne = self.fields['thermal_electron_density']
            out = self.response(ift.Field(self.domain, ne)).val_rw() * output_units
        else:
            raise KeyError
        return out


class VectorTransform(ift.LinearOperator):
    def __init__(self, rg_domain, rg_keys, sphere_keys):
        self._rg_dom = ift.makeDomain(rg_domain)
        assert isinstance(self._rg_dom[0], ift.RGSpace)
        assert len(rg_keys) == len(sphere_keys) == len(self._rg_dom[0].shape)
        self._rg_keys = rg_keys
        self._sphere_keys = sphere_keys
        self._domain = ift.MultiDomain.make({kd: self._rg_dom for kd in self._rg_keys})
        self._target = ift.MultiDomain.make({kd: self._rg_dom for kd in self._sphere_keys})
        s = np.multiply(np.asarray(self._rg_dom[0].shape), np.asarray(self._rg_dom[0].distances))
        self.rg_grid = np.meshgrid(np.arange(-s[0]/2, s[0]/2, self._rg_dom[0].distances[0]),
                              np.arange(-s[1]/2, s[1]/2, self._rg_dom[0].distances[1]),
                              np.arange(-s[2]/2, s[2]/2, self._rg_dom[0].distances[2]))
        self.sin_theta = np.zeros(self._rg_dom[0].shape)
        self.cos_theta = np.zeros(self._rg_dom[0].shape)
        self.sin_phi = np.zeros(self._rg_dom[0].shape)
        self.cos_phi = np.zeros(self._rg_dom[0].shape)
        self.calc_trafo_elements(rg_domain.shape)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def calc_trafo_elements(self, sh):
        for i, j, k in itertools.product(range(sh[0]), range(sh[1]), range(sh[2])):
            if not ((i == sh[0]/2) & (j == sh[1]/2) & (k == sh[2]/2)):
                x = i - sh[0] / 2
                y = j - sh[1] / 2
                z = k - sh[2] / 2
                theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
                phi = np.arctan2(y, x)

                self.sin_theta[i, j, k] = np.sin(theta)
                self.cos_theta[i, j, k] = np.cos(theta)
                self.sin_phi[i, j, k] = np.sin(phi)
                self.cos_phi[i, j, k] = np.cos(phi)

    def apply_transform(self, vec_field):
        vec_new = np.empty(vec_field.shape)
        assert vec_field.shape[0] == 3
        vec_new[0] = self.cos_phi * (self.sin_theta * vec_field[0] + self.cos_theta * vec_field[1]) - \
                     self.sin_phi * vec_field[2]
        vec_new[1] = self.sin_phi * (self.sin_theta * vec_field[0] + self.cos_theta * vec_field[1]) + \
                     self.cos_phi * vec_field[2]
        vec_new[2] = self.cos_theta * vec_field[0] - self.sin_theta * vec_field[1]
        return vec_new

    def apply_adjoint_transform(self, vec_field):
        vec_new = np.empty(vec_field.shape)
        assert vec_field.shape[0] == 3
        vec_new[0] = self.sin_theta * (self.cos_phi * vec_field[0] + self.sin_phi * vec_field[1]) + \
                     self.cos_theta * vec_field[2]
        vec_new[1] = self.cos_theta * (self.cos_phi * vec_field[0] + self.sin_phi * vec_field[1]) - \
                     self.sin_theta * vec_field[2]
        vec_new[2] = - self.sin_phi * vec_field[0] + self.cos_phi * vec_field[1]
        return vec_new

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            kd_d = self._rg_keys
            kd_t = self._sphere_keys
        else:
            kd_t = self._rg_keys
            kd_d = self._sphere_keys
        vec = np.asarray([x.val[y] for y in kd_d])
        if mode == self.TIMES:
            vec_new = self.apply_transform(vec)
            # print(vec_new)
        else:
            vec_new = self.apply_adjoint_transform(vec)
        return ift.MultiField.from_dict({kk: ift.Field(ift.makeDomain(self._rg_dom), vec_new[s])
                                         for kk, s in zip(kd_t, np.arange(3))})
