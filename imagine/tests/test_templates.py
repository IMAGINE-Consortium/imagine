import imagine as img
import astropy.units as u
import numpy as np
import pytest

from imagine.templates.magnetic_field_template import MagneticFieldTemplate
from imagine.templates.thermal_electrons_template import ThermalElectronsDensityTemplate

__all__ = []

def test_magnetic_field_template():
    """
    Tests the MagneticFieldTemplate, including the handling of cartesian units
    and units by Fields.
    """
    grid = img.fields.grid.UniformGrid(box=[[-1*u.kpc,1*u.kpc]]*3, resolution=[2]*3)

    magnetic_field = MagneticFieldTemplate(grid,
                                           parameters={'Parameter_A': 17*u.microgauss,
                                                      'Parameter_B': 2})
    data = magnetic_field.get_data()
    Bx = data[:,:,:,0]
    By = data[:,:,:,1]
    Bz = data[:,:,:,2]

    assert np.all(Bx == 17*u.microgauss)
    assert np.all(By == 2*u.microgauss)
    assert Bz[0,0,0] == -42*u.microgauss
    assert Bz[1,1,1] == 42*u.microgauss


def test_thermal_electrons_template():
    """
    Tests the ThermalElectronsDensityTemplate, including the handling of
    spherical coordinates and ensemble seeds.
    """
    grid = img.fields.grid.UniformGrid(box=[[-1*u.kpc,1*u.kpc]]*3, resolution=[2]*3)

    ne = ThermalElectronsDensityTemplate(grid, ensemble_seeds=[1,2],
                                         parameters={'Parameter_A': 2000*u.pc,
                                                     'Parameter_B': 0.5})
    # The answer combines the
    answer = ([[[-8.92229869e-06, -3.89869352e-06],
                [ 8.92229869e-06,  3.89869352e-06]],
                [[-2.97409956e-06, -1.29956451e-06],
                [ 2.97409956e-06,  1.29956451e-06]]]) * u.cm**-3

    assert(np.allclose(ne.get_data(0).cgs, answer))
    assert(np.allclose(ne.get_data(1).cgs, answer*2))
