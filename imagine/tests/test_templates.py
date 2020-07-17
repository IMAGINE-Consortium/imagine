import imagine as img
import astropy.units as u
import numpy as np
import pytest
__all__ = []

from imagine.templates.magnetic_field_template import MagneticFieldTemplate

def test_magnetic_field_template():
    import os

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
