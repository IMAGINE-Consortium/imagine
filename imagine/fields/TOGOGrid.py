"""
Contains the definition of the BaseGrid class and an example of its
application: a basic uniform grid.

This was strongly based on GalMag's Grid class,
initially developed by Theo Steininger
"""

# %% IMPORTS
# Built-in imports
import abc

# Package imports
import astropy.units as u
import numpy as np

# IMAGINE imports
from imagine.tools import BaseClass

# All declaration
__all__ = ['BaseGrid', 'UniformGrid']

class ParameterSpace()
class ParameterSpaceTuple()


# %% CLASS DEFINITIONS
class BaseGrid(BaseClass, metaclass=abc.ABCMeta):
    """
    Defines a 3D grid object for a given choice of box dimensions
    and resolution.

    This is a base class. To create your own grid, you need to subclass
    `BaseGrid`  and override the method :py:meth:`generate_coordinates`.

    Calling the attributes does the conversion between different coordinate
    systems automatically (spherical, cylindrical and cartesian coordinates
    centred at the galaxy centre).

    Parameters
    ----------
    box : 3x2-array_like
         Box limits

    resolution : 3-array_like
         containing the resolution along each axis.

    Attributes
    ----------
    box : 3x2-array_like
         Box limits

    resolution : 3-array_like
         Containing the resolution along each axis (the *shape* of the grid).
    """
    def __init__(self, box, resolution):
        # Call super constructor
        super().__init__()

        self.box = box

        self.resolution = np.empty((3,), dtype=np.int)
        self.resolution[:] = resolution

        self._coordinates = None
        self._prototype_source = None
        self.grid_type = None

    @property
    def coordinates(self):
        """A dictionary contaning all the coordinates"""
        if self._coordinates is None:
            self._coordinates = self.generate_coordinates()
            # Checks the input units
            for k in self._coordinates:
                assert isinstance(self._coordinates[k],u.Quantity), k+' must be a Quantity'
                if k in ('x','y','z','r_spherical', 'r_cylindrical'):
                    assert self._coordinates[k].unit.is_equivalent(u.kpc), k+' must be in length units'
                elif k in ('theta','phi'):
                    assert self._coordinates[k].unit.is_equivalent(u.rad,
                        equivalencies=u.dimensionless_angles()), k+' must be an angle or dimensionless'

        return self._coordinates

    @property
    def x(self):
        """Horizontal coordinate, :math:`x`"""
        if 'x' not in self.coordinates:
            self.coordinates['x'] = self.r_cylindrical * self.cos_phi

        return self.coordinates['x']

    @property
    def y(self):
        """Horizontal coordinate, :math:`y`"""
        if 'y' not in self.coordinates:
            self.coordinates['y'] = self.r_cylindrical * self.sin_phi
        return self.coordinates['y']

    @property
    def z(self):
        """Vertical coordinate, :math:`z`"""
        if 'z' not in self.coordinates:
            self.coordinates['z'] = self.r_spherical * self.cos_theta

        return self.coordinates['z']

    @property
    def r_spherical(self):
        """Spherical radial coordinate, :math:`r`"""
        if 'r_spherical' not in self.coordinates:
            if 'z' not in self.coordinates:
                raise KeyError('Could not compute r_spherical from available coordinates')

            self.coordinates['r_spherical'] = np.sqrt(self.r_cylindrical**2 +
                                                      self.z**2)
        return self.coordinates['r_spherical']

    @property
    def r_cylindrical(self):
        """Cylindrical radial coordinate, :math:`s`"""
        if 'r_cylindrical' not in self.coordinates:

            if ('x' not in self.coordinates) or ('y' not in self.coordinates):
                self.coordinates['r_cylindrical'] = self.r_spherical * self.sin_theta
            else:
                self.coordinates['r_cylindrical'] = np.sqrt(self.x**2 + self.y**2)

        return self.coordinates['r_cylindrical']

    @property
    def theta(self):
        r"""Polar coordinate, :math:`\theta`"""
        if 'theta' not in self.coordinates:
            self.coordinates['theta'] = np.arccos(self.z/self.r_spherical)
        return self.coordinates['theta']

    @property
    def phi(self):
        r"""Azimuthal coordinate, :math:`\phi`"""
        if 'phi' not in self.coordinates:
            self.coordinates['phi'] = np.arctan2(self.y, self.x)
        return self.coordinates['phi']

    @property
    def sin_theta(self):
        r""":math:`\sin(\theta)`"""
        if 'theta' not in self.coordinates:
            return self.r_cylindrical / self.r_spherical
        return np.sin(self.theta)

    @property
    def cos_theta(self):
        r""":math:`\cos(\theta)`"""
        if 'theta' in self.coordinates:
            return np.cos(self.theta)
        return self.z / self.r_spherical

    @property
    def sin_phi(self):
        r""":math:`\sin(\phi)`"""
        if 'phi' in self.coordinates:
            return np.sin(self.phi)
        else:
            return self.y / self.r_cylindrical

    @property
    def cos_phi(self):
        r""":math:`\cos(\phi)`"""
        if 'phi' in self.coordinates:
            return np.cos(self.phi)
        return self.x / self.r_cylindrical

    @property
    def shape(self):
        """The same as :py:attr:`resolution`"""
        return self.resolution

    @abc.abstractmethod
    def generate_coordinates(self):
        """
        Placeholder for method which uses the information in the attributes
        `box` and `resolution` to return a dictionary containing the values
        of (either) the coordinates ('x','y','z') or
        ('r_cylindrical', 'phi','z'), ('r_spherical','theta', 'phi')

        This method is *automatically* called the first time any coordinate
        is read.
        """
        raise NotImplementedError(("Subclasses should implement this!"))


class UniformGrid(BaseGrid):
    r"""
    Defines a 3D grid object for a given choice of box dimensions
    and resolution. The grid is uniform in the selected coordinate system
    (which is chosen through the parameter `grid_type`.

    Example
    -------
    >>> import magnetizer.grid as grid
    >>> import astropy.units as u
    >>> xlims = [0,4]*u.kpc; ylims = [1,2]*u.kpc; zlims = [1,1]*u.kpc
    >>> g = grid.UniformGrid([xlims, ylims, zlims], [5,2,1])
    >>> g.x
    array([[[0.],[0.]], [[1.],[1.]], [[2.],[2.]], [[3.],[3.]], [[4.],[4.]]])
    >>> g.y
    array([[[1.], [2.]], [[1.], [2.]], [[1.], [2.]], [[1.], [2.]], [[1.], [2.]]])

    Calling the attributes does the conversion between different coordinate
    systems automatically.

    Parameters
    ----------
    box : 3x2-array_like
         Box limits. Each row corresponds to a different coordinate and should
         contain units. For 'cartesian' grid_type, the rows should contain
         (in order)'x','y' and 'z'.
         For 'cylindrical' they should have  'r_cylindrical', 'phi' and 'z'.
         for 'spherical', 'r_spherical','theta' and 'phi'.
    resolution : 3-array_like
         containing the resolution along each axis.
    grid_type : str, optional
        Choice between 'cartesian', 'spherical' and 'cylindrical' *uniform*
        coordinate grids. Default: 'cartesian'
    """
    def __init__(self, box, resolution, grid_type='cartesian'):
        # Base class initialization
        super().__init__(box, resolution)
        # Subclass specific attributes
        self.grid_type = grid_type

    def generate_coordinates(self):
        """
        This method is *automatically* called internally the first time any
        coordinate is requested.

        Generates a uniform grid based on the attributes `box`, `resolution` and
        `grid_type` and returns it in a dictionary.

        Returns
        -------
        coordinates_dict : dict
            Dictionary containing the keys ('x','y','z') if `grid_type` is
            'cartesian`,
            ('r_cylindrical', 'phi','z') if `grid_type` is spherical, and
            ('r_spherical','theta', 'phi') if `grid_type` is 'cylindrical`.
        """

        # Creates array with starting and endpoints as specified in self.box
        # and with self.resolution

        # Stores the dimensions (the begin of the interval is taken as reference)
        ubox = [row[0].unit for row in self.box]
        # Constructs a dimensionless version (to enforce consistent dimensions
        # in the intervals and avoid problems in mgrid)
        box_vals = [ [b.to_value(unit) for b in row]
                     for row, unit in zip(self.box, ubox) ]

        local_slice = (slice(box_vals[0][0], box_vals[0][1], self.resolution[0]*1j),
                       slice(box_vals[1][0], box_vals[1][1], self.resolution[1]*1j),
                       slice(box_vals[2][0], box_vals[2][1], self.resolution[2]*1j))

        local_coordinates = np.mgrid[local_slice]

        if self.grid_type=='cartesian':
            selected_coords = ['x','y','z']
        elif self.grid_type=='spherical':
            selected_coords = ['r_spherical','theta','phi']
        elif self.grid_type=='cylindrical':
            selected_coords = ['r_cylindrical','phi','z']
        else:
            raise ValueError

        coordinates_dict = {k: cgrid*unit for k, cgrid, unit in zip(selected_coords,
                                                                    local_coordinates,
                                                                    ubox)}

        return coordinates_dict
