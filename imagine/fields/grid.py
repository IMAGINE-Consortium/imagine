"""
Contains the definition of the BaseGrid class and an example of its
application: a basic uniform grid.

This was strongly based on GalMag's Grid class, initially developed
by Theo Steininger
"""
import numpy as np

class BaseGrid:
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

        self.box = np.empty((3, 2), dtype=np.float)
        self.resolution = np.empty((3,), dtype=np.int)

        # use numpy upcasting of scalars and dtype conversion
        self.box[:] = box
        self.resolution[:] = resolution

        self._coordinates = None
        self._prototype_source = None

    @property
    def coordinates(self):
        """A dictionary contaning all the coordinates"""
        if self._coordinates is None:
            self._coordinates = self.generate_coordinates()
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
    >>> g = grid.UniformGrid([[0,4],[1,2],[1,1]], [5,2,1])
    >>> g.x
    array([[[0.],[0.]], [[1.],[1.]], [[2.],[2.]], [[3.],[3.]], [[4.],[4.]]])
    >>> g.y
    array([[[1.], [2.]], [[1.], [2.]], [[1.], [2.]], [[1.], [2.]], [[1.], [2.]]])


    Calling the attributes does the conversion between different coordinate
    systems automatically.

    Parameters
    ----------
    box : 3x2-array_like
         Box limits
    resolution : 3-array_like
         containing the resolution along each axis.
    grid_type : str, optional
        Choice between 'cartesian', 'spherical' and 'cylindrical' *uniform*
        coordinate grids. Default: 'cartesian'
    """
    def __init__(self, box, resolution, grid_type='cartesian'):
        # Base class initialization
        super(UniformGrid, self).__init__(box, resolution)
        # Subclass specific attributes
        self.grid_type=grid_type

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
        # (the naming and structure allows later making the arrays distributed)
        box = self.box
        local_slice = (slice(box[0, 0], box[0, 1], self.resolution[0]*1j),
                       slice(box[1, 0], box[1, 1], self.resolution[1]*1j),
                       slice(box[2, 0], box[2, 1], self.resolution[2]*1j))

        local_coordinates = np.mgrid[local_slice]

        if self.grid_type=='cartesian':
            selected_coords = ['x','y','z']
        elif self.grid_type=='spherical':
            selected_coords = ['r_spherical','theta','phi']
        elif self.grid_type=='cylindrical':
            selected_coords = ['r_cylindrical','phi','z']
        else:
            raise ValueError

        coordinates_dict = {k: i for k, i in zip(selected_coords,
                                                 local_coordinates)}

        return coordinates_dict
