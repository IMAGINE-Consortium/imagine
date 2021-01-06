"""
Datasets are auxiliary classes used to facilitate the reading and inclusion of
observational data in the IMAGINE pipeline"""

# %% IMPORTS
# Package imports
from astropy import units as u
import numpy as np

# IMAGINE imports
from imagine.tools import BaseClass, distribute_matrix, peye, req_attr

# All declaration
__all__ = ['Dataset', 'TabularDataset', 'HEALPixDataset', 'ImageDataset',
           'FaradayDepthHEALPixDataset', 'SynchrotronHEALPixDataset',
           'DispersionMeasureHEALPixDataset']


# %% CLASS DEFINITIONS
class Dataset(BaseClass):
    """
    Base class for writing helpers to convert arbitrary
    observational datasets into  IMAGINE's standardized format
    """
    def __init__(self):
        # Call super constructor
        super().__init__()

        self.coords = None
        self.Nside = None
        self.frequency = None
        self.tag = None
        self.cov = None
        self._var = None
        self._error = None
        self._data = None
        self.otype = None

    @property
    @req_attr
    def name(self):
        return(self.NAME)

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        # Converts the frequency to a value in GHz
        if isinstance(frequency, u.Quantity):
            frequency = frequency.to_value(u.GHz, equivalencies=u.spectral())

        self._frequency = frequency

    @property
    def data(self):
        """ Array in the shape (1, N) """
        return self._data[np.newaxis, :]

    @property
    def key(self):
        """Key used in the Observables_dictionary """
        return (self.name, self.frequency, self.Nside, self.tag)


    @property
    def var(self):
        if (self.cov is None) and (self._error is not None):
            self._var = np.ones(self._data.shape) * self._error**2
        return self._var


class TabularDataset(Dataset):
    """
    Base class for tabular datasets, where the data is input in either
    in a Python dictionary-like object
    (:py:class:`dict`, :py:class:`astropy.table.Table`,
    :py:class:`pandas.DataFrame`, etc).

    Parameters
    ----------
    data : dict_like
        Can be a :py:class:`dict`, :py:class:`astropy.table.Table`,
        :py:class:`pandas.DataFrame`, or similar object
        containing the data.
    name : str
        Standard name of this type of observable. E.g. 'fd', 'sync', 'dm'.

    Optional
    --------
    data_col : str or None. Default: None
        Key used to access the relevant dataset from the provided data
        (i.e. data[data_column]).
        If *None*, this value is equal to `name`.
    units : :obj:`~astropy.units.Unit` object, str or None. Default: None
        Units used for the data.
        If *None*, the units are inferred from the given `data_column`.
    coords_type : {'galactic'; 'cartesian'} or None. Default: None
        Type of coordinates used.
        If *None*, type is inferred from present coordinate columns.
    lon_col, lat_col : str. Default: ('lon', 'lat')
        Key used to access the Galactic longitudes/latitudes (in deg) from
        `data`.
    x_col, y_col, z_col : str. Default: ('x', 'y', 'z')
        Keys used to access the coordinates (in kpc) from `data`.
    err_col : str or None. Default: None
        The key used for accessing the error for the data values.
        If *None*, no errors are used.
    frequency : :obj:`~astropy.units.Quantity` object or None. Default: None
        Frequency of the measurement (if relevant)
    tag : str or None. Default: None
        Extra information associated with the observable.
          * 'I' - total intensity (in unit K-cmb)
          * 'Q' - Stokes Q (in unit K-cmb, IAU convention)
          * 'U' - Stokes U (in unit K-cmb, IAU convention)
          * 'PI' - polarisation intensity (in unit K-cmb)
          * 'PA' - polarisation angle (in unit rad, IAU convention)
    """
    def __init__(self, data, name, *, data_col=None, units=None,
                 coords_type=None, lon_col='lon', lat_col='lat',
                 x_col='x', y_col='y', z_col='z', err_col=None,
                 frequency=None, tag=None):
        # Set name
        self.NAME = name

        # Call super constructor
        super().__init__()

        # If data_col is None, set it to name
        if data_col is None:
            data_col = name

        # Obtain data column
        self._data = u.Quantity(data[data_col], units)
        units = self._data.unit

        # Determine the keys for galactic and cartesian coordinates
        gal_keys = {lon_col, lat_col}
        cart_keys = {x_col, y_col, z_col}

        # If coords_type is None, attempt to infer type from provided data
        if coords_type is None:
            # Obtain data keys
            keys = data.keys()

            # If only cart_keys are present, set type to 'cartesian'
            if cart_keys.issubset(keys) and not gal_keys.issubset(keys):
                coords_type = 'cartesian'

            # Else, if only gal_keys are present, set type to 'galactic'
            elif gal_keys.issubset(keys) and not cart_keys.issubset(keys):
                coords_type = 'galactic'

        # Obtain coordinates
        if coords_type is None:
            pass
        elif(coords_type.lower() == 'galactic'):
            self.coords = {'type': 'galactic',
                           'lon': u.Quantity(data[lon_col], unit=u.deg),
                           'lat': u.Quantity(data[lat_col], unit=u.deg)}
        elif(coords_type.lower() == 'cartesian'):
            self.coords = {'type': 'cartesian',
                           'x': u.Quantity(data[x_col], unit=u.kpc),
                           'y': u.Quantity(data[y_col], unit=u.kpc),
                           'z': u.Quantity(data[z_col], unit=u.kpc)}
        else:
            raise ValueError('Unknown coordinates_type!')

        # Obtain errors if provided
        if err_col is not None:
            self._error = u.Quantity(data[err_col], units)

        # Save provided frequency and tag
        self.frequency = frequency
        self.tag = tag

        # Set Nside
        self.Nside = "tab"
        self.otype = "tabular"

class ImageDataset(Dataset):
    """
    Class for simple non-full-sky image data
    """
    def __init__(self, data, name, lon_min, lon_max, lat_min, lat_max,
                 object_id=None, units=None, error=None, cov=None,
                 frequency=None, tag=None):
        self.NAME = name
        super().__init__()


        self.frequency = frequency
        self.object_id = object_id
        self.tag = tag

        self.coords = {'type': 'galactic',
                        'lon_min': lon_min << u.deg,
                        'lon_max': lon_max << u.deg,
                        'lat_min': lat_min << u.deg,
                        'lat_max': lat_max << u.deg,
                        'shape': data.shape}
        self.otype = 'image'
        self.Nside = 'image'

        assert len(data.shape) == 2
        self._data = data.ravel()

        if cov is not None:
            assert error is None

            self._cov = distribute_matrix(cov)
        else:
            self._error = error


class HEALPixDataset(Dataset):
    """
    Base class for HEALPix datasets, which are input as
    a simple 1D-array without explicit coordinate information
    """
    def __init__(self, data, error=None, cov=None, Nside=None):
        super().__init__()

        dataNside = np.uint(np.sqrt(data.size/12))
        if Nside is None:
            Nside = dataNside
        try:
            assert 12*int(Nside)**2 == data.size
        except AssertionError:
            print(12*int(Nside)**2, data.size)
            raise

        self.Nside = Nside
        self.otype = 'HEALPix'

        assert len(data.shape) == 1
        self._data = data

        if cov is not None:
            assert error is None

            self.cov = distribute_matrix(cov)
        else:
            self._error = error


class FaradayDepthHEALPixDataset(HEALPixDataset):
    r"""
    Stores a Faraday depth map into an IMAGINE-compatible
    dataset

    Parameters
    ----------
    data : numpy.ndarray
        1D-array containing the HEALPix map
    Nside : int, optional
        For extra internal consistency checking. If `Nside` is present,
        it will be checked whether :math:`12\times N_{side}^2` matches
    error : float or array
        If errors are uncorrelated, this can be used to specify them
        (a diagonal covariance matrix will then be constructed).
    cov : numpy.ndarray
        2D-array containing the covariance matrix

    Attributes
    -------
    data
        Data in ObservablesDict-compatible shape
    key
        Standard key associated with this observable
    """

    # Class attributes
    NAME = 'fd'


class DispersionMeasureHEALPixDataset(HEALPixDataset):
    r"""
    Stores a dispersion measure map into an IMAGINE-compatible
    dataset

    Parameters
    ----------
    data : numpy.ndarray
        1D-array containing the HEALPix map
    Nside : int, optional
        For extra internal consistency checking. If `Nside` is present,
        it will be checked whether :math:`12\times N_{side}^2` matches
    error : float or array
        If errors are uncorrelated, this can be used to specify them
        (a diagonal covariance matrix will then be constructed).
    cov : numpy.ndarray
        2D-array containing the covariance matrix

    Attributes
    -------
    data
        Data in ObservablesDict-compatible shape
    key
        Standard key associated with this observable
    """

    # Class attributes
    NAME = 'dm'


class SynchrotronHEALPixDataset(HEALPixDataset):
    r"""
    Stores a synchrotron emission map into an IMAGINE-compatible
    dataset. This can be Stokes parameters, total and polarised
    intensity, and polarisation angle.

    The parameter `typ` and the units of the map in `data` must follow:

    * 'I' - total intensity (in unit K-cmb)
    * 'Q' - Stokes Q (in unit K-cmb, IAU convention)
    * 'U' - Stokes U (in unit K-cmb, IAU convention)
    * 'PI' - polarisation intensity (in unit K-cmb)
    * 'PA' - polarisation angle (in unit rad, IAU convention)

    Parameters
    ----------
    data : numpy.ndarray
      1D-array containing the HEALPix map
    frequency : astropy.units.Quantity
        Frequency of the measurement (if relevant)
    Nside : int, optional
      For extra internal consistency checking. If `Nside` is present,
      it will be checked whether :math:`12\times N_{side}^2` matches data.size
    typ : str
      The type of map being supplied in `data`.
    error : float or array
        If errors are uncorrelated, this can be used to specify them
        (a diagonal covariance matrix will then be constructed).
    cov : numpy.ndarray
        2D-array containing the covariance matrix

    Attributes
    -------
    data
      Data in ObservablesDict-compatible shape
    key
      Standard key associated with this observable
    """

    # Class attributes
    NAME = 'sync'

    def __init__(self, data, frequency, typ, **kwargs):
        super().__init__(data, **kwargs)

        self.frequency = frequency

        # Checks whether the typ is valid
        assert typ in ['I', 'Q', 'U', 'PI', 'PA']
        self.tag = typ
