"""
Datasets are auxiliary classes used to facilitate the reading and inclusion of 
observational data in the IMAGINE pipeline"""

from imagine.tools.icy_decorator import icy
import numpy as np
from astropy import units as u

class Dataset:
    """
    Base class for writing helpers to convert arbitrary
    observational datasets into  IMAGINE's standardized format
    """
    def __init__(self):
        self.coords = None
        self.Nside = None
        self.frequency = 'nan'
        self.tag = 'nan'
        self._cov = None
        self._error = None
        self._data = None
        
    @property
    def name(self):
        return None

    @property
    def data(self):
        """ Array in the shape (1, N) """
        return self._data[np.newaxis,:]

    @property
    def key(self):
        """Key used in the Observables_dictionary """
        return (self.name,self.frequency, self.Nside, self.tag)
    
    @property
    def cov(self):
        if self._cov is None:
            self._cov = self._error * np.eye(self._data.size)
        return self._cov
            

@icy 
class TabularDataset(Dataset):
    """
    Base class for tabular datasets, where the data is input in either
    in a Python dictionary-like object 
    (astropy.Tables, pandas.DataFrame, etc).
    
    Parameters
    ----------
    data : dictionary-like
        astropy.Tables, pandas.DataFrame, or similar object
        containing the data.
    data_column : str
        Key used to access the relevant dataset from the provided data
        (i.e. data[data_column]).
    units : astropy.units.Unit or str
        Units used for the data.
    coordinates_type : str
        Type of coordinates used. Can be 'galactic' or 'cartesian'.
    lon_column : str
        Key used to access the Galactic longitudes (in deg) from
        `data`.
    lat_column : str
        Key used to access the Galactic latitudes (in deg) from
        `data`.
    lat_column : str
        Key used to access the Galactic latitudes (in deg) from
        `data`.
    x_column, y_column, z_column : str
        Keys used to access the coordinates (in kpc) from
        `data`.    
    frequency : str
        String with the frequency of the measurement in GHz (if relevant)
    tag : str
    """
    def __init__(self, data, name, data_column=None, units=None,
                 coordinates_type='galactic', lon_column='lon', lat_column='lat', 
                 x_column='x', y_column='y', z_column='z',
                 error_column=None, frequency='nan', tag='nan'):
        super().__init__()
        if data_column is None:
            data_column=name
            
        self._name = name
        self._data = u.Quantity(data[data_column], units, copy=False)
        if coordinates_type == 'galactic':
            self.coords = {'type': coordinates_type,
                           'lon': u.Quantity(data[lon_column], unit=u.deg), 
                           'lat': u.Quantity(data[lat_column], unit=u.deg)}
        elif coordinates_type == 'cartesian':
            self.coords = {'type': coordinates_type,
                           'x': u.Quantity(data[x_column], unit=u.kpc), 
                           'y': u.Quantity(data[y_column], unit=u.kpc),
                           'z': u.Quantity(data[z_column], unit=u.kpc)}
        elif coordinates_type == None:
            pass
        else:
            raise ValueError('Unknown coordinates_type!')
            
        self.frequency = str(frequency)
        self.tag = tag
        if error_column is not None:
            self._error = np.array(data[error_column])
        
        self.Nside = "tab"
    @property
    def name(self):
        return self._name


@icy
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
            
        self.Nside = str(Nside)
        assert len(data.shape)==1
        self._data = data
        self._error = error

@icy
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

    Attributes
    -------
    data
        Data in ObservablesDict-compatible shape
    key
        Standard key associated with this observable
    """
    @property
    def name(self):
        return 'fd'

@icy
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

    Attributes
    -------
    data
        Data in ObservablesDict-compatible shape
    key
        Standard key associated with this observable
    """
    @property
    def name(self):
        return 'dm'

@icy
class SynchrotronHEALPixDataset(HEALPixDataset):
    r"""
    Stores a synchrotron emission map into an IMAGINE-compatible
    dataset. This can be Stokes parameters, total and polarised
    intensity, and polarisation angle.

    The parameter `type` and the units of the map in `data` must follow:

    * 'I' - total intensity (in unit K-cmb)
    * 'Q' - Stokes Q (in unit K-cmb, IAU convention)
    * 'U' - Stokes U (in unit K-cmb, IAU convention)
    * 'PI' - polarisation intensity (in unit K-cmb)
    * 'PA' - polarisation angle (in unit rad, IAU convention)

    Parameters
    ----------
    data : numpy.ndarray
      1D-array containing the HEALPix map
    frequency : float
      Frequency of the radio observation in $\rm GHz$
    Nside : int, optional
      For extra internal consistency checking. If `Nside` is present,
      it will be checked whether :math:`12\times N_{side}^2` matches data.size
    type : str
      The type of map being supplied in `data`.

    Attributes
    -------
    data
      Data in ObservablesDict-compatible shape
    key
      Standard key associated with this observable
    """
    def __init__(self, data, frequency, type, Nside=None):
        super().__init__(data, Nside)

        self.frequency = str(frequency)
        # Checks whether the type is valid
        assert type in ['I','Q','U','PI','PA']
        self.tag = type
    @property
    def name(self):
        return 'sync'

