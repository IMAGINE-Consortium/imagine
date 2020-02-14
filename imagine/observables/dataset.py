from imagine.tools.icy_decorator import icy
import numpy as np

class Dataset:
    """
    Base class for writing helpers to convert arbitrary
    observational datasets into  IMAGINE's standardized format
    """
    def __init__(self):
        self.coords = None
    @property
    def name(self):
        return None

    @property
    def data(self):
        raise NotImplemented

    @property
    def key(self):
        raise NotImplemented

@icy
class HEALPixDataset(Dataset):
    """
    Base class for HEALPix datasets, which are input as
    a simple 1D-array without explicit coordinate information
    """
    def __init__(self, data, Nside=None):
        super().__init__()

        self.frequency = 'nan'
        self.tag = 'nan'

        dataNside = np.uint(np.sqrt(data.size/12))
        if Nside is None:
            Nside = dataNside
        assert Nside == dataNside
        self.Nside = str(Nside)

        assert len(data.shape)==1
        self._data = data

    @property
    def data(self):
        """ Array in the shape (1, N) """
        return data[np.newaxis,:]

    @property
    def key(self):
        """Key used in the Observables_dictionary """
        return (self.name,self.frequency, self.Nside, self.tag)

@icy
class FaradayDepthHEALPixDataset(HEALPixDataset):
      r"""
      Stores a Faraday depth map into IMAGINE-compatible
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
      Stores a Dispersion measure map into IMAGINE-compatible
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
      Stores a synchrotron emission maps into IMAGINE-compatible
      dataset. These include Stokes parameters, total and polarised
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
          Frequency of the radio observation in $\rm cm$

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


