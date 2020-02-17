"""
For convenience we define dictionary of Observable objects as
ObservableDict from which one can define define the classes Measurements,
Covariances, Simulations and Masks, which can be used to store:

    * measured data sets
    * measured/simulated covariances
    * simulated ensemble sets
    * mask "maps" (but actually mask lists)


Conventions for observables entries
    * **Faraday depth:** `('fd','nan',str(size/Nside),'nan')`
    * **Dispersion measure**: `('dm','nan',str(pix),'nan')`
    * **Synchrotron emission**: `('sync',str(freq),str(pix),X)`
        where X stands for:
            * 'I' - total intensity (in unit K-cmb)
            * 'Q' - Stokes Q (in unit K-cmb, IAU convention)
            * 'U' - Stokes U (in unit K-cmb, IAU convention)
            * 'PI' - polarisation intensity (in unit K-cmb)
            * 'PA' - polarisation angle (in unit rad, IAU convention)

    Remarks:
        * `str(freq)`, polarisation-related-flag are redundant for Faraday depth
            and dispersion measure so we put 'nan' instead
        * `str(pix/nside)` stores either Healpix Nside, or just number of
            pixels/points we do this for flexibility, in case users have non-HEALPix
            based in/output

Remarks on the observable tags
    str(freq), polarisation-related-flag are redundant for Faraday depth and dispersion measure so we put 'nan' instead

    str(pix/nside) stores either Healpix Nisde, or just number of pixels/points
    we do this for flexibility, in case users have non-HEALPix-map-like in/output

Masking convention
    masked erea associated with pixel value 0,
    unmasked area with pixel value 1

Masking method
    mask only applies to observables/covariances
    observable after masking will be re-recorded as plain data type


.. note:: Distribution with MPI

    * all data are either distributed or copied, where "copied" means each node
      stores the identical copy which is convenient for hosting measured data
      and mask map

    * Covariances has Field object with global shape "around"
      (data_size//mpisize, data_size) "around" means to distribute matrix
      correctly as described in "imagine/tools/mpi_helper.py"
"""
import numpy as np
import logging as log
from mpi4py import MPI

from imagine.observables.observable import Observable
from imagine.tools.masker import mask_obs, mask_cov
from imagine.tools.icy_decorator import icy


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

@icy
class ObservableDict(object):
    """
    Base class from which `imagine.observables.observable_dict.Measurements`,
    `imagine.observables.observable_dict.Covariances`, `imagine.observables.observable_dict.Simulations`
    and `imagine.observables.observable_dict.Masks` are derived.

    See `imagine.observables.observable_dict` module documentation for
    further details.
    """
    def __init__(self):
        self._archive = dict()

    @property
    def archive(self):
        return self._archive

    def keys(self):
        return self._archive.keys()

    def __getitem__(self, key):
        return self._archive[key]

    def append(self, name, data, plain=False):
        """
        Adds/updates name and data

        Parameters
        ----------
        name : str tuple
            Should follow the convention:
            ``(data-name,str(data-freq),str(data-Nside/size),str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U','PI','PA', 'nan' or other customized tags.
        data
            distributed/copied ndarray/Observable
        plain : bool
            If True, means unstructured data.
            If False (default case), means HEALPix-like sky map.
        """
        pass

    def apply_mask(self, mask_dict):
        """
        Parameters
        ----------
        mask_dict : imagine.observables.observable_dict.Masks
            Masks object
        """
        pass


@icy
class Masks(ObservableDict):
    """
    Stores HEALPix mask maps

    See `imagine.observables.observable_dict` module documentation for
    further details.
    """
    def __init__(self):
        super(Masks, self).__init__()

    def append(self, name, new, plain=False):
        """
        Adds/updates name and data

        Parameters
        ----------
        name : str tuple
            Should follow the convention:
            ``(data-name,str(data-freq),str(data-Nside/size),str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U','PI','PA', 'nan' or other customized tags.
        data
            distributed/copied ndarray/Observable
        plain : bool
            If True, means unstructured data.
            If False (default case), means HEALPix-like sky map.
        """
        log.debug('@ observable_dict::Masks::append')
        assert (len(name) == 4)
        if isinstance(new, Observable):
            assert (new.dtype == 'measured')
            if plain:
                assert (new.shape[1] == np.uint(name[2]))
            else:
                assert (new.size == 12*np.uint(name[2])**2)
            self._archive.update({name: new})
        elif isinstance(new, np.ndarray):
            assert (new.shape[0] == 1)
            if plain:
                assert (new.shape[1] == np.uint(name[2]))
            else:
                assert (new.shape[1] == 12*np.uint(name[2])*np.uint(name[2]))
            self._archive.update({name: Observable(new, 'measured')})
        else:
            raise TypeError('unsupported data type')


@icy
class Measurements(ObservableDict):
    """
    Stores observational data sets

    See `imagine.observables.observable_dict` module documentation for
    further details.
    """
    def __init__(self):
        super(Measurements, self).__init__()

    def append(self, name=None, new=None, plain=False, dataset=None):
        """
        Adds/updates name and data

        Parameters
        ----------
        name : str tuple
            Should follow the convention:
            ``(data-name,str(data-freq),str(data-Nside/size),str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U','PI','PA', 'nan' or other customized tags.
        data
            distributed/copied ndarray/Observable
        plain : bool
            If True, means unstructured data.
            If False (default case), means HEALPix-like sky map.
        """
        log.debug('@ observable_dict::Measurements::append')
        
        if dataset is not None:
            name = dataset.key
            new = dataset.data
        
        assert (len(name) == 4)
        if isinstance(new, Observable):
            assert (new.dtype == 'measured')
            if plain:
                assert (new.size == np.uint(name[2]))
            else:
                assert (new.size == 12*np.uint(name[2])**2)
            self._archive.update({name: new})  # rw
        elif isinstance(new, np.ndarray):
            if plain:
                assert (new.shape[1] == np.uint(name[2]))
            else:
                assert (new.shape[1] == 12*np.uint(name[2])**2)
            self._archive.update({name: Observable(new, 'measured')})  # rw
        else:
            raise TypeError('unsupported data type')

    def apply_mask(self, mask_dict=None):
        log.debug('@ observable_dict::Measurements::apply_mask')
        if mask_dict is None:
            pass
        else:
            assert isinstance(mask_dict, Masks)
            for name, msk in mask_dict._archive.items():
                if name in self._archive.keys():
                    masked = mask_obs(self._archive[name].data, msk.data)
                    new_name = (name[0], name[1], str(masked.shape[1]), name[3])
                    self._archive.pop(name, None)  # pop out obsolete data
                    self.append(new_name, masked, plain=True)  # append new as plain data


@icy
class Simulations(ObservableDict):
    """
    Stores simulated ensemble sets

    See `imagine.observables.observable_dict` module documentation for
    further details.
    """
    def __init__(self):
        super(Simulations, self).__init__()

    def append(self, name, new_data, plain=False):
        """
        Adds/updates name and data

        Parameters
        ----------
        name : str tuple
            Should follow the convention:
            ``(data-name,str(data-freq),str(data-Nside/size),str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U','PI','PA', 'nan' or other customized tags.
        new_data
            distributed/copied ndarray/Observable
        plain : bool
            If True, means unstructured data.
            If False (default case), means HEALPix-like sky map.
        """
        log.debug('@ observable_dict::Simulations::append')
        assert (len(name) == 4)
        if name in self._archive.keys():  # app
            self._archive[name].rw_flag = False
            self._archive[name].append(new_data)
        else:  # new_data
            if isinstance(new_data, Observable):
                if plain:
                    assert (new_data.size == np.uint(name[2]))
                else:
                    assert (new_data.size == 12*np.uint(name[2])**2)
                self._archive.update({name: new_data})
            elif isinstance(new_data, np.ndarray):  # distributed data
                if plain:
                    assert (new_data.shape[1] == np.uint(name[2]))
                else:
                    assert (new_data.shape[1] == 12*np.uint(name[2])**2)
                self._archive.update({name: Observable(new_data, 'simulated')})
            else:
                raise TypeError('unsupported data type')

    def apply_mask(self, mask_dict):
        log.debug('@ observable_dict::Simulations::apply_mask')
        if mask_dict is None:
            pass
        else:
            assert isinstance(mask_dict, Masks)
            for name, msk in mask_dict._archive.items():
                if name in self._archive.keys():
                    masked = mask_obs(self._archive[name].data, msk.data)
                    new_name = (name[0], name[1], str(masked.shape[1]), name[3])
                    self._archive.pop(name, None)  # pop out obsolete
                    self.append(new_name, masked, plain=True)  # append new as plain data


@icy
class Covariances(ObservableDict):
    """
    Stores observational covariances

    See `imagine.observables.observable_dict` module documentation for
    further details.
    """
    def __init__(self):
        super(Covariances, self).__init__()

    def append(self, name=None, new=None, plain=False, dataset=None):
        """
        Adds/updates name and data

        Parameters
        ----------
        name : str tuple
            Should follow the convention:
            ``(data-name,str(data-freq),str(data-Nside/size),str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U','PI','PA', 'nan' or other customized tags.
        data
            distributed/copied ndarray/Observable
        plain : bool
            If True, means unstructured data.
            If False (default case), means HEALPix-like sky map.
        """
        log.debug('@ observable_dict::Covariances::append')
        
        if dataset is not None:
            name = dataset.key
            new = dataset.cov        
        
        assert (len(name) == 4)
        if isinstance(new, Observable):  # always rewrite
            if plain:
                assert (new.size == np.uint(name[2]))
            else:
                assert (new.size == 12*np.uint(name[2])**2)
            self._archive.update({name: new})  # rw
        elif isinstance(new, np.ndarray):
            if plain:
                assert (new.shape[1] == np.uint(name[2]))
            else:
                assert (new.shape[1] == 12*np.uint(name[2])**2)
            self._archive.update({name: Observable(new, 'covariance')})
        else:
            raise TypeError('unsupported data type')

    def apply_mask(self, mask_dict):
        log.debug('@ observable_dict::Covariances::apply_mask')
        if mask_dict is None:
            pass
        else:
            assert isinstance(mask_dict, Masks)
            for name, msk in mask_dict._archive.items():
                if name in self._archive.keys():
                    masked = mask_cov(self._archive[name].data, msk.data)
                    new_name = (name[0], name[1], str(masked.shape[1]), name[3])
                    self._archive.pop(name, None)  # pop out obsolete
                    self.append(new_name, masked, plain=True)  # append new as plain data
