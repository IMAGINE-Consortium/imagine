"""
for convenience we define dictionary of Observable objects as
ObservableDict and inherit from which,
we define Measurements, Covariances, Simulations and Masks
for storing:
    * measured data sets
    * measured/simulated covariances
    * simulated ensemble sets
    * mask "maps" (but actually mask lists)
separately

observable name/unit convention:
    
    * ('fd','nan',str(size/Nside),'nan')
        Faraday depth (in unit ?

    * ('dm','nan',str(pix),'nan')
        dispersion measure (in unit ?

    * ('sync',str(freq),str(pix),X)
        synchrotron emission
        
        X can be:
            * 'I', total intensity (in unit K-cmb)
            * 'Q', Stokes Q (in unit K-cmb, IAU convention)
            * 'U', Stokes U (in unit K-cmb, IAU convention)
            * 'PI', polarisation intensity (in unit K-cmb)
            * 'PA', polarisation angle (in unit rad, IAU convention)

remarks on the observable tags:
    
    str(freq), polarisation-related-flag are redundant for Faraday depth and dispersion measure
    so we put 'nan' instead
    
    str(pix/nside) stores either Healpix Nisde, or just number of pixels/points
    we do this for flexibility, in case users have non-HEALPix-map-like in/output

masking convention:
    
    masked erea associated with pixel value 0,
    unmasked area with pixel value 1
    
masking method:
    
    mask only applies to observables/covariances
    observable after masking will be re-recorded as plain data type

distribution with MPI:
    
    * all data are either distributed or copied, where "copied" means each node
    stores the identical copy which is convenient for hosting measured data and
    mask map
    
    * Covariances has Field object with global shape "around" 
    (data_size//mpisize, data_size) "around" means to distribute matrix correctly
    as described in "imagine/tools/mpi_helper.py"
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
        add/update name and data
        
        parameters
        ----------
        
        name
            str tuple
            name should follow convention
            (data-name,str(data-freq),str(data-Nside/size),str(ext))
            if data is independent from frequency, set 'nan'
            ext can be 'I','Q','U','PI','PA', 'nan' or other customized tags

        data
            distributed/copied ndarray/Observable

        plain
            if True, means unstructured data
            if False(default), means HEALPix-like sky map
        """
        pass

    def apply_mask(self, mask_dict):
        """
        apply mask maps
        
        parameters
        ----------
        
        mask_dict
            Masks object
        """
        pass


@icy
class Masks(ObservableDict):

    def __init__(self):
        super(Masks, self).__init__()

    def append(self, name, new, plain=False):
        """
        
        parameters
        ----------
        
        name
            tag tuple
            
        data
            numpy.ndarray or Observable
            distributed mask data
            
        plain
            denoting either plain or HEALPix-like data
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

    def __init__(self):
        super(Measurements, self).__init__()

    def append(self, name, new, plain=False):
        """
        
        parameters
        ----------
        
        name
            tag tuple
            
        new
            numpy.ndarray or Observable
            distributed mask data
            
        plain
            denoting either plain or HEALPix-like data
        """
        log.debug('@ observable_dict::Measurements::append')
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

    def __init__(self):
        super(Simulations, self).__init__()

    def append(self, name, new, plain=False):
        """
        
        parameters
        ----------
        
        name
            tag tuple
            
        new
            numpy.ndarray or Observable
            distributed mask data
            
        plain
            denoting either plain or HEALPix-like data
        """
        log.debug('@ observable_dict::Simulations::append')
        assert (len(name) == 4)
        if name in self._archive.keys():  # app
            self._archive[name].rw_flag = False
            self._archive[name].append(new)
        else:  # new
            if isinstance(new, Observable):
                if plain:
                    assert (new.size == np.uint(name[2]))
                else:
                    assert (new.size == 12*np.uint(name[2])**2)
                self._archive.update({name: new})
            elif isinstance(new, np.ndarray):  # distributed data
                if plain:
                    assert (new.shape[1] == np.uint(name[2]))
                else:
                    assert (new.shape[1] == 12*np.uint(name[2])**2)
                self._archive.update({name: Observable(new, 'simulated')})
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

    def __init__(self):
        super(Covariances, self).__init__()

    def append(self, name, new, plain=False):
        """
        
        parameters
        ----------
        
        name
            tag tuple
            
        new
            numpy.ndarray
            distributed covariance matrix
            
        plain
            denoting either plain or HEALPix-like data
        """
        log.debug('@ observable_dict::Covariances::append')
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
