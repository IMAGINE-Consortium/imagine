"""
For convenience we define dictionary of Observable objects as
ObservableDict from which one can define define the classes Measurements,
Covariances, Simulations and Masks, which can be used to store:

    * measured data sets
    * measured/simulated covariances
    * simulated ensemble sets
    * mask "maps"

Conventions for observables key values
    * **Faraday depth:** `('fd', None, size/Nside, None)`
    * **Dispersion measure**: `('dm', None, size/Nsize, None)`
    * **Synchrotron emission**: `('sync', freq, size/Nside, tag)`
        where tag stands for:
            * 'I' - total intensity (in unit K-cmb)
            * 'Q' - Stokes Q (in unit K-cmb, IAU convention)
            * 'U' - Stokes U (in unit K-cmb, IAU convention)
            * 'PI' - polarisation intensity (in unit K-cmb)
            * 'PA' - polarisation angle (in unit rad, IAU convention)

    Remarks:
        * `freq`, frequency in GHz
        * `str(pix/nside)` stores either Healpix Nside, or just number of
            pixels/points

Masking convention
    masked area associated with pixel value 0,
    unmasked area with pixel value 1

Masking
    After applying a mask, the Observables change `otype` from
    `'HEALPix'` to `'plain'`.
"""
# %% IMPORTS
# Built-in imports
import abc
import logging as log

# Package imports
import numpy as np

# IMAGINE imports
from imagine.observables.dataset import Dataset, HEALPixDataset
from imagine.observables.observable import Observable
from imagine.tools import BaseClass, mask_cov, mask_var, mask_obs, req_attr, oas_cov
import imagine.tools.visualization as visu

# All declaration
__all__ = ['ObservableDict', 'Masks', 'Measurements', 'Simulations',
           'Covariances']


# %% CLASS DEFINITIONS
class ObservableDict(BaseClass, metaclass=abc.ABCMeta):
    """
    Base class from which :py:class:`Measurements`, :py:class:`Covariances`,
    :py:class:`Simulations` and class :py:class:`Masks` classes are derived.

    Parameters
    ----------
    datasets : imagine.observables.Dataset, optional
        If present, Datasets that are appended to this
        :py:obj:`ObservableDict` object after initialization.
    """
    def __init__(self, *datasets):
        # Call super constructor
        super().__init__()

        # Initialize archive dict
        self._archive = {}

        # Append all provided datasets
        for dataset in datasets:
            self.append(dataset=dataset)

    @property
    def archive(self):
        return self._archive

    def keys(self):
        return self._archive.keys()

    def __iter__(self):
        return iter(self._archive)

    def __getitem__(self, key):
        return self._archive[key]

    def show(self, **kwargs):
        """
        Shows the contents of this ObservableDict using
        :py:func:`imagine.tools.visualization.show_observable_dict`
        """
        return visu.show_observable_dict(self, **kwargs)

    @abc.abstractmethod
    def append(self, dataset=None, *, name=None, data=None, cov_data=None,
               otype=None, coords=None):
        """
        Adds/updates name and data

        Parameters
        ----------
        dataset : imagine.observables.dataset.Dataset
            The IMAGINE dataset already adjusts the format of the data and sets the
            adequate key. If `dataset` is present, all other arguments will be ignored.
        name : str tuple
            Should follow the convention:
            ``(data-name, data-freq, data-Nside/"tab", ext)``.
            If data is independent from frequency, set None.
            `ext` can be 'I','Q','U','PI','PA', None or other customized tags.
        data : numpy.ndarray or imagine.observables.observable.Observable
            distributed/copied :py:class:`numpy.ndarray` or :py:class:`Observable <imagine.observables.observable.Observable>`
        otype : str
            Type of observable. May be: 'HEALPix', for HEALPix-like sky map;
            'tabular', for tabular data; or 'plain' for unstructured data.
        coords : dict
            A dictionary containing the coordinates of tabular data
        """

        if dataset is not None:
            assert isinstance(dataset, Dataset)
            name = dataset.key
            data = dataset.data
            cov_data = dataset.cov
            if cov_data is None:
                cov_data = dataset.var
            otype = dataset.otype
            coords = dataset.coords

        assert (len(name) == 4), 'Wrong format for Observable key!'
        return name, data, cov_data, otype, coords



class Masks(ObservableDict):
    """
    Stores HEALPix mask maps.

    After the Masks dictionary is assembled it can be applied to any other
    observables dictionary to return a dictionary containing masked maps
    (see below).

    Example
    --------
    >>> import imagine.observables as obs
    >>> import numpy as np
    >>> meas, cov, mask = obs.Measurements(), obs.Covariances(), obs.Masks()
    >>> key = ('test','nan','4','nan')
    >>> meas.append(name=key, data=np.array([[1,2,3,4.]]), otype='plain')
    >>> mask.append(name=key, data=np.array([[1,2,3,4.]]), otype='plain')
    >>> masked_meas = mask(meas)
    >>> print(masked_meas[('test','nan','2','nan')].data)
    [[1. 2.]]
    >>> cov.append(name=key, cov_data=np.diag((1,2,3,4.)), otype='plain')
    >>> masked_cov = mask(cov)
    >>> print(masked_cov[('test',None, 2 ,None)].data)
    [[1. 0.]
     [0. 2.]]
    """

    def append(self, *args, **kwargs):
        log.debug('@ observable_dict::Masks::append')
        name, data, _, otype, _ = super().append(*args, **kwargs)

        if isinstance(data, Observable):
            assert (data.dtype == 'measured')
            if otype == 'HEALPix':
                assert (data.size == 12*np.uint(name[2])**2)
            self._archive.update({name: data})
        elif isinstance(data, np.ndarray):
            assert (data.shape[0] == 1)
            if otype == 'HEALPix':
                assert (data.shape[1] == _Nside_to_Npixels(name[2]))
            self._archive.update({name: Observable(data, 'measured')})
        else:
            raise TypeError('unsupported data type')

    def __call__(self, observable_dict):
        """
        Applies the masks

        Parameters
        ----------
        observable_dict : imagine.observables.ObservableDict
            Dictionary containing (some) entries where one wants to apply the
            masks.

        Returns
        -------
        masked_dict : imagine.observables.ObservableDict
            New observables dictionary containing masked entries (any entries
            in the original dictionary for which no mask was specified are
            referenced in `masked_dict` without modification).
        """
        assert isinstance(observable_dict,
                          (Measurements, Simulations, Covariances))

        # Creates an empty ObservableDict of the same type/subclass
        masked_dict = type(observable_dict)()

        for name, observable in observable_dict._archive.items():
            if name not in self._archive:
                # Saves reference to any observables where the masks are
                # not available
                if isinstance(observable_dict, Covariances):
                    masked_dict.append(name=name,
                                       cov_data=observable)
                else:
                    masked_dict.append(name=name,
                                       data=observable)
            else:
                # Reads the mask
                mask = self._archive[name].data
                # Applies appropriate function
                if isinstance(observable_dict, Covariances):
                    if observable_dict[name].dtype == 'variance':
                        masked_data = mask_var(observable_dict[name].var, mask)
                    else:
                        masked_data = mask_cov(observable_dict[name].data, mask)
                else:
                    masked_data = mask_obs(observable_dict[name].data, mask)
                # Prepares key
                if observable_dict[name].dtype != 'variance':
                    masked_shape = masked_data.shape[1]
                else:
                    masked_shape = masked_data.size

                new_name = (name[0], name[1], masked_shape, name[3])
                # Appends the masked Observable
                if isinstance(observable_dict, Covariances):
                    masked_dict.append(name=new_name, cov_data=masked_data,
                                       otype='plain')
                else:
                    masked_dict.append(name=new_name, data=masked_data,
                                       otype='plain')
                # Copies refs to units/coords
                masked_dict.coords = observable.coords
                masked_dict.unit = observable.unit

        return masked_dict


class Measurements(ObservableDict):
    """
    Stores observational data.


    Parameters
    ----------
    datasets : imagine.observables.Dataset, optional
        If present, Datasets that are appended to this
        :py:obj:`ObservableDict` object after initialization.

    Attributes
    ----------
    cov : imagine.observables.observable_dict.Covariances
        The :py:obj:`Covariances <imagine.observables.observable_dict.Covariances>`
        object associated with these measurements.
    """

    def __init__(self, *datasets):
        # Call super constructor
        self.cov = None
        super().__init__(*datasets)

    def append(self, *args, **kwargs):
        log.debug('@ observable_dict::Measurements::append')
        name, data, cov, otype, coords = super().append(*args, **kwargs)

        if cov is not None:
            if self.cov is None:
                self.cov = Covariances()
            self.cov.append(*args, **kwargs)

        if isinstance(data, Observable):
            assert (data.dtype == 'measured')
            self._archive.update({name: data})
        elif isinstance(data, np.ndarray):
            if otype == 'HEALPix':
                assert (data.shape[1] == _Nside_to_Npixels(name[2]))
            self._archive.update({name: Observable(data=data,
                                                   dtype='measured',
                                                   coords=coords,
                                                   otype=otype)})
        else:
            raise TypeError('Unsupported data type')


class Simulations(ObservableDict):
    """
    Stores simulated ensemble sets

    See `imagine.observables.observable_dict` module documentation for
    further details.
    """
    def append(self, *args, **kwargs):
        log.debug('@ observable_dict::Simulations::append')
        name, data, _, otype, coords = super().append(*args, **kwargs)

        if name in self._archive.keys():  # app
            self._archive[name].rw_flag = False
            self._archive[name].append(data)
        else:  # data
            if isinstance(data, Observable):
                self._archive.update({name: data})
            elif isinstance(data, np.ndarray):  # distributed data
                self._archive.update({name: Observable(data=data,
                                                       dtype='simulated',
                                                       coords=coords,
                                                       otype=otype)})
            else:
                raise TypeError('unsupported data type')


    def estimate_covariances(self, cov_est=oas_cov):
        """
        Produces a Covariances object based on the present Simulations

        Parameters
        ----------
        cov_est : func
            A function that computes the covariance given a
            matrix of data

        Returns
        -------
        covs : imagine.observables.Covariances
            IMAGINE Covariances object
        """
        covs = Covariances()

        for k in self:
            obs = self[k]
            d = cov_est(obs.data)*(obs.unit)**2
            covs.append(name=k, cov_data=d, otype=obs.otype)

        return covs


    def sub_sim(self, indices):
        """
        Creates a new :py:obj:`Simulations <imagine.observables.Simulations>` object
        based on a subset of the ensemble of a larger :py:obj:`Simulations <imagine.observables.Simulations>`.

        Parameters
        ----------
        indices
            A tuple of indices numbers, a slice object or a boolean array which
            will be used to select the data for the sub-simulation

        Returns
        -------
        sims_subset : imagine.observables.Simulations
            The selected sub-simulation
        """
        sims_subset = Simulations()

        for k in self:
            sims_subset.append(name=k,
                               data=self[k].global_data[indices,:],
                               otype=self[k].otype)
        return sims_subset


class Covariances(ObservableDict):
    """
    Stores observational covariances

    See `imagine.observables.observable_dict` module documentation for
    further details.
    """
    def append(self, *args, **kwargs):
        log.debug('@ observable_dict::Covariances::append')
        name, _, data, otype, coords = super().append(*args, **kwargs)

        if isinstance(data, Observable):
            self._archive.update({name: data})
        elif isinstance(data, np.ndarray):
            # Covariances case
            if len(data.shape)==2:
                if otype == 'HEALPix':
                    assert (data.shape[1] == _Nside_to_Npixels(name[2]))
                self._archive.update({name: Observable(data,
                                                       dtype='covariance',
                                                       coords=coords,
                                                       otype=otype)})
            # Variances case
            else:
                self._archive.update({name: Observable(data,
                                                       dtype='variance',
                                                       coords=coords,
                                                       otype=otype)})
        else:
            raise TypeError('unsupported data type')

    def show_variances(self, **kwargs):
        """
        Shows the contents of this ObservableDict using
        :py:func:`imagine.tools.visualization.show_observable_dict`
        """
        return visu.show_observable_dict(self, show_variances=True, **kwargs)

def _Nside_to_Npixels(Nside):
    return 12*int(Nside)**2
