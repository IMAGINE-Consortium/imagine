"""
This module defines methods related to masking out distributed data
and/or the associated covariance matrix.
For the testing suits, please turn to "imagine/tests/tools_tests.py".

Implemented with numpy.ndarray raw data.
"""

# %% IMPORTS
# Built-in imports
from copy import deepcopy
import logging as log

# Package imports
import numpy as np

# IMAGINE imports
from imagine.tools.mpi_helper import mpi_arrange

# All declaration
__all__ = ['mask_cov', 'mask_obs']


# %% FUNCTION DEFINITIONS
def mask_obs(obs, mask):
    """
    Applies a mask to an observable.

    Parameters
    ----------
    data : distributed numpy.ndarray
        Ensemble of observables, in global shape (ensemble size, data size)
        each node contains part of the global rows.

    mask : numpy.ndarray
        Copied mask map in shape (1, data size) on each node.

    Returns
    -------
    numpy.ndarray
        Masked observable of shape (ensemble size, masked data size).
    """
    log.debug('@ masker::mask_data')
    assert (obs.shape[0] >= 1)
    assert (mask.shape[0] == 1)
    assert (obs.shape[1] == mask.shape[1])

    # Creates a boolean mask
    bool_mask = mask[0].astype(bool)

    return obs[:, bool_mask]


def mask_cov(cov, mask):
    """
    Applies mask to the observable covariance.

    Parameters
    ----------
    cov : (distributed) numpy.ndarray
        Covariance matrix of observables in global shape (data size, data size)
        each node contains part of the global rows
        (if `imagine.rc['distributed_arrays']=True`).
    mask : numpy.ndarray
        Copied mask map in shape (1, data size).

    Returns
    -------
    masked_cov : numpy.ndarray
        Masked covariance matrix of shape (masked data size, masked data size).
    """
    log.debug('@ masker::mask_cov')
    assert (mask.shape[0] == 1)
    assert (cov.shape[1] == mask.shape[1])

    # Creates a 1D boolean mask
    bool_mask_1D = mask[0].astype(bool)
    # Constructs a 2D boolean mask and replaces 1D mask
    bool_mask = np.outer(bool_mask_1D, bool_mask_1D)

    # If mpi distributed_arrays are being used, the shape of the mask
    # needs to be adjusted, as each node accesses only some rows
    row_min, row_max = mpi_arrange(bool_mask_1D.size)
    nrows, ncolumns = bool_mask_1D[row_min:row_max].sum(), bool_mask_1D.sum()
    bool_mask = bool_mask[row_min:row_max, :]

    # Applies the mask and reshapes
    masked_cov = cov[bool_mask].reshape((nrows, ncolumns))

    return masked_cov
