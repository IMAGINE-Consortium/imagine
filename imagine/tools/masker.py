"""
This module defines methods related to masking out distributed data
and/or the associated covariance matrix.
For the testing suits, please turn to "imagine/tests/tools_tests.py".

Implemented with numpy.ndarray raw data.
"""
import numpy as np
from copy import deepcopy
import logging as log
from imagine.tools.mpi_helper import mpi_arrange


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
    assert isinstance(obs, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert (obs.shape[0] >= 1)
    assert (mask.shape[0] == 1)
    assert (obs.shape[1] == mask.shape[1])
    new_obs = deepcopy(obs)
    raw_mask = (deepcopy(mask)).astype(np.bool)
    #
    idx = int(0)
    for ptr in raw_mask[0]:
        if not ptr:
            new_obs = np.delete(new_obs, idx, 1)
        else:
            idx += int(1)
    assert (new_obs.shape[1] == idx)
    return new_obs

def mask_cov(cov, mask):
    """
    Applies mask to the observable covariance.

    Parameters
    ----------
    cov : distributed numpy.ndarray
        Covariance matrix of observalbes in global shape (data size, data size)
        each node contains part of the global rows.
    mask : numpy.ndarray
        Copied mask map in shape (1, data size).

    Returns
    -------
    numpy.ndarray
        Masked covariance matrix of shape (masked data size, masked data size).
    """
    log.debug('@ masker::mask_cov')
    assert isinstance(cov, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert (mask.shape[0] == 1)
    assert (cov.shape[1] == mask.shape[1])
    new_cov = deepcopy(cov)
    raw_mask = (deepcopy(mask)).astype(np.bool)
    # masking cols
    col_idx = int(0)
    for ptr in raw_mask[0]:
        if not ptr:
            new_cov = np.delete(new_cov, col_idx, 1)
        else:
            col_idx += int(1)
    assert (new_cov.shape[1] == col_idx)
    # masking rows
    row_idx = int(0)
    row_min, row_max = mpi_arrange(raw_mask.shape[1])
    for ptr in raw_mask[0, row_min : row_max]:
        if ptr == 0:
            new_cov = np.delete(new_cov, row_idx, 0)
        else:
            row_idx += int(1)
    return new_cov
