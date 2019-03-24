"""
methods related to masking out HEALPix maps
and associated covariance matrix

implemented with numpy ndarray raw data
"""

import numpy as np
from copy import deepcopy

from mpi4py import MPI

mpisize = MPI.COMM_WORLD.Get_size()
mpirank = MPI.COMM_WORLD.Get_rank()


def mask_obs(_obs, _mask):
    """
    mask observable sample
    :param _obs: ensemble of observables, in shape (ensemble_size, data_size)
    :param _mask: mask map in shape (1, data_size)
    :return: masked sample
    """
    if _obs.shape[0] != 0:  # if not empty
        raw_obs = deepcopy(_obs)
        raw_mask = deepcopy(_mask)
        assert (raw_mask.shape[0] == 1)
        assert (raw_obs.shape[1] == raw_obs.shape[1])
        idx = int(0)
        for ptr in raw_mask[0]:
            if ptr == 0:
                raw_obs = np.delete(raw_obs, idx, 1)
            else:
                idx += int(1)
        assert (raw_obs.shape[1] == idx)
        return raw_obs
    else:
        return np.zeros((0, _mask.sum()))

def mask_cov(_cov, _mask):
    """
    mask observable sample
    :param _cov: cov matrix of observables, in shape (data_size, data_size)
    :param _mask: mask map in shape (1, data_size)
    :return: masked sample
    """
    raw_cov = deepcopy(_cov)
    raw_mask = deepcopy(_mask)
    assert (raw_mask.shape[0] == 1)
    assert (raw_cov.shape[1] == raw_mask.shape[1])
    # masking cols
    col_idx = int(0)
    for ptr in raw_mask[0]:
        if ptr == 0:
            raw_cov = np.delete(raw_cov, col_idx, 1)
        else:
            col_idx += int(1)
    assert (raw_cov.shape[1] == col_idx)
    # masking rows
    row_idx = int(0)
    row_min, row_max = mpi_row_lim(raw_mask.shape[1], mpisize, mpirank)
    for ptr in raw_mask[0, row_min : row_max]:
        if ptr == 0:
            raw_cov = np.delete(raw_cov, row_idx, 0)
        else:
            row_idx += int(1)
    return raw_cov

def mpi_row_lim(_tot, _size, _rank):
    """
    global row index limit for covariance matrix on given node
    """
    res = min(_rank, _tot%_size)
    ave = _tot//_size
    return res + _rank*ave, res + (_rank+1)*ave + int(_rank < _tot%_size)
