"""
methods related to masking out HEALPix maps
and associated covariance matrix

implemented with numpy ndarray raw data
"""

import numpy as np
from copy import deepcopy


def mask_obs(_obs, _mask):
    """
    mask observable sample
    :param _obs: ensemble of observables, in shape (ensemble_size, data_size)
    :param _mask: mask map in shape (1, data_size)
    :return: masked sample
    """
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
    assert (raw_cov.shape[0] == raw_mask.shape[1])
    idx = int(0)
    for ptr in raw_mask[0]:
        if ptr == 0:
            raw_cov = np.delete(raw_cov, idx, 0)
            raw_cov = np.delete(raw_cov, idx, 1)
        else:
            idx += int(1)
    assert (raw_cov.shape[0] == raw_cov.shape[1])
    assert (raw_cov.shape[0] == idx)
    return raw_cov
