"""
This module contains a set of tools for manipulating observables
"""
# %% IMPORTS

# Package imports
from e13tools import add_to_all
import numpy as np

# IMAGINE imports
import imagine as img

# All declaration
__all__ = []


# %% FUNCTION DEFINITIONS
@add_to_all
def estimate_covariances(simulations, cov_est=img.tools.oas_cov):
    """
    Convenience function that, given a Simulations object,
    produces a Covariances object.

    Parameters
    ----------
    simulations : imagine.observables.Simulations
        Simulations
    cov_est : func
        A function that computes the covariance given a
        matrix of data

    Returns
    -------
    covs : imagine.observables.Covariances
        IMAGINE Covariances object
    """
    assert isinstance(simulations, img.observables.Simulations)

    covs = img.observables.Covariances()
    for k in simulations.keys():
        obs = simulations[k]
        d = cov_est(obs.data)*obs.unit
        covs.append(name=k, cov_data=d, otype=obs.otype)

    return covs

@add_to_all
def extract_simulation_subset(sims, indices):
    """
    Creates a new :py:obj:`Simulations <imagine.observables.Simulations>` object
    based on a subset of the ensemble of a larger :py:obj:`Simulations <imagine.observables.Simulations>`.

    Parameters
    ----------
    sims : imagine.observables.Simulations
        A `Simulations` object containing an ensemble
    indices
        A tuple of indices numbers, a slice object or a boolean array which
        will be used to select the data for the sub-simulation

    Returns
    -------
    sims_subset : imagine.observables.Simulations
        The selected sub-simulation
    """
    assert isinstance(sims, img.observables.Simulations)

    sims_subset = img.observables.Simulations()

    for k in sims.keys():
        sims_subset.append(name=k,
                          data=sims[k].global_data[indices,:],
                          otype=sims[k].otype)
    return sims_subset
