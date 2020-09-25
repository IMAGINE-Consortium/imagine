# %% IMPORTS
# Built-in imports
import os
from copy import copy, deepcopy
import logging as log
import warnings

# Package imports
import hickle
import dill
from mpi4py import MPI
import numpy as np

# IMAGINE imports
from imagine.tools import rc

__all__ = ['save_pipeline', 'load_pipeline']

# GLOBALS
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()

# %% FUNCTION DEFINITIONS
def save_pipeline(pipeline, use_dill=True):

    # Works on a (shallow) copy
    pipeline = copy(pipeline)

    # Adjusts paths: we want everything to be relative to run_directory
    # Hidden variables are used to avoid checks
    run_directory, pipeline._run_directory = pipeline._run_directory, '.'
    pipeline._chains_directory = os.path.relpath(pipeline._chains_directory,
                                                run_directory)

    # Adjusts observational data, if using distributed arrays
    if rc['distributed_arrays']:
        # Covariances need to be "undistributed"
        # First, makes sure we are working on a copy
        # (this is done as shallow as possible to save memory)
        pipeline.likelihood = copy(pipeline.likelihood)
        pipeline.likelihood.covariance_dict = deepcopy(pipeline.likelihood.covariance_dict)

        # Gathers all distributed data -- i.e. turns global (distributed) into local
        for k in pipeline.likelihood.covariance_dict.keys():
            pipeline.likelihood.covariance_dict[k]._data = pipeline.likelihood.covariance_dict[k].global_data
            # NB any processes with mpirank!=0 will store None in the above operation

    if mpirank == 0:
        if use_dill:
            with open(os.path.join(run_directory,'pipeline.pkl'), 'wb') as f:
                dill.dump(pipeline, f)
        else:
            hickle.dump(pipeline, os.path.join(run_directory,'pipeline.hkl'))

    return pipeline

def load_pipeline(directory_path='.', use_dill=True):
    if use_dill:
        with open(os.path.join(directory_path,'pipeline.pkl'), 'rb') as f:
            pipeline = dill.load(f)
    else:
        pipeline = hickle.load(os.path.join(directory_path, 'pipeline.hkl'))

    # Adjusts paths (hidden variables are used to avoid checks)
    pipeline._run_directory = os.path.join(directory_path, pipeline._run_directory)
    pipeline._chains_directory = os.path.join(directory_path, pipeline._chains_directory)

    # Adjust observational data, if using distributed arrays
    if rc['distributed_arrays']:
        # Distributes the covariance data
        for k in pipeline.likelihood.covariance_dict.keys():
            cov = pipeline.likelihood.covariance_dict[k]._data
            pipeline.likelihood.covariance_dict[k]._data = distribute_matrix(cov)

    return pipeline
