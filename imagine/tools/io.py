# %% IMPORTS
# Built-in imports
import os
from copy import copy, deepcopy
import logging as log
import warnings

# Package imports
import cloudpickle
import hickle
from mpi4py import MPI
import numpy as np

# IMAGINE imports
from imagine.tools import rc
from imagine.tools.parallel_ops import distribute_matrix

__all__ = ['save_pipeline', 'load_pipeline']

# GLOBALS
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()

# %% FUNCTION DEFINITIONS
def save_pipeline(pipeline, use_hickle=False):
    """
    Saves the state of a Pipeline object

    Parameters
    ----------
    pipeline : imagine.pipelines.pipeline.Pipeline
        The pipeline object one would like to save
    use_hickle : bool
        If `False` (default) the state is saved using the `cloudpickle` package.
        Otherwise, experimental support to `hickle` is enabled.
    """
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
            # NB any process with mpirank!=0 will store None in the above operation

    # Hammurabi-specific path adjustment
    if hasattr(pipeline.simulator, 'hamx_path'):
        # In the case hamx path is the system default, it will use the
        # system default the next time it is loaded.
        pipeline.simulator = copy(pipeline.simulator)
        if pipeline.simulator.hamx_path == rc['hammurabi_hamx_path']:
            pipeline.simulator._hamx_path = None
            pipeline.simulator._ham._exe_path = None

    if mpirank == 0:
        if not use_hickle:
            with open(os.path.join(run_directory,'pipeline.pkl'), 'wb') as f:
                cloudpickle.dump(pipeline, f)
        else:
            hickle.dump(pipeline, os.path.join(run_directory,'pipeline.hkl'))

    return pipeline

def load_pipeline(directory_path='.'):
    """
    Loads the state of a Pipeline object

    Parameters
    ----------
    directory_path : str
        Path to the directory where the Pipeline state should be saved
    """
    if os.path.isfile(os.path.join(directory_path, 'pipeline.hkl')):
        pipeline = hickle.load(os.path.join(directory_path, 'pipeline.hkl'))
    else:
        with open(os.path.join(directory_path,'pipeline.pkl'), 'rb') as f:
            pipeline = cloudpickle.load(f)

    # Adjusts paths (hidden variables are used to avoid checks)
    pipeline._run_directory = os.path.join(directory_path, pipeline._run_directory)
    pipeline._chains_directory = os.path.join(directory_path, pipeline._chains_directory)

    # Adjust observational data, if using distributed arrays
    if rc['distributed_arrays']:
        # Distributes the covariance data
        for k in pipeline.likelihood.covariance_dict.keys():
            cov = pipeline.likelihood.covariance_dict[k]._data
            pipeline.likelihood.covariance_dict[k]._data = distribute_matrix(cov)

    # Hammurabi-specific path adjustment
    if hasattr(pipeline.simulator, 'hamx_path'):
        # In the case hamx path is the system default, it will use the
        # system default the next time it is loaded.
        if pipeline.simulator.hamx_path is None:
            pipeline.simulator.hamx_path = rc['hammurabi_hamx_path']
        # The following refreshes the path to the XML template internally
        # using the xml_path property setter
        if pipeline.simulator.hamx_path is None:
            pipeline.xml_path = None

    # Avoids synchronization problems after loading the pipeline when using MPI
    comm.Barrier()

    return pipeline
