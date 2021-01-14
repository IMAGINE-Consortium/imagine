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

__all__ = ['save_pipeline', 'load_pipeline']

# GLOBALS
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()

# %% FUNCTION DEFINITIONS
def save_pipeline(pipeline, save_obs_data_separately=True):
    """
    Saves the state of a Pipeline object

    Parameters
    ----------
    pipeline : imagine.pipelines.pipeline.Pipeline
        The pipeline object one would like to save
    save_obs_data_separately : bool
        If `True` (default) observable dictionaries (the Measurements,
        Covariances and Masks objects linked to the pipeline's Likelihood
        object) are saved separately and compressed. Otherwise, they are
        serialized together with the remainder of the Pipeline object.
    """
    # Works on a (shallow) copy
    pipeline = copy(pipeline)

    # Adjusts paths: we want everything to be relative to run_directory
    # Hidden variables are used to avoid checks
    run_directory, pipeline._run_directory = pipeline._run_directory, '.'
    pipeline._chains_directory = os.path.relpath(pipeline._chains_directory,
                                                run_directory)

    # Shallow copy of the likelihood object to allow later manipulation
    pipeline.likelihood = copy(pipeline.likelihood)

    # Adjusts observational data, if using distributed arrays
    if rc['distributed_arrays']:
        # Covariances need to be "undistributed"
        pipeline.likelihood.covariance_dict = deepcopy(pipeline.likelihood.covariance_dict)

        # Gathers distributed covariance data
        # i.e. turns global (distributed) into local
        for k in pipeline.likelihood.covariance_dict:
            if pipeline.likelihood.covariance_dict[k].dtype == 'covariance':
                pipeline.likelihood.covariance_dict[k]._data = pipeline.likelihood.covariance_dict[k].global_data
                # NB any process with mpirank!=0 will store None in the above operation

    # Stores measurements, covariances and masks separately
    if save_obs_data_separately and mpirank==0:
        for obs_dict_name in ('covariance', 'measurement','mask'):

            obs_dict = getattr(pipeline.likelihood, obs_dict_name + '_dict')

            if obs_dict is not None:
                obs_dict = copy(obs_dict)
                obs_dict._archive = copy(obs_dict._archive)

                # Takes the actual data arrays out of the Observable_Dict
                # for saving (hickle works better if supplied a python dict
                # comprising only numpy arrays as values)
                data_dict = {}
                for k, obs in obs_dict._archive.items():
                    obs = copy(obs)
                    # Serializes the key (to avoid hickle hickups)
                    serial_key = cloudpickle.dumps(k)
                    data_dict[serial_key] = obs._data
                    obs._data = None
                    obs_dict._archive[k] = obs

                # Finally, save the dictionary
                hickle.dump(data_dict,
                            os.path.join(run_directory, obs_dict_name+'.hkl'),
                            mode='w', compression='gzip')

                setattr(pipeline.likelihood, obs_dict_name + '_dict', obs_dict)

    # Hammurabi-specific path adjustment
    if hasattr(pipeline.simulator, 'hamx_path'):
        # In the case hamx path is the system default, it will use the
        # system default the next time it is loaded.
        pipeline.simulator = copy(pipeline.simulator)
        if pipeline.simulator.hamx_path == rc['hammurabi_hamx_path']:
            pipeline.simulator._hamx_path = None
            pipeline.simulator._ham._exe_path = None

    if mpirank == 0:
        with open(os.path.join(run_directory,'pipeline.pkl'), 'wb') as f:
            cloudpickle.dump(pipeline, f)

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

    # Loads observable dictionaries, if available
    for obs_dict_name in ('covariance', 'measurement','mask'):
        filename = os.path.join(directory_path, obs_dict_name + '.hkl')
        if os.path.isfile(filename):
            data_dict = hickle.load(filename)
            obs_dict = getattr(pipeline.likelihood, obs_dict_name +'_dict')
            for skey, data in data_dict.items():
                # Deserializes the key
                key = cloudpickle.loads(eval(skey))
                obs_dict[key]._data = data

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

    return pipeline
