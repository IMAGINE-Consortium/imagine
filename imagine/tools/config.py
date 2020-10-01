"""

IMAGINE global configuration
----------------------------

The default behaviour of some aspects of IMAGINE can be set using
global `rc` configuration variables.

These can be accessed and modified using the
:py:data:`imagine.rc <imagine.config.rc>` dictionary or setting the
corresponding environment variables (named 'IMAGINE\_'+RC_VAR_NAME).

For example to set the default path for the hamx executable,
one can either do::

    import imagine
    imagine.rc.hammurabi_hamx_path = 'my_desired_path'

or, alternatively, set this as an environment variable
before the exectution of the script::

    export IMAGINE_HAMMURABI_HAMX_PATH='my_desired_path'

The following list describes all the available global settings variables.

IMAGINE rc variables
    temp_dir
        Default temporary directory used by IMAGINE. If not set, a temporary
        directory will be created at /tmp/ with a safe name.
    distributed_arrays
        If `True`, arrays containing covariances are distributed among
        different MPI processes (and so are the corresponding array operations).
    pipeline_default_seed
        The default value for the master seed used by a Pipeline object
        (see :py:data:`Pipeline.master_seed <imagine.pipelines.Pipeline.master_seed>`).
    pipeline_distribute_ensemble
        The default value of
        (see :py:data:`Pipeline.distribute_ensemble <imagine.pipelines.Pipeline.distribute_ensemble>`).
    hammurabi_hamx_path
        Default location of the Hammurabi X executable file, `hamx`.
"""

# %% IMPORTS
# Built-in imports
import os
import tempfile

# All declaration
__all__ = ['rc']

# Sets default values of configuration parameters
rc = {'temp_dir': None,
      'distributed_arrays': False,
      'pipeline_default_seed': 1,
      'pipeline_distribute_ensemble': False,
      'hammurabi_hamx_path': None,
      'max_dense_matrix_size': 200}


# %% FUNCTION DEFINITIONS
def _str_to_python(v):
    """
    Attempts to convert a string to a python basic type
    """
    # Tries to convert to a number
    try:
        v = float(v)
        # Converts to integer if needed
        if v.is_integer():
            v = int(v)
    except ValueError:
        pass

    # Converts to boolean if needed
    if v in ('True','T','TRUE'):
        v = True
    elif v in ('False','F','FALSE'):
        v = False

    return v


def read_rc_from_env():
    """
    Updates the rc configuration dictionary using current
    environment variables
    """
    global rc
    for var in rc:
        env_var = 'IMAGINE_'+var.upper()

        try:
            rc[var] = _str_to_python(os.environ[env_var])
        except KeyError:
            pass


read_rc_from_env()

# If a temp directory was not set using environment variables, create one
if rc['temp_dir'] is None:
    rc['temp_dir_obj'] = tempfile.TemporaryDirectory(prefix='imagine_')
    rc['temp_dir'] = rc['temp_dir_obj'].name
