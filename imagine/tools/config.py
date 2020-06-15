"""
This module is responsible for setting the `rc` configuration variables.

These can be accessed and modified using the
:py:data:`imagine.rc <imagine.config.rc>` dictionary or setting the
corresponding environment variables (named 'IMAGINE_'+RC_VAR_NAME).
"""
import tempfile
import os

# Sets default values of configuration parameters
rc = {'temp_dir': None,
      'distributed_arrays': False}


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
