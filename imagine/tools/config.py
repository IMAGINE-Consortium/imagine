"""
This module is responsible for setting the `rc` configuration variables.

These can be accessed and modified using the
:py:data:`imagine.rc <imagine.config.rc>` dictionary or setting the
corresponding environment variables (named 'IMAGINE_'+RC_VAR_NAME).
"""
import tempfile
import os

# Sets default values of configuration parameters
rc = {'temp_dir': None}


def read_rc_from_env():
    """
    Updates the rc configuration dictionary using current
    environment variables
    """
    global rc
    for var in rc:
        env_var = 'IMAGINE_'+var.upper()

        try:
            rc[var] = os.environ[env_var]
        except KeyError:
            pass


read_rc_from_env()

# If a temp directory was not set using environment variables, create one
if rc['temp_dir'] is None:
    rc['temp_dir_obj'] = tempfile.TemporaryDirectory(prefix='imagine_')
    rc['temp_dir'] = rc['temp_dir_obj'].name
