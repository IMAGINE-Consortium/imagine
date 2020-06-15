""" 
This module is responsible for setting the `rc` configuration variables.

These can be accessed and modified using the 
:py:data:`imagine.rc <imagine.config.rc>` dictionary or setting the 
corresponding environment variables (named 'IMAGINE_'+RC_VAR_NAME). 
"""
import tempfile
import os

# Sets default values of configuration parameters
rc = {'temp_dir': None, 'test':0}



for var in rc:
    env_var = 'IMAGINE_'+var.upper()

    try:
        rc[var] = os.environ[env_var]
    except KeyError:
        pass


# If a temp directory was not set using envirnment variables, create one
if rc['temp_dir'] is None:
    rc['temp_dir_obj'] = tempfile.TemporaryDirectory(prefix='imagine_')
    # Default values that require previous values
    rc['temp_dir'] = rc['temp_dir_obj'].name
