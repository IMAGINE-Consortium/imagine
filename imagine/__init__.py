# %% IMPORTS
# Version import
from .__version__ import __version__

# Global configuration and settings
from .tools.config import rc
# Global configuration and settings
from .tools.io import load_pipeline, save_pipeline

# check ptional dependencies
model_library_installed = True
try:
    import ImagineModels
    
except ImportError:
    print('No ImagineModels installation found, model library not available')
    model_library_installed = False

hampyx_installed = True
try:
    import hampyx
    
except ImportError:
    print('No hampyx installation found,Hammurabi simulator and fields not available')
    hampyx_installed = False
    
    
nifty_installed = True
try:
    import nifty8
    
except ImportError:
    print('No hampyx installation found,Hammurabi simulator and fields not available')
    nifty_installed = False
    



# Import subpackages
from . import (
    fields, likelihoods, observables, pipelines, priors, simulators, tools)

# All declaration
__all__ = ['fields', 'likelihoods', 'observables', 'pipelines', 'priors',
           'simulators', 'rc', 'tools', 'load_pipeline', 'save_pipeline']
