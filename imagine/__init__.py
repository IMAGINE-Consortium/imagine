# %% IMPORTS
# Version import
from .__version__ import __version__

# Global configuration and settings
from .tools.config import rc

# Import subpackages
from . import (
    fields, likelihoods, observables, pipelines, priors, simulators, tools)

# All declaration
__all__ = ['fields', 'likelihoods', 'observables', 'pipelines', 'priors',
           'simulators', 'rc', 'tools']
