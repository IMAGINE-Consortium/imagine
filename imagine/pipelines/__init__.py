# -*- coding utf-8 -*-

# %% IMPORTS
# Import core modules
from . import pipeline
from .pipeline import *

# Import base modules
from . import (
    dynesty_pipeline, emcee_pipeline, multinest_pipeline, ultranest_pipeline)
from .dynesty_pipeline import *
from .multinest_pipeline import *
from .ultranest_pipeline import *
from .emcee_pipeline import *

# All declaration
__all__ = ['dynesty_pipeline', 'multinest_pipeline', 'pipeline',
           'emcee_pipeline', 'ultranest_pipeline']
__all__.extend(dynesty_pipeline.__all__)
__all__.extend(multinest_pipeline.__all__)
__all__.extend(pipeline.__all__)
__all__.extend(ultranest_pipeline.__all__)
