# -*- coding utf-8 -*-

# %% IMPORTS
# Import core modules
from . import pipeline
from .pipeline import *

# Import base modules
dynesty_installed = True
try:
    from . import dynesty_pipeline
    from .dynesty_pipeline import *

except ImportError:
    print('No Dynesty installation found, dynesty pipeline not available')
    dynesty_installed = False


emcee_installed = True
try:
    from . import emcee_pipeline
    from .emcee_pipeline import *

except ImportError:
    print('No emcee installation found, emcee pipeline not available')
    emcee_installed = False

multinest_installed = True
try:
    from . import multinest_pipeline
    from .multinest_pipeline import *

except ImportError:
    print('No pymultinest installation found, multinest pipeline not available')
    multinest_installed = False

ultranest_installed = True
try:
    from . import ultranest_pipeline
    from .ultranest_pipeline import *

except ImportError:
    print('No ultranest installation found, ultranest pipeline not available')
    ultranest_installed = False


# All declaration
__all__ = ['pipeline', ]
__all__.extend(pipeline.__all__)

if dynesty_installed:
    __all__.append('dynesty_pipeline')
    __all__.extend(dynesty_pipeline.__all__)
if emcee_installed:
    __all__.append('emcee_pipeline')
    __all__.extend(emcee_pipeline.__all__)
if multinest_installed:
    __all__.append('multinest_pipeline')
    __all__.extend(multinest_pipeline.__all__)
if ultranest_installed:
    __all__.append('ultranest_pipeline')
    __all__.extend(ultranest_pipeline.__all__)