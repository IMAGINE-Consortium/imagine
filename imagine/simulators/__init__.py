# -*- coding utf-8 -*-

# %% IMPORTS
# Import core modules
from . import simulator
from .simulator import *

# Import base modules
hammurabi_installed = True
try:
    from . import hammurabi
    from .hammurabi import *

except ImportError:
    print('No hampyx installation found, Hammurabi simulator not available')
    hammurabi_installed = False
    
nifty_installed = True
try:
    from . import LOSresponse
    from .LOSresponse import *
        
    from . import rmlos
    from .rmlos import *
    
    from . import synchrotronlos
    from .synchrotronlos import *
    

except ImportError:
    print('No nifty installation found, nifty based simulators not available')
    nifty_installed = False
   

from . import test_simulator
from .test_simulator import *

# All declaration
__all__ = ['simulator', 'test_simulator']
__all__.extend(simulator.__all__)
__all__.extend(test_simulator.__all__)

if hammurabi_installed:
    __all__.append('hammurabi')
    __all__.extend(hammurabi.__all__)
    
    
if nifty_installed:
    __all__.append(['rmlos', 'synchrotronlos'])
    __all__.extend(rmlos.__all__)
    __all__.extend(synchrotronlos.__all__)