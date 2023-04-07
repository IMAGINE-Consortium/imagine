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

from . import test_simulator
from .test_simulator import *

# All declaration
__all__ = ['simulator', 'test_simulator']
__all__.extend(simulator.__all__)
__all__.extend(test_simulator.__all__)

if hammurabi_installed:
    all.append('hammurabi')
    __all__.extend(hammurabi.__all__)