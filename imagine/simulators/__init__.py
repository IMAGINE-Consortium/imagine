# -*- coding utf-8 -*-

# %% IMPORTS
# Import core modules
import importlib


from . import simulator
from .simulator import *

from . import test_simulator
from .test_simulator import *


# Import base modules
spec = importlib.util.find_spec('hampyx')
if spec is not None:
    from . import (hammurabi)
    from .hammurabi import *

# All declaration
# __all__ = ['hammurabi', 'simulator', 'test_simulator']
# __all__.extend(hammurabi.__all__)
__all__ = ['simulator', 'test_simulator']
__all__.extend(simulator.__all__)

__all__.extend(test_simulator.__all__)
