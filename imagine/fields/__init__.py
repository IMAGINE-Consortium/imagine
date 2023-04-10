# -*- coding utf-8 -*-

# %% IMPORTS
# Import core modules
from . import field, field_factory
from .field import *
from .field_factory import *

# Import base modules
from . import (
    base_fields, basic_fields, grid, test_field)
from .base_fields import *
from .basic_fields import *
from .grid import *
from .test_field import *

# Import subpackages


from . import hamx

# Import base modules
model_library_installed = True
try:
    from . import library
    
except ImportError:
    print('No ImagineModels installation found, model library not available')
    model_library_installed = False

# All declaration
__all__ = ['base_fields', 'basic_fields', 'field', 'field_factory', 'grid',
           'hamx', 'test_field']
__all__.extend(base_fields.__all__)
__all__.extend(basic_fields.__all__)
__all__.extend(field.__all__)
__all__.extend(field_factory.__all__)
__all__.extend(grid.__all__)
__all__.extend(test_field.__all__)
