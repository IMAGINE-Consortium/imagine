#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from inspect import currentframe

# All declaration
__all__ = ['BaseClass', 'req_attr']


# %% CLASS DEFINITIONS
# Define a base class that automatically checks for missing attributes
class BaseClass(object):
    # Class attributes
    REQ_ATTRS = []

    def __init__(self):
        # Check if all required class attributes are defined
        self._check_class_attrs()

    # This function checks if all required class attributes are available
    def _check_class_attrs(self):
        # Loop over all required attributes in REQ_ATTRS and check if it exists
        for attr in self.REQ_ATTRS:
            if not hasattr(self, attr):
                # Raise error if attribute is not found
                raise AttributeError("Required class attribute %r is not "
                                     "defined!" % (attr))


# %% FUNCTION DEFINITIONS
def req_attr(meth):
    # Obtain the REQ_ATTRS attribute of the class of given 'meth'
    frame = currentframe().f_back
    req_attrs = frame.f_locals.get('REQ_ATTRS')

    # If req_attrs is None, add it to the class first
    if req_attrs is None:
        req_attrs = []
        frame.f_locals['REQ_ATTRS'] = req_attrs

    # Add capitalized version of method name to req_attrs
    req_attrs.append(meth.__name__.upper())

    # Return method
    return(meth)
