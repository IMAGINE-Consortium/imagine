# -*- coding: utf-8 -*-

# %% IMPORTS
# Package imports
import matplotlib as mpl
from py.path import local
import _pytest
import pytest


# Set MPL backend
mpl.use('Agg')


# %% PYTEST CUSTOM CONFIGURATION PLUGINS
# This makes the pytest report header mention the tested IMAGINE version
def pytest_report_header(config):
    from imagine.__version__ import __version__
    return("IMAGINE: %s" % (__version__))


# Add the incremental marker
def pytest_configure(config):
    config.addinivalue_line("markers",
                            "incremental: Mark test suite to xfail all "
                            "remaining tests when one fails.")


# This introduces a marker that auto-fails tests if a previous one failed
def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if(call.excinfo is not None and
           call.excinfo.type is not _pytest.outcomes.Skipped):
            parent = item.parent
            parent._previousfailed = item


# This makes every marked test auto-fail if a previous one failed as well
def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("Previous test failed (%s)" % (previousfailed.name))


# %% PYTEST SETTINGS
# Set the current working directory to the temporary directory
local.get_temproot().chdir()
