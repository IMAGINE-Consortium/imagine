# Settings for the test_templates test_templates
# Ignore this file unless you know what you are doing!
import sys
import numpy as np
import astropy.units as u
from imagine.fields import DummyField
import imagine.tests.mocks_for_templates as mock

# Here we set up several mock modules which allow testing the template files
sys.modules['MY_GMF_MODEL'] = mock.MY_GMF_MODEL
sys.modules['MY_GALAXY_MODEL'] = mock.MY_GALAXY_MODEL
sys.modules['MY_PACKAGE'] = mock.MY_PACKAGE
sys.modules['MY_SIMULATOR'] = mock.MY_SIMULATOR
sys.modules['MY_SAMPLER'] = mock.MY_SAMPLER
