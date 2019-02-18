from .version import __version__
from likelihoods import *
from fields import *
from observables import *
from simulators import *
from priors import *
from pymultinest_importer import pymultinest
from pipelines import *
'''
from sample import Sample
'''
import nifty
nifty.nifty_configuration['default_distribution_strategy'] = 'equal'
