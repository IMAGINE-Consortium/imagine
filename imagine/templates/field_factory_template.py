from imagine.fields import FieldFactory
from imagine.priors import FlatPrior, GaussianPrior
# Substitute this by your own code
from MY_PACKAGE import MY_FIELD_CLASS

class YourFieldFactory(FieldFactory):
    """Example: field factory for YourFieldClass"""

    # Class attributes
    # Field class this factory uses
    FIELD_CLASS = MY_FIELD_CLASS

    # Default values are used for inactive parameters
    DEFAULT_PARAMETERS = {'param_A': param_A_value,
                          'param_B': param_B_value}

    # All parameters need a range and a prior
    PRIORS = {'param_A': FlatPrior(prior_args, interval=[A_min, A_max])
              'param_B': GaussianPrior(prior_args, interval=[B_min, B_max])}
