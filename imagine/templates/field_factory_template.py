from imagine.fields import FieldFactory
from imagine.priors import FlatPrior, GaussianPrior
# Substitute this by your own code
from MY_PACKAGE import MY_FIELD_CLASS
from MY_PACKAGE import A_std_val, B_std_val, A_min, A_max, B_min, B_max, B_sig

class FieldFactoryTemplate(FieldFactory):
    """Example: field factory for YourFieldClass"""

    # Class attributes
    # Field class this factory uses
    FIELD_CLASS = MY_FIELD_CLASS

    # Default values are used for inactive parameters
    DEFAULT_PARAMETERS = {'Parameter_A': A_std_val,
                          'Parameter_B': B_std_val}

    # All parameters need a range and a prior
    PRIORS = {'Parameter_A': FlatPrior(xmin=A_min, xmax=A_max),
              'Parameter_B': GaussianPrior(mu=B_std_val, sigma=B_sig)}
