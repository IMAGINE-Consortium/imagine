from imagine.fields import FieldFactory

class YourFieldFactory(FieldFactory):
    """Example: field factory for YourFieldClass"""

    # Class attributes
    # Field class this factory uses
    FIELD_CLASS = YourFieldClass

    # Default values are used for inactive parameters
    DEFAULT_PARAMETERS = {'param_A': param_A_value,
                          'param_B': param_B_value,
                          ... }

    # All parameters need a range and a prior
    PRIORS = {'param_A': PriorA(prior_args, interval=[A_min, A_max])
              'param_B': PriorB(prior_args, interval=[B_min, B_max])
              ... }
