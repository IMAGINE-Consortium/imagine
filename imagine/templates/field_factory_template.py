from imagine.fields import FieldFactory
from imagine.priors import FlatPrior, GaussianPrior
# Substitute this by your own code
from MY_PACKAGE import MY_FIELD_CLASS
from MY_PACKAGE import A_std_val, B_std_val, A_min, A_max, B_min, B_max, B_sig

class FieldFactoryTemplate(FieldFactory):
    """Example: field factory for YourFieldClass"""

    # "Recomended" default values for inactive parameters
    # (i.e. if the user does not set a default value, these are instead)
    DEFAULT_PARAMETERS = {'Parameter_A': A_std_val,
                          'Parameter_B': B_std_val}

    # Default priors (the recomended priors for this field)
    PRIORS = {'Parameter_A': FlatPrior(xmin=A_min, xmax=A_max),
              'Parameter_B': GaussianPrior(mu=B_std_val, sigma=B_sig)}

    def __init__(self, active_parameters=(), default_parameters={}, priors={},
                 grid=None, field_kwargs={}, **kwargs):
        default_parameters_actual = self.DEFAULT_PARAMETERS.copy()
        default_parameters_actual.update(default_parameters)

        priors_actual = self.PRIORS.copy()
        priors_actual.update(priors)

        super().__init__(field_class=MY_FIELD_CLASS,
                         active_parameters=active_parameters,
                         default_parameters=default_parameters_actual,
                         priors=priors_actual,
                         grid=grid, field_kwargs=field_kwargs, **kwargs)

