# %% IMPORTS
import astropy.units as u
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory
from imagine.priors import GaussianPrior

# All declaration
__all__ = ['BregCart', 'BregCartFactory']


# %% CLASS DEFINITIONS
class BregCart(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin regular uniform magnetic field.
    """

    # Class attributes
    NAME = 'breg_cart'
    FIELD_CHECKLIST = {'bx': (['magneticfield', 'regular', 'cart', 'bx'],
                              'value'),
                       'by': (['magneticfield', 'regular', 'cart', 'by'],
                              'value'),
                       'bz': (['magneticfield', 'regular', 'cart', 'bz'],
                              'value'),
                       }
    SIMULATOR_CONTROLLIST = {'cue': (['magneticfield', 'regular'],
                                     {'cue': '1'}),
                             'type': (['magneticfield', 'regular'],
                                      {'type': 'cart'})}


class BregCartFactory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`BregCart`
    (see its docs for details).
    """

    # Class attributes
    FIELD_CLASS = BregCart
    DEFAULT_PARAMETERS = {'bx': 0*u.Gauss, 'by': 0*u.Gauss, 'bz': 0*u.Gauss, }
    PRIORS = {'bx': GaussianPrior(mu=0., sigma=1e-6, unit=u.Gauss),
              'by': GaussianPrior(mu=0., sigma=1e-6, unit=u.Gauss),
              'bz': GaussianPrior(mu=0., sigma=1e-6, unit=u.Gauss)
            }
