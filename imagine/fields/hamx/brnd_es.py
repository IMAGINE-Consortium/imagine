# %% IMPORTS
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory
from imagine.priors import FlatPrior

# All declaration
__all__ = ['BrndES', 'BrndESFactory']


# %% CLASS DEFINITIONS
class BrndES(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin random magnetic field
    ES random GMF
    """

    # Class attributes
    NAME = 'breg_wmap'

    def __init__(self, *args, grid_nx=None, grid_ny=None, grid_nz=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Default controllist
        self._controllist = {'cue': (['magneticfield', 'random'], {'cue': '1'}),
                             'type': (['magneticfield', 'random'], {'type': 'global'}),
                             'method': (['magneticfield', 'random', 'global'], {'type': 'es'})}
        self.set_grid_size(nx=grid_nx, ny=grid_ny, nz=grid_nz)

    def set_grid_size(self, nx=None, ny=None, nz=None):
        """
        Changes the size of the grid used for the evaluation of the random field
        """
        if nx is not None:
            self._controllist['box_brnd_nx'] = (['grid', 'box_brnd', 'nx'],{'value': str(nx)})
        if ny is not None:
            self._controllist['box_brnd_ny'] = (['grid', 'box_brnd', 'ny'],{'value': str(ny)})
        if nz is not None:
            self._controllist['box_brnd_nz'] = (['grid', 'box_brnd', 'nz'],{'value': str(nz)})

    @property
    def field_checklist(self):
        """
        Hammurabi XML locations of physical parameters
        """
        checklist = {'rms': (['magneticfield', 'random', 'global', 'es', 'rms'], 'value'),
                     'k0': (['magneticfield', 'random', 'global', 'es', 'k0'], 'value'),
                     'a0': (['magneticfield', 'random', 'global', 'es', 'a0'], 'value'),
                     'k1': (['magneticfield', 'random', 'global', 'es', 'k1'], 'value'),
                     'a1': (['magneticfield', 'random', 'global', 'es', 'a1'], 'value'),
                     'rho': (['magneticfield', 'random', 'global', 'es', 'rho'], 'value'),
                     'r0': (['magneticfield', 'random', 'global', 'es', 'r0'], 'value'),
                     'z0': (['magneticfield', 'random', 'global', 'es', 'z0'], 'value'),
                     'random_seed': (['magneticfield', 'random'], 'seed')}
        return checklist

    @property
    def simulator_controllist(self):
        """
        Hammurabi XML locations of logical parameters
        """
        return self._controllist


class BrndESFactory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`BrndES`
    (see its docs for details).
    """

    # Class attributes
    FIELD_CLASS = BrndES
    DEFAULT_PARAMETERS = {'rms': 2,
                          'k0': 10,
                          'a0': 1.7,
                          'k1': 0.1,
                          'a1': 0,
                          'rho': 0.5,
                          'r0': 8,
                          'z0': 1}
    PRIORS = {'rms': FlatPrior(xmin=0, xmax=4),
              'k0': FlatPrior(xmin=0.1, xmax=1),
              'a0': FlatPrior(xmin=1, xmax=3),
              'k1': FlatPrior(xmin=0.01, xmax=1),
              'a1': FlatPrior(xmin=0, xmax=3),
              'rho': FlatPrior(xmin=0, xmax=1),
              'r0': FlatPrior(xmin=2, xmax=10),
              'z0': FlatPrior(xmin=0.1, xmax=3)}

    def __init__(self, *args, grid_nx=None,
                 grid_ny=None, grid_nz=None, **kwargs):
        super().__init__(*args, **kwargs,
                         field_kwargs={'grid_nx': grid_nx,
                                       'grid_ny': grid_ny,
                                       'grid_nz': grid_nz})
