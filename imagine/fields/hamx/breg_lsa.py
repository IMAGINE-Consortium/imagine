import logging as log
from imagine.fields.field import GeneralField
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.tools.icy_decorator import icy


@icy
class BregLSA(GeneralField):
    """
    hammurabiX WMAP-3yr LSA GMF
    """
    field_name = 'breg_lsa'
    field_type = 'dummy'

    @property
    def field_checklist(self):
        """
        record XML location of physical parameters

        return
        ------
        dict of XML locations
        """
        checklist = {'b0': (['magneticfield', 'regular', 'lsa', 'b0'], 'value'),
                     'psi0': (['magneticfield', 'regular', 'lsa', 'psi0'], 'value'),
                     'psi1': (['magneticfield', 'regular', 'lsa', 'psi1'], 'value'),
                     'chi0': (['magneticfield', 'regular', 'lsa', 'chi0'], 'value')}
        return checklist

    @property
    def field_controllist(self):
        """
        record XML location of logical parameters

        return
        ------
        dict of XML locations
        """
        controllist = {'cue': (['magneticfield', 'regular'], {'cue': '1'}),
                       'type': (['magneticfield', 'regular'], {'type': 'lsa'})}
        return controllist


@icy
class BregLSAFactory(GeneralFieldFactory):
    """
    hammurabiX WMAP-3yr LSA GMF factory
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(BregLSAFactory, self).__init__(boxsize, resolution)
        self.field_class = BregLSA
        self.default_parameters = {'b0': 6.0, 'psi0': 27.0, 'psi1': 0.9, 'chi0': 25.0}
        self.parameter_ranges = {'b0': [0., 10.],
                                 'psi0': [0., 50.],
                                 'psi1': [0., 5.],
                                 'chi0': [-25., 50.]}
        self.active_parameters = active_parameters
