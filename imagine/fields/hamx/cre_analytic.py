import logging as log
from imagine.fields.field import GeneralField
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.tools.icy_decorator import icy

@icy
class CREAna(GeneralField):
    """
    hammurabiX analytic CRE field
    """
    field_name = 'cre_ana'
    field_type = 'dummy'

    @property
    def field_checklist(self):
        """
        record XML location of physical parameters

        return
        ------
        dict of XML locations
        """
        checklist = {'alpha': (['cre', 'analytic', 'alpha'], 'value'),
                     'beta': (['cre', 'analytic', 'beta'], 'value'),
                     'theta': (['cre', 'analytic', 'theta'], 'value'),
                     'r0': (['cre', 'analytic', 'r0'], 'value'),
                     'z0': (['cre', 'analytic', 'z0'], 'value'),
                     'E0': (['cre', 'analytic', 'E0'], 'value'),
                     'j0': (['cre', 'analytic', 'j0'], 'value')}
        return checklist

    @property
    def field_controllist(self):
        """
        record XML location of logical parameters

        return
        ------
        dict of XML locations
        """
        controllist = {'cue': (['cre'], {'cue': '1'}),
                       'type': (['cre'], {'type': 'analytic'})}
        return controllist


@icy
class CREAnaFactory(GeneralFieldFactory):
    """
    hammurabiX analytic CRE field
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(CREAnaFactory, self).__init__(boxsize, resolution)
        self.field_class = CREAna
        self.default_parameters = {'alpha': 3.0,
                                   'beta': 0.0,
                                   'theta': 0.0,
                                   'r0': 5.0,
                                   'z0': 1.0,
                                   'E0': 20.6,
                                   'j0': 0.0217}
        self.parameter_ranges = {'alpha': [2., 4.],
                                 'beta': [-1., 1.],
                                 'theta': [-1., 1.],
                                 'r0': [0.1, 10.],
                                 'z0': [0.1, 3.],
                                 'E0': [10., 30.],
                                 'j0': [0., 0.1]}
        self.active_parameters = active_parameters
