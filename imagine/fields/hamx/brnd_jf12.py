# %% IMPORTS
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory
from imagine.priors import FlatPrior

# All declaration
__all__ = ['BrndJF12', 'BrndJF12Factory']

class BrndJF12(DummyField):
    """ This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin random magnetic field
    for the Jansson & Farrar model, https://ui.adsabs.harvard.edu/abs/2012ApJ...761L..11J/abstract
    """
    NAME = 'brnd_jf12'
    
    FIELD_CHECKLIST = {'rms': (['magneticfield', 'random', 'global','jf12', 'rms'], 'value'),
                     'k0': (['magneticfield', 'random', 'global','jf12', 'k0'], 'value'),
                     'k1': (['magneticfield', 'random', 'global','jf12', 'k1'], 'value'),
                     'a0': (['magneticfield', 'random', 'global','jf12', 'a0'], 'value'),
                     'a1': (['magneticfield', 'random', 'global','jf12', 'a1'], 'value'),
                     'rho': (['magneticfield', 'random', 'global','jf12', 'rho'], 'value'),
                     'b0_1': (['magneticfield', 'random', 'global','jf12', 'b0_1'], 'value'),
                     'b0_2': (['magneticfield', 'random', 'global','jf12', 'b0_2'], 'value'),
                     'b0_3': (['magneticfield', 'random', 'global','jf12', 'b0_3'], 'value'),
                     'b0_4': (['magneticfield', 'random', 'global','jf12', 'b0_4'], 'value'),
                     'b0_5': (['magneticfield', 'random', 'global','jf12', 'b0_5'], 'value'),
                     'b0_6': (['magneticfield', 'random', 'global','jf12', 'b0_6'], 'value'),
                     'b0_7': (['magneticfield', 'random', 'global','jf12', 'b0_7'], 'value'),
                     'b0_8': (['magneticfield', 'random', 'global','jf12', 'b0_8'], 'value'),
                     'b0_int': (['magneticfield', 'random', 'global','jf12', 'b0_int'], 'value'),
                     'z0_spiral': (['magneticfield', 'random', 'global','jf12', 'z0_spiral'], 'value'),
                     'b0_halo': (['magneticfield', 'random', 'global','jf12', 'b0_halo'], 'value'),
                     'r0_halo': (['magneticfield', 'random', 'global','jf12', 'r0_halo'], 'value'),
                     'z0_halo': (['magneticfield', 'random', 'global','jf12', 'z0_halo'], 'value'),
                     'random_seed':(['magneticfield','random'],'seed')}
    
    SIMULATOR_CONTROLLIST = {'cue': (['magneticfield', 'random'], {'cue': '1'}),
                'type': (['magneticfield', 'random'], {'type': 'global'}),
                'method':(['magneticfield','random','global'],{'type':'jf12'})}


class BrndJF12Factory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`BrndJF12`
    (see its docs for details).
    """
    FIELD_CLASS = BrndJF12
    DEFAULT_PARAMETERS = {'rms':1.0, 'k0':10.0,'k1':0.1,'a0':1.7, 'a1':0.0, 'rho':1.0,
                             'b0_1':10.81, 'b0_2':6.96,'b0_3':9.59,'b0_4':6.96, 'b0_5':1.96, 'b0_6':16.34, 'b0_7':37.29, 'b0_8':10.35,
                             'b0_int':7.63, 'z0_spiral':0.61, 'b0_halo':4.68, 'r0_halo':10.97, 'z0_halo': 2.84}
    PRIORS = { 'rms': FlatPrior(xmin =0.5, xmax =2.5),
                   'k0': FlatPrior(xmin =9.0, xmax =11.0),
                   'k1': FlatPrior(xmin =0.0, xmax =2.0),
                   'a0': FlatPrior(xmin =1.0, xmax =3.0),
                   'a1': FlatPrior(xmin =0.0, xmax =1.0),
                   'rho': FlatPrior(xmin =0.0, xmax =2.0),
                   'b0_1': FlatPrior(xmin =10.0, xmax =12.0),
                   'b0_2': FlatPrior(xmin =6.0, xmax =8.0),
                   'b0_3': FlatPrior(xmin =9.0, xmax =11.0),
                   'b0_4': FlatPrior(xmin =6.0, xmax =8.0),
                   'b0_5': FlatPrior(xmin =1.0, xmax =3.0),
                   'b0_6': FlatPrior(xmin =15.0, xmax =17.0),
                   'b0_7': FlatPrior(xmin =36.0, xmax =38.0),
                   'b0_8': FlatPrior(xmin =10.0, xmax =12.0),
                   'b0_int': FlatPrior(xmin =7.0, xmax =9.0),
                   'z0_spiral': FlatPrior(xmin =0.0, xmax =2.0),
                   'b0_halo': FlatPrior(xmin =4.0, xmax =6.0),
                   'r0_halo': FlatPrior(xmin =10.0, xmax =12.0),
                   'z0_halo': FlatPrior(xmin =2.0, xmax =4.0)}
