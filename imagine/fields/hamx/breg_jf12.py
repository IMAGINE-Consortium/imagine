# %% IMPORTS
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory
from imagine.priors import FlatPrior

# All declaration
__all__ = ['BregJF12', 'BregJF12Factory']

class BregJF12(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin regular magnetic field
    for the Jansson & Farrar model, https://ui.adsabs.harvard.edu/abs/2012ApJ...757...14J/abstract
    """
    NAME = 'breg_jf12'
        
    FIELD_CHECKLIST = {'b_arm1': (['magneticfield', 'regular', 'jf12', 'b_arm1'], 'value'),
                     'b_arm2': (['magneticfield', 'regular', 'jf12', 'b_arm2'], 'value'),
                     'b_arm3': (['magneticfield', 'regular', 'jf12', 'b_arm3'], 'value'),
                     'b_arm4': (['magneticfield', 'regular', 'jf12', 'b_arm4'], 'value'),
                     'b_arm5': (['magneticfield', 'regular', 'jf12', 'b_arm5'], 'value'),
                     'b_arm6': (['magneticfield', 'regular', 'jf12', 'b_arm6'], 'value'),
                     'b_arm7': (['magneticfield', 'regular', 'jf12', 'b_arm7'], 'value'),
                     'b_ring': (['magneticfield', 'regular', 'jf12', 'b_ring'], 'value'),
                     'h_disk': (['magneticfield', 'regular', 'jf12', 'h_disk'], 'value'),
                     'w_disk': (['magneticfield', 'regular', 'jf12', 'w_disk'], 'value'),
                     'Bn': (['magneticfield', 'regular', 'jf12', 'Bn'], 'value'),
                     'Bs': (['magneticfield', 'regular', 'jf12', 'Bs'], 'value'),
                     'rn': (['magneticfield', 'regular', 'jf12', 'rn'], 'value'),
                     'rs': (['magneticfield', 'regular', 'jf12', 'rs'], 'value'),
                     'wh': (['magneticfield', 'regular', 'jf12', 'wh'], 'value'),
                     'z0': (['magneticfield', 'regular', 'jf12', 'z0'], 'value'),
                     'B0_X': (['magneticfield', 'regular', 'jf12', 'B0_X'], 'value'),
                     'Xtheta': (['magneticfield', 'regular', 'jf12', 'Xtheta'], 'value'),
                     'rpc_X': (['magneticfield', 'regular', 'jf12', 'rpc_X'], 'value'),
                     'r0_X': (['magneticfield', 'regular', 'jf12', 'r0_X'], 'value')}
        
    SIMULATOR_CONTROLLIST = {'cue': (['magneticfield', 'regular'], {'cue': '1'}),
                'type': (['magneticfield', 'regular'], {'type': 'jf12'})}

class BregJF12Factory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`BregJF12`
    (see its docs for details).
    """
    FIELD_CLASS = BregJF12
    DEFAULT_PARAMETERS = {'b_arm1': 0.1, 'b_arm2': 3.0, 'b_arm3': -0.9, 'b_arm4': -0.8, 
                'b_arm5': -2.0, 'b_arm6': -4.2, 'b_arm7': 0.0, 'b_ring': 0.1, 'h_disk': 0.40, 'w_disk': 0.27,
                'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': 0.20, 'z0': 5.3, 
                'B0_X': 4.6,'Xtheta': 49, 'rpc_X': 4.8, 'r0_X': 2.9}
    PRIORS = { 'b_arm1': FlatPrior(xmin =-1., xmax =1.),
                        'b_arm2': FlatPrior(xmin = 2., xmax = 4.),
                        'b_arm3': FlatPrior(xmin =-1.,xmax =1.),
                        'b_arm4': FlatPrior(xmin =-1.,xmax =1.), 
                        'b_arm5': FlatPrior(xmin =-3.,xmax =-1.),
                        'b_arm6': FlatPrior(xmin =-5.,xmax =-3.),
                        'b_arm7': FlatPrior(xmin =-1.,xmax =1.),
                        'b_ring': FlatPrior(xmin =-1.,xmax =1.),
                        'h_disk': FlatPrior(xmin =-1.,xmax =1.),
                        'w_disk': FlatPrior(xmin =-1.,xmax =1.),
                        'Bn'    : FlatPrior(xmin =0.,xmax =2.),
                        'Bs'    : FlatPrior(xmin =-2.,xmax =0.),
                        'rn'    : FlatPrior(xmin =8.,xmax =10.),
                        'rs'    : FlatPrior(xmin =15.,xmax =17.),
                        'wh'    : FlatPrior(xmin =0.,xmax =2.),
                        'z0'    : FlatPrior(xmin =4.,xmax =6.),
                        'B0_X'  : FlatPrior(xmin =4.,xmax =6.),
                        'Xtheta': FlatPrior(xmin =48.,xmax =50.),
                        'rpc_X' : FlatPrior(xmin =4.,xmax =6.),
                        'r0_X': FlatPrior(xmin =2., xmax =4.)}
