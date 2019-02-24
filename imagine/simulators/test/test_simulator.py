'''
test simulator is built only for testing purpose
'''

import numpy as np
import logging as log

from imagine.simulators.simulator import Simulator
from imagine.observables.observable_list import Measurements

class TestSimulator(Simulator):

    '''
    measurement_keys
        -- keys in Measruments object
    '''
    def __init__(self, measurement_keys):
        self.output_checklist = measurement_keys

    
    @property
    def output_checklist(self):
        return self._output_checklist

    
