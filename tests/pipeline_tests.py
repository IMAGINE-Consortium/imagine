import unittest
import numpy as np

from imagine.observables.observable_dict import Simulations, Measurements, Covariances
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.priors.flat_prior import FlatPrior
from imagine.simulators.test.test_simulator import TestSimulator
from imagine.pipelines.pipeline import Pipeline

class PipelineTests(unittest.TestCase):

    def test_init(self):
        # mock measures
        arr = np.random.rand(1,3)
        measuredict = Measurements()
        measuredict.append(('test','nan','3','nan'),arr,True)
        # simulator
        simer = TestSimulator(measuredict)
        # mock factory list
        tf = TestFieldFactory(active_parameters=tuple('a'))
        flist = (tf,)
        # mock likelihood
        lh = EnsembleLikelihood (measuredict)
        # mock prior
        pr = FlatPrior()
        # pipeline
        pipe = Pipeline(simer,flist,lh,pr,5)

        self.assertEqual (pipe.active_parameters, ('test_a',))
        self.assertEqual (pipe.factory_list, (tf,))
        self.assertEqual (pipe.simulator, simer)
        self.assertEqual (pipe.likelihood, lh)
        self.assertEqual (pipe.prior, pr)
        self.assertEqual (pipe.ensemble_size, 5)
        self.assertEqual (pipe.pymultinest_parameter_dict, {'verbose': True,
                                                            'n_iter_before_update': 100,
                                                            'n_live_points': 400,
                                                            'resume': False})
        pipe.pymultinest_parameter_dict = {'verbose':False}
        self.assertEqual (pipe.pymultinest_parameter_dict, {'verbose': False,
                                                            'n_iter_before_update': 100,
                                                            'n_live_points': 400,
                                                            'resume': False})
        self.assertEqual (pipe.sample_callback, False)
        pipe.sample_callback = True
        self.assertEqual (pipe.sample_callback, True)
        self.assertEqual (pipe.likelihood_rescaler, 1.)
        pipe.likelihood_rescaler = 0.5
        self.assertEqual (pipe.likelihood_rescaler, 0.5)
        self.assertEqual (pipe.check_threshold, False)
        pipe.check_threshold = True
        self.assertEqual (pipe.check_threshold, True)
        self.assertEqual (pipe.likelihood_threshold, 0.)
        pipe.likelihood_threshold = -0.2
        self.assertEqual (pipe.likelihood_threshold, -0.2)
        self.assertEqual (pipe.random_seed, 0)
        pipe.random_seed = int(23)
        self.assertEqual(pipe.random_seed, int(23))


if __name__ == '__main__':
    unittest.main()
