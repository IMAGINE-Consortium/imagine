import unittest
import numpy as np

from imagine.observables.observable_dict import Simulations, Measurements, Covariances
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.priors.flat_prior import FlatPrior
from imagine.simulators.test.li_simulator import LiSimulator
from imagine.simulators.test.bi_simulator import BiSimulator
from imagine.pipelines.multinest_pipeline import MultinestPipeline
from imagine.pipelines.dynesty_pipeline import DynestyPipeline


class TestPipelines(unittest.TestCase):

    def test_multinest(self):
        # mock measures
        arr = np.random.rand(1, 3)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '3', 'nan'), arr, True)
        # simulator
        simer = LiSimulator(measuredict)
        # mock factory list
        tf = TestFieldFactory(active_parameters=tuple('a'))
        flist = (tf,)
        # mock likelihood
        lh = EnsembleLikelihood(measuredict)
        # mock prior
        pr = FlatPrior()
        # pipeline
        pipe = MultinestPipeline(simer, flist, lh, pr, 5)

        self.assertEqual(pipe.active_parameters, ('test_a',))
        self.assertEqual(pipe.factory_list, (tf,))
        self.assertEqual(pipe.simulator, simer)
        self.assertEqual(pipe.likelihood, lh)
        self.assertEqual(pipe.prior, pr)
        self.assertEqual(pipe.ensemble_size, 5)
        self.assertEqual(pipe.sampling_controllers, {})
        pipe.sampling_controllers = {'verbose': False}
        self.assertEqual(pipe.sampling_controllers, {'verbose': False})
        self.assertEqual(pipe.sample_callback, False)
        pipe.sample_callback = True
        self.assertEqual(pipe.sample_callback, True)
        self.assertEqual(pipe.likelihood_rescaler, 1.)
        pipe.likelihood_rescaler = 0.5
        self.assertEqual(pipe.likelihood_rescaler, 0.5)
        self.assertEqual(pipe.check_threshold, False)
        pipe.check_threshold = True
        self.assertEqual(pipe.check_threshold, True)
        self.assertEqual(pipe.likelihood_threshold, 0.)
        pipe.likelihood_threshold = -0.2
        self.assertEqual(pipe.likelihood_threshold, -0.2)
        self.assertEqual(pipe.random_seed, 0)
        pipe.random_seed = int(23)
        self.assertEqual(pipe.random_seed, int(23))

    def test_dynesty(self):
        # mock measures
        arr = np.random.rand(1, 3)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '3', 'nan'), arr, True)
        # simulator
        simer = BiSimulator(measuredict)
        # mock factory list
        tf = TestFieldFactory(active_parameters=tuple('a'))
        flist = (tf,)
        # mock likelihood
        lh = EnsembleLikelihood(measuredict)
        # mock prior
        pr = FlatPrior()
        # pipeline
        pipe = DynestyPipeline(simer, flist, lh, pr, 5)

        self.assertEqual(pipe.active_parameters, ('test_a',))
        self.assertEqual(pipe.factory_list, (tf,))
        self.assertEqual(pipe.simulator, simer)
        self.assertEqual(pipe.likelihood, lh)
        self.assertEqual(pipe.prior, pr)
        self.assertEqual(pipe.ensemble_size, 5)
        self.assertEqual(pipe.sampling_controllers, {})
        pipe.sampling_controllers = {'nlive': 1000}
        self.assertEqual(pipe.sampling_controllers, {'nlive': 1000})
        self.assertEqual(pipe.sample_callback, False)
        pipe.sample_callback = True
        self.assertEqual(pipe.sample_callback, True)
        self.assertEqual(pipe.likelihood_rescaler, 1.)
        pipe.likelihood_rescaler = 0.5
        self.assertEqual(pipe.likelihood_rescaler, 0.5)
        self.assertEqual(pipe.check_threshold, False)
        pipe.check_threshold = True
        self.assertEqual(pipe.check_threshold, True)
        self.assertEqual(pipe.likelihood_threshold, 0.)
        pipe.likelihood_threshold = -0.2
        self.assertEqual(pipe.likelihood_threshold, -0.2)
        self.assertEqual(pipe.random_seed, 0)
        pipe.random_seed = int(23)
        self.assertEqual(pipe.random_seed, int(23))


if __name__ == '__main__':
    unittest.main()
