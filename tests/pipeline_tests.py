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
        self.assertEqual(pipe._ensemble_seeds, None)
        self.assertEqual(pipe.seed_tracer, int(0))
        self.assertEqual(pipe.random_type, 'free')

        # test free random seed, full randomness
        pipe._randomness()
        s1 = pipe._ensemble_seeds
        self.assertTrue(s1 is None)
        # test controllable random seed, with top level seed controllable
        pipe.random_type = 'controllable'
        pipe.seed_tracer = int(3)  # controlling seed at top level
        pipe._randomness()  # core func in assigning ensemble seeds, before calling simulator
        s1 = pipe._ensemble_seeds
        pipe._randomness()  # 2nd call of sampeler
        s2 = pipe._ensemble_seeds
        pipe = MultinestPipeline(simer, flist, lh, pr, 5)  # init a new sampler
        pipe.random_type = 'controllable'
        pipe.seed_tracer = int(3)  # repeat the controlling seed
        pipe._randomness()
        s1re = pipe._ensemble_seeds
        pipe._randomness()
        s2re = pipe._ensemble_seeds
        self.assertListEqual(list(s1), list(s1re))  # should get the same seeds
        self.assertListEqual(list(s2), list(s2re))
        pipe = MultinestPipeline(simer, flist, lh, pr, 5)
        pipe.random_type = 'controllable'
        pipe.seed_tracer = int(4)  # different controlling seed
        pipe._randomness()
        s1new = pipe._ensemble_seeds
        for i in range(len(s1)):
            self.assertNotEqual(s1[i], s1new[i])  # should get different seeds
        # test fixed random seed
        pipe.random_type = 'fixed'
        pipe.seed_tracer = int(5)
        pipe._randomness()  # 1st time seed assignment
        s1 = pipe._ensemble_seeds
        pipe._randomness()  # 2nd time seed assignment
        s1re = pipe._ensemble_seeds
        self.assertListEqual(list(s1), list(s1re))  # should get the same seeds
        

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
        self.assertEqual(pipe._ensemble_seeds, None)
        self.assertEqual(pipe.seed_tracer, int(0))
        self.assertEqual(pipe.random_type, 'free')

        # test free random seed
        pipe._randomness()
        s1 = pipe._ensemble_seeds
        self.assertTrue(s1 is None)
        # test controllable random seed
        pipe.random_type = 'controllable'
        pipe.seed_tracer = int(3)
        pipe._randomness()
        s1 = pipe._ensemble_seeds
        pipe._randomness()
        s2 = pipe._ensemble_seeds
        pipe = DynestyPipeline(simer, flist, lh, pr, 5)
        pipe.random_type = 'controllable'
        pipe.seed_tracer = int(3)
        pipe._randomness()
        s1re = pipe._ensemble_seeds
        pipe._randomness()
        s2re = pipe._ensemble_seeds
        self.assertListEqual(list(s1), list(s1re))
        self.assertListEqual(list(s2), list(s2re))
        pipe = DynestyPipeline(simer, flist, lh, pr, 5)
        pipe.random_type = 'controllable'
        pipe.seed_tracer = int(4)
        pipe._randomness()
        s1new = pipe._ensemble_seeds
        for i in range(len(s1)):
            self.assertNotEqual(s1[i], s1new[i])
        # test fixed random seed
        pipe.random_type = 'fixed'
        pipe.seed_tracer = int(5)
        pipe._randomness()
        s1 = pipe._ensemble_seeds
        pipe._randomness()
        s1re = pipe._ensemble_seeds
        self.assertListEqual(list(s1), list(s1re))


if __name__ == '__main__':
    unittest.main()
