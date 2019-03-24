import unittest
import numpy as np
import mpi4py
from nifty5 import Field, UnstructuredDomain, RGSpace, HPSpace, DomainTuple
from imagine.observables.observable import Observable
from imagine.observables.observable_dict import Simulations, Measurements, Covariances
from imagine.likelihoods.likelihood import Likelihood
from imagine.likelihoods.simple_likelihood import SimpleLikelihood
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood

comm = mpi4py.MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()


class TestSimpleLikeli(unittest.TestCase):

    def test_without_cov(self):
        simdict = Simulations()
        meadict = Measurements()
        # mock measurements
        dtuple = DomainTuple.make((RGSpace(1), HPSpace(nside=2)))
        arr_a = np.random.rand(1, 48)
        comm.Bcast(arr_a, root=0)
        mea = Observable(dtuple, arr_a)
        meadict.append(('test', 'nan', '2', 'nan'), mea)
        # mock sims
        dtuple = DomainTuple.make((RGSpace(3*mpisize), HPSpace(nside=2)))
        arr_b = np.random.rand(3, 48)
        sim = Observable(dtuple, arr_b)
        simdict.append(('test', 'nan', '2', 'nan'), sim)
        # no covariance
        lh = SimpleLikelihood(meadict)
        # calc by likelihood
        rslt = lh(simdict)  # feed variable value, not parameter value
        # calc by hand
        arr_b = sim.to_global_data()  # global arr_b
        diff = (np.mean(arr_b, axis=0) - arr_a)
        baseline = -float(0.5)*float(np.vdot(diff, diff))
        # comapre
        self.assertAlmostEqual(rslt, baseline)
    
    def test_with_cov(self):
        simdict = Simulations()
        meadict = Measurements()
        covdict = Covariances()
        # mock measurements
        dtuple = DomainTuple.make((RGSpace(1), RGSpace(12)))
        arr_a = np.random.rand(1, 12)
        comm.Bcast(arr_a, root=0)
        mea = Observable(dtuple, arr_a)
        meadict.append(('test', 'nan', '12', 'nan'), mea, True)
        # mock sims
        dtuple = DomainTuple.make((RGSpace(5*mpisize), RGSpace(12)))
        arr_b = np.random.rand(5, 12)
        sim = Observable(dtuple, arr_b)
        simdict.append(('test', 'nan', '12', 'nan'), sim, True)
        # mock covariance
        arr_c = np.random.rand(12, 12)
        comm.Bcast(arr_c, root=0)
        dtuple = DomainTuple.make((RGSpace(shape=arr_c.shape)))
        cov = Field.from_global_data(dtuple, arr_c)
        covdict.append(('test', 'nan', '12', 'nan'), cov, True)
        # with covariance
        lh = SimpleLikelihood(meadict, covdict)
        # calc by likelihood
        rslt = lh(simdict)  # feed variable value, not parameter value
        # calc by hand
        arr_b = sim.to_global_data()  # get global arr_b
        diff = (np.mean(arr_b, axis=0) - arr_a)
        (sign, logdet) = np.linalg.slogdet(arr_c*2.*np.pi)
        baseline = -float(0.5)*float(np.vdot(diff, np.linalg.solve(arr_c, diff.T))+sign*logdet)
        self.assertAlmostEqual(rslt, baseline)
    

class TestEnsembleLikeli(unittest.TestCase):
    
    def test_without_simcov(self):
        simdict = Simulations()
        meadict = Measurements()
        covdict = Covariances()
        # mock measurements
        dtuple = DomainTuple.make((RGSpace(1), HPSpace(nside=2)))
        arr_a = np.random.rand(1, 48)
        comm.Bcast(arr_a, root=0)
        mea = Observable(dtuple, arr_a)
        meadict.append(('test', 'nan', '2', 'nan'), mea)
        # mock covariance
        dtuple = DomainTuple.make((RGSpace(shape=(48, 48))))
        arr_c = np.random.rand(48, 48)
        comm.Bcast(arr_c, root=0)
        cov = Field.from_global_data(dtuple, arr_c)
        covdict.append(('test', 'nan', '2', 'nan'), cov)
        # mock observable with repeated single realisation
        dtuple = DomainTuple.make((RGSpace(5*mpisize), HPSpace(nside=2)))
        arr_b = np.random.rand(1, 48)
        comm.Bcast(arr_b, root=0)
        arr_ens = np.zeros((5, 48))
        for i in range(len(arr_ens)):
            arr_ens[i] = arr_b
        sim = Observable(dtuple, arr_ens)
        simdict.append(('test', 'nan', '2', 'nan'), sim)
        # simplelikelihood
        lh_simple = SimpleLikelihood(meadict, covdict)
        rslt_simple = lh_simple(simdict)
        # ensemblelikelihood
        lh_ensemble = EnsembleLikelihood(meadict, covdict)
        rslt_ensemble = lh_ensemble(simdict)
        self.assertEqual(rslt_ensemble, rslt_simple)
        
    def test_without_cov(self):
        simdict = Simulations()
        meadict = Measurements()
        # mock measurements
        dtuple = DomainTuple.make((RGSpace(1), HPSpace(nside=2)))
        arr_a = np.random.rand(1, 48)
        comm.Bcast(arr_a, root=0)
        mea = Observable(dtuple, arr_a)
        meadict.append(('test', 'nan', '2', 'nan'), mea)
        # mock observable with repeated single realisation
        dtuple = DomainTuple.make((RGSpace(5*mpisize), HPSpace(nside=2)))
        arr_b = np.random.rand(1, 48)
        comm.Bcast(arr_b, root=0)
        arr_ens = np.zeros((5, 48))
        for i in range(len(arr_ens)):
            arr_ens[i] = arr_b
        sim = Observable(dtuple, arr_ens)
        simdict.append(('test', 'nan', '2', 'nan'), sim)
        # simplelikelihood
        lh_simple = SimpleLikelihood(meadict)
        rslt_simple = lh_simple(simdict)
        # ensemblelikelihood
        lh_ensemble = EnsembleLikelihood(meadict)
        rslt_ensemble = lh_ensemble(simdict)
        self.assertEqual(rslt_ensemble, rslt_simple)


if __name__ == '__main__':
    unittest.main()
