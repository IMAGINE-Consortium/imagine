import unittest
import numpy as np
from mpi4py import MPI
from imagine.observables.observable import Observable
from imagine.observables.observable_dict import Simulations, Measurements, Covariances
from imagine.likelihoods.simple_likelihood import SimpleLikelihood
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

class TestSimpleLikeli(unittest.TestCase):

    def test_without_cov(self):
        simdict = Simulations()
        meadict = Measurements()
        # mock measurements
        arr_a = np.random.rand(1, 48)
        comm.Bcast(arr_a, root=0)
        mea = Observable(arr_a, 'measured')
        meadict.append(('test', 'nan', '2', 'nan'), mea)
        # mock sims
        arr_b = np.random.rand(3, 48)
        sim = Observable(arr_b, 'simulated')
        simdict.append(('test', 'nan', '2', 'nan'), sim)
        # no covariance
        lh = SimpleLikelihood(meadict)
        # calc by likelihood
        rslt = lh(simdict)  # feed variable value, not parameter value
        # calc by hand
        full_b = np.vstack(comm.allgather(arr_b))  # global arr_b
        diff = (np.mean(full_b, axis=0) - arr_a)
        baseline = -float(0.5)*float(np.vdot(diff, diff))
        # comapre
        self.assertAlmostEqual(rslt, baseline)
    
    def test_with_cov(self):
        simdict = Simulations()
        meadict = Measurements()
        covdict = Covariances()
        # mock measurements
        arr_a = np.random.rand(1, 4*mpisize)
        comm.Bcast(arr_a, root=0)
        mea = Observable(arr_a, 'measured')
        meadict.append(('test', 'nan', str(4*mpisize), 'nan'), mea, True)
        # mock sims
        arr_b = np.random.rand(5, 4*mpisize)
        sim = Observable(arr_b, 'simulated')
        simdict.append(('test', 'nan', str(4*mpisize), 'nan'), sim, True)
        # mock covariance
        arr_c = np.random.rand(4, 4*mpisize)
        cov = Observable(arr_c, 'covariance')
        covdict.append(('test', 'nan', str(4*mpisize), 'nan'), cov, True)
        # with covariance
        lh = SimpleLikelihood(meadict, covdict)
        # calc by likelihood
        rslt = lh(simdict)  # feed variable value, not parameter value
        # calc by hand
        full_b = np.vstack(comm.allgather(arr_b))  # global arr_b
        diff = (np.mean(full_b, axis=0) - arr_a)
        full_cov = np.vstack(comm.allgather(arr_c))  # global covariance
        (sign, logdet) = np.linalg.slogdet(full_cov*2.*np.pi)
        baseline = -float(0.5)*float(np.vdot(diff, np.linalg.solve(full_cov, diff.T))+sign*logdet)
        self.assertAlmostEqual(rslt, baseline)
    

class TestEnsembleLikeli(unittest.TestCase):
    
    def test_without_simcov(self):
        simdict = Simulations()
        meadict = Measurements()
        covdict = Covariances()
        # mock measurements
        arr_a = np.random.rand(1, 4*mpisize)
        comm.Bcast(arr_a, root=0)
        mea = Observable(arr_a, 'measured')
        meadict.append(('test', 'nan', str(4*mpisize), 'nan'), mea, True)
        # mock covariance
        arr_c = np.random.rand(4, 4*mpisize)
        cov = Observable(arr_c, 'covariance')
        covdict.append(('test', 'nan', str(4*mpisize), 'nan'), cov, True)
        # mock observable with repeated single realisation
        arr_b = np.random.rand(1, 4*mpisize)
        comm.Bcast(arr_b, root=0)
        arr_ens = np.zeros((2, 4*mpisize))
        for i in range(len(arr_ens)):
            arr_ens[i] = arr_b
        sim = Observable(arr_ens, 'simulated')
        simdict.append(('test', 'nan', str(4*mpisize), 'nan'), sim, True)
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
        arr_a = np.random.rand(1, 12*mpisize**2)
        comm.Bcast(arr_a, root=0)
        mea = Observable(arr_a, 'measured')
        meadict.append(('test', 'nan', str(mpisize), 'nan'), mea)
        # mock observable with repeated single realisation
        arr_b = np.random.rand(1, 12*mpisize**2)
        comm.Bcast(arr_b, root=0)
        if not mpirank:
            arr_ens = np.zeros((3, 12*mpisize**2))
        else:
            arr_ens = np.zeros((2, 12*mpisize**2))
        for i in range(len(arr_ens)):
            arr_ens[i] = arr_b
        sim = Observable(arr_ens, 'simulated')
        simdict.append(('test', 'nan', str(mpisize), 'nan'), sim)
        # simplelikelihood
        lh_simple = SimpleLikelihood(meadict)
        rslt_simple = lh_simple(simdict)
        # ensemblelikelihood
        lh_ensemble = EnsembleLikelihood(meadict)
        rslt_ensemble = lh_ensemble(simdict)
        self.assertEqual(rslt_ensemble, rslt_simple)
        

if __name__ == '__main__':
    unittest.main()
