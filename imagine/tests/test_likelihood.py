# %% IMPORTS
# Package imports
from mpi4py import MPI
import numpy as np
import pytest

# IMAGINE imports
from imagine.observables import (
    Observable, Simulations, Measurements, Covariances)
from imagine.likelihoods import (
  SimpleLikelihood, EnsembleLikelihood, EnsembleLikelihoodDiagonal)
from imagine.tools.covariance_estimator import diagonal_mcov

# Globals
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

# Marks tests in this module as quick
pytestmark = pytest.mark.quick

# %% PYTEST DEFINITIONS
class TestSimpleLikeli(object):
    def test_without_cov(self):
        simdict = Simulations()
        meadict = Measurements()
        # mock measurements
        arr_a = np.random.rand(1, 48)
        comm.Bcast(arr_a, root=0)
        mea = Observable(arr_a, 'measured')
        meadict.append(name=('test', None, 2, None), data=mea, otype='HEALPix')
        # mock sims
        arr_b = np.random.rand(3, 48)
        sim = Observable(arr_b, 'simulated')
        simdict.append(name=('test', None, 2, None), data=sim, otype='HEALPix')
        # no covariance
        lh = SimpleLikelihood(meadict)
        # calc by likelihood
        result = lh(simdict)  # feed variable value, not parameter value
        # calc by hand
        full_b = np.vstack(comm.allgather(arr_b))  # global arr_b
        diff = (np.mean(full_b, axis=0) - arr_a)
        baseline = -float(0.5)*float(np.vdot(diff, diff))
        # comapre
        assert np.allclose(result, baseline)

    def test_with_cov(self):
        simdict = Simulations()
        meadict = Measurements()
        covdict = Covariances()
        # mock measurements
        arr_a = np.random.rand(1, 4*mpisize)
        comm.Bcast(arr_a, root=0)
        mea = Observable(arr_a, 'measured')
        meadict.append(name=('test', None, 4*mpisize, None),
                       data=mea, otype='plain')
        # mock sims
        arr_b = np.random.rand(5, 4*mpisize)
        sim = Observable(arr_b, 'simulated')
        simdict.append(name=('test', None, 4*mpisize, None),
                       data=sim, otype='plain')
        # mock covariance
        arr_c = np.random.rand(4, 4*mpisize)
        cov = Observable(arr_c, 'covariance')
        covdict.append(name=('test', None, 4*mpisize, None),
                       cov_data=cov)
        # with covariance
        lh = SimpleLikelihood(meadict, covdict)
        # calc by likelihood
        result = lh(simdict)  # feed variable value, not parameter value
        # calc by hand
        full_b = np.vstack(comm.allgather(arr_b))  # global arr_b
        diff = (np.mean(full_b, axis=0) - arr_a)
        full_cov = np.vstack(comm.allgather(arr_c))  # global covariance
        (sign, logdet) = np.linalg.slogdet(full_cov*2.*np.pi)
        baseline = -0.5*(np.vdot(diff, np.linalg.solve(full_cov, diff.T))+sign*logdet)
        assert np.allclose(result, baseline)


class TestEnsembleLikeli(object):
    def test(self):
        simdict = Simulations()
        meadict = Measurements()
        covdict = Covariances()
        # mock measurements
        arr_a = np.random.rand(1, 4*mpisize)
        comm.Bcast(arr_a, root=0)
        mea = Observable(arr_a, 'measured')
        meadict.append(name=('test', None, 4*mpisize, None),
                       data=mea, otype='plain')
        # mock covariance
        arr_c = np.random.rand(4, 4*mpisize)
        cov = Observable(arr_c, 'covariance')
        covdict.append(name=('test', None, 4*mpisize, None),
                       cov_data=cov)
        # mock observable with repeated single realisation
        arr_b = np.random.rand(1, 4*mpisize)
        comm.Bcast(arr_b, root=0)
        arr_ens = np.zeros((2, 4*mpisize))
        for i in range(len(arr_ens)):
            arr_ens[i] = arr_b
        sim = Observable(arr_ens, 'simulated')
        simdict.append(name=('test', None, 4*mpisize, None),
                       data=sim, otype='plain')

        # simplelikelihood
        lh_simple = SimpleLikelihood(meadict, covdict)
        result_simple = lh_simple(simdict)
        # ensemblelikelihood
        lh_ensemble = EnsembleLikelihood(meadict, covdict,
                                         use_trace_approximation=False)
        result_ensemble = lh_ensemble(simdict)
        assert result_ensemble == result_simple

    def test_with_trace_approximation(self):
        simdict = Simulations()
        meadict = Measurements()
        covdict = Covariances()
        # mock measurements
        arr_a = np.random.rand(1, 4*mpisize)
        comm.Bcast(arr_a, root=0)
        mea = Observable(arr_a, 'measured')
        meadict.append(name=('test', None, 4*mpisize, None),
                       data=mea, otype='plain')
        # mock covariance (NB for the trace approximation to work, the data
        # covariance needs to be diagonal)
        arr_c = np.diag(np.random.rand(4))

        cov = Observable(arr_c, 'covariance')
        covdict.append(name=('test', None, 4*mpisize, None),
                       cov_data=cov)
        # mock observable with repeated single realisation
        arr_b = np.random.rand(1, 4*mpisize)
        comm.Bcast(arr_b, root=0)
        arr_ens = np.zeros((2, 4*mpisize))
        for i in range(len(arr_ens)):
            arr_ens[i] = arr_b
        sim = Observable(arr_ens, 'simulated')
        simdict.append(name=('test', None, 4*mpisize, None),
                       data=sim, otype='plain')

        # simplelikelihood
        lh_simple = SimpleLikelihood(meadict, covdict)
        result_simple = lh_simple(simdict)
        # ensemblelikelihood
        lh_ensemble = EnsembleLikelihood(meadict, covdict,
                                         use_trace_approximation=True)
        result_ensemble = lh_ensemble(simdict)
        assert result_ensemble == result_simple

    def test_diag(self):
        simdict = Simulations()
        meadict = Measurements()
        covdict = Covariances()
        # mock measurements
        arr_a = np.random.rand(1, 40)
        mea = Observable(arr_a, 'measured')
        meadict.append(name=('test', None, 40, None),
                       data=mea, otype='plain')
        # mock (diagonal) covariance
        arr_var = np.random.rand(40)
        cov = Observable(np.diag(arr_var), 'covariance')
        covdict.append(name=('test', None, 40, None),
                       cov_data=cov)
        # mock observable
        arr_ens = np.random.rand(10, 40)

        sim = Observable(arr_ens, 'simulated')
        simdict.append(name=('test', None, 40, None),
                       data=sim, otype='plain')
        # ensemblelikelihood + diagonal_covcov
        lh_ens = EnsembleLikelihood(meadict, covdict, cov_func=diagonal_mcov)
        result_ens = lh_ens(simdict)
        # EnsembleLikelihoodDiagonal
        lh_diag = EnsembleLikelihoodDiagonal(meadict, covdict)
        result_diag = lh_diag(simdict)

        assert np.allclose(result_diag, result_ens)
