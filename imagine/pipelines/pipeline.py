import numpy as np
import logging as log
from imagine.likelihoods.likelihood import Likelihood
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.priors.prior import GeneralPrior
from imagine.simulators.simulator import Simulator
from imagine.tools.random_seed import ensemble_seed_generator
from imagine.tools.icy_decorator import icy
import imagine.tools.misc as misc
from astropy.table import QTable

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

class Pipeline(object):
    """
    Base class used for for initialing Bayesian analysis pipeline

    Attributes
    ----------
    dynesty_parameter_dict : dict
        extra parameters for controlling Dynesty
        i.e., 'nlive', 'bound', 'sample'
    sample_callback : bool
        not implemented yet
    likelihood_rescaler : double
        Rescale log-likelihood value
    random_type : str
        If set to 'fixed', the exact same set of ensemble seeds will be used for
        the evaluation of all fields, generated using the `master_seed`.
        If set to 'controllable', each individual field will get their own set of
        ensemble fields, but multiple runs will lead to the same results, as
        they are based on the same `master_seed`.
        If set to 'free', every time the pipeline is run, the `master_seed` is
        reset to a different value, and the ensemble seeds for each individual
        field are drawn based on this.
    master_seed : int
        Master seed used by the random number generators


    Parameters
    ----------
    simulator : imagine.simulators.simulator.Simulator
        Simulator object
    factory_list : list
        List or tuple of field factory objects
    likelihood : imagine.likelihoods.likelihood.Likelihood
        Likelihood object
    prior : imagine.priors.prior.Prior
        Prior object
    ensemble_size : int
        Number of observable realizations PER COMPUTING NODE to be generated in simulator
    """
    def __init__(self, simulator, factory_list, likelihood, ensemble_size=1):

        self.factory_list = factory_list
        # NB setting the factory list automatically sets: the active parameters,
        # parameter ranges and priors, based on the list
        self.simulator = simulator
        self.likelihood = likelihood
        self.ensemble_size = ensemble_size
        self.sampling_controllers = dict()
        self.sample_callback = False

        self.distribute_ensemble = True

        # rescaling total likelihood in _core_likelihood
        self.likelihood_rescaler = 1.
        # Checking likelihood threshold
        self.check_threshold = False
        self.likelihood_threshold = 0.

        # This changes on every execution is random_type=='free'
        self.master_seed = 1
        self.random_type = 'controllable'
        # The ensemble_seeds are fixed in the case of the 'fixed' random_type;
        # or are regenerated on each Field evaluation, in the 'free' and
        # 'controllable' cases
        self.ensemble_seeds = None

        # Place holders
        self.sampler = None
        self.results = None
        self._evidence = None
        self._evidence_err = None
        self._posterior_summary = None
        self._samples_array = None
        self._samples = None


    @property
    def active_parameters(self):
        """
        List of all the active parameters
        """
        # The user should not be able to set this attribute manually
        return self._active_parameters

    @property
    def active_ranges(self):
        """
        Ranges of all active parameters
        """
        # The user should not be able to set this attribute manually
        return self._active_ranges

    @property
    def priors(self):
        """
        Dictionary containing priors for all active parameters
        """
        # The user should not be able to set this attribute manually
        return self._priors

    @property
    def posterior_summary(self):
        r"""
        A dictionary containing a summary of posterior statistics for each of
        the active parameters.  These are: 'median', 'errlo'
        (15.87th percentile), 'errup' (84.13th percentile), 'mean' and 'stdev'.
        """

        if self._posterior_summary is None:
            samp = self.samples

            self._posterior_summary = {}

            for column in samp.columns:
                percentiles = np.percentile(samp[column], [15.87, 50,  84.13])
                column_dict = {s:  percentiles[i]
                               for i,s in enumerate(('errlo', 'median', 'errup'))}
                column_dict['mean'] = np.mean(samp[column])
                column_dict['stdev'] = np.std(samp[column])
                self._posterior_summary[column] = column_dict

        return self._posterior_summary

    def posterior_report(self, significant_digits=2):
        """
        Displays the best fit values and 1-sigma errors for each active parameter.

        If running on a jupyter-notebook, a nice LaTeX display is used.

        Parameters
        ----------
        significant_digits : int
            The number of significant digits to be used
        """
        from IPython.display import display, Math
        """Reports the """
        out = ''

        for param, pdict in self.posterior_summary.items():
            if misc.is_notebook():
                out += r'\\ \text{ '
            out += param
            med, low, up = (pdict[a].value for a in ['median', 'errlo', 'errup'])

            v, l, u = misc.adjust_error_intervals(med, low, up, digits=significant_digits)
            if misc.is_notebook():
                # Extracts LaTeX representation from astropy unit object
                unit_latex = pdict['median'].unit._repr_latex_().replace('$','')
                # Prepares LaTeX output string
                out += r': }}\; {0}_{{ {1} }}^{{ +{2} }}\,'.format(v, l, u)
                out += unit_latex +r'\\'
            else:
                out += r': {0} ({1})/(+{2}) '.format(v, l, u)
                out += str(pdict['median'].unit)+'\n'

        if misc.is_notebook():
            display(Math(out))
        else:
            print(out)

    @property
    def log_evidence(self):
        r"""
        Natural logarithm of the *marginal likelihood* or *Bayesian model evidence*,
        :math:`\ln\mathcal{Z}`, where

        .. math::
            \mathcal{Z} = P(d|m) = \int_{\Omega_\theta} P(d | \theta, m) P(\theta | m) \mathrm{d}\theta .

        Note
        ----
        Available only after the pipeline is run.
        """
        if self._evidence is None:
            raise ValueError('Evidence not set! Have you run the pipeline?')
        else:
            return self._evidence

    @property
    def log_evidence_err(self):
        """
        Error estimate in the natural logarithm of the *Bayesian model evidence*.
        Available once the pipeline is run.

        Note
        ----
        Available only after the pipeline is run.
        """
        assert self._evidence_err is not None, 'Evidence error not set! Did you run the pipeline?'

        return self._evidence_err

    @property
    def samples_scaled(self):
        """
        An :py:class:`astropy.table.QTable` object containing parameter values of the samples
        produced in the run, scaled to the interval [0,1].
        """
        assert self._samples_array is not None, 'Samples not available. Did you run the pipeline?'
        return QTable(data=self._samples_array, names=self.active_parameters)

    @property
    def samples(self):
        """
        An :py:class:`astropy.table.QTable` object containing parameter values of the samples
        produced in the run.
        """
        if self._samples is None:
            self._samples = self.samples_scaled

            for param in self.active_parameters:
                pmin, pmax = self.active_ranges[param]
                self._samples[param] = self._samples[param]*(pmax - pmin)+pmin

        return self._samples

    @property
    def factory_list(self):
        """
        List of the
        :py:class:`Field Factories <imagine.fields.field_factory.GeneralFieldFactory>`
        currently being used.

        Updating the factory list automatically extracts active_parameters,
        parameter ranges and priors from each field factory.
        """
        return self._factory_list

    @factory_list.setter
    def factory_list(self, factory_list):
        # Notice that the parameter/variable ordering is fixed wrt
        # factory ordering. This is useful for recovering variable logic value
        # for each factory and necessary to construct the common prior function.
        assert isinstance(factory_list, (list, tuple)), 'Factory list must be a tuple or list'
        self._active_parameters = tuple()
        self._active_ranges = dict()
        self._priors = dict()

        for factory in factory_list:
            assert isinstance(factory, GeneralFieldFactory)
            for ap_name in factory.active_parameters:
                assert isinstance(ap_name, str)
                # Sets the parameters and ranges
                self._active_parameters += (str(factory.name+'_'+ap_name),)
                self._active_ranges[str(factory.name+'_'+ap_name)] = factory.parameter_ranges[ap_name]
                # Sets the Prior
                prior = factory.priors[ap_name]
                assert isinstance(prior, GeneralPrior)
                self._priors[str(factory.name+'_'+ap_name)] = prior
        self._factory_list = factory_list

    @property
    def sampler_supports_mpi(self):
        raise NotImplementedError('Value of property must be set in sub-class!')

    @property
    def simulator(self):
        """
        The :py:class:`Simulator <imagine.simulators.simulator.Simulator>`
        object used by the pipeline
        """
        return self._simulator

    @simulator.setter
    def simulator(self, simulator):
        assert isinstance(simulator, Simulator)
        self._simulator = simulator

    @property
    def likelihood(self):
        """
        The :py:class:`Likelihood <imagine.likelihoods.likelihood.Likelihood>`
        object used by the pipeline
        """
        return self._likelihood

    @likelihood.setter
    def likelihood(self, likelihood):
        assert isinstance(likelihood, Likelihood)
        self._likelihood = likelihood


    def prior_pdf(self, cube):
        """
        Probability distribution associated with the all parameters being used by
        the multiple Field Factories

        Parameters
        ----------
        cube : array
            Each row of the array corresponds to a different parameter in the sampling.

        Returns
        -------
        cube_rtn
            The modified cube
        """
        cube_rtn = np.empty_like(cube)
        for i, parameter in enumerate(self.priors):
            cube_rtn[i] = self.priors[parameter].pdf(cube_rtn[i])
        return cube_rtn


    def prior_transform(self, cube):
        """
        Prior transform cube (i.e. MultiNest style prior).

        Takes a cube containing a uniform sampling of  values and maps then onto
        a distribution compatible with the priors specified in the
        Field Factories.

        Parameters
        ----------
        cube : array
            Each row of the array corresponds to a different parameter in the sampling.
            Warning: the function will modify `cube` inplace.

        Returns
        -------
        cube
            The modified cube
        """
        for i, parameter in enumerate(self.priors):
            cube[i] = self.priors[parameter](cube[i])
        return cube

    @property
    def distribute_ensemble(self):
        """
        If True, whenever the sampler requires a likelihood evaluation,
        the ensemble of stochastic fields realizations is distributed among
        all the nodes.

        Otherwise, each likelihood evaluations will go through the whole
        ensemble size on a single node. See :doc:`parallel` for details.
        """
        return self._distribute_ensemble

    @distribute_ensemble.setter
    def distribute_ensemble(self, distr_ensemble):
        if distr_ensemble:
            self._likelihood_function = self._mpi_likelihood
        else:
            self._likelihood_function = self._core_likelihood
        self._distribute_ensemble = distr_ensemble

    @property
    def ensemble_size(self):
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, ensemble_size):
        ensemble_size = int(ensemble_size)
        assert (ensemble_size > 0)
        self._ensemble_size = ensemble_size
        log.debug('set ensemble size to %i' % int(ensemble_size))

    @property
    def sampling_controllers(self):
        return self._sampling_controllers

    @sampling_controllers.setter
    def sampling_controllers(self, pp_dict):
        try:
            self._sampling_controllers.update(pp_dict)
        except AttributeError:
            self._sampling_controllers = pp_dict

    def tidy_up(self):
        """
        Resets internal state before a new run
        """
        self.results = None
        self._evidence = None
        self._evidence_err = None
        self._posterior_summary = None
        self._samples_array = None
        self._samples = None
        self._randomness()



    def _randomness(self):
        """
        Manipulate random seed(s)
        isolating this process for convenience of testing
        """
        log.debug('@ pipeline::_randomness')

        assert self.random_type in ('free', 'controllable', 'fixed')

        if self.random_type == 'free':
            # Refreshes the master seed
            self.master_seed = np.random.randint(0, 2**32)

        # Updates numpy random accordingly
        np.random.seed(self.master_seed)

        if self.random_type == 'fixed':
            self.ensemble_seeds = ensemble_seed_generator(self.ensemble_size)
        else:
            self.ensemble_seeds = None


    def _core_likelihood(self, cube):
        """
        core log-likelihood calculator

        Parameters
        ----------
        cube
            list of variable values

        Returns
        -------
        log-likelihood value
        """
        log.debug('@ pipeline::_core_likelihood')
        log.debug('sampler at %s' % str(cube))
        # security boundary check
        if np.any(cube > 1.) or np.any(cube < 0.):
            log.debug('cube %s requested. returned most negative possible number' % str(cube))
            return np.nan_to_num(-np.inf)
        # return active variables from pymultinest cube to factories
        # and then generate new field objects
        head_idx = 0
        tail_idx = 0
        field_list = tuple()

        # the ordering in factory list and variable list is vital
        for factory in self._factory_list:
            variable_dict = dict()
            tail_idx = head_idx + len(factory.active_parameters)
            factory_cube = cube[head_idx:tail_idx]
            for i, av in enumerate(factory.active_parameters):
                variable_dict[av] = factory_cube[i]
            field_list += (factory.generate(variables=variable_dict,
                                            ensemble_size=self.ensemble_size,
                                            ensemble_seeds=self.ensemble_seeds),)
            log.debug('create '+factory.name+' field')
            head_idx = tail_idx
        assert(head_idx == len(self._active_parameters))

        observables = self._simulator(field_list)
        # apply mask
        observables.apply_mask(self.likelihood.mask_dict)
        # add up individual log-likelihood terms
        current_likelihood = self.likelihood(observables)
        # check likelihood value until negative (or no larger than given threshold)
        if self.check_threshold and current_likelihood > self.likelihood_threshold:
            raise ValueError('log-likelihood beyond threshold')
        return current_likelihood * self.likelihood_rescaler


    def _mpi_likelihood(self, cube):
        """
        mpi log-likelihood calculator
        PyMultinest supports execution with MPI
        where sampler on each node follows DIFFERENT journeys in parameter space
        but keep in communication
        so we need to firstly register parameter position on each node
        and calculate log-likelihood value of each node with joint force of all nodes
        in this way, ensemble size is multiplied by the number of working nodes

        Parameters
        ----------
        cube
            list of variable values

        Returns
        -------
        log-likelihood value
        """

        if self.sampler_supports_mpi:

            log.debug('@ multinest_pipeline::_mpi_likelihood')

            # Gathers cubes from all nodes
            cube_local_size = cube.size
            cube_pool = np.empty(cube_local_size*mpisize, dtype=np.float64)
            comm.Allgather([cube, MPI.DOUBLE], [cube_pool, MPI.DOUBLE])

            # Calculates log-likelihood for each node
            loglike_pool = np.empty(mpisize, dtype=np.float64)
            for i in range(mpisize):  # loop through nodes
                cube_local = cube_pool[i*cube_local_size : (i+1)*cube_local_size]

                loglike_pool[i] = self._core_likelihood(cube_local)

            # Scatters log-likelihood to each node
            loglike_local = np.empty(1, dtype=np.float64)
            comm.Scatter([loglike_pool, MPI.DOUBLE], [loglike_local, MPI.DOUBLE], root=0)

            return loglike_local[0] # Some samplers require a scalar value

        else:

            log.debug('@ dynesty_pipeline::_mpi_likelihood')
            # gather cubes from all nodes
            cube_local_size = cube.size
            cube_pool = np.empty(cube_local_size*mpisize, dtype=np.float64)
            comm.Allgather([cube, MPI.DOUBLE], [cube_pool, MPI.DOUBLE])
            # check if all nodes are at the same parameter-space position
            assert ((cube_pool == np.tile(cube_pool[:cube_local_size], mpisize)).all())
            return self._core_likelihood(cube)
