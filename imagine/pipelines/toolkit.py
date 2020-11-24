"""
This module contains a set of tools for manipulating pipelines
"""
# %% IMPORTS
# Built-in imports
from collections import defaultdict

# Package imports
from e13tools import add_to_all
import numpy as np

# IMAGINE imports
import imagine as img

# All declaration
__all__ = []


# %% FUNCTION DEFINITIONS
@add_to_all
def get_central_value(pipeline):
    """
    Gets central point in the parameter space of a given pipeline

    The ranges are extracted from each prior. The result is a pure list of
    numbers corresponding to the values in the native units of each prior.

    For non-finite ranges (e.g. [-inf,inf]), a zero central value is assumed.

    Parameters
    ----------
    pipeline : imagine.pipelines.pipeline.Pipeline
        An imagine pipeline

    Returns
    -------
    central_values : list
        A list of parameter values at the centre of each
        parameter range
    """
    central_values = []
    for param in pipeline.active_parameters:
        centre = np.mean(pipeline.priors[param].range)
        # Removes units
        if hasattr(centre,'value'):
            centre = centre.value
        # Deals with priors with undefined ranges
        if not np.isfinite(centre):
            centre = 0.
        central_values.append(centre)
    return central_values


@add_to_all
def likelihood_convergence_report(pipeline, min_Nens=4, max_Nens=40,
                                  n_seeds=10, n_points=1, include_centre=True,
                                  verbose=True):
    """
    Constructs a report dataset based on a given Pipeline setup, which can
    be used for studying the *likelihood convergence* in a particular problem

    The pipeline's ensemble size is temporarily set to `Nens*n_seeds`, and
    (for each point) the present pipeline setup is used to compute a
    Simulations dictionary object.
    Subsets of this simulations object are then produced and the likelihood
    computed.

    The end result is a dictionary containing the likelihood values for:
    different choices of seed, ensemble size and points.

    Parameters
    ----------
    pipeline : imagine.pipelines.pipeline.Pipeline
        An imagine pipeline
    min_Nens : int
        Minimum ensemble size to be considered
    max_Nens : int
        Maximum ensemble size to be examined
    n_seeds : int
        Number of different (master) random seeds to be used
    n_points : int
        Number of points to be evaluated. Points are randomly drawn
        from the *prior* distribution (but see `include_centre`).
    include_centre : bool
        If `True`, the first point is taken as the value corresponding to
        the centre of each parameter range.

    Returns
    -------
    results : dict
        A dictionary containing the report data in the form of aligned lists.
        For easier manipulation, consider using this to generate a
        `pandas.DataFrame` object.
    """
    # Saves original choice for ensemble size
    original_size = pipeline.ensemble_size

    pipeline.ensemble_size = max_Nens * n_seeds

    n_params = len(pipeline.active_parameters)

    results = defaultdict(list)

    # Samples the prior distributions
    for i_point in range(1,n_points+1):
        if (i_point == 1) and include_centre:
            values = get_central_value(pipeline)
        else:
            # Draws a random point from the prior distributions
            values = pipeline.prior_transform(np.random.random_sample(n_params))

        if verbose:
            print('Working on point:', values, flush=True)

        # Produces max_Nens*n_seeds outputs from this
        maps = pipeline._get_observables(values)

        # Now, we construct smaller Simulations dicts based on a
        # subset of `maps`
        for i_seed in range(n_seeds):
            for Nens in range(min_Nens,max_Nens+1):
                # Constructs a slice corresponding to a subset
                indices = slice(i_seed*Nens, (i_seed+1)*Nens)
                # Constructs the subset Simulations and computes likelihood
                maps_subset = img.observables.extract_simulation_subset(maps,
                                                                        indices)
                L_value = pipeline.likelihood(maps_subset)

                # Stores everything
                results['indices'].append(indices)
                results['likelihood'].append(L_value)
                results['ensemble_size'].append(Nens)
                results['iseed'].append(i_seed)
                results['ipoint'].append(i_point)
                results['param_values'].append(values)

    # Restores original ensemble size
    pipeline.ensemble_size = original_size

    return results


