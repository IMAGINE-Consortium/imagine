import numpy as np
import logging as log

import pymultinest

from imagine.pipelines.pipeline import Pipeline
from imagine.tools.icy_decorator import icy


@icy
class MultinestPipeline(Pipeline):

    def __init__(self, simulator, factory_list, likelihood, prior, ensemble_size=1):
        """
        """
        super(MultinestPipeline, self).__init__(simulator, factory_list, likelihood, prior, ensemble_size)

    def __call__(self):
        """

        :return: pyMultinest sampling results
        """
        # run pymultinest
        results = pymultinest.solve(LogLikelihood=self._core_likelihood,
                                    Prior=self.prior,
                                    n_dims=len(self._active_parameters),
                                    **self._sampling_controllers)
        return results
