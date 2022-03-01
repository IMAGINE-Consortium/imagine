import imagine as img
import nifty7 as ift
import astropy.units as u
import numpy as np
import os

from functions_nifty import get_los_response
from imagine_wrappers import RandomField, SpiralDensityField, SpiralMagneticLosField,\
    RandomFieldFactory, SpiralDensityFactory, SpiralMagneticLosFactory, MockSpiralSimulator,\
    make_observable_dict
from plot import field_plotter, data_plotter, oned_samples, twod_samples

if __name__ == '__main__':
    n_x = n_y = 500
    n_data = 100
    ift.random.push_sseq_from_seed(4243)

    interactive = False
    path = './evals/test5/'
    if not os.path.exists(path):
        os.makedirs(path)

    two_d_grid = img.fields.UniformGrid(box=[[0, n_x] * u.kpc,
                                             [0, n_y] * u.kpc,
                                             [0, 0] * u.kpc
                                             ],
                                        resolution=[n_x, n_y, 1])

    factory_list = []
    spiral_electron = SpiralDensityFactory(two_d_grid)
    spiral_electron.priors = {'k':  img.priors.GaussianPrior(mu=2.3, sigma=.2), }
    spiral_electron.active_parameters = ('k',)
    spiral_electron.default_parameters = {'log_a': .4}

    spiral_magnetic = SpiralMagneticLosFactory(two_d_grid)
    spiral_magnetic.priors = {'k':  img.priors.GaussianPrior(mu=1.2, sigma=.1),}
    spiral_magnetic.active_parameters = ('k',)
    spiral_magnetic.default_parameters = {'k': .7, 'log_a': .7}

    fixed_parameters = {'density_log_a': .4, 'magnetic_log_a': 2.5}
    ground_truth = {'magnetic_k': 1.3, 'density_k': 2.4}

    noise_sigma = 0.01

    domain = ift.makeDomain(ift.RGSpace((n_x, n_y)))
    starts = list(ift.random.current_rng().random((n_data, 2)).T)
    ang_pos = np.arctan2(starts[0], starts[1])
    indices = np.argsort(ang_pos)
    starts[0] = starts[0][indices]
    starts[1] = starts[1][indices]
    ang_pos = ang_pos[indices]
    response = get_los_response(domain, starts, [0.5, 0.2, ], n_data)

    model_dict = {'SpiralDensity': SpiralDensityField.get_function(domain),
                  'SpiralMagneticLos': SpiralMagneticLosField.get_function(domain)}

    imagine_data, nifty_data, groundtruth_dict = make_observable_dict(response, noise_sigma, model_dict,
                                                                      {**ground_truth, **fixed_parameters})
    simer = MockSpiralSimulator(grid=two_d_grid, response=response, measurements=imagine_data)

    groundtruth_dict.update({'summed': ift.utilities.my_sum(list(groundtruth_dict.values()))})
    groundtruth_dict.update({'backprojected_data': response.adjoint(nifty_data['data'])})

    field_plotter({'Magnetic': groundtruth_dict['SpiralMagneticLos']}, 'RdBu', interactive, path + 'magnetic_truth',)
    field_plotter({'Density': groundtruth_dict['SpiralDensity']}, 'magma', interactive, path + 'density_truth',)
    field_plotter({'backprojected_data': groundtruth_dict['backprojected_data']}, 'cmr.apple',
                  interactive, path + 'backprop', {'backprojected_data': {'vmin': -0.005, 'vmax': 0.005}})
    field_plotter({'combined': groundtruth_dict['SpiralDensity']*groundtruth_dict['SpiralMagneticLos']},
                  'cmr.pride', interactive, path + 'combined_truth',)
    data_plotter({'sres_truth': nifty_data['sres'], 'data': nifty_data['data']}, ang_pos, interactive,
                  path + 'truth_sres_vs_data')
    data_plotter({'sres_truth': nifty_data['sres']}, ang_pos, interactive, path + 'truth_sres+')
    data_plotter({'data': nifty_data['data']}, ang_pos, interactive, path + 'data')

    likelihood = img.likelihoods.SimpleLikelihood(imagine_data)

    pipeline = img.pipelines.MultinestPipeline(simulator=simer,
                                               run_directory=path,
                                               factory_list=[spiral_electron, spiral_magnetic],
                                               likelihood=likelihood,
                                               ensemble_size=2,
                                               show_progress_reports=False,

                                               )

    pipeline.sampling_controllers = {'n_live_points': 500, 'max_ncalls': 12000,  'evidence_tolerance' : 0.5}
    results = pipeline()
    td = ift.Field(response.target, np.asarray(list(imagine_data.archive.values())[0].data[0]))
    if len(ground_truth.values()) == 2:

    twod_samples(pipeline.samples, *ground_truth.values())
    #oned_samples({'log_a': pipeline.samples.values()[0].to_value()},
    #             ground_truth if ground_truth is not None else {}, False, path + 'hist_' +
    scalar_domain = ift.makeDomain(ift.DomainTuple.scalar_domain())
    density_position = ift.MultiField.from_dict({'density_k': ift.Field(,
                                                              np.mean(pipeline.samples['density_k']).value),
                                                 'density_log_a': ift.Field(ift.makeDomain(ift.DomainTuple.scalar_domain()),
                                                                            fixed_parameters['density_log_a'])
                                                                            })
    magnetic_position = ift.MultiField.from_dict({'magnetic_k': ift.Field(ift.makeDomain(ift.DomainTuple.scalar_domain()),
                                                                          fixed_parameters['magnetic_k']),
                                                  'magnetic_log_a': ift.Field(ift.makeDomain(ift.DomainTuple.scalar_domain()),
                                                                        np.mean(pipeline.samples['magnetic_log_a']).value),
                                                  })
    print(density_position.val)
    density_field = model_dict['SpiralDensity'].force(density_position)
    magnetic_field = model_dict['SpiralMagneticLos'].force(magnetic_position)
    field_plotter({'magnetic_field': magnetic_field}, 'RdBu', interactive,
                  path + 'field_' + str('img'))
    field_plotter({'spiral_density': density_field}, 'magma', interactive,
                  path + 'field_' + str('img'))
    data_plotter({'sres': (response(magnetic_field*density_field)), 'data': td}, ang_pos, interactive,
                 path + 'sres_vs_data_' + str('img'))
