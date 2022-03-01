import numpy as np
import nifty7 as ift
import os

import functions_nifty as nfc
from imagine_wrappers import make_observable_dict
from plot import field_plotter, data_plotter
from imagine_evaluation import imagine_evaluation


def evaluation(param_dict, interactive):
    assert set(param_dict.keys()) == {'name', 'n_grid', 'n_data', 'noise_sigma', 'seed', 'response_type',
                                       'eval_imagine', 'components', }

    joint_path = './evals/' + param_dict['name'] + '/'
    nifty_path = joint_path + 'nifty/'
    imagine_path = joint_path + 'imagine/'
    for path in [joint_path, nifty_path, imagine_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    n_grid = param_dict['n_grid']
    n_data = param_dict['n_data']
    response_type = param_dict['response_type']

    if response_type == 'los':
        n_x = n_grid
        n_y = n_grid

    elif response_type == 'mask':
        n_x = n_data
        n_y = n_grid
    else:
        raise NotImplementedError('response type ' + response_type + ' not implemented')
    domain = ift.makeDomain(ift.RGSpace((n_x, n_y)))

    models = {}
    ground_truth = {}
    fixed_params = {}
    for key, d in param_dict['components'].items():
        if d['switched_on']:
            fixed_params.update(**d['fixed_param_dict'])
            ground_truth.update(**d['ground_truth'])
            models.update({key: {'fixed_params': d['fixed_param_dict'], 'active_params': d['active_param_tuple']}})

    fixed_params.update(ground_truth)

    if response_type == 'los':
        starts = list(ift.random.current_rng().random((n_data, 2)).T)
        ang_pos = np.arctan2(starts[0], starts[1])
        indices = np.argsort(ang_pos)
        starts[0] = starts[0][indices]
        starts[1] = starts[1][indices]
        ang_pos = ang_pos[indices]
        response = nfc.get_los_response(domain, starts, [0.5, 0.2, ], n_data)
    elif response_type == 'mask':
        endpoints = np.random.uniform(n_y/ 2, n_y - 1, n_x)
        mask_op = nfc.get_mask_op(domain, endpoints)
        ang_pos = np.linspace(n_x)
        response = nfc.get_mask_response(domain, mask_op)

    noise_sigma = param_dict['noise_sigma']

    imagine_data, nifty_data, groundtruth_dict = make_observable_dict(response, noise_sigma, models, fixed_params)

    groundtruth_dict.update({'summed': ift.utilities.my_sum(list(groundtruth_dict.values()))})
    groundtruth_dict.update({'backprojected_data': response.adjoint(nifty_data['data'])})

    field_plotter(ift.utilities.my_sum(list(groundtruth_dict.values())), 'cmr.gothic', interactive,  joint_path + 'field_truth',
                  {'backprojected_data': {'vmin': 0, 'vmax': 0.005}})
    field_plotter(groundtruth_dict[''], 'cmr.gothic', interactive,  joint_path + 'field_truth',
                  {'backprojected_data': {'vmin': 0, 'vmax': 0.005}})
    data_plotter({'sres_truth': nifty_data['sres'], 'data': nifty_data['data']}, ang_pos, interactive,
                 joint_path + 'truth_sres_vs_data')
    data_plotter({'sres_truth': nifty_data['sres']}, ang_pos, interactive, joint_path + 'truth_sres+')
    data_plotter({'data': nifty_data['data']}, ang_pos, interactive, joint_path + 'data')

    seed = param_dict['seed']

    if param_dict['eval_imagine']:
        imagine_evaluation(response=response,
                           interactive=interactive, img_obs=imagine_data, model_dict=models, n_x=n_x, n_y=n_y,
                           pos=ang_pos, path=imagine_path)


if __name__ == '__main__':
    parameters = {
        'name': 'test4',
        'n_grid': 100,
        'n_data': 500,
        'noise_sigma': 0.001,
        'seed': 324,
        'response_type': 'los',
        'eval_nifty': False,
        'eval_imagine': True,
        'components': {
            'random': {
                'switched_on': False,
                'fixed_param_dict': {'a0': 0.1, 'k0': 1, 'p': -3},
                'active_param_tuple': tuple(),
            },

            'spiral_density': {
                'switched_on': True,
                'fixed_param_dict': {'density_log_amplitude': .4, },
                'active_param_tuple': ('density_k',),
                'ground_truth': {'density_k': 2., },
            },
            'spiral_sign': {
                'switched_on': True,
                'fixed_param_dict': {'sign_k': .7},
                'active_param_tuple': ('sign_log_amplitude',),
                'ground_truth': {'sign_log_amplitude': 1.,},
            },
        }
    }

    evaluation(parameters, False)
