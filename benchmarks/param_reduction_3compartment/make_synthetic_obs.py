"""Synthetic observations of 3compartment at a hand-set truth, for the parameter-reduction study.

Writes into ``<out>/inputs/``:

* ``3compartment_module_array.csv`` and ``3compartment_parameters.json`` (copied from resources);
* ``3compartment_reduction_params_for_id.json`` -- the four parameters of
  ``3compartment_params_for_id.json`` and three more: ``venous_svc/C``, ``aortic_root/R`` and
  ``systemic_T/R_T``;
* ``3compartment_reduction_obs_data.json`` -- the six features of ``3compartment_obs_data.json``
  computed at :data:`TRUTH`, each with a std of 10% of its value and the same weights;
* ``truth.json`` -- the truth, and the default each parameter has in the parameters file.

Usage: ``python make_synthetic_obs.py <out>``
"""

import copy
import json
import os
import shutil
import sys

import numpy as np

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'resources')
PREFIX = '3compartment'
OBS_FILE = f'{PREFIX}_reduction_obs_data.json'
PARAMS_FOR_ID_FILE = f'{PREFIX}_reduction_params_for_id.json'
STD_FRACTION = 0.1

# (target, min, max, name_for_plotting)
PARAMETERS = [
    ('global/q_lv_init', 200e-6, 1500e-6, 'q_{sbv}'),
    ('aortic_root/C', 1e-9, 5e-8, 'C_{ao}'),
    ('global/E_lv_A', 1e8, 5e8, 'E_{lvA}'),
    ('global/E_lv_B', 1e6, 5e7, 'E_{lvB}'),
    ('venous_svc/C', 2e-7, 5e-6, 'C_{svc}'),
    ('aortic_root/R', 3e5, 5e6, 'R_{ao}'),
    ('systemic_T/R_T', 5e7, 2.5e8, 'R_{T}'),
]
#: Set by hand inside the bounds, each away from its parameters-file default, so that a
#: reduction fixing a parameter at its default does not fix it at the truth.
TRUTH = {
    'global/q_lv_init': 1.0e-3,
    'aortic_root/C': 2.0e-8,
    'global/E_lv_A': 3.0e8,
    'global/E_lv_B': 1.5e7,
    'venous_svc/C': 1.5e-6,
    'aortic_root/R': 1.5e6,
    'systemic_T/R_T': 1.3e8,
}


def default_values(parameters_path):
    with open(parameters_path) as f:
        doc = json.load(f)
    return {target: doc[target.split('/')[0]][target.split('/')[1]]['value']
            for target, *_ in PARAMETERS}


def study_inputs(inputs_dir, obs_path, params_for_id_path, generated_dir, output_dir):
    """The user_inputs of every run of the study."""
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    inp = {'file_prefix': PREFIX, 'input_param_file': f'{PREFIX}_parameters.json',
           'resources_dir': inputs_dir, 'generated_models_dir': generated_dir,
           'param_id_output_dir': output_dir, 'model_type': 'cellml',
           'solver': 'CVODE_myokit', 'solver_info': {'MaximumStep': 0.001}, 'DEBUG': False,
           'pre_time': 20, 'sim_time': 2, 'dt': 0.01,
           'param_id_obs_path': obs_path, 'params_for_id_path': params_for_id_path}
    return YamlFileParser().parse_user_inputs_file(inp, obs_path_needed=True,
                                                   do_generation_with_fit_parameters=False)


def build_param_id(inputs_dir, obs_path, params_for_id_path, generated_dir, output_dir):
    """A CVS0DParamID of the study's model, generated into ``generated_dir`` if needed."""
    from libcuflynx.param_id.paramID import CVS0DParamID
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)
    inp = study_inputs(inputs_dir, obs_path, params_for_id_path, generated_dir, output_dir)
    if not os.path.isfile(inp['model_path']):
        assert generate_with_new_architecture(inp_data_dict=dict(inp)), 'generation failed'
    pid = CVS0DParamID.init_from_dict(inp)
    with open(obs_path) as f:
        pid.set_ground_truth_data(json.load(f))
    pid.param_id_info = pid.obs_and_param_parser.get_param_id_info(params_for_id_path)
    pid.param_id.set_param_id_info(pid.param_id_info)
    return pid


def features_at(pid, values):
    """The data_item features (one experiment, one sub-experiment) at parameter ``values``,
    in the obs_data's order; None when the simulation fails."""
    _, operands, _ = pid.param_id.get_cost_obs_and_pred_from_params(np.asarray(values, float))
    if not operands or operands[0] is None:
        return None
    with pid.param_id.evaluating_segment(0, 0):
        obs = pid.param_id.get_obs_output_dict(operands[0])
    return [float(v) for v in obs['const']]


def main(out):
    inputs = os.path.join(out, 'inputs')
    os.makedirs(inputs, exist_ok=True)
    for name in (f'{PREFIX}_module_array.csv', f'{PREFIX}_parameters.json'):
        shutil.copy(os.path.join(RESOURCES, name), inputs)
    params_for_id_path = os.path.join(inputs, PARAMS_FOR_ID_FILE)
    with open(params_for_id_path, 'w') as f:
        json.dump({'version': 1, 'params': [
            {'targets': [t], 'min': lo, 'max': hi, 'param_type': 'const',
             'name_for_plotting': label} for t, lo, hi, label in PARAMETERS]}, f, indent=1)

    # the shipped obs_data's features, at the truth
    with open(os.path.join(RESOURCES, f'{PREFIX}_obs_data.json')) as f:
        template = json.load(f)
    pid = build_param_id(inputs, os.path.join(RESOURCES, f'{PREFIX}_obs_data.json'),
                         params_for_id_path, os.path.join(out, 'generated_models'),
                         os.path.join(out, 'synthetic'))
    truth = [TRUTH[t] for t, *_ in PARAMETERS]
    values = features_at(pid, truth)
    obs_data = copy.deepcopy(template)
    for item, value in zip(obs_data['data_items'], values):
        item['value'] = value
        item['std'] = abs(value) * STD_FRACTION
        item['data_reference'] = (f'synthetic: 3compartment at the truth in truth.json, '
                                  f'std {STD_FRACTION:.0%} of the value')
    with open(os.path.join(inputs, OBS_FILE), 'w') as f:
        json.dump(obs_data, f, indent=1)
    with open(os.path.join(inputs, 'truth.json'), 'w') as f:
        json.dump({'truth': TRUTH,
                   'defaults': default_values(os.path.join(inputs, f'{PREFIX}_parameters.json')),
                   'features': {item['data_item_name']: v for item, v in
                                zip(obs_data['data_items'], values)},
                   'std_fraction': STD_FRACTION}, f, indent=1)
    print(json.dumps(dict(zip([i['data_item_name'] for i in obs_data['data_items']], values)),
                     indent=1))


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]))
