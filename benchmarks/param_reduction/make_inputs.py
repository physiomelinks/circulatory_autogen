"""Inputs of one model of the parameter-reduction study: ``python make_inputs.py <model> <out>``.

Writes into ``<out>/inputs/``: the model's resources, ``full_params_for_id.json`` (the
benchmark's parameters and the extras), ``synthetic_obs_data.json`` (the benchmark's series,
simulated at the spec's truth, with the benchmark's stds), ``binned_obs_data.json`` (their bins,
for the Sobol analysis) and ``truth.json`` (the truth and the model's own defaults). The model's
CellML goes in ``<out>/generated_models/``.
"""

import copy
import json
import os
import sys

import numpy as np

from models import RESOURCES, SPECS, binned_obs, build_param_id, outputs_at, paths, \
    prepare_model, write_params_for_id


def main(model, out):
    spec = SPECS[model]
    p = paths(out)
    prepare_model(spec, out)
    write_params_for_id(spec, p['params_for_id'])
    template_path = os.path.join(RESOURCES, f'{spec.prefix}_obs_data.json')
    with open(template_path) as f:
        template = json.load(f)
    pid = build_param_id(spec, out, template_path, p['params_for_id'],
                         os.path.join(out, 'synthetic'))
    defaults = pid.param_id.sim_helper.get_init_param_vals(pid.param_id_info['param_names'])
    truth = [spec.truth[t] for t in spec.targets()]
    _, series = outputs_at(pid, truth)
    dt = spec.settings['dt']
    obs = copy.deepcopy(template)
    for item, trace in zip(obs['data_items'], series):
        step = int(round(float(item['obs_dt']) / dt))
        values = trace[::step][:len(item['value'])]
        assert len(values) == len(item['value']), (item['data_item_name'], len(values))
        item['value'] = [float(v) for v in values]
        item['data_reference'] = (f'synthetic: {spec.prefix} at the truth in truth.json, '
                                  f'the benchmark obs_data\'s std')
    with open(p['obs'], 'w') as f:
        json.dump(obs, f, indent=1)
    with open(p['sa_obs'], 'w') as f:
        json.dump(binned_obs(obs, spec.settings['sim_time'], spec.bin_points), f, indent=1)
    with open(p['truth'], 'w') as f:
        json.dump({'model': model, 'truth': spec.truth,
                   'defaults': dict(zip(spec.targets(), map(float, defaults))),
                   'extras': list(spec.extras), 'reasons': spec.reason,
                   'bin_points': spec.bin_points}, f, indent=1)
    print(f'{spec.name}: {len(obs["data_items"])} series, '
          f'{sum(len(i["value"]) for i in obs["data_items"])} points; defaults '
          f'{dict(zip(spec.targets(), np.round(defaults, 4)))}')


if __name__ == '__main__':
    main(sys.argv[1], os.path.abspath(sys.argv[2]))
