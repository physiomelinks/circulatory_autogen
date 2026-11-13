"""The benchmark models of the parameter-reduction study, and what every step shares.

Each :class:`Spec` is one of CA's optimiser benchmarks (``benchmarks/benchmark_specs.py``):
the model and solver settings of that benchmark, its calibrated parameters plus extra ones
that make the full calibration unidentifiable, and a hand-set truth inside the bounds, each
value away from the model's own default (so a reduction that fixes a parameter at its default
does not fix it at the truth).

The benchmarks' obs_data are time series. The Sobol analysis needs scalar outputs, so it reads
a second obs_data, made by :func:`binned_obs`: every series cut into bins of ``bin_points``
observed points, each bin's mean a ``mean_in_range`` feature with std ``sigma / sqrt(points)``
-- the information of those points if the trace is smooth over the bin. The calibrations fit
the series themselves.
"""

import copy
import json
import os
import shutil
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESOURCES = os.path.join(REPO, 'resources')


@dataclass
class Spec:
    name: str
    prefix: str
    source: str                    # 'module_array' (generated) or 'cellml' (vendored file)
    settings: dict                 # user_inputs: solver, times
    parameters: list               # (target, min, max, label), benchmark's first, then extras
    truth: dict
    extras: tuple                  # the targets added to the benchmark's
    bin_points: int
    reason: dict = field(default_factory=dict)  # why each extra makes the full set harder

    def targets(self):
        return [p[0] for p in self.parameters]


SPECS = {
    'fitzhugh_nagumo': Spec(
        name='FitzHugh-Nagumo', prefix='FitzHugh_Nagumo', source='module_array',
        settings={'model_type': 'cellml', 'solver': 'CVODE_myokit',
                  'solver_info': {'MaximumStep': 0.01}, 'pre_time': 0.0, 'sim_time': 60.0,
                  'dt': 0.2},
        parameters=[('fhn/a', 0.05, 1.0, 'a'), ('fhn/b', 0.05, 1.0, 'b'),
                    ('fhn/c', 0.5, 6.0, 'c'),
                    ('fhn/V_init', -2.0, -0.2, 'V_0'), ('fhn/R_init', 0.2, 2.0, 'R_0')],
        truth={'fhn/a': 0.2, 'fhn/b': 0.2, 'fhn/c': 3.0, 'fhn/V_init': -1.5,
               'fhn/R_init': 0.5},
        extras=('fhn/V_init', 'fhn/R_init'), bin_points=10,
        reason={'fhn/V_init': 'initial state: moves only the transient before the limit cycle',
                'fhn/R_init': 'initial state: moves only the transient before the limit cycle'}),
    'goodwin': Spec(
        name='Goodwin oscillator', prefix='Goodwin', source='cellml',
        settings={'model_type': 'cellml', 'solver': 'CVODE_myokit',
                  'solver_info': {'MaximumStep': 0.01, 'rtol': 1e-8, 'atol': 1e-8},
                  'pre_time': 0.0, 'sim_time': 40.0, 'dt': 0.1},
        parameters=[('Xi/ai', 20.0, 150.0, 'a_i'), ('Xi/bi', 0.5, 6.0, 'b_i'),
                    ('Xi/Ai', 10.0, 80.0, 'A_i'),
                    ('Xi/ki', 0.3, 3.0, 'k_i'), ('Yi/alpha_i', 0.3, 3.0, 'alpha_i')],
        truth={'Xi/ai': 100.0, 'Xi/bi': 3.0, 'Xi/Ai': 25.0, 'Xi/ki': 1.5,
               'Yi/alpha_i': 0.7},
        extras=('Xi/ki', 'Yi/alpha_i'), bin_points=4,
        reason={'Xi/ki': 'a_i / (A_i + k_i Y) is unchanged when a_i, A_i and k_i are scaled '
                         'together: an exact symmetry',
                'Yi/alpha_i': 'rate of Y production from X'}),
    'teusink': Spec(
        name='Teusink glycolysis', prefix='Teusink', source='cellml',
        settings={'model_type': 'cellml', 'solver': 'CVODE_myokit',
                  'solver_info': {'MaximumStep': 0.005, 'rtol': 1e-9, 'atol': 1e-9},
                  'pre_time': 0.0, 'sim_time': 5.0, 'dt': 0.05},
        parameters=[('v_GLK/v_max', 75.0, 680.0, 'Vmax_GLK'),
                    ('v_PGI/v_max', 113.0, 1020.0, 'Vmax_PGI'),
                    ('v_PYK/v_max', 360.0, 3270.0, 'Vmax_PYK'),
                    ('v_GAPDH/v_max_f', 395.0, 3550.0, 'Vmax_GAPDH_f'),
                    ('v_PFK/v_max', 60.0, 550.0, 'Vmax_PFK'),
                    ('v_ALD/v_max', 107.0, 967.0, 'Vmax_ALD'),
                    ('v_PDC/v_max', 58.0, 523.0, 'Vmax_PDC')],
        truth={'v_GLK/v_max': 300.0, 'v_PGI/v_max': 450.0, 'v_PYK/v_max': 1500.0,
               'v_GAPDH/v_max_f': 900.0, 'v_PFK/v_max': 240.0, 'v_ALD/v_max': 250.0,
               'v_PDC/v_max': 220.0},
        extras=('v_PFK/v_max', 'v_ALD/v_max', 'v_PDC/v_max'), bin_points=2,
        reason={'v_PFK/v_max': 'enzyme capacity; glycolytic flux is shared among the enzymes',
                'v_ALD/v_max': 'enzyme capacity; glycolytic flux is shared among the enzymes',
                'v_PDC/v_max': 'enzyme capacity downstream of the observed metabolites'}),
}

PARAMS_FOR_ID = 'full_params_for_id.json'
OBS = 'synthetic_obs_data.json'          # the series the calibrations fit
SA_OBS = 'binned_obs_data.json'          # their bins, the Sobol analysis's outputs


def paths(out):
    inputs = os.path.join(out, 'inputs')
    return {'out': out, 'inputs': inputs, 'generated': os.path.join(out, 'generated_models'),
            'params_for_id': os.path.join(inputs, PARAMS_FOR_ID),
            'obs': os.path.join(inputs, OBS), 'sa_obs': os.path.join(inputs, SA_OBS),
            'truth': os.path.join(inputs, 'truth.json'),
            'sa': os.path.join(out, 'sensitivity'), 'fits': os.path.join(out, 'fits'),
            'reductions': os.path.join(out, 'reductions'),
            'compare': os.path.join(out, 'compare')}


def raw_inputs(spec, out, obs_path, params_for_id_path, output_dir, **extra):
    """The user_inputs dict of a run of ``spec`` (unparsed, as a script takes it)."""
    p = paths(out)
    inp = {'file_prefix': spec.prefix, 'input_param_file': f'{spec.prefix}_parameters.json',
           'resources_dir': p['inputs'], 'generated_models_dir': p['generated'],
           'param_id_output_dir': output_dir, 'DEBUG': False, 'do_uq': False, 'do_ia': False,
           'UQ_options': {},
           'plot_predictions': False, 'param_id_obs_path': obs_path,
           # absolute, so the parser's join with resources_dir leaves it as it is
           'params_for_id_file': os.path.abspath(params_for_id_path),
           **copy.deepcopy(spec.settings)}
    inp.update(extra)
    return inp


def parsed_inputs(spec, out, obs_path, params_for_id_path, output_dir):
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    return YamlFileParser().parse_user_inputs_file(
        raw_inputs(spec, out, obs_path, params_for_id_path, output_dir),
        obs_path_needed=True, do_generation_with_fit_parameters=False)


def prepare_model(spec, out):
    """Copy the model's resources into ``<out>/inputs`` and put its CellML where
    ``model_path`` points (generating it from the module array, or copying the vendored file)."""
    p = paths(out)
    os.makedirs(p['inputs'], exist_ok=True)
    model_dir = os.path.join(p['generated'], spec.prefix)
    model = os.path.join(model_dir, f'{spec.prefix}.cellml')
    if spec.source == 'module_array':
        for name in (f'{spec.prefix}_module_array.csv', f'{spec.prefix}_parameters.json'):
            shutil.copy(os.path.join(RESOURCES, name), p['inputs'])
        if not os.path.isfile(model):
            from libcuflynx.scripts.script_generate_with_new_architecture import (
                generate_with_new_architecture)
            inp = parsed_inputs(spec, out, os.path.join(RESOURCES, f'{spec.prefix}_obs_data.json'),
                                os.path.join(RESOURCES, f'{spec.prefix}_params_for_id.json'),
                                os.path.join(out, 'generation'))
            assert generate_with_new_architecture(inp_data_dict=dict(inp)), 'generation failed'
    else:
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy(os.path.join(RESOURCES, f'{spec.prefix}.cellml'), model)
    return model


def build_param_id(spec, out, obs_path, params_for_id_path, output_dir):
    """A CVS0DParamID of ``spec``'s model with this obs_data and params_for_id."""
    from libcuflynx.param_id.paramID import CVS0DParamID
    pid = CVS0DParamID.init_from_dict(parsed_inputs(spec, out, obs_path, params_for_id_path,
                                                    output_dir))
    with open(obs_path) as f:
        pid.set_ground_truth_data(json.load(f))
    pid.param_id_info = pid.obs_and_param_parser.get_param_id_info(params_for_id_path)
    pid.param_id.set_param_id_info(pid.param_id_info)
    return pid


def outputs_at(pid, values):
    """``(const, series)`` of the data_items at parameter ``values`` (one experiment, one
    sub-experiment): the scalar features, and each series item's trace at its obs times."""
    _, operands, _ = pid.param_id.get_cost_obs_and_pred_from_params(np.asarray(values, float))
    if not operands or operands[0] is None:
        raise RuntimeError(f'simulation failed at {list(values)}')
    with pid.param_id.evaluating_segment(0, 0):
        obs = pid.param_id.get_obs_output_dict(operands[0])
    const = [float(v) for v in (obs['const'] if obs['const'] is not None else [])]
    series = [np.asarray(s, float).ravel() for s in (obs['series'] or [])]
    return const, series


def write_params_for_id(spec, path):
    with open(path, 'w') as f:
        json.dump({'version': 1, 'params': [
            {'targets': [t], 'min': lo, 'max': hi, 'name_for_plotting': label}
            for t, lo, hi, label in spec.parameters]}, f, indent=1)


def binned_obs(series_obs, sim_time, bin_points):
    """The Sobol analysis's obs_data: each series item of ``series_obs`` as ``mean_in_range``
    features over bins of ``bin_points`` observed points (the last bin takes what is left),
    each with std ``rms(sigma) / sqrt(points)`` and the item's weight."""
    out = {'protocol_info': copy.deepcopy(series_obs['protocol_info']), 'data_items': []}
    for item in series_obs['data_items']:
        values = np.asarray(item['value'], float)
        stds = np.broadcast_to(np.asarray(item['std'], float), values.shape)
        dt = float(item['obs_dt'])
        n = len(values)
        starts = list(range(0, n, bin_points))
        if len(starts) > 1 and n - starts[-1] < bin_points / 2:
            starts.pop()                       # a short last bin joins the one before
        for k, first in enumerate(starts):
            last = (starts[k + 1] if k + 1 < len(starts) else n) - 1
            idx = slice(first, last + 1)
            t0 = max(0.0, (first - 0.5) * dt)
            t1 = min(sim_time, (last + 0.5) * dt)
            points = last - first + 1
            out['data_items'].append({
                'data_item_name': f'{item["data_item_name"]} bin {k}',
                'operands': list(item['operands']), 'data_type': 'constant',
                'operation': 'mean_in_range',
                'operation_kwargs': {'start_frac': t0 / sim_time, 'end_frac': t1 / sim_time},
                'unit': item.get('unit', 'dimensionless'),
                'value': float(values[idx].mean()),
                'std': float(np.sqrt(np.mean(stds[idx] ** 2)) / np.sqrt(points)),
                'weight': float(item.get('weight', 1.0)), 'cost_type': 'gaussian_MLE',
                'data_reference': f'mean of points {first}-{last} of {item["data_item_name"]}'})
    return out
