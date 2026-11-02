'''
A workflow's parameter sweep: one model run at each chosen set of parameter values.

A step's ``sweep`` is ``{"design", "outputs", "parameters"}``. ``parameters`` maps a parameter
(``<vessel>/<parameter>`` as a params_for_id names it) to the values it takes:

* ``[min, max]`` or ``{"min", "max", "points", "scale"}``: evenly spaced, or by ratio with
  ``"scale": "log"`` (default 5 points);
* ``{"values": [...]}``: these values;
* ``{"factors": [...]}``: these multiples of the value the step's model gives it.

``design`` is ``grid`` (every combination) or ``one_at_a_time`` (each parameter's values with
the others at the model's values). ``zip`` lists groups of parameters that move together: the
i-th point of a group takes each member's i-th value, and the design treats the group as one
parameter. With no parameters the sweep is one run of the protocol. ``outputs`` are variables
recorded over time.

Each point sets the parameters at the start of every experiment of the protocol, which is the
step's ``obs_data`` protocol, or one run of ``sim_time`` after ``pre_time``. Under MPI each rank
runs a contiguous block of points. A point that fails keeps its row, as NaN, and the error is
recorded. An ``obs_data``'s prediction items give a scalar feature per point, and each feature is
drawn against the parameters as a ``feature_vs_input`` plot.

Output (``<step_dir>/sweep/``)::

    sweep_results.npz     points (points, parameters), failed, parameters, outputs, features,
                          feature_values (points, features), and per experiment e
                          time_<e> and trace_<e> (points, outputs, times)
    sweep_manifest.json   the design, each parameter's values, and what failed
    <plot name>.png       the feature plots
'''

import itertools
import json
import os

import numpy as np

from libcuflynx.calibration_workflow.spec import WorkflowError

SWEEP_DIR = 'sweep'
RESULTS_FILE = 'sweep_results.npz'
MANIFEST_FILE = 'sweep_manifest.json'
DEFAULT_POINTS = 5
DEFAULT_SOLVER = 'CVODE_myokit'


def parameter_values(spec, nominal=None, where=''):
    '''The values one sweep ``spec`` gives a parameter whose value in the model is ``nominal``;
    None for ``factors`` when ``nominal`` is not known yet.'''
    if isinstance(spec, list):
        spec = {'min': spec[0], 'max': spec[1]}
    if 'values' in spec:
        return np.asarray(spec['values'], dtype=float)
    if 'factors' in spec:
        if nominal is None:
            return None
        if nominal == 0:
            raise WorkflowError(f'{where}: "factors" multiply the model\'s value, which is 0; '
                                f'give "values" or a range.')
        return float(nominal) * np.asarray(spec['factors'], dtype=float)
    low, high = float(spec['min']), float(spec['max'])
    points = int(spec.get('points', DEFAULT_POINTS))
    if not low < high:
        raise WorkflowError(f'{where}: min {low:g} is not below max {high:g}.')
    if spec.get('scale') == 'log':
        if low <= 0:
            raise WorkflowError(f'{where}: a log range needs a positive min, got {low:g}.')
        return np.geomspace(low, high, points)
    return np.linspace(low, high, points)


def axis_groups(sweep):
    '''The sweep's axes: each a list of the parameters that move together, in the order the
    parameters are listed (a ``zip`` group sits where its first member is).'''
    zipped = {name: group for group in sweep.get('zip', []) for name in group}
    groups, seen = [], set()
    for name in sweep['parameters']:
        if name in seen:
            continue
        group = zipped.get(name, [name])
        groups.append(group)
        seen.update(group)
    return groups


def _check_zip(sweep, where):
    names = [name for group in sweep.get('zip', []) for name in group]
    unknown = sorted(set(names) - set(sweep['parameters']))
    if unknown:
        raise WorkflowError(f'{where}: "zip" names {unknown}, which it does not sweep.')
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise WorkflowError(f'{where}: "zip" puts {twice} in more than one group.')


def _zipped_axis(group, axes, where):
    lengths = {name: len(axes[name]) for name in group}
    if len(set(lengths.values())) > 1:
        raise WorkflowError(f'{where}: the parameters zipped together take different numbers of '
                            f'values: {lengths}.')
    return np.column_stack([axes[name] for name in group])


def check_sweep(sweep, where):
    '''Check what can be checked of a ``sweep`` without the model.'''
    _check_zip(sweep, where)
    known = {}
    for name, spec in sweep['parameters'].items():
        values = parameter_values(spec, where=f'{where}: "{name}"')
        if values is not None:
            known[name] = values
    for group in sweep.get('zip', []):
        if all(name in known for name in group):
            _zipped_axis(group, known, where)


def sweep_points(sweep, nominal, where=''):
    '''``(names, points)``: the swept parameters and the ``(points, parameters)`` array of
    values, by the sweep's design. ``nominal`` is ``{parameter: its value in the model}``. With
    no parameters it is one point with no values.'''
    names = list(sweep['parameters'])
    if not names:
        return names, np.zeros((1, 0))
    values = {name: parameter_values(sweep['parameters'][name], nominal[name], f'{where}: "{name}"')
              for name in names}
    groups = axis_groups(sweep)
    axes = [_zipped_axis(group, values, where) for group in groups]
    column = {name: names.index(name) for name in names}
    rows = []
    if sweep['design'] == 'grid':
        for combination in itertools.product(*axes):
            row = [0.0] * len(names)
            for group, group_values in zip(groups, combination):
                for name, value in zip(group, group_values):
                    row[column[name]] = value
            rows.append(row)
        return names, np.array(rows, dtype=float)
    for group, axis in zip(groups, axes):
        for group_values in axis:
            row = [nominal[name] for name in names]
            for name, value in zip(group, group_values):
                row[column[name]] = value
            rows.append(row)
    return names, np.array(rows, dtype=float)


class PointRunner:
    '''Runs one point of a sweep; each rank has its own.'''

    def __init__(self, model_path, settings, obs_data_path, outputs):
        from libcuflynx.param_id.operation_funcs import get_operation_funcs_dict_for_mode
        from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
        from libcuflynx.protocol_runners.protocol_runner import ProtocolRunner

        self.outputs = list(outputs)
        solver = settings.get('solver') or DEFAULT_SOLVER
        solver_info = dict(settings.get('solver_info') or {})
        if solver == 'CVODE_myokit':
            # Without a MaximumStep the calibration runs CVODE uncapped; ProtocolRunner would
            # cap it at 1e-3, which over a long pre_time costs millions of steps per point.
            solver_info.setdefault('MaximumStep', None)
        self.runner = ProtocolRunner(
            model_path, inp_data_dict={'dt': settings['dt'], 'solver_info': solver_info},
            solver=solver,
            model_type=settings.get('model_type'))
        self.prediction_info = None
        if obs_data_path:
            parsed = ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=obs_data_path)
            self.protocol = parsed['protocol_info']
            prediction_info = parsed['prediction_info']
            if (prediction_info or {}).get('data_item_names'):
                self.prediction_info = prediction_info
                self.funcs = get_operation_funcs_dict_for_mode(
                    'numpy', settings.get('operation_funcs_external_path'))
        else:
            self.protocol = {'sim_times': [[settings['sim_time']]],
                             'pre_times': [settings.get('pre_time') or 0.0],
                             'params_to_change': {}}

    def __call__(self, names, values):
        '''``(times, traces, features)`` at one point: per experiment the recorded time and
        ``(outputs, times)`` array, and ``{feature: value}``. Raises when the point fails.'''
        from libcuflynx.protocol_runners.predictions import predict_on

        values = [float(value) for value in values]
        times, traces, features = [], [], {}
        try:
            if self.outputs:
                times, recorded, _ = self.runner.run_protocols(
                    protocol_info=self.protocol, id_param_names=names, id_param_vals=values,
                    result_variables=[[name] for name in self.outputs])
                traces = [np.asarray(columns, dtype=float) for columns in recorded]
            if self.prediction_info is not None:
                self.runner._apply_protocol_info(self.protocol)
                features = predict_on(self.runner._executor, self.protocol, self.prediction_info,
                                      self.funcs, id_param_names=names,
                                      id_param_vals=values).features
        except Exception:
            self.runner.sim_helper.reset_and_clear()
            raise
        return [np.asarray(t, dtype=float) for t in times], traces, dict(features)


def run_on_ranks(evaluate, count, comm):
    '''``evaluate(index)`` for ``range(count)``, each rank a contiguous block. On rank 0
    ``{index: (result, error)}``, with ``error`` the message of a point that raised; None
    elsewhere.'''
    from libcuflynx.emulators.emulator_trainer import _block_for_rank

    start, end = _block_for_rank(count, comm.Get_rank(), comm.Get_size())
    done = []
    for index in range(start, end):
        try:
            done.append((index, evaluate(index), None))
        except Exception as exc:  # noqa: BLE001 - a failed point keeps its row
            done.append((index, None, str(exc)))
    gathered = comm.gather(done, root=0)
    if comm.Get_rank() != 0:
        return None
    return {index: (result, error) for block in gathered for index, result, error in block}


def _results_arrays(names, points, outputs, by_point):
    '''The arrays of ``sweep_results.npz`` from :func:`run_on_ranks`' ``by_point``.'''
    count = len(points)
    failed = np.array([by_point[i][1] is not None for i in range(count)], dtype=bool)
    good = next(by_point[i][0] for i in range(count) if not failed[i])
    times, traces, features = good
    feature_names = list(features)
    arrays = {'points': points, 'failed': failed, 'parameters': np.array(names),
              'outputs': np.array(outputs, dtype=str), 'features': np.array(feature_names,
                                                                            dtype=str),
              'feature_values': np.full((count, len(feature_names)), np.nan)}
    for experiment, (time, trace) in enumerate(zip(times, traces)):
        arrays[f'time_{experiment}'] = time
        arrays[f'trace_{experiment}'] = np.full((count,) + trace.shape, np.nan)
    for index in np.flatnonzero(~failed):
        _, point_traces, point_features = by_point[index][0]
        for experiment, trace in enumerate(point_traces):
            arrays[f'trace_{experiment}'][index] = trace
        arrays['feature_values'][index] = [point_features[name] for name in feature_names]
    return arrays


def _plot_groups(sweep, names, points, nominal):
    '''``[(x parameter index, series parameter index or None, point indices)]``, a figure's
    points by the design: one parameter at a time, or a grid of at most two parameters.'''
    axes = axis_groups(sweep)
    if not names:
        return []
    if sweep['design'] == 'one_at_a_time':
        groups = []
        start = 0
        for group in axes:
            size = len(parameter_values(sweep['parameters'][group[0]], nominal[group[0]]))
            groups.append((names.index(group[0]), None, list(range(start, start + size))))
            start += size
        return groups
    if len(axes) > 2:
        return []
    return [(names.index(axes[0][0]), names.index(axes[1][0]) if len(axes) == 2 else None,
             list(range(len(points))))]


def plot_features(sweep, names, points, nominal, arrays, out_dir):
    '''Draw each feature against the swept parameters as ``feature_vs_input`` plots; a failed
    point has no mark. Returns the paths.'''
    import pandas as pd
    from libcuflynx.param_id.prediction_plots import plot_prediction_plots

    paths = []
    for x_index, series_index, group in _plot_groups(sweep, names, points, nominal):
        group = [i for i in group if not arrays['failed'][i]]
        if not group:
            continue
        protocol = {'sim_times': [[1.0]] * len(group),
                    'params_to_change': {name: [[float(points[i, column])] for i in group]
                                         for column, name in enumerate(names)}}
        items, values, plots = [], {}, []
        for feature_index, feature in enumerate(arrays['features']):
            feature = str(feature)
            for experiment, point in enumerate(group):
                item = f'{feature}_{point}'
                items.append({'item_name_for_plotting': feature, 'data_item_name': item,
                              'experiment_idx': experiment, 'value': None, 'std': None,
                              'data_type': 'constant'})
                values[item] = float(arrays['feature_values'][point, feature_index])
            plots.append({'name': f'{feature} vs {names[x_index]}', 'kind': 'feature_vs_input',
                          'y': feature,
                          'x': {'params_to_change': names[x_index], 'subexperiment_idx': 0},
                          'series': ({'params_to_change': names[series_index],
                                      'subexperiment_idx': 0}
                                     if series_index is not None else None)})
        paths += plot_prediction_plots(plots, pd.DataFrame(items), None, protocol, values, out_dir)
    return paths


def run_points(sweep, generated, settings, obs_data_path, nominal, comm, where):
    '''Run every point of ``sweep`` on the model ``generated``, spread across ranks. Returns
    ``(names, points, by_point)`` with ``by_point`` as :func:`run_on_ranks` gives it. Every rank
    must call it, and only :func:`write_results` writes.'''
    names, points = sweep_points(sweep, nominal, where)
    runner = PointRunner(generated['model_path'], settings, obs_data_path, sweep['outputs'])
    if not sweep['outputs'] and runner.prediction_info is None:
        raise WorkflowError(f'{where}: records nothing: give "outputs", or prediction items in '
                            f'its obs_data.')
    return names, points, run_on_ranks(lambda index: runner(names, points[index]), len(points),
                                       comm)


def write_results(sweep, names, points, by_point, nominal, units, out_dir, where):
    '''Rank 0: write what :func:`run_points` gave to ``out_dir``. Returns ``{sweep_dir,
    sweep_results, sweep_manifest, design, parameters, n_points, n_failed, outputs, features,
    plots}``. WorkflowError when every point failed: the run is misconfigured, and the first
    error says how.'''
    errors = {i: by_point[i][1] for i in range(len(points)) if by_point[i][1] is not None}
    if len(errors) == len(points):
        raise WorkflowError(f'{where}: every one of the {len(points)} points failed; the first '
                            f'error: {errors[0]}')
    arrays = _results_arrays(names, points, sweep['outputs'], by_point)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, RESULTS_FILE)
    np.savez(results_path, **arrays)
    features = [str(name) for name in arrays['features']]
    plots = plot_features(sweep, names, points, nominal, arrays, out_dir) if features else []
    manifest = {
        'design': sweep['design'], 'zip': sweep.get('zip', []),
        'n_points': len(points), 'n_failed': len(errors),
        'parameters': [{'name': name, 'units': units.get(name, ''), 'nominal': nominal[name],
                        'values': sorted({float(v) for v in points[:, column]})}
                       for column, name in enumerate(names)],
        'outputs': list(sweep['outputs']), 'features': features,
        'failures': [{'point': int(index), 'values': points[index].tolist(), 'error': error}
                     for index, error in sorted(errors.items())],
        'results': RESULTS_FILE, 'plots': [os.path.basename(path) for path in plots]}
    manifest_path = os.path.join(out_dir, MANIFEST_FILE)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=1)
    return {'sweep_dir': out_dir, 'sweep_results': results_path, 'sweep_manifest': manifest_path,
            'design': sweep['design'], 'parameters': names, 'n_points': len(points),
            'n_failed': len(errors), 'outputs': list(sweep['outputs']), 'features': features,
            'plots': plots}
