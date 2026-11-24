'''
A workflow's figure steps: a Python function that draws figures from the results of other steps.

A figure step names a function of a script file and one typed input per parameter of that
function, each read from a named step::

    {"id": "dose_response", "kind": "figure",
     "function": "figures/dose_response.py:draw",
     "inputs": {"curves": {"sweep": "scan"}, "fit": {"values": "fit_a"}},
     "files": ["figures/style.py"]}

The function's parameters are exactly the input names plus ``out_dir``. libcuflynx loads each
input (:data:`INPUT_KINDS`) and calls ``draw(curves=..., fit=..., out_dir=...)``; the function
writes its files into ``out_dir`` and returns their paths. It runs no simulation: every
simulation is a step (a ``sweep`` or ``simulate`` step), so figures only read step results. It is
called with Matplotlib's settings as the rc file gives them, whatever was drawn or imported
before, so it sets its own style inside the function.

The script also has a ``main``, so it runs on its own on a finished run, with the same loaders
and so the same objects::

    def main(argv=None):
        from libcuflynx.calibration_workflow.figures import figure_main
        return figure_main(draw, argv)

    if __name__ == '__main__':
        main()

    python figures/dose_response.py <run directory> [--step dose_response] [--out-dir DIR]

A figure step is out of date when its script, a file it lists in ``files`` (helpers the script
imports, a style module) or its declaration changed, or a step it reads ran again after it. A run
draws every figure step that is out of date and whose inputs are current.

Output (``<step_dir>/figure/``): the files the function wrote; ``step_result.json`` lists them
under ``figures``.

A new input kind is one :func:`input_kind` entry here; the workflow schema's list of kinds is
generated from it (``python -m libcuflynx.schemas.registries --write``).

libcuflynx ships a script of standard figures (``standard_figures.py``: sweep traces and
features, an obs_data's prediction_plots, the validation against held-out data, Sobol indices, a calibration's progress and best fit, posteriors, a
Laplace approximation).
:func:`write_standard_figures` copies it beside a workflow, where a figure step names it like any
other script, so it is hashed and packed with the workflow and a later libcuflynx does not change
a finished run's figures.
'''

import argparse
import ast
import contextlib
import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

from libcuflynx.calibration_workflow.spec import WORKFLOW_FILE_NAME, WorkflowError

FIGURE_DIR = 'figure'
#: the parameter of a figure function that is not an input: where it writes
OUT_DIR = 'out_dir'
STEP_RESULT = 'step_result.json'
#: the keys of a figure step's recorded inputs that are files: the script, and each of ``files``
SCRIPT_INPUT = 'script'
FILE_INPUT_PREFIX = 'file:'
_MODEL_KINDS = ('calibration', 'sensitivity', 'emulation', 'identifiability', 'sweep', 'simulate')


@dataclass(frozen=True)
class InputKind:
    '''One kind of figure input: the step kinds it reads from and how it is loaded.'''
    name: str
    step_kinds: tuple
    load: object          # (run_dir, step_id, step result) -> the object the function gets
    description: str


#: the kinds of input a figure step may name, by name
INPUT_KINDS = {}


def input_kind(name, step_kinds, description):
    '''Register ``load(run_dir, step_id, result)`` as the loader of input kind ``name``, read
    from steps of ``step_kinds``.'''
    def register(load):
        INPUT_KINDS[name] = InputKind(name, tuple(step_kinds), load, description)
        return load
    return register


#: the file name of the script of standard figures libcuflynx ships
STANDARD_FIGURES = 'standard_figures.py'


def write_standard_figures(directory, overwrite=False):
    '''Copy libcuflynx's script of standard figures into ``directory`` (made if missing); its
    path. A figure step names one of its functions, e.g. ``"function":
    "figures/standard_figures.py:sweep_traces"``. An existing copy is kept unless ``overwrite``:
    replacing it makes every figure step that uses it out of date.'''
    import shutil
    path = os.path.join(directory, STANDARD_FIGURES)
    if overwrite or not os.path.isfile(path):
        os.makedirs(directory, exist_ok=True)
        shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     STANDARD_FIGURES), path)
    return path


def _existing(path, what, step_id):
    if not path or not os.path.isfile(path):
        raise WorkflowError(f'step "{step_id}" has no {what} to read ({path}).')
    return path


@input_kind('sweep', ('sweep', 'simulate'),
            'the arrays of sweep_results.npz, as a dict: points, failed, parameters, outputs, '
            'features, feature_values, and time_<e> / trace_<e> per experiment; and manifest, '
            'sweep_manifest.json as its JSON document')
def _load_sweep(run_dir, step_id, result):
    from libcuflynx.calibration_workflow.sweep import MANIFEST_FILE, RESULTS_FILE, SWEEP_DIR
    directory = os.path.join(run_dir, step_id, SWEEP_DIR)
    path = _existing(os.path.join(directory, RESULTS_FILE), 'sweep results', step_id)
    with np.load(path) as arrays:
        loaded = {name: arrays[name] for name in arrays.files}
    with open(_existing(os.path.join(directory, MANIFEST_FILE), 'sweep manifest', step_id)) as f:
        loaded['manifest'] = json.load(f)
    return loaded


@input_kind('validation', ('sweep', 'simulate'),
            'a one-point step\'s validation_results.json, as its JSON document: "items", per '
            'prediction item that carries a value its data_item_name, operation, unit, '
            'data_type, n_points, rmse, nrmse, mean_abs_z, within_2std, and t, data, std and '
            'model at the observation times')
def _load_validation(run_dir, step_id, result):
    path = result.get('validation_results')
    if not path:
        raise WorkflowError(f'step "{step_id}" compared no prediction item with data: a step of '
                            f'one point validates the prediction items of its obs_data that '
                            f'carry a value.')
    with open(_existing(path, 'validation results', step_id)) as f:
        return json.load(f)


@input_kind('values', ('calibration', 'set'),
            'the values a step calibrated or set by hand, {"<module>/<parameter>": value}, '
            'named as a params_for_id names them (a nested module by its path, heart/lv/E; a '
            'global as global/R)')
def _load_values(run_dir, step_id, result):
    from libcuflynx.utilities.params_for_id_file import target_name
    records = result.get('calibrated', []) + result.get('set', [])
    return {target_name(record['vessel'], record['param']): float(record['value'])
            for record in records}


@input_kind('progress', ('calibration',),
            'a calibration\'s optimiser history (param_id.run_history.read_run_history): '
            'param_labels, cost_history (per generation, best first), param_history_norm '
            '(normalised 0-1), grad_history, starts (per multi-start: cost, params as actual '
            'values), best_cost; and param_mins / param_maxs, the params_for_id bounds')
def _load_progress(run_dir, step_id, result):
    from libcuflynx.param_id.run_history import find_run_dir, read_run_history
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
    output_dir = result.get('param_id_output_dir')
    if find_run_dir(output_dir) is None:
        raise WorkflowError(f'step "{step_id}" has no optimiser history to read ({output_dir}).')
    info = ObsAndParamDataParser().get_param_id_info(
        _existing(result.get('params_for_id_used'), 'params_for_id', step_id))
    progress = read_run_history(output_dir, info)
    progress['param_mins'] = [float(v) for v in np.ravel(info['param_mins'])]
    progress['param_maxs'] = [float(v) for v in np.ravel(info['param_maxs'])]
    return progress


@input_kind('fit_errors', ('calibration',),
            'the best fit\'s error per data_item, which a calibration writes when it ends '
            '(param_id.plot_outputs.ERROR_VEC_FILES): names, percent_error (100 (model - '
            'data) / data), std_error ((model - data) / std; a series: averaged over its '
            'points), data_references, in obs_data order')
def _load_fit_errors(run_dir, step_id, result):
    from libcuflynx.param_id.plot_outputs import ERROR_VEC_FILES
    output_dir = result.get('param_id_output_dir') or ''
    loaded = {}
    for key, name in ERROR_VEC_FILES.items():
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            raise WorkflowError(
                f'step "{step_id}" has no best-fit errors to read ({path}): run the step again '
                f'with this libcuflynx, or write them for its best fit with '
                f'CVS0DParamID.save_best_fit_errors().')
        values = np.load(path, allow_pickle=True)
        loaded[key] = [str(v) for v in values] if values.dtype.kind in 'OUS' else values
    return loaded


#: the file a calibration writes per experiment it fits: every model output at the best fit
BEST_FIT_OUTPUTS = 'all_outputs_with_best_param_vals_exp_{}.npz'


def _resolve_output(name, outputs):
    '''The key of ``outputs`` (a backend's all-outputs dict: Myokit qnames or
    ``component/variable``) that ``name``, as an obs_data operand, names; None if none does.'''
    from libcuflynx.solver_wrappers.name_resolver import VariableNameResolver
    for separator in ('.', '/'):
        _, key = VariableNameResolver.resolve_key(name, [('output', outputs)], separator)
        if key is not None:
            return key
    return None


@input_kind('best_fit', ('calibration',),
            'the best fit against its data, from the outputs a calibration saves per '
            'experiment it fits (all_outputs_with_best_param_vals_exp_<e>.npz) and its '
            'obs_data: "items", per data_item its data_item_name, trace_name (what its page is '
            'about), data_type, operation, operation_kwargs, plot_type (a constant without '
            'one: horizontal, as the calibration\'s plots read it; "None" is None), unit, '
            'experiment_idx, subexperiment_idx, window (its subexperiment\'s start and end '
            'times), value, std, data_reference, t and model (its first operand over its '
            'subexperiment; None when the solver did not log it), obs_t (a series: '
            'the times of its values) and fit (a constant: the calibrated model\'s value, '
            'from the best fit\'s std_error; None without one); "experiment_labels"')
def _load_best_fit(run_dir, step_id, result):
    from libcuflynx.param_id.plot_outputs import ERROR_VEC_FILES
    obs = _load_obs_data(run_dir, step_id, result)
    protocol = obs.get('protocol_info') or {}
    sim_times = protocol.get('sim_times') or []
    output_dir = result.get('param_id_output_dir') or ''
    data_items = obs.get('data_items') or []
    # the best fit's std errors, in data_item order (their names are plot labels)
    path = os.path.join(output_dir, ERROR_VEC_FILES['std_error'])
    std_error = np.load(path) if os.path.isfile(path) else np.array([])
    if len(std_error) != len(data_items):
        std_error = [None] * len(data_items)
    outputs = {}
    items = []
    for entry, error in zip(data_items, std_error):
        e = int(entry.get('experiment_idx', 0))
        k = int(entry.get('subexperiment_idx', 0))
        if e not in outputs:
            path = _existing(os.path.join(output_dir, BEST_FIT_OUTPUTS.format(e)),
                             f'best-fit outputs of experiment {e}', step_id)
            with np.load(path) as arrays:
                outputs[e] = {name: np.asarray(arrays[name], dtype=float)
                              for name in arrays.files}
        durations = [float(d) for d in sim_times[e]]
        start, end = sum(durations[:k]), sum(durations[:k + 1])
        operands = list(entry.get('operands') or [])
        key = _resolve_output(operands[0], outputs[e]) if operands else None
        t = model = None
        if key is not None:
            trace = outputs[e][key]
            if trace.size == 1:
                # a constant: not logged over time, so drawn flat over the subexperiment
                t, model = np.array([start, end]), np.full(2, float(trace.ravel()[0]))
            else:
                time = np.linspace(0.0, sum(durations), trace.size)
                inside = (time >= start - 1e-9 * (end or 1.0)) & (
                    time <= end + 1e-9 * (end or 1.0))
                t, model = time[inside], trace[inside]
        name = str(entry['data_item_name'])
        value, std = entry.get('value'), entry.get('std')
        data_type = entry.get('data_type', 'constant')
        plot_type = entry.get('plot_type', 'horizontal' if data_type == 'constant' else None)
        if str(plot_type).lower() in ('none', 'null'):
            plot_type = None
        obs_t = fit = None
        if data_type == 'series' and value is not None:
            obs_t = np.linspace(start, end, len(value))
        elif (data_type == 'constant' and error is not None and value is not None
              and std is not None and np.isfinite(std) and std != 0):
            fit = float(value) + float(error) * float(std)
        items.append({
            'data_item_name': name,
            'trace_name': str(entry.get('trace_name_for_plotting') or
                              (operands[0] if operands else name)),
            'data_type': data_type, 'operation': entry.get('operation'),
            'operation_kwargs': entry.get('operation_kwargs') or {},
            'plot_type': plot_type, 'unit': entry.get('unit', ''),
            'experiment_idx': e, 'subexperiment_idx': k, 'window': (start, end),
            'value': value, 'std': std,
            'data_reference': entry.get('data_reference'), 't': t, 'model': model,
            'obs_t': obs_t, 'fit': fit})
    if not items:
        raise WorkflowError(f'step "{step_id}" fitted no data_item: its obs_data has none.')
    return {'items': items, 'experiment_labels': list(protocol.get('experiment_labels') or [])}


@input_kind('mcmc_chain', ('calibration',),
            'a calibration\'s MCMC chain (do_uq, store_distribution): chain, an array (steps, '
            'walkers, parameters) cut to the draws every walker reached; param_labels; and '
            'burn_in_index, the first step after the burn-in its UQ options give')
def _load_mcmc_chain(run_dir, step_id, result):
    from libcuflynx.calibration_workflow.runner import burn_in_index
    from libcuflynx.param_id.paramID import drop_unsampled_draws
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser, param_entry_labels
    path = _existing(os.path.join(result.get('param_id_output_dir') or '', 'mcmc_chain.npy'),
                     'MCMC chain', step_id)
    chain = drop_unsampled_draws(np.load(path))
    info = ObsAndParamDataParser().get_param_id_info(
        _existing(result.get('params_for_id_used'), 'params_for_id', step_id))
    return {'chain': chain, 'param_labels': param_entry_labels(info),
            'burn_in_index': burn_in_index(result.get('settings') or {}, chain.shape[0])}


@input_kind('emulator_validation', ('emulation',),
            'an emulation step\'s held-out points (emulator_validation.npz) in real units: '
            'theta (points, parameters), y_true and y_pred (points, features), '
            'feature_labels, param_entry_labels; and feature_r2 / feature_rmse / '
            'feature_nrmse from emulator_metadata.json (None where not recorded)')
def _load_emulator_validation(run_dir, step_id, result):
    from libcuflynx.emulators.emulator_bundle import METADATA_FILE, VALIDATION_FILE
    from libcuflynx.calibration_workflow.runner import EMULATOR_DIR
    directory = os.path.join(run_dir, step_id, EMULATOR_DIR)
    path = _existing(os.path.join(directory, VALIDATION_FILE), 'emulator validation points',
                     step_id)
    with np.load(path, allow_pickle=True) as data:
        loaded = {name: np.asarray(data[name], dtype=float)
                  for name in ('theta', 'y_true', 'y_pred')}
        for name in ('feature_labels', 'param_entry_labels'):
            loaded[name] = [str(label) for label in data[name]]
    with open(_existing(os.path.join(directory, METADATA_FILE), 'emulator metadata',
                        step_id)) as f:
        meta = json.load(f)
    for name in ('feature_r2', 'feature_rmse', 'feature_nrmse'):
        loaded[name] = meta.get(name)
    return loaded


@input_kind('obs_data', _MODEL_KINDS, 'the step\'s obs_data file, as its JSON document')
def _load_obs_data(run_dir, step_id, result):
    path = _existing((result.get('inputs', {}).get('obs_data') or {}).get('path'), 'obs_data',
                     step_id)
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


@input_kind('params_for_id', ('calibration', 'sensitivity', 'emulation', 'identifiability',
                              'reduction'),
            'the params_for_id a step ran with (a reduction: the one it wrote), as its JSON '
            'document')
def _load_params_for_id(run_dir, step_id, result):
    from libcuflynx.utilities.params_for_id_file import read_params_for_id
    path = result.get('params_for_id_used') or result.get('params_for_id')
    return read_params_for_id(_existing(path, 'params_for_id', step_id))


@input_kind('sensitivity', ('sensitivity',),
            'a sensitivity step\'s indices: Sobol, {"S1": {output: {parameter: value}}, '
            '"ST": {...}} (empty for a local analysis); and "local", {"relative": {output: '
            '{parameter: value}}, "absolute": {...}} from a local analysis, else None')
def _load_sensitivity(run_dir, step_id, result):
    from libcuflynx.sensitivity_analysis.sobol_ranking import read_sobol_indices
    import pandas as pd
    local = {}
    for key in ('relative', 'absolute'):
        path = os.path.join(result.get('sensitivity_dir') or '', f'local_sensitivity_{key}.csv')
        if os.path.isfile(path):
            table = pd.read_csv(path, index_col='output')
            local[key] = {str(output): {str(p): float(v) for p, v in row.items()}
                          for output, row in table.iterrows()}
    if not result.get('sobol_indices') and not local:
        raise WorkflowError(f'step "{step_id}" has no sensitivity indices to read '
                            f'({result.get("sensitivity_dir")}).')
    indices = (read_sobol_indices(_existing(result['sobol_indices'], 'Sobol indices', step_id))
               if result.get('sobol_indices') else {'S1': {}, 'ST': {}})
    indices['local'] = local or None
    return indices


@input_kind('reduction', ('reduction',),
            'a reduction step\'s report (param_reduction.json: reduction, selected, ...) and, '
            'for a PCA reduction, "modes", [{mode, variance, variance_share, informative, '
            'loadings: {feature: value}}], and "mode_indices", {"S1": {"mode_<k>": '
            '{parameter: value}}, "ST": {...}}; both None for a Sobol reduction')
def _load_reduction(run_dir, step_id, result):
    import pandas as pd
    from libcuflynx.sensitivity_analysis.parameter_reduction import (PCA_MODE_INDICES_FILE,
                                                                     PCA_MODES_FILE)
    from libcuflynx.sensitivity_analysis.sobol_ranking import read_sobol_indices
    with open(_existing(result.get('report'), 'reduction report', step_id)) as f:
        loaded = json.load(f)
    directory = os.path.dirname(result['report'])
    modes_path = os.path.join(directory, PCA_MODES_FILE)
    loaded['modes'] = loaded['mode_indices'] = None
    if os.path.isfile(modes_path):
        table = pd.read_csv(modes_path)
        loaded['modes'] = [
            {'mode': int(row['mode']), 'variance': float(row['variance']),
             'variance_share': float(row['variance_share']),
             'informative': bool(row['informative']),
             'loadings': {column[len('loading_'):]: float(row[column])
                          for column in table.columns if column.startswith('loading_')}}
            for _, row in table.iterrows()]
        loaded['mode_indices'] = read_sobol_indices(
            _existing(os.path.join(directory, PCA_MODE_INDICES_FILE), 'PCA mode indices',
                      step_id))
    return loaded


@input_kind('distribution', ('calibration',),
            'the posterior a calibration stored (store_distribution), {"names": [...], '
            '"samples": (samples, parameters) array}')
def _load_distribution(run_dir, step_id, result):
    from libcuflynx.calibration_workflow.distributions import (DISTRIBUTION_DIR, MANIFEST,
                                                               StoredDistribution)
    directory = os.path.join(run_dir, step_id, DISTRIBUTION_DIR)
    _existing(os.path.join(directory, MANIFEST), 'stored distribution', step_id)
    distribution = StoredDistribution.load(directory)
    return {'names': list(distribution.names), 'samples': np.asarray(distribution.samples)}


@input_kind('laplace', ('identifiability',),
            'an identifiability step\'s Laplace approximation, {"parameters": [...], "mean", '
            '"covariance", "std"} as arrays')
def _load_laplace(run_dir, step_id, result):
    laplace = result.get('laplace')
    if not laplace:
        raise WorkflowError(f'step "{step_id}" wrote no Laplace approximation to read.')
    return {'parameters': list(result.get('parameters') or []),
            **{key: np.asarray(laplace[key], dtype=float)
               for key in ('mean', 'covariance', 'std')}}


# --------------------------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------------------------

def parse_figure(value, base_dir, label):
    '''A figure step's ``{script, function, inputs: {name: {kind, step}}, files}`` from the step
    as the file writes it; paths are relative to ``base_dir``.'''
    script, _, function = value['function'].strip().rpartition(':')
    inputs = {}
    for name, source in value['inputs'].items():
        if name == OUT_DIR:
            raise WorkflowError(f'{label}: "{OUT_DIR}" is where the function writes; name the '
                                f'input something else.')
        (kind, step_id), = source.items()
        inputs[name] = {'kind': kind, 'step': step_id.strip()}

    def path(text):
        text = os.path.expanduser(text.strip())
        return text if os.path.isabs(text) else os.path.normpath(os.path.join(base_dir, text))

    return {'script': path(script), 'function': function, 'inputs': inputs,
            'files': [path(name) for name in value.get('files', [])]}


def check_figure_step(workflow, step):
    '''Check a figure step's inputs against the steps they name (the file alone).'''
    for name, source in step.figure['inputs'].items():
        kind, wanted = INPUT_KINDS[source['kind']], workflow.step(source['step'])
        where = f'figure step "{step.id}": input "{name}"'
        if wanted.kind not in kind.step_kinds:
            raise WorkflowError(f'{where} reads "{kind.name}" from "{wanted.id}", a {wanted.kind} '
                                f'step; it is read from a {" or ".join(kind.step_kinds)} step.')
        if kind.name == 'distribution' and not wanted.store_distribution:
            raise WorkflowError(f'{where} reads the distribution of "{wanted.id}", which does '
                                f'not store one. Set "store_distribution": true on "{wanted.id}".')
        if kind.name == 'obs_data' and not wanted.obs_data:
            raise WorkflowError(f'{where} reads the obs_data of "{wanted.id}", which has none.')


def declaration(step):
    '''What a figure step's ``settings`` digest covers: its function and inputs, and the names of
    its files beside the script. The script's and files' contents are inputs of their own.'''
    figure = step.figure
    here = os.path.dirname(figure['script'])
    return {'function': f'{os.path.basename(figure["script"])}:{figure["function"]}',
            'inputs': figure['inputs'],
            'files': [os.path.relpath(path, here) for path in figure['files']]}


def input_files(step):
    '''``{recorded input key: path}`` of a figure step's script and ``files``.'''
    figure = step.figure
    here = os.path.dirname(figure['script'])
    files = {SCRIPT_INPUT: figure['script']}
    for path in figure['files']:
        files[FILE_INPUT_PREFIX + os.path.relpath(path, here)] = path
    return files


def check_script(step):
    '''Check a figure step's script without importing it: the file and its ``files`` exist, the
    function is defined at the top of the script with exactly the input names and ``out_dir`` as
    its parameters, and the script has a ``main``.'''
    figure, where = step.figure, f'figure step "{step.id}"'
    for path in [figure['script']] + figure['files']:
        if not os.path.isfile(path):
            raise WorkflowError(f'{where}: its file {path} does not exist.')
    with open(figure['script'], encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=figure['script'])
        except SyntaxError as exc:
            raise WorkflowError(f'{where}: {figure["script"]} is not valid Python: {exc}') from exc
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if figure['function'] not in functions:
        raise WorkflowError(f'{where}: {figure["script"]} defines no function '
                            f'"{figure["function"]}" (its functions: {sorted(functions)}).')
    args = functions[figure['function']].args
    if args.vararg or args.kwarg:
        raise WorkflowError(f'{where}: "{figure["function"]}" takes *args or **kwargs; name '
                            f'each input and {OUT_DIR} as a parameter.')
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    _check_parameters(names, figure, where)
    if 'main' not in functions:
        raise WorkflowError(
            f'{where}: {figure["script"]} has no "main". A figure script runs on its own too: '
            f'add\n    def main(argv=None):\n        from '
            f'libcuflynx.calibration_workflow.figures import figure_main\n        return '
            f'figure_main({figure["function"]}, argv)')


def _check_parameters(names, figure, where):
    wanted = sorted(set(figure['inputs']) | {OUT_DIR})
    if sorted(names) != wanted:
        raise WorkflowError(f'{where}: the parameters of "{figure["function"]}" are '
                            f'{sorted(names)}, but the step gives {wanted} (its inputs and '
                            f'{OUT_DIR}).')


# --------------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------------

def _step_result(run_dir, step_id):
    path = os.path.join(run_dir, step_id, STEP_RESULT)
    if not os.path.isfile(path):
        raise WorkflowError(f'{run_dir} holds no result of step "{step_id}".')
    with open(path) as f:
        result = json.load(f)
    if result.get('status') != 'done':
        raise WorkflowError(f'step "{step_id}" in {run_dir} has not finished.')
    return result


def load_inputs(run_dir, inputs):
    '''``{name: object}`` for a figure's ``inputs`` (``{name: {kind, step}}``) from the workflow
    run in ``run_dir``, each by its kind's loader. Reads the run directory alone.'''
    loaded = {}
    for name, source in inputs.items():
        if source['kind'] not in INPUT_KINDS:
            raise WorkflowError(f'input "{name}": no input kind "{source["kind"]}" (kinds: '
                                f'{sorted(INPUT_KINDS)}).')
        result = _step_result(run_dir, source['step'])
        kind = INPUT_KINDS[source['kind']]
        if result.get('kind', 'calibration') not in kind.step_kinds:
            raise WorkflowError(f'input "{name}" reads "{kind.name}" from "{source["step"]}", a '
                                f'{result.get("kind")} step; it is read from a '
                                f'{" or ".join(kind.step_kinds)} step.')
        loaded[name] = kind.load(run_dir, source['step'], result)
    return loaded


def load_function(script, function):
    '''The function ``function`` of the file ``script``, imported by path. The script's
    directory is on ``sys.path`` while it loads, so it can import the helpers beside it.'''
    script = os.path.abspath(script)
    name = f'_cuflynx_figure_{abs(hash(script))}'
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(script))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(os.path.dirname(script))
    if not callable(getattr(module, function, None)):
        raise WorkflowError(f'{script} defines no function "{function}".')
    return getattr(module, function)


def call_figure(function, loaded, out_dir, where=''):
    '''Call a figure ``function`` with its ``loaded`` inputs and ``out_dir``; the absolute paths
    of the files it wrote, as it returned them. Matplotlib's settings are its rc file's during
    the call, so a figure sets its style inside the function. WorkflowError when its parameters
    are not the inputs and ``out_dir``, or it returns a file that is not in ``out_dir``.'''
    where = where or f'figure function "{function.__name__}"'
    _check_parameters(list(inspect.signature(function).parameters),
                      {'function': function.__name__, 'inputs': loaded}, where)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with _fresh_matplotlib():
        written = function(**loaded, out_dir=out_dir)
    if isinstance(written, (str, os.PathLike)):
        written = [written]
    paths = []
    for path in written or []:
        path = os.path.abspath(os.path.join(out_dir, os.fspath(path)))
        if os.path.commonpath([path, out_dir]) != out_dir or not os.path.isfile(path):
            raise WorkflowError(f'{where}: it returned {path}, which is not a file it wrote in '
                                f'{out_dir}.')
        paths.append(path)
    if not paths:
        raise WorkflowError(f'{where}: it returned no files; a figure function returns the paths '
                            f'of the files it wrote in out_dir.')
    return paths


def run_figure(run_dir, step, out_dir):
    '''Draw the figure step ``step`` (a spec.Step) of the run in ``run_dir`` into ``out_dir``;
    the paths written.'''
    figure = step.figure
    function = load_function(figure['script'], figure['function'])
    return call_figure(function, load_inputs(run_dir, figure['inputs']), out_dir,
                       f'figure step "{step.id}"')


@contextlib.contextmanager
def _fresh_matplotlib():
    '''Matplotlib's settings as its rc file gives them, put back afterwards. Importing
    libcuflynx and the plots drawn earlier in a run change them, so without this a figure would
    depend on what ran before it and differ between a workflow and its script run alone.'''
    import matplotlib
    with matplotlib.rc_context():
        matplotlib.rc_file_defaults()
        yield


#: the image files compared by their pixels; any other file is compared byte for byte
PIXEL_SUFFIXES = ('.png', '.jpg', '.jpeg')


def output_differences(first_dir, second_dir):
    '''How the files a figure wrote into ``first_dir`` differ from those in ``second_dir``, as
    a list of sentences; empty when they are the same: the same files, each image equal pixel by
    pixel (an image file's metadata differs between runs) and every other file byte for byte.
    A figure drawn by a workflow and by its script run on its own gives none.'''
    def files(directory):
        return sorted(os.path.relpath(os.path.join(root, name), directory)
                      for root, _, names in os.walk(directory) for name in names)

    first, second = files(first_dir), files(second_dir)
    differences = [f'{name} is only in {directory}'
                   for names, others, directory in ((first, second, first_dir),
                                                    (second, first, second_dir))
                   for name in names if name not in others]
    for name in (n for n in first if n in second):
        one, other = os.path.join(first_dir, name), os.path.join(second_dir, name)
        if name.lower().endswith(PIXEL_SUFFIXES):
            from matplotlib.image import imread
            pixels, others = imread(one), imread(other)
            same = pixels.shape == others.shape and np.array_equal(pixels, others)
        else:
            with open(one, 'rb') as f, open(other, 'rb') as g:
                same = f.read() == g.read()
        if not same:
            differences.append(f'{name} differs')
    return differences


# --------------------------------------------------------------------------------------------
# a figure script run on its own
# --------------------------------------------------------------------------------------------

def _run_steps(run_dir):
    '''The figure steps of the workflow run in ``run_dir``, as its copy of the workflow writes
    them.'''
    path = os.path.join(run_dir, WORKFLOW_FILE_NAME)
    if not os.path.isfile(path):
        raise WorkflowError(f'{run_dir} is not a calibration workflow run (no '
                            f'{WORKFLOW_FILE_NAME}).')
    with open(path, encoding='utf-8-sig') as f:
        steps = json.load(f).get('steps', [])
    return [s for s in steps if s.get('kind') == 'figure']


def _declared_inputs(raw_step):
    inputs = {}
    for name, source in raw_step['inputs'].items():
        (kind, step_id), = source.items()
        inputs[name] = {'kind': kind, 'step': step_id}
    return inputs


def _given_inputs(texts):
    inputs = {}
    for text in texts:
        name, _, source = text.partition('=')
        kind, _, step_id = source.partition(':')
        if not (name and kind and step_id):
            raise WorkflowError(f'--input {text}: write it as <name>=<kind>:<step>.')
        inputs[name] = {'kind': kind, 'step': step_id}
    return inputs


def figure_main(functions, argv=None):
    '''The ``main`` of a figure script: draw a figure of a finished workflow run with the same
    loaders a run uses. ``functions`` is the script's figure function, or a list of them.

    ``<run_dir>`` alone draws the run's figure step that names one of ``functions`` in this
    script; ``--step`` picks one when several do. ``--input name=kind:step`` (repeated) gives the
    inputs directly, for a run whose workflow declares no such step (``--function`` then picks
    among several functions). Files go to ``--out-dir`` (default ``./<step id or function>``).
    Prints and returns the paths written.'''
    functions = functions if isinstance(functions, (list, tuple)) else [functions]
    by_name = {function.__name__: function for function in functions}
    script = os.path.basename(inspect.getsourcefile(functions[0]) or '')
    parser = argparse.ArgumentParser(
        description='Draw a figure from a calibration workflow run, as the workflow draws it.')
    parser.add_argument('run_dir', help='the workflow run\'s output directory')
    parser.add_argument('--step', help='the figure step to draw')
    parser.add_argument('--input', action='append', default=[], metavar='NAME=KIND:STEP',
                        help=f'an input given directly; kinds: {", ".join(sorted(INPUT_KINDS))}')
    parser.add_argument('--function', choices=sorted(by_name),
                        help='with --input: which function to call')
    parser.add_argument('--out-dir', help='where the files go')
    args = parser.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    try:
        if args.input:
            if args.function is None and len(by_name) > 1:
                raise WorkflowError(f'--input: name the function with --function (one of '
                                    f'{sorted(by_name)}).')
            name = args.function or next(iter(by_name))
            inputs, label = _given_inputs(args.input), name
        else:
            steps = [s for s in _run_steps(run_dir)
                     if s['function'].rpartition(':')[2] in by_name
                     and os.path.basename(s['function'].rpartition(':')[0]) == script]
            if args.step:
                steps = [s for s in steps if s['id'] == args.step]
            if len(steps) != 1:
                found = [s['id'] for s in steps]
                raise WorkflowError(
                    f'{run_dir}: ' + (f'steps {found} draw with {script}; pick one with --step.'
                                      if found else
                                      f'no figure step{" " + args.step if args.step else ""} '
                                      f'draws with {script}; give the inputs with --input.'))
            name = steps[0]['function'].rpartition(':')[2]
            inputs, label = _declared_inputs(steps[0]), steps[0]['id']
        paths = call_figure(by_name[name], load_inputs(run_dir, inputs),
                            args.out_dir or os.path.join(os.getcwd(), label))
    except WorkflowError as exc:
        parser.exit(2, f'{parser.prog}: {exc}\n')
    for path in paths:
        print(path)
    return paths
