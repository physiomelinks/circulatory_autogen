"""libcuflynx's script of standard figures (calibration_workflow/standard_figures.py).

Each function is drawn twice: in this process, as a workflow run draws a figure step, and by the
script copied beside the workflow and run on its own; the files must be the same. The analysis
figures read hand-written step results; the sweep and simulation figures read a real run of the
workflow library's ``chain`` (z = 2 p_A^2 + k q^2).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from libcuflynx.calibration_workflow import figures, run_calibration_workflow
from libcuflynx.calibration_workflow.resolve import resolve_workflow
from libcuflynx.calibration_workflow.spec import WORKFLOW_FILE_NAME, load_workflow
from libcuflynx.external_testing.workflow_library import Y_A, build_workflow_library

SCRIPT = f'figures/{figures.STANDARD_FIGURES}'


def _write_json(path, raw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(raw, f, indent=1)
    return path


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _figure_step(step_id, function, inputs):
    return {'id': step_id, 'kind': 'figure', 'function': f'{SCRIPT}:{function}',
            'inputs': {name: {kind: step} for name, (kind, step) in inputs.items()}}


def _draw_both_ways(run_dir, step, tmp_path):
    """The files the step's function writes in this process and those its script writes run
    on its own; asserts they are the same and returns the first list."""
    script = os.path.join(run_dir, SCRIPT)
    function = figures.load_function(script, step['function'].rpartition(':')[2])
    inputs = {name: {'kind': kind, 'step': step_id}
              for name, source in step['inputs'].items() for kind, step_id in source.items()}
    here = str(tmp_path / f'{step["id"]}_workflow')
    paths = figures.call_figure(function, figures.load_inputs(run_dir, inputs), here)
    alone = str(tmp_path / f'{step["id"]}_alone')
    done = subprocess.run([sys.executable, script, run_dir, '--step', step['id'],
                           '--out-dir', alone],
                          capture_output=True, text=True, timeout=300, env=dict(os.environ))
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.split() == [os.path.join(alone, os.path.basename(p)) for p in paths]
    assert figures.output_differences(here, alone) == []
    return paths


# --------------------------------------------------------------------------------------------
# the analysis figures, from hand-written step results
# --------------------------------------------------------------------------------------------

def _analysis_run(root):
    """A finished run with a sensitivity, a calibration that stored its distribution and an
    identifiability step, and a figure step per analysis function."""
    from libcuflynx.calibration_workflow.distributions import (DISTRIBUTION_DIR,
                                                               DistributionParameter,
                                                               StoredDistribution)
    run = str(root)
    figures.write_standard_figures(os.path.join(run, 'figures'))
    indices = os.path.join(run, 'sobol', 'all_outputs_Sobol_indices.csv')
    os.makedirs(os.path.dirname(indices))
    with open(indices, 'w') as f:
        f.write('Parameter,S1_z,ST_z,S1_w,ST_w\nA/p,0.6,0.7,0.1,0.2\nC/q,0.3,0.35,0.8,0.85\n')
    _write_json(os.path.join(run, 'sobol', 'step_result.json'),
                {'kind': 'sensitivity', 'status': 'done', 'sobol_indices': indices})
    rng = np.random.default_rng(3)
    StoredDistribution(
        [DistributionParameter([['A', 'p']], 0.0, 10.0, 'logit'),
         DistributionParameter([['C', 'q']], 0.0, np.inf, 'log')],
        np.column_stack([rng.normal(2.0, 0.3, 400), rng.lognormal(0.0, 0.2, 400)]),
    ).save(os.path.join(run, 'fit', DISTRIBUTION_DIR))
    _write_json(os.path.join(run, 'fit', 'step_result.json'),
                {'kind': 'calibration', 'status': 'done'})
    _write_json(os.path.join(run, 'ia', 'step_result.json'),
                {'kind': 'identifiability', 'status': 'done', 'parameters': ['A/p', 'C/q'],
                 'laplace': {'mean': [2.0, 1.0], 'covariance': [[0.04, 0.0], [0.0, 1e300]],
                             'std': [0.2, float('inf')]}})
    steps = [_figure_step('indices', 'sobol_indices', {'sensitivity': ('sensitivity', 'sobol')}),
             _figure_step('posterior', 'posterior', {'distribution': ('distribution', 'fit')}),
             _figure_step('laplace', 'laplace', {'laplace': ('laplace', 'ia')})]
    _write_json(os.path.join(run, WORKFLOW_FILE_NAME), {'steps': steps})
    return run, {step['id']: step for step in steps}


@pytest.fixture(scope='module')
def analysis_run(tmp_path_factory):
    return _analysis_run(tmp_path_factory.mktemp('analysis_run'))


@pytest.mark.unit
@pytest.mark.parametrize('step_id, files', [
    ('indices', ['sobol_z.png', 'sobol_w.png', 'sobol_ST_heatmap.png']),
    ('posterior', ['posterior.png']),
    ('laplace', ['laplace.png']),
])
def test_an_analysis_figure_is_the_same_drawn_alone(analysis_run, tmp_path, step_id, files):
    run, steps = analysis_run
    paths = _draw_both_ways(run, steps[step_id], tmp_path)
    assert [os.path.basename(p) for p in paths] == files


@pytest.mark.unit
def test_the_standard_script_is_copied_once(tmp_path):
    directory = str(tmp_path / 'figures')
    path = figures.write_standard_figures(directory)
    assert path == os.path.join(directory, 'standard_figures.py')
    with open(path, 'a') as f:
        f.write('# edited\n')
    assert figures.write_standard_figures(directory) == path
    with open(path) as f:
        assert f.read().endswith('# edited\n')
    figures.write_standard_figures(directory, overwrite=True)
    with open(path) as f:
        assert not f.read().endswith('# edited\n')


# --------------------------------------------------------------------------------------------
# the sweep and simulation figures, from a run
# --------------------------------------------------------------------------------------------

def _features_obs(path, plots=None):
    """Two experiments with q = 1 and 2, z's maximum as a feature in each."""
    return _write_json(path, {
        'protocol_info': {'pre_times': [0.0, 0.0], 'sim_times': [[1.0], [1.0]],
                          'params_to_change': {'mod__C/q': [[1.0], [2.0]]}},
        'data_items': [],
        'prediction_items': [
            {'data_item_name': f'z_max_{e}', 'item_name_for_plotting': 'z_max',
             'operands': ['mod__C/z'], 'unit': 'dimensionless', 'operation': 'max',
             'data_type': 'constant', 'experiment_idx': e} for e in range(2)],
        'prediction_plots': plots or [
            {'name': 'z_max vs q', 'kind': 'feature_vs_input', 'y': 'z_max',
             'x': {'params_to_change': 'mod__C/q', 'subexperiment_idx': 0}},
            {'name': 'q and z', 'kind': 'input_and_output',
             'input': {'params_to_change': 'mod__C/q'}, 'output': 'mod__C/z'}]})


def _with_standard_figures(library):
    raw = _read_json(library['chain_rest'])
    rest = raw['steps'][1]
    here = os.path.dirname(library['chain_rest'])
    # outside the parameterisation's directory, whose obs_data is its own
    data = os.path.join(os.path.dirname(library['modules']), 'data')
    obs = _features_obs(os.path.join(data, 'features_obs.json'))
    scan_obs = _write_json(os.path.join(data, 'scan_obs.json'), {
        'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]]}, 'data_items': [],
        'prediction_items': [{'data_item_name': 'z_max', 'operands': ['mod__C/z'],
                              'unit': 'dimensionless', 'operation': 'max',
                              'data_type': 'constant'}]})
    figures.write_standard_figures(os.path.join(here, 'figures'))
    raw['steps'] += [
        {'id': 'scan', 'kind': 'sweep', 'model': rest['model'], 'fixed_from': ['fit_a'],
         'obs_data': scan_obs, 'settings': {},
         'sweep': {'outputs': ['mod__C/z'], 'design': 'grid',
                   'parameters': {'mod__C/k': {'values': [1.0, 3.0]},
                                  'mod__C/q': {'values': [1.0, 2.0]}}}},
        {'id': 'run', 'kind': 'simulate', 'model': rest['model'], 'fixed_from': ['fit_a'],
         'obs_data': obs, 'settings': {}, 'simulate': {'outputs': ['mod__C/z']}},
        _figure_step('traces', 'sweep_traces', {'sweep': ('sweep', 'scan')}),
        _figure_step('features', 'sweep_features', {'sweep': ('sweep', 'scan')}),
        _figure_step('predictions', 'prediction_plots',
                     {'simulation': ('sweep', 'run'), 'obs_data': ('obs_data', 'run')})]
    return _write_json(library['chain_rest'], raw)


@pytest.fixture(scope='module')
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp('standard_figures_run')
    path = _with_standard_figures(build_workflow_library(str(root / 'library')))
    out = str(root / 'out')
    run_calibration_workflow(path, output_dir=out)
    return path, out


@pytest.mark.integration
@pytest.mark.parametrize('step_id, files', [
    ('traces', ['traces_exp_0.png']),
    ('features', ['z_max_vs_mod__C_k.png']),
    ('predictions', ['z_max_vs_q.png', 'q_and_z_exp_0.png', 'q_and_z_exp_1.png']),
])
def test_a_standard_figure_step_is_the_same_drawn_alone(finished, tmp_path, step_id, files):
    path, out = finished
    result = _read_json(os.path.join(out, step_id, 'step_result.json'))
    assert result['status'] == 'done'
    assert [os.path.basename(p) for p in result['figures']] == files
    step = next(s for s in _read_json(path)['steps'] if s['id'] == step_id)
    # the run's copy of the workflow names the script beside the workflow, which the run
    # directory does not hold: draw from a copy of the script there
    figures.write_standard_figures(os.path.join(out, 'figures'))
    paths = _draw_both_ways(out, step, tmp_path)
    assert [os.path.basename(p) for p in paths] == files
    assert figures.output_differences(os.path.join(out, step_id, 'figure'),
                                      os.path.dirname(paths[0])) == []


@pytest.mark.integration
def test_the_simulation_gives_the_features_drawn(finished):
    _, out = finished
    loaded = figures.load_inputs(out, {'run': {'kind': 'sweep', 'step': 'run'}})['run']
    np.testing.assert_allclose(loaded['feature_values'][0], [Y_A + 1.0, Y_A + 4.0], rtol=1e-6)
    assert loaded['manifest']['plot_groups'] == []


@pytest.mark.unit
def test_a_time_course_output_the_step_did_not_record_is_named(tmp_path):
    from libcuflynx.calibration_workflow import standard_figures
    obs = _read_json(_features_obs(str(tmp_path / 'obs.json'), plots=[
        {'name': 'q and w', 'kind': 'input_and_output',
         'input': {'params_to_change': 'mod__C/q'}, 'output': 'mod__C/w'}]))
    simulation = {'points': np.zeros((1, 0)), 'failed': np.array([False]),
                  'outputs': np.array(['mod__C/z']), 'features': np.array([], dtype=str),
                  'feature_values': np.zeros((1, 0))}
    with pytest.raises(ValueError, match="draws 'mod__C/w', which the step did not record"):
        standard_figures.prediction_plots(simulation, obs, str(tmp_path / 'out'))
    with pytest.raises(ValueError, match='reads one simulation .* has 2 points'):
        standard_figures.prediction_plots(dict(simulation, points=np.zeros((2, 0))), obs,
                                          str(tmp_path / 'out'))


@pytest.mark.unit
def test_every_standard_figure_passes_the_script_check(tmp_path):
    """Each function, named by a figure step with the inputs its docstring gives, passes the
    check a workflow makes when it is loaded."""
    import inspect
    from types import SimpleNamespace
    from libcuflynx.calibration_workflow import standard_figures
    script = figures.write_standard_figures(str(tmp_path))
    for function in standard_figures.FIGURES:
        names = [n for n in inspect.signature(function).parameters if n != 'out_dir']
        figures.check_script(SimpleNamespace(
            id=function.__name__, figure={'script': script, 'function': function.__name__,
                                          'inputs': {n: {} for n in names}, 'files': []}))
