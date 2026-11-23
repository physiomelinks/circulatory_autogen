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

def _optimiser_histories(run):
    """Calibration steps ``ga`` (a genetic algorithm's top-3 costs and normalised values per
    generation) and ``starts`` (two starts of a multi-start run, in actual values), as the
    optimisers write them, with p in [0, 10] and q in [1, 3]."""
    from libcuflynx.utilities import params_for_id_file
    params_for_id = os.path.join(run, 'params_for_id.json')
    params_for_id_file.write_params_for_id(params_for_id, [
        {'targets': [params_for_id_file.target_name(vessel, param)], 'min': lo, 'max': hi,
         'name_for_plotting': param} for vessel, param, lo, hi in (('A', 'p', 0.0, 10.0),
                                                                   ('C', 'q', 1.0, 3.0))])
    files = {
        'ga': {'best_cost_history.csv': '4.0,5.0,9.0\n2.0,2.5,4.0\n0.5,1.0,2.0\n',
               'best_param_vals_history.csv': 'A/p,C/q\n0.9,0.1\n0.5,0.4\n0.2,0.5\n'},
        'starts': {'multi_start_cost_history.csv':
                   'start_idx, iteration, cost\n0, 0, 3.0\n0, 1, 1.0\n1, 0, 6.0\n1, 1, 0.5\n',
                   'multi_start_param_vals_history.csv':
                   'start_idx, iteration, A/p, C/q\n0, 0, 8.0, 1.5\n0, 1, 4.0, 2.0\n'
                   '1, 0, 1.0, 2.8\n1, 1, 2.0, 2.0\n'}}
    for step_id, written in files.items():
        directory = os.path.join(run, step_id, 'param_id')
        os.makedirs(directory)
        for name, text in written.items():
            with open(os.path.join(directory, name), 'w') as f:
                f.write(text)
        np.save(os.path.join(directory, 'best_cost.npy'), np.array(0.5))
        if step_id == 'ga':
            # the best fit's errors, as save_error_vectors writes them: one within 2 stds,
            # one 3.5 stds off
            np.save(os.path.join(directory, 'percent_error_vec.npy'), np.array([1.5, -12.0]))
            np.save(os.path.join(directory, 'std_error_vec.npy'), np.array([0.4, -3.5]))
            np.save(os.path.join(directory, 'error_vec_names.npy'),
                    np.array(['z_max', 'w_mean']))
            np.save(os.path.join(directory, 'error_vec_data_references.npy'),
                    np.array(['Smith 2020', '']))
        _write_json(os.path.join(run, step_id, 'step_result.json'),
                    {'kind': 'calibration', 'status': 'done', 'param_id_output_dir': directory,
                     'params_for_id_used': params_for_id})


def _mcmc_step(run):
    """A calibration step ``mc`` with a 200-step chain of four walkers over p and q (p mixes,
    q drifts), its last 10 steps unsampled by one walker (NaN), and a burn-in of 0.25."""
    rng = np.random.default_rng(11)
    chain = np.empty((200, 4, 2))
    chain[:, :, 0] = 2.0 + 0.3 * rng.standard_normal((200, 4))
    chain[:, :, 1] = np.linspace(1.0, 3.0, 200)[:, None] + 0.01 * rng.standard_normal((200, 4))
    chain[190:, 3, :] = np.nan
    directory = os.path.join(run, 'mc', 'param_id')
    os.makedirs(directory)
    np.save(os.path.join(directory, 'mcmc_chain.npy'), chain)
    _write_json(os.path.join(run, 'mc', 'step_result.json'),
                {'kind': 'calibration', 'status': 'done', 'param_id_output_dir': directory,
                 'params_for_id_used': os.path.join(run, 'params_for_id.json'),
                 'settings': {'UQ_options': {'burn_in': 0.25}}})


def _emulation_step(run):
    """An emulation step ``emu`` with 12 held-out points of two parameters and two features,
    z (all truths positive) and w (a truth of 0)."""
    rng = np.random.default_rng(5)
    theta = rng.uniform([0.0, 1.0], [10.0, 3.0], (12, 2))
    y_true = np.column_stack([2 * theta[:, 0] ** 2 + 1, theta[:, 1] - theta[0, 1]])
    y_pred = y_true * (1 + 0.02 * rng.standard_normal(y_true.shape)) + 0.01
    directory = os.path.join(run, 'emu', 'emulator')
    os.makedirs(directory)
    np.savez(os.path.join(directory, 'emulator_validation.npz'), theta=theta, y_true=y_true,
             y_pred=y_pred, feature_labels=np.array(['z', 'w'], dtype=object),
             param_entry_labels=np.array(['A/p', 'C/q'], dtype=object))
    _write_json(os.path.join(directory, 'emulator_metadata.json'),
                {'feature_r2': [0.998, 0.91], 'feature_nrmse': [0.012, 0.05]})
    _write_json(os.path.join(run, 'emu', 'step_result.json'),
                {'kind': 'emulation', 'status': 'done'})


def _local_and_reduction_steps(run):
    """A local sensitivity step ``local`` (relative and absolute CSVs, no Sobol indices), a PCA
    reduction step ``reduce`` (two modes over features z and w, the second below the noise
    floor) and a simulate step ``proto`` whose obs_data changes q (a number, then a ramp
    trace) and k (a pulse shape) over two experiments."""
    directory = os.path.join(run, 'local', 'sensitivity')
    os.makedirs(directory)
    for key, scale in (('relative', 1.0), ('absolute', 4.0)):
        with open(os.path.join(directory, f'local_sensitivity_{key}.csv'), 'w') as f:
            f.write(f'output,A/p,C/q\nz,{0.9 * scale},{-0.2 * scale}\n'
                    f'w,{0.05 * scale},{-1.1 * scale}\n')
    _write_json(os.path.join(run, 'local', 'step_result.json'),
                {'kind': 'sensitivity', 'status': 'done', 'method': 'local',
                 'sensitivity_dir': directory, 'sobol_indices': None})
    reduce = os.path.join(run, 'reduce')
    os.makedirs(reduce)
    with open(os.path.join(reduce, 'pca_modes.csv'), 'w') as f:
        f.write('mode,variance,variance_share,informative,loading_z,loading_w\n'
                '1,3.0,0.75,True,0.8,-0.6\n2,1.0,0.25,False,0.6,0.8\n')
    with open(os.path.join(reduce, 'pca_mode_sobol_indices.csv'), 'w') as f:
        f.write('Parameter,S1_mode_1,ST_mode_1,S1_mode_2,ST_mode_2\n'
                'A/p,0.7,0.8,0.1,0.15\nC/q,0.1,0.2,0.6,0.9\n')
    report = _write_json(os.path.join(reduce, 'param_reduction.json'),
                         {'reduction': {'method': 'pca'}, 'selected': [{'parameter': 'A/p'}]})
    _write_json(os.path.join(reduce, 'step_result.json'),
                {'kind': 'reduction', 'status': 'done', 'report': report})
    obs = _write_json(os.path.join(run, 'data', 'proto_obs.json'), {
        'protocol_info': {
            'pre_times': [0.0, 1.0], 'sim_times': [[1.0, 2.0], [1.5]],
            'experiment_labels': ['step and ramp', 'pulse'],
            'params_to_change': {'mod__C/q': [[1.0, 'ramp'], [2.0]],
                                 'mod__C/k': [[3.0, 3.0], ['pulse']]},
            'protocol_traces': {'ramp': {'t': [0.0, 1.0], 'values': [0.0, 1.0]}},
            'protocol_shapes': {'pulse': {'events': [
                {'level': 1.0, 'start': 0.2, 'length': 0.3, 'period': 0, 'multiplier': 0}]}}},
        'data_items': []})
    _write_json(os.path.join(run, 'proto', 'step_result.json'),
                {'kind': 'simulate', 'status': 'done', 'inputs': {'obs_data': {'path': obs}}})


def _uq_step(run):
    """A uq step ``uq`` whose posterior predictive check kept one trace: 40 draws of three
    scalar observables (the third has no std) and 5 traces of v in its one segment, saved by
    the check's own save; and a uq step ``uq_failed`` whose check failed."""
    from libcuflynx.calibration_workflow.runner import POSTERIOR_PREDICTIVE_DIR
    from libcuflynx.param_id import posterior_predictive as pp
    rng = np.random.default_rng(5)
    data, std = np.array([1.0, 10.0, 3.0]), np.array([0.1, 1.0, 0.0])
    predictions = data + std * rng.standard_normal((40, 3)) + [0.0, 2.5, 0.2]
    t = np.linspace(0.0, 1.0, 50)
    traces = np.sin(2 * np.pi * t) + 0.05 * rng.standard_normal((5, 50))
    meta = {'time': {'0': t.tolist()}, 'experiment_labels': ['rest'], 'segments': {'0': [0, 0]},
            'observables': [{'kind': 'series', 'segment': 0, 'variable': 'C/v',
                             'label': 'v recorded', 'time': t[::5].tolist(),
                             'values': np.sin(2 * np.pi * t[::5]).tolist()}]}
    directory = os.path.join(run, 'uq', POSTERIOR_PREDICTIVE_DIR)
    result = pp.PosteriorPredictiveResult(
        rng.standard_normal((40, 2)), predictions, data, std, ['z_max', 'w_mean', 'u_min'],
        pp.coverage(predictions, data, std), {'source': 'chain', 'n_steps': 100,
                                              'burn_in_steps': 50, 'pool': 200},
        0, False, series={(0, 'C/v'): traces}, series_meta=meta)
    draws, coverage = result.save(directory)
    _write_json(os.path.join(run, 'uq', 'step_result.json'),
                {'kind': 'uq', 'status': 'done', 'posterior_predictive_dir': directory,
                 'posterior_predictive_draws': draws,
                 'posterior_predictive_coverage': coverage,
                 'posterior_predictive_series': os.path.join(directory, pp.SERIES_FILE)})
    _write_json(os.path.join(run, 'uq_failed', 'step_result.json'),
                {'kind': 'uq', 'status': 'done',
                 'posterior_predictive_summary': {'error': 'no chain'}})


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
    _optimiser_histories(run)
    _emulation_step(run)
    _local_and_reduction_steps(run)
    _mcmc_step(run)
    _uq_step(run)
    steps = [_figure_step('predictive', 'posterior_predictive',
                          {'predictive': ('posterior_predictive', 'uq')}),
             _figure_step('emulator', 'emulator_validation',
                          {'emulator': ('emulator_validation', 'emu')}),_figure_step('indices', 'sobol_indices', {'sensitivity': ('sensitivity', 'sobol')}),
             _figure_step('ga_progress', 'calibration_progress', {'progress': ('progress', 'ga')}),
             _figure_step('fit', 'fit_errors', {'errors': ('fit_errors', 'ga')}),
             _figure_step('starts_progress', 'calibration_progress',
                          {'progress': ('progress', 'starts')}),
             _figure_step('posterior', 'posterior', {'distribution': ('distribution', 'fit')}),
             _figure_step('laplace', 'laplace', {'laplace': ('laplace', 'ia')}),
             _figure_step('local', 'local_sensitivity', {'sensitivity': ('sensitivity', 'local')}),
             _figure_step('modes', 'pca_modes', {'reduction': ('reduction', 'reduce')}),
             _figure_step('inputs', 'protocol_inputs', {'obs_data': ('obs_data', 'proto')}),
             _figure_step('mcmc', 'mcmc_diagnostics', {'chain': ('mcmc_chain', 'mc')})]
    _write_json(os.path.join(run, WORKFLOW_FILE_NAME), {'steps': steps})
    return run, {step['id']: step for step in steps}


@pytest.fixture(scope='module')
def analysis_run(tmp_path_factory):
    return _analysis_run(tmp_path_factory.mktemp('analysis_run'))


@pytest.mark.unit
@pytest.mark.parametrize('step_id, files', [
    ('indices', ['sobol_z.png', 'sobol_w.png', 'sobol_ST_heatmap.png']),
    ('ga_progress', ['cost_history.png', 'param_history.png']),
    ('fit', ['fit_errors.csv', 'fit_percent_error.png', 'fit_std_error.png']),
    ('starts_progress', ['cost_history.png', 'param_history.png']),
    ('emulator', ['emulator_parity.png', 'emulator_residuals_z.png',
                  'emulator_residuals_w.png']),
    ('posterior', ['posterior.png']),
    ('laplace', ['laplace.png']),
    ('local', ['local_sensitivity_heatmap.png']),
    ('modes', ['pca_mode_1.png', 'pca_mode_2.png', 'pca_mode_ST_heatmap.png']),
    ('inputs', ['inputs_exp_0.png', 'inputs_exp_1.png']),
    ('mcmc', ['mcmc_chain.png', 'mcmc_cumulative_mean.png', 'mcmc_autocorrelation.png']),
    ('predictive', ['posterior_predictive_intervals.csv', 'posterior_predictive_intervals.png',
                    'posterior_predictive_0_C_v.png']),
])
def test_an_analysis_figure_is_the_same_drawn_alone(analysis_run, tmp_path, step_id, files):
    run, steps = analysis_run
    paths = _draw_both_ways(run, steps[step_id], tmp_path)
    assert [os.path.basename(p) for p in paths] == files


@pytest.mark.unit
def test_the_progress_input_reads_the_history_and_the_bounds(analysis_run):
    run, _ = analysis_run
    loaded = figures.load_inputs(run, {'ga': {'kind': 'progress', 'step': 'ga'},
                                       'starts': {'kind': 'progress', 'step': 'starts'}})
    ga, starts = loaded['ga'], loaded['starts']
    assert ga['param_labels'] == ['A/p', 'C/q']
    assert [row[0] for row in ga['cost_history']] == [4.0, 2.0, 0.5]
    np.testing.assert_allclose(ga['param_history'][-1], [2.0, 2.0])
    assert (ga['param_mins'], ga['param_maxs']) == ([0.0, 1.0], [10.0, 3.0])
    assert [s['cost'] for s in starts['starts']] == [[3.0, 1.0], [6.0, 0.5]]


@pytest.mark.unit
def test_the_emulator_input_reads_the_points_and_the_scores(analysis_run):
    run, _ = analysis_run
    loaded = figures.load_inputs(run, {'e': {'kind': 'emulator_validation', 'step': 'emu'}})['e']
    assert loaded['theta'].shape == (12, 2) and loaded['y_pred'].shape == (12, 2)
    assert (loaded['feature_labels'], loaded['param_entry_labels']) == (['z', 'w'],
                                                                        ['A/p', 'C/q'])
    assert loaded['feature_r2'] == [0.998, 0.91] and loaded['feature_rmse'] is None


@pytest.mark.unit
def test_the_sensitivity_input_reads_local_and_sobol_indices(analysis_run):
    run, _ = analysis_run
    loaded = figures.load_inputs(run, {'local': {'kind': 'sensitivity', 'step': 'local'},
                                       'sobol': {'kind': 'sensitivity', 'step': 'sobol'}})
    local, sobol = loaded['local'], loaded['sobol']
    assert (local['S1'], local['ST']) == ({}, {})
    assert local['local']['relative'] == {'z': {'A/p': 0.9, 'C/q': -0.2},
                                          'w': {'A/p': 0.05, 'C/q': -1.1}}
    assert local['local']['absolute']['w']['C/q'] == pytest.approx(-4.4)
    assert sobol['local'] is None and sobol['ST']['z'] == {'A/p': 0.7, 'C/q': 0.35}


@pytest.mark.unit
def test_the_mcmc_input_reads_the_sampled_chain_and_the_burn_in(analysis_run):
    run, _ = analysis_run
    loaded = figures.load_inputs(run, {'c': {'kind': 'mcmc_chain', 'step': 'mc'}})['c']
    assert loaded['chain'].shape == (190, 4, 2) and np.isfinite(loaded['chain']).all()
    assert loaded['param_labels'] == ['A/p', 'C/q']
    assert loaded['burn_in_index'] == 47


@pytest.mark.unit
def test_the_posterior_predictive_input_reads_the_draws_coverage_and_traces(analysis_run,
                                                                           tmp_path):
    """The check's draws, data and coverage, and its kept traces by segment and variable;
    the figure's table puts each observable's data against the draws' intervals."""
    import csv
    from libcuflynx.calibration_workflow import standard_figures
    run, _ = analysis_run
    loaded = figures.load_inputs(run, {'p': {'kind': 'posterior_predictive', 'step': 'uq'}})['p']
    assert loaded['labels'] == ['z_max', 'w_mean', 'u_min']
    assert loaded['predictions'].shape == (40, 3)
    assert loaded['ground_truth'].tolist() == [1.0, 10.0, 3.0]
    assert set(loaded['coverage']['coverage']['levels']) == {'0.8', '0.95'}
    assert list(loaded['series']['traces']) == ['0|C/v']
    assert loaded['series']['traces']['0|C/v'].shape == (5, 50)
    assert loaded['series']['time']['0'].shape == (50,)
    assert loaded['series']['meta']['observables'][0]['label'] == 'v recorded'
    paths = standard_figures.posterior_predictive(loaded, str(tmp_path))
    with open(paths[0]) as f:
        rows = {row['observable']: row for row in csv.DictReader(f)}
    assert list(rows) == ['z_max', 'w_mean', 'u_min']
    # w_mean's draws sit 2.5 stds above its data: outside the 95% interval
    assert rows['z_max']['inside_95'] == 'true' and rows['w_mean']['inside_95'] == 'false'
    assert float(rows['w_mean']['median']) == pytest.approx(12.5, abs=0.5)
    with pytest.raises(figures.WorkflowError, match='no posterior predictive check to read: '
                                                    'it failed \\(no chain\\)'):
        figures.load_inputs(run, {'p': {'kind': 'posterior_predictive', 'step': 'uq_failed'}})


@pytest.mark.unit
def test_the_fit_errors_input_reads_what_a_calibration_saved(analysis_run):
    """The four files a calibration writes when it ends, names and references as text; a
    calibration without them (its run stopped before it could write them) says how to get
    them."""
    run, _ = analysis_run
    loaded = figures.load_inputs(run, {'e': {'kind': 'fit_errors', 'step': 'ga'}})['e']
    assert loaded['names'] == ['z_max', 'w_mean']
    assert loaded['data_references'] == ['Smith 2020', '']
    assert loaded['percent_error'].tolist() == [1.5, -12.0]
    assert loaded['std_error'].tolist() == [0.4, -3.5]
    with pytest.raises(figures.WorkflowError, match='no best-fit errors to read.*'
                                                    'save_best_fit_errors'):
        figures.load_inputs(run, {'e': {'kind': 'fit_errors', 'step': 'starts'}})


@pytest.mark.unit
def test_the_autocorrelation_title_says_whether_a_walker_mixed(tmp_path):
    """p is white noise and stays within the guides; q drifts and stays correlated."""
    from unittest import mock
    from libcuflynx.calibration_workflow import standard_figures
    rng = np.random.default_rng(2)
    chain = np.stack([rng.standard_normal((300, 3)),
                      np.cumsum(rng.standard_normal((300, 3)), axis=0)], axis=2)
    titles = []
    with mock.patch.object(standard_figures, '_mcmc_axes',
                           side_effect=lambda ax, title, xlabel: titles.append(title)):
        standard_figures.mcmc_diagnostics(
            {'chain': chain, 'param_labels': ['p', 'q'], 'burn_in_index': 150},
            str(tmp_path))
    assert titles[-2:] == ['p: within', 'q: still correlated']


@pytest.mark.unit
def test_the_reduction_input_reads_the_report_and_the_modes(analysis_run):
    run, _ = analysis_run
    loaded = figures.load_inputs(run, {'r': {'kind': 'reduction', 'step': 'reduce'}})['r']
    assert loaded['selected'] == [{'parameter': 'A/p'}]
    assert [(m['mode'], m['informative'], m['loadings']) for m in loaded['modes']] == [
        (1, True, {'z': 0.8, 'w': -0.6}), (2, False, {'z': 0.6, 'w': 0.8})]
    assert loaded['mode_indices']['ST']['mode_2'] == {'A/p': 0.15, 'C/q': 0.9}


@pytest.mark.unit
def test_an_analysis_figure_without_its_results_says_what_it_needs(tmp_path):
    from libcuflynx.calibration_workflow import standard_figures
    out = str(tmp_path / 'out')
    with pytest.raises(ValueError, match='no local sensitivities'):
        standard_figures.local_sensitivity({'S1': {}, 'ST': {'z': {'A/p': 0.5}},
                                            'local': None}, out)
    with pytest.raises(ValueError, match="no PCA modes \\(method 'sobol'\\)"):
        standard_figures.pca_modes({'reduction': {'method': 'sobol'}, 'modes': None,
                                    'mode_indices': None}, out)
    with pytest.raises(ValueError, match='changes no parameter'):
        standard_figures.protocol_inputs(
            {'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]]}, 'data_items': []},
            out)


@pytest.mark.unit
def test_a_protocol_trace_is_drawn_over_its_sub_experiment():
    """A trace is cut to its sub-experiment, keeps a jump (a repeated time) and is held at its
    ends, as Myokit holds it."""
    from libcuflynx.calibration_workflow.standard_figures import _trace_points
    t, v = _trace_points(np.array([0.0, 0.5, 0.5, 2.0]), np.array([1.0, 1.0, 3.0, 3.0]), 1.0)
    np.testing.assert_allclose(t, [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(v, [1.0, 1.0, 3.0, 3.0])
    t, v = _trace_points(np.array([0.2, 0.4]), np.array([2.0, 4.0]), 1.0)
    np.testing.assert_allclose(t, [0.0, 0.2, 0.4, 1.0])
    np.testing.assert_allclose(v, [2.0, 2.0, 4.0, 4.0])


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


#: held-out data on z (Y_A + q^2: q = 1 in experiment 0, 2 in experiment 1): two scalars, one
#: 1.5 stds off, and a series 0.1 above the model at each of its points after the first
_HELD_OUT = [
    {'data_item_name': 'z_max_0', 'operands': ['mod__C/z'], 'unit': 'dimensionless',
     'operation': 'max', 'data_type': 'constant', 'experiment_idx': 0,
     'value': Y_A + 1.15, 'std': 0.1},
    {'data_item_name': 'z_max_1', 'operands': ['mod__C/z'], 'unit': 'dimensionless',
     'operation': 'max', 'data_type': 'constant', 'experiment_idx': 1,
     'value': Y_A + 4.0, 'std': 0.1},
    {'data_item_name': 'z_course', 'operands': ['mod__C/z'], 'unit': 'dimensionless',
     'data_type': 'series', 'experiment_idx': 1, 'obs_dt': 0.25, 'std': 0.05,
     'value': [Y_A + 4.0] + [Y_A + 4.1] * 4}]


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
    check_obs = _write_json(os.path.join(data, 'check_obs.json'), {
        'protocol_info': {'pre_times': [0.0, 0.0], 'sim_times': [[1.0], [1.0]],
                          'params_to_change': {'mod__C/q': [[1.0], [2.0]]}},
        'data_items': [], 'prediction_items': _HELD_OUT})
    figures.write_standard_figures(os.path.join(here, 'figures'))
    raw['steps'] += [
        {'id': 'check', 'kind': 'simulate', 'model': rest['model'], 'fixed_from': ['fit_a'],
         'obs_data': check_obs, 'settings': {}, 'simulate': {'outputs': []}},
        _figure_step('valid', 'validation', {'validation': ('validation', 'check')}),
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
    ('valid', ['validation_scores.csv', 'validation_parity.png', 'validation_z_course.png']),
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


@pytest.mark.integration
def test_the_validation_input_scores_the_held_out_items(finished):
    _, out = finished
    items = {item['data_item_name']: item for item in figures.load_inputs(
        out, {'v': {'kind': 'validation', 'step': 'check'}})['v']['items']}
    assert sorted(items) == ['z_course', 'z_max_0', 'z_max_1']
    np.testing.assert_allclose(items['z_max_0']['mean_abs_z'], 1.5, rtol=1e-4)
    assert items['z_max_0']['within_2std'] == 1.0
    np.testing.assert_allclose(items['z_max_1']['model'], [Y_A + 4.0], rtol=1e-6)
    course = items['z_course']
    np.testing.assert_allclose(course['t'], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(course['model'], [Y_A + 4.0] * 5, rtol=1e-6)
    np.testing.assert_allclose(course['rmse'], 0.1 * np.sqrt(0.8), rtol=1e-4)
    with open(os.path.join(out, 'valid', 'figure', 'validation_scores.csv')) as f:
        assert f.readline().strip() == ('item,operation,unit,n_points,rmse,nrmse,mean_abs_z,'
                                        'within_2std')


@pytest.mark.integration
def test_a_step_of_several_points_has_no_validation(finished):
    _, out = finished
    with pytest.raises(figures.WorkflowError, match='step "scan" compared no prediction item'):
        figures.load_inputs(out, {'v': {'kind': 'validation', 'step': 'scan'}})


@pytest.mark.unit
def test_a_validation_without_items_says_so(tmp_path):
    from libcuflynx.calibration_workflow import standard_figures
    with pytest.raises(ValueError, match='no prediction item carries a value'):
        standard_figures.validation({'items': []}, str(tmp_path))


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
