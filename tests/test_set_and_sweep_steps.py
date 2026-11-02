"""The ``set`` and ``sweep`` kinds of a calibration workflow.

The fixture library is libcuflynx.external_testing.workflow_library: in ``chain``, C reads A's
output, z = 2 p_A^2 + k q^2, so the value a set step gives p_A decides what a later step fits for
q, and what a sweep of q records is known in closed form.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from libcuflynx.calibration_workflow import (WorkflowError, load_workflow_run, plan_workflow,
                                             run_calibration_workflow)
from libcuflynx.calibration_workflow import sweep as sweep_module
from libcuflynx.calibration_workflow.pack import pack_workflow
from libcuflynx.calibration_workflow.spec import parse_workflow
from libcuflynx.calibration_workflow.sweep import parameter_values, run_on_ranks, sweep_points
from libcuflynx.external_testing.workflow_library import Y_A, Z_C, build_workflow_library
from libcuflynx.reporting.methods import workflow_methods_latex
from libcuflynx.reporting.provenance import provenance_graph, provenance_svg
from libcuflynx.utilities import mpi_utils
from libcuflynx.utilities.parameters_file import model_parameter_rows, read_parameters_file

REASON = 'the fitted amount is too low for the sugar flux'


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write(library, raw):
    with open(library['chain_rest'], 'w') as f:
        json.dump(raw, f, indent=1)
    return library['chain_rest']


def _raw(library):
    return _read_json(library['chain_rest'])


def _merged(result):
    return {r['parameterisation_name']: r for r in result['merged_parameters']}


def _parse(library, raw):
    return parse_workflow(raw, path=library['chain_rest'])


def _set_step(step_id='hand', entries=None):
    return {'id': step_id, 'kind': 'set', 'set': entries or [
        {'parameter': 'mod__A/p', 'value': 1.5, 'reason': REASON}]}


def _with_set_step(library, entries=None):
    '''chain_rest with a set step giving A's value in place of fit_a.'''
    raw = _raw(library)
    rest = raw['steps'][1]
    raw['steps'] = [_set_step(entries=entries), dict(rest, fixed_from=['hand'])]
    return _write(library, raw)


def _sweep_step(raw, **sweep):
    rest = raw['steps'][1]
    return {'id': 'scan', 'kind': 'sweep', 'model': rest['model'],
            'settings': {'sim_time': 1.0},
            'sweep': dict({'outputs': ['mod__C/z'],
                           'parameters': {'mod__C/q': {'values': [1.0, 2.0, 3.0]}}}, **sweep)}


# --------------------------------------------------------------------------------------------
# the points of a sweep
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_grid_is_every_combination():
    sweep = {'design': 'grid', 'parameters': {'a': [0.0, 1.0], 'b': {'values': [5.0, 6.0, 7.0]}}}
    sweep['parameters']['a'] = {'min': 0.0, 'max': 1.0, 'points': 2}
    names, points = sweep_points(sweep, {'a': 9.0, 'b': 9.0})
    assert names == ['a', 'b']
    assert points.tolist() == [[0, 5], [0, 6], [0, 7], [1, 5], [1, 6], [1, 7]]


@pytest.mark.unit
def test_one_at_a_time_holds_the_others_at_the_models_values():
    sweep = {'design': 'one_at_a_time',
             'parameters': {'a': {'values': [1.0, 2.0]}, 'b': {'factors': [0.5, 2.0]}}}
    _, points = sweep_points(sweep, {'a': 7.0, 'b': 4.0})
    assert points.tolist() == [[1, 4], [2, 4], [7, 2], [7, 8]]


@pytest.mark.unit
def test_zipped_parameters_move_together_as_one_axis():
    sweep = {'design': 'grid', 'zip': [['apical', 'basolateral']],
             'parameters': {'apical': {'values': [0.0, 0.1, 1.0]},
                            'glucose': {'values': [15.0, 40.0]},
                            'basolateral': {'values': [1.0, 1.0, 10.0]}}}
    names, points = sweep_points(sweep, {'apical': 9.0, 'glucose': 9.0, 'basolateral': 9.0})
    assert names == ['apical', 'glucose', 'basolateral']
    assert points.tolist() == [[0, 15, 1], [0, 40, 1], [0.1, 15, 1], [0.1, 40, 1],
                               [1, 15, 10], [1, 40, 10]]
    sweep['design'] = 'one_at_a_time'
    _, points = sweep_points(sweep, {'apical': 9.0, 'glucose': 5.0, 'basolateral': 8.0})
    assert points.tolist() == [[0, 5, 1], [0.1, 5, 1], [1, 5, 10], [9, 15, 8], [9, 40, 8]]


@pytest.mark.unit
def test_a_sweep_of_nothing_is_one_run_of_the_protocol():
    names, points = sweep_points({'design': 'grid', 'parameters': {}}, {})
    assert names == [] and points.shape == (1, 0)


@pytest.mark.unit
def test_a_sweep_on_the_target_takes_every_calibration_and_set_step(library):
    raw = _raw(library)
    raw['steps'] = [{'id': 'scan', 'kind': 'sweep', 'on_target': True,
                     'settings': {'sim_time': 1.0},
                     'sweep': {'outputs': ['mod__C/z'], 'parameters': {}}}] + raw['steps'] + \
        [_set_step()]
    workflow = _parse(library, raw)
    scan = workflow.step('scan')
    assert scan.model == workflow.target
    # in the order the steps run, so a later step wins as it does in the target
    assert scan.fixed_from == scan.depends_on[:0] + ['fit_a', 'rest', 'hand']
    assert [s.id for s in workflow.topological_order()][-1] == 'scan'


@pytest.mark.unit
def test_on_target_is_for_a_sweep(library):
    raw = _raw(library)
    raw['steps'][1]['on_target'] = True
    with pytest.raises(WorkflowError, match='"on_target" is for a sweep step'):
        _parse(library, raw)


@pytest.mark.unit
def test_the_range_forms_match_cams():
    assert parameter_values([0.0, 1.0]).tolist() == [0, 0.25, 0.5, 0.75, 1]
    np.testing.assert_allclose(
        parameter_values({'min': 1.0, 'max': 100.0, 'points': 3, 'scale': 'log'}), [1, 10, 100])
    assert parameter_values({'factors': [2.0]}) is None
    assert parameter_values({'factors': [2.0]}, 3.0).tolist() == [6.0]


@pytest.mark.unit
@pytest.mark.parametrize('spec, nominal, message', [
    ({'min': 1.0, 'max': 1.0}, None, 'not below max'),
    ({'min': 0.0, 'max': 1.0, 'scale': 'log'}, None, 'positive min'),
    ({'factors': [2.0]}, 0.0, 'which is 0'),
])
def test_an_unusable_range_says_why(spec, nominal, message):
    with pytest.raises(WorkflowError, match=message):
        parameter_values(spec, nominal, 'here')


@pytest.mark.unit
def test_a_raising_point_keeps_its_place():
    class OneRank:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def gather(self, block, root=0):
            return [block]

    def evaluate(index):
        if index == 1:
            raise RuntimeError('no solution')
        return index * 10

    assert run_on_ranks(evaluate, 3, OneRank()) == {0: (0, None), 1: (None, 'no solution'),
                                                    2: (20, None)}


# --------------------------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------------------------

def _no_reason(step):
    del step['set'][0]['reason']


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (_no_reason, "'reason' is a required property"),
    (lambda s: s['set'][0].update(reason='   '), r"at steps\[2\] \('hand'\)"),
    (lambda s: s['set'][0].update(scale=2.0), 'set'),
    (lambda s: s['set'][0].update(of='fit_a'), 'set'),
    (lambda s: s.update(obs_data='o.json'), r'has no \[\'obs_data\'\]'),
    (lambda s: s.pop('set'), "'set' is a required property"),
    (lambda s: s['set'].append(dict(s['set'][0])), 'more than once'),
    (lambda s: s.update(fixed_from=['fit_a']), 'no "fixed_from"'),
])
def test_invalid_set_steps_say_why(library, edit, message):
    raw = _raw(library)
    step = _set_step()
    edit(step)
    raw['steps'].append(step)
    with pytest.raises(WorkflowError, match=message):
        _parse(library, raw)


@pytest.mark.unit
def test_a_scale_of_a_sweep_or_an_unknown_step_is_refused(library):
    raw = _raw(library)
    raw['steps'] += [_sweep_step(raw), _set_step(entries=[
        {'parameter': 'mod__A/p', 'scale': 2.0, 'of': 'scan', 'reason': REASON}])]
    with pytest.raises(WorkflowError, match='takes a calibration or set step'):
        _parse(library, raw)
    raw['steps'][-1]['set'][0]['of'] = 'nope'
    with pytest.raises(WorkflowError, match='unknown step "nope"'):
        _parse(library, raw)


@pytest.mark.unit
def test_a_sweep_cannot_pass_values_on(library):
    raw = _raw(library)
    raw['steps'].append(_sweep_step(raw))
    raw['steps'][1]['fixed_from'] = ['scan']
    with pytest.raises(WorkflowError, match='a sweep step; it takes a calibration or set step'):
        _parse(library, raw)


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda s: s.update(params_for_id='p.json'), 'this combination of keys is not allowed'),
    (lambda s: s['sweep'].update(design='random'), 'not one of'),
    (lambda s: s['sweep'].update(zip=[['mod__C/q', 'mod__C/k']]),
     r"\"zip\" names \['mod__C/k'\], which it does not sweep"),
    (lambda s: s['sweep'].update(parameters={'mod__C/q': {'values': [1.0, 2.0]},
                                             'mod__C/k': {'values': [1.0]}},
                                 zip=[['mod__C/q', 'mod__C/k']]),
     'take different numbers of values'),
    (lambda s: s.update(on_target=True), r"names no \['model'\]"),
    (lambda s: s['sweep'].update(outputs=[]), 'records nothing'),
    (lambda s: s['sweep']['parameters'].update({'mod__C/q': {'min': 2.0, 'max': 1.0}}),
     'not below max'),
    (lambda s: s.update(settings={}), r"needs \['sim_time'\]"),
    (lambda s: s.update(model={'select': ['A']}), 'not as a selection'),
])
def test_invalid_sweep_steps_say_why(library, edit, message):
    raw = _raw(library)
    step = _sweep_step(raw)
    edit(step)
    raw['steps'].append(step)
    with pytest.raises(WorkflowError, match=message):
        _parse(library, raw)


@pytest.mark.unit
def test_a_set_step_names_a_vessel_of_the_target(library):
    with pytest.raises(WorkflowError, match='"mod__X/p" names vessel "mod__X"'):
        run_calibration_workflow(_with_set_step(library, [
            {'parameter': 'mod__X/p', 'value': 1.0, 'reason': REASON}]), dry_run=True)


@pytest.mark.unit
def test_the_plan_shows_what_each_new_kind_does(library):
    raw = _raw(library)
    raw['steps'] += [_sweep_step(raw), _set_step()]
    plan = plan_workflow(_write(library, raw))
    steps = {s['id']: s for s in plan['steps']}
    assert steps['hand']['set'][0]['target_model_name'] == 'p_mod__A'
    assert steps['scan']['sweep']['parameters'] == {'mod__C/q': {'values': [1.0, 2.0, 3.0]}}


# --------------------------------------------------------------------------------------------
# running a set step
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_a_set_value_reaches_a_later_step_and_the_target(library, tmp_path):
    out = str(tmp_path / 'out')
    result = run_calibration_workflow(_with_set_step(library), output_dir=out)
    merged = _merged(result)
    assert merged['A/p']['value'] == 1.5
    assert (merged['A/p']['origin'], merged['A/p']['reason']) == ('set_by_hand', REASON)
    assert 'origin' not in merged['C/q']
    assert merged['C/q']['value'] == pytest.approx(np.sqrt(Z_C - 2 * 1.5 ** 2), rel=1e-4)

    rest = _read_json(os.path.join(out, 'rest', 'step_result.json'))
    assert [(f['model_name'], f['from_step'], f['origin']) for f in rest['fixed']] == [
        ('p_mod__A', 'hand', 'set_by_hand')]
    target = {r['variable_name']: r for r in model_parameter_rows(
        result['target_model']['parameters_path'])}
    assert float(target['p_mod__A']['value']) == 1.5
    assert (target['p_mod__A']['origin'], target['p_mod__A']['reason']) == ('set_by_hand', REASON)


@pytest.mark.integration
def test_a_scale_multiplies_a_steps_value_or_the_inherited_one(library, tmp_path):
    raw = _raw(library)
    raw['steps'] += [_set_step('hand', [
        {'parameter': 'mod__A/p', 'scale': 2.0, 'of': 'fit_a', 'reason': REASON},
        {'parameter': 'mod__C/k', 'scale': 3.0, 'of': 'inherited', 'reason': REASON},
        {'parameter': 'mod__C/q', 'value': 4.0, 'reason': REASON}])]
    raw['steps'][1]['fixed_from'] = ['fit_a']
    with pytest.warns(UserWarning, match='calibrated by more than one step'):
        result = run_calibration_workflow(_write(library, raw), output_dir=str(tmp_path / 'out'))
    merged = _merged(result)
    assert merged['A/p']['value'] == pytest.approx(2.0 * np.sqrt(Y_A / 2), rel=1e-4)
    assert merged['C/k']['value'] == 3.0
    # the set step runs last here, so its value is the one merged
    assert merged['C/q']['value'] == 4.0
    hand = _read_json(str(tmp_path / 'out' / 'hand' / 'step_result.json'))
    by_name = {r['model_name']: r for r in hand['set']}
    assert by_name['k_mod__C']['base'] == 1.0
    assert by_name['p_mod__A']['of'] == 'fit_a'
    assert by_name['k_mod__C']['units'] == 'dimensionless'


@pytest.mark.integration
def test_a_set_step_is_stale_when_its_source_changes(library, tmp_path):
    raw = _raw(library)
    raw['steps'] = [raw['steps'][0], _set_step('hand', [
        {'parameter': 'mod__A/p', 'scale': 2.0, 'of': 'fit_a', 'reason': REASON}]),
        dict(raw['steps'][1], fixed_from=['hand'])]
    path, out = _write(library, raw), str(tmp_path / 'out')
    run_calibration_workflow(path, output_dir=out)
    raw['steps'][1]['set'][0]['scale'] = 3.0
    _write(library, raw)
    with pytest.raises(WorkflowError, match='its settings has changed since "hand" ran'):
        run_calibration_workflow(path, output_dir=out, from_step='rest')


@pytest.mark.integration
def test_the_calibrated_parameters_file_says_what_was_set_by_hand(library, tmp_path):
    result = run_calibration_workflow(_with_set_step(library), output_dir=str(tmp_path / 'out'),
                                      write_calibrated=True)
    doc = read_parameters_file(result['calibrated_parameters_file'], kind='module')
    leaf = doc['A']['p']
    assert (leaf['origin'], leaf['reason']) == ('set_by_hand', REASON)
    assert leaf['data_reference'].startswith('set by hand in workflow chain_rest, step hand')
    assert doc['C']['q']['data_reference'].startswith('calibrated by workflow')


@pytest.mark.integration
def test_the_diagram_shows_a_set_value_with_its_reason_and_never_as_calibrated(library, tmp_path):
    out = str(tmp_path / 'out')
    run_calibration_workflow(_with_set_step(library), output_dir=out)
    graph = provenance_graph(out)
    nodes = {n['id']: n for n in graph['nodes']}
    hand = nodes['hand']
    assert (hand['kind'], hand['mark'], hand['counts']['set_by_hand']) == ('values', 'set_by_hand',
                                                                          1)
    assert hand['parameters'][0]['reason'] == REASON
    assert hand['parameters'][0]['value'] == 1.5
    assert REASON in hand['lines'][0]
    assert graph['counts']['set_by_hand'] == 1
    assert graph['counts']['calibrated'] == 1
    assert '__set_by_hand__' not in nodes
    assert '1 calibrated value' in nodes['__target__']['lines']
    assert '1 value set by hand' in nodes['__target__']['lines']
    edges = {(e['from'], e['to']): e for e in graph['edges']}
    assert ('hand', '__target__') in edges
    assert edges[('hand', 'rest')]['label'] == 'fixed'
    assert ('__source__', 'hand') not in edges
    given = next(v for v in nodes['rest']['parameters'] if v['from_step'] == 'hand')
    assert given['origin'] == 'set_by_hand'
    assert 'SET BY HAND' in provenance_svg(graph)


# --------------------------------------------------------------------------------------------
# running a sweep
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_a_sweep_records_every_point_and_feeds_nothing_forward(library, tmp_path):
    raw = _raw(library)
    raw['steps'].append(dict(_sweep_step(raw), fixed_from=['fit_a']))
    out = str(tmp_path / 'out')
    result = run_calibration_workflow(_write(library, raw), output_dir=out)
    assert set(_merged(result)) == {'A/p', 'C/q'}
    assert _merged(result)['C/q']['value'] == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)

    step = _read_json(os.path.join(out, 'scan', 'step_result.json'))
    assert (step['n_points'], step['n_failed'], step['calibrated']) == (3, 0, [])
    assert [f['from_step'] for f in step['fixed']] == ['fit_a']
    with np.load(step['sweep_results']) as results:
        assert results['points'].ravel().tolist() == [1.0, 2.0, 3.0]
        assert results['outputs'].tolist() == ['mod__C/z']
        assert not results['failed'].any()
        z = results['trace_0'][:, 0, -1]
        assert results['trace_0'].shape[1] == 1
        assert len(results['time_0']) == results['trace_0'].shape[2]
    np.testing.assert_allclose(z, Y_A + np.array([1.0, 2.0, 3.0]) ** 2, rtol=1e-6)
    manifest = _read_json(step['sweep_manifest'])
    assert manifest['parameters'][0]['nominal'] == 1.0
    assert manifest['failures'] == []

    graph = provenance_graph(out)
    scan = next(n for n in graph['nodes'] if n['id'] == 'scan')
    assert scan['kind'] == 'analysis' and scan['mark'] is None
    assert '3 points, 0 failed' in scan['lines']
    assert '`mod__C/q` {1, 2, 3}' in scan['lines']
    # it passes no values on: its only edges out are to the files it wrote
    assert {(e['to'], e['style']) for e in graph['edges'] if e['from'] == 'scan'} == {
        ('scan/sweep_results.npz', 'writes')}
    figure = next(n for n in graph['nodes'] if n['id'] == 'scan/sweep_results.npz')
    assert (figure['kind'], figure['status']) == ('figure', 'done')
    assert figure['lines'] == ['`scan/sweep/sweep_results.npz`']
    assert [e['label'] for e in graph['edges'] if e['to'] == 'scan'] == ['fixed']


@pytest.mark.integration
def test_a_sweep_on_the_target_runs_with_the_merged_values(library, tmp_path):
    """fit_a and the later set step both give A's p; on the target the set step's 1.5 wins, as
    it does in the merged target, so z = 2 * 1.5^2 + q^2 (k = 1)."""
    raw = _raw(library)
    raw['steps'] += [_set_step(), {
        'id': 'scan', 'kind': 'sweep', 'on_target': True, 'settings': {'sim_time': 1.0},
        'sweep': {'outputs': ['mod__C/z'], 'parameters': {'mod__C/q': {'values': [1.0, 2.0]}}}}]
    out = str(tmp_path / 'out')
    with pytest.warns(UserWarning, match='calibrated by more than one step'):
        result = run_calibration_workflow(_write(library, raw), output_dir=out)
    assert _merged(result)['A/p']['value'] == 1.5

    step = _read_json(os.path.join(out, 'scan', 'step_result.json'))
    by_name = {f['model_name']: f for f in step['fixed']}
    assert (by_name['p_mod__A']['value'], by_name['p_mod__A']['from_step']) == (1.5, 'hand')
    with np.load(step['sweep_results']) as results:
        np.testing.assert_allclose(results['trace_0'][:, 0, -1], 2 * 1.5 ** 2 + np.array([1.0, 4.0]),
                                   rtol=1e-6)

    graph = provenance_graph(out)
    scan = next(n for n in graph['nodes'] if n['id'] == 'scan')
    assert scan['lines'][:2] == ['the merged target', '`mod__C/q` {1, 2}']
    # drawn below the merged box, not from each step it took values from
    assert [(e['from'], e['label']) for e in graph['edges'] if e['to'] == 'scan'] == [
        ('__target__', 'runs on')]
    assert ('scan', 'scan/sweep_results.npz') in {(e['from'], e['to']) for e in graph['edges']}
    assert 'the merged target' in provenance_svg(graph)


@pytest.mark.integration
def test_a_protocol_run_with_nothing_swept(library, tmp_path):
    raw = _raw(library)
    raw['steps'].append(dict(_sweep_step(raw, parameters={}), fixed_from=['fit_a']))
    out = str(tmp_path / 'out')
    run_calibration_workflow(_write(library, raw), output_dir=out)
    step = _read_json(os.path.join(out, 'scan', 'step_result.json'))
    assert (step['n_points'], step['n_failed'], step['parameters']) == (1, 0, [])
    with np.load(step['sweep_results']) as results:
        assert results['points'].shape == (1, 0)
        np.testing.assert_allclose(results['trace_0'][0, 0, -1], Y_A + 1.0, rtol=1e-6)
    scan = next(n for n in provenance_graph(out)['nodes'] if n['id'] == 'scan')
    assert 'the protocol, nothing swept' in scan['lines']


@pytest.mark.integration
def test_a_sweep_runs_at_a_calibrated_result(library, tmp_path):
    raw = _raw(library)
    raw['steps'].append(dict(_sweep_step(raw, parameters={'mod__A/p': {'factors': [1.0, 2.0]}}),
                             result_from='rest'))
    out = str(tmp_path / 'out')
    run_calibration_workflow(_write(library, raw), output_dir=out)
    step = _read_json(os.path.join(out, 'scan', 'step_result.json'))
    with np.load(step['sweep_results']) as results:
        np.testing.assert_allclose(results['points'].ravel(),
                                   [np.sqrt(Y_A / 2), 2 * np.sqrt(Y_A / 2)], rtol=1e-4)
        z = results['trace_0'][:, 0, -1]
    np.testing.assert_allclose(z, [Z_C, Z_C + 3 * Y_A], rtol=1e-4)


@pytest.mark.integration
def test_a_failed_point_stays_a_nan_row(library, tmp_path, monkeypatch):
    raw = _raw(library)
    raw['steps'].append(_sweep_step(raw))
    call = sweep_module.PointRunner.__call__

    def fails_at_two(self, names, values):
        if values[0] == 2.0:
            raise RuntimeError('the solver gave up')
        return call(self, names, values)

    monkeypatch.setattr(sweep_module.PointRunner, '__call__', fails_at_two)
    out = str(tmp_path / 'out')
    run_calibration_workflow(_write(library, raw), output_dir=out)
    step = _read_json(os.path.join(out, 'scan', 'step_result.json'))
    assert step['n_failed'] == 1
    with np.load(step['sweep_results']) as results:
        assert results['failed'].tolist() == [False, True, False]
        assert np.isnan(results['trace_0'][1]).all()
        assert np.isfinite(results['trace_0'][[0, 2]]).all()
    assert _read_json(step['sweep_manifest'])['failures'] == [
        {'point': 1, 'values': [2.0], 'error': 'the solver gave up'}]


@pytest.mark.integration
def test_every_point_failing_is_an_error(library, tmp_path, monkeypatch):
    raw = _raw(library)
    raw['steps'] = [_sweep_step(raw)]

    def fails(self, names, values):
        raise RuntimeError('misconfigured')

    monkeypatch.setattr(sweep_module.PointRunner, '__call__', fails)
    with pytest.raises(WorkflowError, match='every one of the 3 points failed.*misconfigured'):
        run_calibration_workflow(_write(library, raw), output_dir=str(tmp_path / 'out'))


def _obs_with_features(path):
    with open(path, 'w') as f:
        json.dump({'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]]},
                   'data_items': [],
                   'prediction_items': [
                       {'data_item_name': 'z_max', 'operands': ['mod__C/z'], 'unit': 'dimensionless',
                        'operation': 'max', 'data_type': 'constant'}]}, f)
    return path


@pytest.mark.integration
@pytest.mark.parametrize('design, sweep_parameters, figures, expected', [
    ('one_at_a_time', {'mod__C/q': {'values': [1.0, 2.0]}, 'mod__C/k': {'values': [3.0]}},
     2, [2 + 1, 2 + 4, 2 + 3]),
    ('grid', {'mod__C/q': {'values': [1.0, 2.0]}, 'mod__C/k': {'values': [1.0, 3.0]}},
     1, [2 + 1, 2 + 3, 2 + 4, 2 + 12]),
])
def test_an_obs_datas_features_are_drawn_against_the_swept_parameters(
        library, tmp_path, design, sweep_parameters, figures, expected):
    raw = _raw(library)
    obs = _obs_with_features(str(tmp_path / 'features_obs.json'))
    step = _sweep_step(raw, design=design, outputs=[], parameters=sweep_parameters)
    raw['steps'] = [dict(step, obs_data=obs, settings={})]
    out = str(tmp_path / 'out')
    run_calibration_workflow(_write(library, raw), output_dir=out)
    result = _read_json(os.path.join(out, 'scan', 'step_result.json'))
    assert result['features'] == ['z_max']
    with np.load(result['sweep_results']) as results:
        np.testing.assert_allclose(results['feature_values'].ravel(), expected, rtol=1e-6)
    assert len(result['plots']) == figures
    assert all(os.path.getsize(path) > 0 for path in result['plots'])


@pytest.mark.integration
def test_the_methods_describe_set_and_sweep_steps(library, tmp_path):
    raw = _raw(library)
    raw['steps'] = [_set_step(), dict(raw['steps'][1], fixed_from=['hand']), _sweep_step(raw)]
    path, out = _write(library, raw), str(tmp_path / 'out')
    run_calibration_workflow(path, output_dir=out)
    latex = workflow_methods_latex(path, run_dir=out)
    assert 'This step sets values by hand' in latex
    assert 'Set by hand in step' in latex
    assert 'set by hand in hand' in latex
    assert 'Its values were passed to no other step' in latex
    assert 'Values set by hand were merged with them' in latex


@pytest.mark.integration
def test_a_packed_workflow_with_both_kinds_runs(library, tmp_path):
    raw = _raw(library)
    obs = _obs_with_features(str(tmp_path / 'features_obs.json'))
    raw['steps'] = [_set_step(), dict(raw['steps'][1], fixed_from=['hand']),
                    dict(_sweep_step(raw), obs_data=obs)]
    packed = pack_workflow(_write(library, raw), str(tmp_path / 'pack'))
    shutil.rmtree(library['modules'])
    result = run_calibration_workflow(packed['workflow'], output_dir=str(tmp_path / 'out'))
    assert _merged(result)['A/p']['value'] == 1.5
    assert load_workflow_run(str(tmp_path / 'out'))['steps']['scan']['n_points'] == 3


_MPI_SCRIPT = """
import sys
from libcuflynx.calibration_workflow import run_calibration_workflow
run_calibration_workflow(sys.argv[1], output_dir=sys.argv[2])
"""


@pytest.mark.integration
@pytest.mark.skipif(not mpi_utils.mpi_available() or shutil.which('mpiexec') is None,
                    reason='needs mpi4py and mpiexec')
def test_a_sweep_spreads_over_ranks_and_keeps_its_order(library, tmp_path):
    raw = _raw(library)
    raw['steps'] = [_sweep_step(raw, parameters={
        'mod__C/q': {'min': 1.0, 'max': 3.0, 'points': 5}})]
    path, out = _write(library, raw), str(tmp_path / 'out')
    done = subprocess.run(['mpiexec', '-n', '2', sys.executable, '-c', _MPI_SCRIPT, path, out],
                          capture_output=True, text=True, timeout=300, env=dict(os.environ))
    assert done.returncode == 0, done.stdout + done.stderr
    step = _read_json(os.path.join(out, 'scan', 'step_result.json'))
    with np.load(step['sweep_results']) as results:
        q = results['points'].ravel()
        np.testing.assert_allclose(results['trace_0'][:, 0, -1], 2.0 + q ** 2, rtol=1e-6)
    assert q.tolist() == [1.0, 1.5, 2.0, 2.5, 3.0]


# --------------------------------------------------------------------------------------------
# figure scripts outside the workflow
# --------------------------------------------------------------------------------------------

def _external(**entry):
    return dict({'id': 'paper_plots', 'command': 'python tools/paper_plots.py {run}',
                 'from': 'target', 'writes': ['plots/fig1.png']}, **entry)


@pytest.mark.unit
@pytest.mark.parametrize('entry, message', [
    (_external(id='rest'), 'has the id of a step'),
    (_external(**{'from': 'nowhere'}), 'neither a step nor "target"'),
    (_external(writes=[]), 'writes'),
    (_external(command=''), 'command'),
])
def test_invalid_external_figures_say_why(library, entry, message):
    raw = _raw(library)
    raw['external_figures'] = [entry]
    with pytest.raises(WorkflowError, match=message):
        _parse(library, raw)


@pytest.mark.unit
def test_two_external_figures_need_their_own_ids(library):
    raw = _raw(library)
    raw['external_figures'] = [_external(), _external()]
    with pytest.raises(WorkflowError, match='another external figure'):
        _parse(library, raw)


@pytest.mark.unit
def test_an_external_path_is_absolute_or_in_the_output_directory(tmp_path):
    from libcuflynx.calibration_workflow.spec import external_path
    assert external_path('plots/a.png', '/runs/w') == '/runs/w/plots/a.png'
    assert external_path(str(tmp_path / 'a.png'), '/runs/w') == str(tmp_path / 'a.png')
    assert external_path('~/a.png', '/runs/w') == os.path.expanduser('~/a.png')


@pytest.mark.integration
def test_a_declared_figure_script_is_current_then_stale_then_missing(library, tmp_path):
    """Declared after the run, the script leaves the run current; touching a file it reads makes
    it stale, and deleting a file it writes leaves it not run."""
    from libcuflynx.schemas import PROVENANCE_GRAPH_SCHEMA
    from libcuflynx.schemas.validation import validate
    out = str(tmp_path / 'out')
    path = _write(library, _raw(library))
    run_calibration_workflow(path, output_dir=out)

    data = os.path.join(out, 'plots', 'data.csv')
    figure = os.path.join(out, 'plots', 'Fig1', 'a.png')
    elsewhere = tmp_path / 'paper' / 'Fig2' / 'a.png'
    for file in (data, figure, str(elsewhere)):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            f.write('x')
    now = os.path.getmtime(os.path.join(out, 'workflow_result.json'))
    os.utime(data, (now + 10, now + 10))
    os.utime(figure, (now + 20, now + 20))
    os.utime(elsewhere, (now + 20, now + 20))
    raw = _raw(library)
    raw['external_figures'] = [_external(reads=['plots/data.csv'],
                                         writes=['plots/Fig1/a.png', str(elsewhere)],
                                         description='the paper figures')]
    _write(library, raw)

    graph = provenance_graph(out)
    validate(PROVENANCE_GRAPH_SCHEMA, graph, source='graph', what='provenance graph')
    assert graph['workflow']['file_status'] == 'current' and not graph['workflow']['stale']
    script = next(n for n in graph['nodes'] if n['id'] == 'paper_plots')
    assert (script['kind'], script['status'], script['stale']) == ('external', 'done', False)
    assert script['lines'] == ['`python tools/paper_plots.py {run}`', 'the paper figures']
    assert [(e['key'], e['status']) for e in script['inputs']] == [
        ('reads', 'current'), ('writes', 'current'), ('writes', 'current')]
    edges = {(e['from'], e['to'], e['style']) for e in graph['edges']}
    assert ('__target__', 'paper_plots', 'reads') in edges
    written = [e['to'] for e in graph['edges'] if e['from'] == 'paper_plots']
    assert len(written) == len(set(written)) == 2      # both files are a.png
    assert 'OUTSIDE THE WORKFLOW' in provenance_svg(graph)

    os.utime(data, (now + 30, now + 30))
    script = next(n for n in provenance_graph(out)['nodes'] if n['id'] == 'paper_plots')
    assert script['stale'] and script['inputs'][0]['status'] == 'stale'

    os.remove(figure)
    graph = provenance_graph(out)
    script = next(n for n in graph['nodes'] if n['id'] == 'paper_plots')
    assert (script['status'], script['stale']) == ('not_run', False)
    assert script['inputs'][1]['status'] == 'missing'
    assert '1 of 2 files not written' in script['lines']
    assert 'FILES MISSING' in provenance_svg(graph)


_DECAY_CELLML = '''<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" name="decay">
  <component name="mod">
    <variable name="t" units="dimensionless" interface="public_and_private"/>
    <variable name="x" units="dimensionless" initial_value="1" interface="public_and_private"/>
    <variable name="k" units="dimensionless" initial_value="0.001" interface="public_and_private"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML"
          xmlns:cellml="http://www.cellml.org/cellml/2.0#">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><times/><apply><minus/><ci>k</ci></apply><ci>x</ci></apply></apply>
    </math>
  </component>
</model>
'''


@pytest.mark.integration
def test_a_sweep_point_runs_as_fast_as_a_calibration_evaluation(tmp_path):
    '''With no MaximumStep given, a point runs uncapped as the calibration does; a 1e-3 cap
    over this 5000 s pre_time would cost 5e6 CVODE steps.'''
    import time
    from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
    from libcuflynx.solver_wrappers import get_simulation_helper

    model = tmp_path / 'decay.cellml'
    model.write_text(_DECAY_CELLML)
    obs = tmp_path / 'obs.json'
    obs.write_text(json.dumps({
        'protocol_info': {'pre_times': [5000.0], 'sim_times': [[10.0]]}, 'data_items': [],
        'prediction_items': [{'data_item_name': 'x_max', 'operands': ['mod/x'],
                              'unit': 'dimensionless', 'operation': 'max',
                              'data_type': 'constant'}]}))
    solver_info = {'rtol': 1e-8, 'atol': 1e-10}
    runner = sweep_module.PointRunner(str(model), {'dt': 1.0, 'solver_info': solver_info},
                                      str(obs), [])
    assert runner.runner.sim_helper.simulation._dtmax == 0

    calibration = get_simulation_helper(
        solver='CVODE_myokit', model_type='cellml', model_path=str(model), dt=1.0, sim_time=1.0,
        solver_info=dict(solver_info, solver='CVODE_myokit'), pre_time=0.0)
    start = time.perf_counter()
    ProtocolExecutor(calibration).run_protocol(runner.protocol)
    one_evaluation = time.perf_counter() - start

    start = time.perf_counter()
    _, _, features = runner(['mod/k'], [0.001])
    one_point = time.perf_counter() - start
    np.testing.assert_allclose(features['x_max'], np.exp(-5.0), rtol=1e-5)
    assert one_point < 10 * one_evaluation + 0.5, (one_point, one_evaluation)
