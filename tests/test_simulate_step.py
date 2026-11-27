"""The ``simulate`` kind of a calibration workflow: one run of a model through its protocol.

The fixture library is libcuflynx.external_testing.workflow_library: in ``chain``, C reads A's
output, z = 2 p_A^2 + k q^2 (k = 1, q = 1 unless a step sets it).
"""
import json
import os
import shutil

import numpy as np
import pytest

from libcuflynx.calibration_workflow import (WorkflowError, load_workflow_run, plan_workflow,
                                             run_calibration_workflow)
from libcuflynx.calibration_workflow.pack import pack_workflow
from libcuflynx.calibration_workflow.spec import parse_workflow
from libcuflynx.external_testing.workflow_library import Y_A, Z_C, build_workflow_library
from libcuflynx.reporting.methods import workflow_methods_latex
from libcuflynx.reporting.provenance import provenance_graph


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _raw(library):
    return _read_json(library['chain_rest'])


def _write(library, raw):
    with open(library['chain_rest'], 'w') as f:
        json.dump(raw, f, indent=1)
    return library['chain_rest']


def _parse(library, raw):
    return parse_workflow(raw, path=library['chain_rest'])


def _simulate_step(raw, **keys):
    return dict({'id': 'run', 'kind': 'simulate', 'model': raw['steps'][1]['model'],
                 'settings': {'sim_time': 1.0}, 'simulate': {'outputs': ['mod__C/z']}}, **keys)


def _obs_with_features(path):
    with open(path, 'w') as f:
        json.dump({'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]]},
                   'data_items': [],
                   'prediction_items': [
                       {'data_item_name': 'z_max', 'operands': ['mod__C/z'],
                        'unit': 'dimensionless', 'operation': 'max', 'data_type': 'constant'}]}, f)
    return path


@pytest.mark.unit
def test_a_simulate_step_is_a_sweep_of_no_parameters(library):
    raw = _raw(library)
    raw['steps'].append(_simulate_step(raw, fixed_from=['fit_a']))
    step = _parse(library, raw).step('run')
    assert step.kind == 'simulate'
    assert step.sweep == {'design': 'grid', 'outputs': ['mod__C/z'], 'parameters': {}, 'zip': []}
    assert step.depends_on == ['fit_a']
    plan = {s['id']: s for s in plan_workflow(_write(library, raw))['steps']}
    assert plan['run']['sweep']['parameters'] == {}


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda s: s.update(sweep={'parameters': {}}), '"sweep" .* is for a sweep step'),
    (lambda s: s.update(params_for_id='p.json'), 'this combination of keys is not allowed'),
    (lambda s: s.pop('simulate'), 'records nothing'),
    (lambda s: s.update(settings={}), r"simulate step \"run\" needs \['sim_time'\]"),
    (lambda s: s.update(model={'select': ['A']}), 'not as a selection'),
    (lambda s: s.update(on_target=True), r"names no \['model'\]"),
    (lambda s: s['simulate'].update(parameters={}), "'parameters' was unexpected"),
    (lambda s: s.pop('model'), 'model'),
])
def test_invalid_simulate_steps_say_why(library, edit, message):
    raw = _raw(library)
    step = _simulate_step(raw)
    edit(step)
    raw['steps'].append(step)
    with pytest.raises(WorkflowError, match=message):
        _parse(library, raw)


@pytest.mark.unit
def test_simulate_options_are_for_a_simulate_step(library):
    raw = _raw(library)
    raw['steps'][1]['simulate'] = {'outputs': ['mod__C/z']}
    with pytest.raises(WorkflowError, match='"simulate" .* is for a simulate step'):
        _parse(library, raw)


@pytest.mark.unit
def test_a_simulate_step_cannot_pass_values_on(library):
    raw = _raw(library)
    raw['steps'].append(_simulate_step(raw))
    raw['steps'][1]['fixed_from'] = ['run']
    with pytest.raises(WorkflowError, match='a simulate step; it takes a calibration or set'):
        _parse(library, raw)


@pytest.mark.integration
def test_a_simulate_step_writes_what_a_sweep_of_nothing_writes(library, tmp_path):
    raw = _raw(library)
    raw['steps'].append(_simulate_step(raw, fixed_from=['fit_a']))
    out = str(tmp_path / 'out')
    result = run_calibration_workflow(_write(library, raw), output_dir=out)
    assert sorted(r['parameterisation_name'] for r in result['merged_parameters']) == ['A/p',
                                                                                       'C/q']
    step = _read_json(os.path.join(out, 'run', 'step_result.json'))
    assert (step['kind'], step['n_points'], step['n_failed'], step['parameters']) == (
        'simulate', 1, 0, [])
    assert [f['from_step'] for f in step['fixed']] == ['fit_a']
    with np.load(step['sweep_results']) as results:
        assert results['points'].shape == (1, 0)
        assert results['outputs'].tolist() == ['mod__C/z']
        np.testing.assert_allclose(results['trace_0'][0, 0, -1], Y_A + 1.0, rtol=1e-6)
    graph = provenance_graph(out)
    node = next(n for n in graph['nodes'] if n['id'] == 'run')
    assert node['kind'] == 'analysis' and node['mark'] is None
    assert 'one run of the protocol' in node['lines']
    assert ('run', 'run/files') in {(e['from'], e['to']) for e in graph['edges']}
    latex = workflow_methods_latex(library['chain_rest'], run_dir=out)
    assert 'This step simulates' in latex
    assert 'was run through the protocol once' in latex


@pytest.mark.integration
def test_a_simulate_step_on_the_target_runs_the_merged_model(library, tmp_path):
    """The target has fit_a's p and rest's q, so z is the value rest was fitted to."""
    raw = _raw(library)
    obs = _obs_with_features(str(tmp_path / 'features_obs.json'))
    raw['steps'].append({'id': 'run', 'kind': 'simulate', 'on_target': True, 'obs_data': obs,
                         'settings': {'dt': 0.01}})
    out = str(tmp_path / 'out')
    run_calibration_workflow(_write(library, raw), output_dir=out)
    step = _read_json(os.path.join(out, 'run', 'step_result.json'))
    assert step['features'] == ['z_max'] and step['outputs'] == []
    assert sorted({f['from_step'] for f in step['fixed']}) == ['fit_a', 'rest']
    with np.load(step['sweep_results']) as results:
        np.testing.assert_allclose(results['feature_values'].ravel(), [Z_C], rtol=1e-3)
    node = next(n for n in provenance_graph(out)['nodes'] if n['id'] == 'run')
    assert node['lines'][:2] == ['the merged target', 'one run of the protocol']


@pytest.mark.integration
def test_a_packed_workflow_with_a_simulate_step_runs(library, tmp_path):
    raw = _raw(library)
    obs = _obs_with_features(str(tmp_path / 'features_obs.json'))
    raw['steps'].append(_simulate_step(raw, obs_data=obs, fixed_from=['fit_a']))
    packed = pack_workflow(_write(library, raw), str(tmp_path / 'pack'))
    shutil.rmtree(library['modules'])
    run_calibration_workflow(packed['workflow'], output_dir=str(tmp_path / 'out'))
    assert load_workflow_run(str(tmp_path / 'out'))['steps']['run']['features'] == ['z_max']
