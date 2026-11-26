"""protocol_info ``params_to_change_factors``: factors on the values a workflow step's model gives
its parameters, resolved into absolute ``params_to_change`` once per step.

The fixture library is libcuflynx.external_testing.workflow_library: in ``chain``,
z = 2 p_A^2 + k q^2 with k = 1, fit_a gives p_A = 2 (2 p_A^2 = Y_A) and rest fits
q = sqrt(Z_C - Y_A) = 3.
"""
import json
import os

import numpy as np
import pytest

from libcuflynx.calibration_workflow import run_calibration_workflow, workflow_model
from libcuflynx.external_testing.workflow_library import Y_A, Z_C, build_workflow_library
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.reporting.provenance import provenance_graph

Q = np.sqrt(Z_C - Y_A)


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
    return path


def _protocol_obs(path, **protocol):
    '''An obs_data of three one-second experiments and no data items.'''
    return _write_json(path, {'protocol_info': dict(
        {'pre_times': [0.0] * 3, 'sim_times': [[1.0]] * 3}, **protocol), 'data_items': []})


def _run(library, raw, out):
    _write_json(library['chain_rest'], raw)
    run_calibration_workflow(library['chain_rest'], output_dir=out)
    return _read_json(os.path.join(out, 'scan', 'step_result.json'))


def _z(step):
    with np.load(step['sweep_results']) as results:
        return [float(results[f'trace_{e}'][0, 0, -1]) for e in range(3)]


def _on_target_scan(obs):
    '''One run of the protocol on the merged target: an on_target sweep of nothing.'''
    return {'id': 'scan', 'kind': 'sweep', 'on_target': True, 'obs_data': obs,
            'settings': {'dt': 0.1}, 'sweep': {'outputs': ['mod__C/z'], 'parameters': {}}}


@pytest.mark.integration
def test_factors_on_the_target_run_as_the_same_values_written_out(library, tmp_path):
    """x1, x2, x3 of the q that rest calibrated give the z of q, 2q and 3q written as absolute
    values, and the step records the q they multiplied and that it came from rest."""
    raw = _read_json(library['chain_rest'])
    factors = _protocol_obs(str(tmp_path / 'factors.json'),
                            params_to_change_factors={'mod__C/q': [[1.0], [2.0], [3.0]]})
    step = _run(library, dict(raw, steps=raw['steps'] + [_on_target_scan(factors)]),
                str(tmp_path / 'out_factors'))
    [record] = step['protocol_factors']
    assert (record['parameter'], record['model_name'], record['from_step']) == (
        'mod__C/q', 'q_mod__C', 'rest')
    assert record['base'] == pytest.approx(Q, rel=1e-4)
    z = _z(step)
    np.testing.assert_allclose(z, [Y_A + (f * record['base']) ** 2 for f in (1, 2, 3)],
                               rtol=1e-6)

    absolute = _protocol_obs(str(tmp_path / 'absolute.json'), params_to_change={
        'mod__C/q': [[f * record['base']] for f in (1.0, 2.0, 3.0)]})
    written_out = _run(library, dict(raw, steps=raw['steps'] + [_on_target_scan(absolute)]),
                       str(tmp_path / 'out_absolute'))
    assert 'protocol_factors' not in written_out
    np.testing.assert_allclose(z, _z(written_out), rtol=1e-12)

    # the step read a resolved copy; its own obs_data is unchanged
    assert 'params_to_change_factors' in _read_json(factors)['protocol_info']
    scan = next(n for n in provenance_graph(str(tmp_path / 'out_factors'))['nodes']
                if n['id'] == 'scan')
    assert any(line.startswith('`mod__C/q` in the protocol: factors ×') and
               line.endswith('from `rest`') for line in scan['lines'])


@pytest.mark.integration
def test_factors_on_a_simulate_step_multiply_a_fixed_value(library, tmp_path):
    """A sweep of nothing on the chain, fixed from fit_a: x1, x0.5, x0 of p_A = 2, with q at
    its default 1."""
    raw = _read_json(library['chain_rest'])
    obs = _protocol_obs(str(tmp_path / 'obs.json'),
                        params_to_change_factors={'mod__A/p': [[1.0], [0.5], [0.0]]})
    scan = {'id': 'scan', 'kind': 'sweep', 'model': raw['steps'][1]['model'],
            'fixed_from': ['fit_a'], 'obs_data': obs, 'settings': {'dt': 0.1},
            'sweep': {'outputs': ['mod__C/z'], 'parameters': {}}}
    raw['steps'] = [raw['steps'][0], scan]
    out = str(tmp_path / 'out')
    step = _run(library, raw, out)
    [record] = step['protocol_factors']
    assert record['from_step'] == 'fit_a'
    p = record['base']
    assert p == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)
    np.testing.assert_allclose(_z(step), [2 * (f * p) ** 2 + 1.0 for f in (1.0, 0.5, 0.0)],
                               rtol=1e-6)

    # the view CUFLynx opens hands over the obs_data resolved against its own model
    view = workflow_model(library['chain_rest'], 'scan', output_dir=out,
                          work_dir=str(tmp_path / 'views'), validate=False)
    info = _read_json(view['obs_data_path'])['protocol_info']
    assert 'params_to_change_factors' not in info
    np.testing.assert_allclose([row[0] for row in info['params_to_change']['mod__A/p']],
                               [p, 0.5 * p, 0.0], rtol=1e-12)


def _refused(library, tmp_path, edit, message):
    raw = _read_json(library['chain_rest'])
    edit(raw, tmp_path)
    _write_json(library['chain_rest'], raw)
    with pytest.raises(Exception, match=message):
        run_calibration_workflow(library['chain_rest'], output_dir=str(tmp_path / 'out'))


def _scan_with(protocol, **sweep):
    def edit(raw, tmp_path):
        obs = _protocol_obs(str(tmp_path / 'obs.json'), **protocol)
        raw['steps'].append(dict(_on_target_scan(obs), sweep=dict(
            {'outputs': ['mod__C/z'], 'parameters': {}}, **sweep)))
    return edit


@pytest.mark.integration
@pytest.mark.parametrize('edit, message', [
    (_scan_with({'params_to_change': {'mod__C/q': [[1.0]] * 3},
                 'params_to_change_factors': {'mod__C/q': [[1.0]] * 3}}),
     'both params_to_change and params_to_change_factors'),
    (_scan_with({'params_to_change_factors': {'mod__C/q': [[1.0]] * 3}},
                parameters={'mod__C/q': {'values': [1.0, 2.0]}}),
     'which the step sweeps'),
    (_scan_with({'params_to_change_factors': {'mod__C/nope': [[1.0]] * 3}}),
     'no parameter "nope_mod__C"'),
    (_scan_with({'params_to_change_factors': {'mod__C/q': [[1.0]] * 2}}),
     'params_to_change_factors shape mismatch'),
])
def test_factors_a_step_cannot_resolve_are_refused(library, tmp_path, edit, message):
    _refused(library, tmp_path, edit, message)


@pytest.mark.integration
def test_factors_on_a_parameter_the_step_calibrates_are_refused(library, tmp_path):
    raw = _read_json(library['chain_rest'])
    rest = raw['steps'][1]
    obs_path = os.path.join(os.path.dirname(library['chain_rest']), rest['obs_data'])
    obs = _read_json(obs_path)
    obs['protocol_info']['params_to_change_factors'] = {'mod__C/q': [[2.0]]}
    _write_json(obs_path, obs)
    with pytest.raises(Exception, match='"mod__C/q", which the step calibrates'):
        run_calibration_workflow(library['chain_rest'], output_dir=str(tmp_path / 'out'))


@pytest.mark.unit
def test_factors_outside_a_workflow_step_are_refused():
    obs = {'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]],
                             'params_to_change_factors': {'C/q': [[2.0]]}},
           'data_items': []}
    with pytest.raises(ValueError, match='resolved by the workflow'):
        ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=obs)
