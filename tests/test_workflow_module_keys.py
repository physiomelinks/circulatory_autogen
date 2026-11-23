"""A calibration workflow names a step's place ``module`` / ``module_map`` and each value record's
``module``; the older ``vessel`` forms are refused, naming ``cuflynx-upgrade workflow``, which
converts workflow files and the step results of earlier runs in place."""

import json
import os

import pytest

from libcuflynx.calibration_workflow import (WorkflowError, load_workflow_run, plan_workflow,
                                             run_calibration_workflow, workflow_status)
from libcuflynx.calibration_workflow.spec import parse_workflow
from libcuflynx.external_testing.workflow_library import build_workflow_library
from libcuflynx.upgrade import main as upgrade_main

RECORD_KEYS = ('calibrated', 'fixed', 'set')


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def _placed_workflow(library, key='module'):
    '''The chain array workflow with its first step the library module lin_a, placed at A.'''
    raw = _read_json(library['chain_array'])
    directory = library['lin_a/fit_a']
    raw['steps'][0] = {'id': 'a-alone',
                       'model': {'module_type': 'lin_a', 'version': 'v1',
                                 'parameterisation': 'fit_a'},
                       'obs_data': os.path.join(directory, 'fit_a_obs_data.json'),
                       'params_for_id': os.path.join(directory, 'fit_a_params_for_id.json'),
                       key: 'A'}
    return raw


@pytest.mark.unit
@pytest.mark.parametrize('old, value', [('vessel', 'A'), ('vessel_map', {'A': 'A'})])
def test_a_step_naming_vessel_is_refused_naming_the_upgrade(library, old, value):
    raw = _read_json(library['chain_array'])
    raw['steps'][0][old] = value
    path = library['chain_array']
    with pytest.raises(WorkflowError, match=f'"{old}" \\(now "module(_map)?"\\).*'
                                            f'cuflynx-upgrade workflow {path}'):
        parse_workflow(raw, path=path)


@pytest.mark.unit
def test_the_upgrade_renames_a_workflow_files_keys_and_keeps_its_formatting(library, capsys):
    path = os.path.join(os.path.dirname(library['chain_array']), 'placed_workflow.json')
    _write_json(path, _placed_workflow(library, key='vessel'))
    with open(path) as f:
        before = f.read()
    assert upgrade_main(['workflow', '--dry-run', path]) == 0
    with open(path) as f:
        assert f.read() == before
    assert upgrade_main(['workflow', path]) == 0
    assert 'step "a-alone": vessel -> module' in capsys.readouterr().out
    with open(path) as f:
        assert f.read() == before.replace('"vessel":', '"module":')
    plan = plan_workflow(parse_workflow(_read_json(path), path=path))
    assert plan['steps'][0]['place'] == 'A'
    # a second run finds nothing to change
    assert upgrade_main(['workflow', path]) == 0
    assert '0 of 1 file(s) changed' in capsys.readouterr().out


@pytest.mark.unit
def test_the_upgrade_refuses_a_step_with_both_keys(library):
    path = os.path.join(os.path.dirname(library['chain_array']), 'both.json')
    raw = _placed_workflow(library)
    raw['steps'][0]['vessel'] = 'A'
    _write_json(path, raw)
    with pytest.raises(SystemExit, match='both "vessel" and "module"'):
        upgrade_main(['workflow', path])


def _as_older_run(out):
    '''Rewrite the run in ``out`` as an earlier libcuflynx wrote it: "vessel" for "module".'''
    snapshot = os.path.join(out, 'calibration_workflow.json')
    raw = _read_json(snapshot)
    raw['steps'] = [{('vessel' if k == 'module' else k): v for k, v in s.items()}
                    for s in raw['steps']]
    _write_json(snapshot, raw)
    renamed = 0
    for step in raw['steps']:
        path = os.path.join(out, step['id'], 'step_result.json')
        result = _read_json(path)
        for key in RECORD_KEYS:
            if key not in result:
                continue
            records = result[key]
            result[key] = [{('vessel' if k == 'module' else k): v for k, v in r.items()}
                           for r in records]
            renamed += len(records)
        _write_json(path, result)
    return renamed


@pytest.mark.integration
def test_an_older_run_is_refused_and_the_upgrade_converts_it_in_place(library, tmp_path):
    workflow = os.path.join(os.path.dirname(library['chain_array']), 'placed_workflow.json')
    _write_json(workflow, _placed_workflow(library))
    out = str(tmp_path / 'run')
    run_calibration_workflow(workflow, output_dir=out)
    current = load_workflow_run(out)
    records = current['steps']['a-alone']['calibrated']
    assert [(r['module'], r['param']) for r in records] == [('mod', 'p')]
    assert [(r['module'], r['model_name']) for r in current['steps']['chain']['fixed']] == \
        [('A', 'p_A')]
    assert _as_older_run(out) > 0

    with pytest.raises(WorkflowError, match=f'"vessel" \\(now "module"\\).*'
                                            f'cuflynx-upgrade workflow'):
        load_workflow_run(out)
    os.remove(os.path.join(out, 'calibration_workflow.json'))
    _write_json(os.path.join(out, 'calibration_workflow.json'), _placed_workflow(library))
    with pytest.raises(WorkflowError, match=f'names the module of each value "vessel" '
                                            f'\\(now "module"\\).*cuflynx-upgrade workflow {out}'):
        load_workflow_run(out)
    with pytest.raises(WorkflowError, match='cuflynx-upgrade workflow'):
        workflow_status(workflow, out)

    assert upgrade_main(['workflow', out]) == 0
    assert load_workflow_run(out)['steps'] == current['steps']
    # the converted run is current: nothing would run again
    status = workflow_status(workflow, out)
    assert status['complete']
    assert all(s['status'] == 'done' and not s['stale'] for s in status['steps'].values())
