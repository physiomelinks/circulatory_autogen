"""Modify steps of a calibration workflow (``"kind": "modify"``): a model's module array changed,
every value of its parameters file kept.

The fixture library's ``a_only`` array (one record, ``A``, lin_a) stands for a model calibrated
earlier: its parameters file holds A/p = 2, which gives y_A = 2 p^2 = Y_A. The modify step
``add_c`` adds ``C`` (lin_c, z = 2 p_A^2 + q^2) fed by ``A``; a calibration of C/q on that model
finds q = sqrt(Z_C - Y_A) = 3 only if A/p kept its value (the library default, 1, gives
sqrt(15)).
"""
import copy
import json
import os

import numpy as np
import pytest

from _parameters import read_parameters, write_parameters
from libcuflynx.calibration_workflow import (WorkflowError, load_workflow, plan_workflow,
                                             resolve_workflow, run_calibration_workflow,
                                             workflow_status)
from libcuflynx.calibration_workflow.spec import ModifiedArray, parse_workflow
from libcuflynx.external_testing.workflow_library import Y_A, Z_C, build_workflow_library
from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA
from libcuflynx.schemas.validation import SchemaValidationError, validate

P_A = float(np.sqrt(Y_A / 2))      # 2: the value the base model was calibrated to
ADD_C = {'add': {'name': 'C', 'module_type': 'lin_c', 'module_subtype': 'v1',
                 'parameterisation': 'default', 'inp_instances': ['A']},
         'reason': 'C reads the output of A'}


@pytest.fixture
def library(tmp_path):
    paths = build_workflow_library(str(tmp_path / 'library'))
    write_parameters(os.path.join(paths['arrays'], 'a_only_parameters.json'),
                     f'variable_name,units,value,data_reference\n'
                     f'A/p,dimensionless,{P_A!r},calibrated earlier\n')
    return paths


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _raw(library, changes=(ADD_C,), target=None, steps=None):
    raw = _read_json(library['chain_array'])
    raw['workflow_name'] = 'modified'
    raw['target'] = target or {'from': 'add_c'}
    raw['steps'] = [{'id': 'add_c', 'kind': 'modify', 'model': 'a_only_module_array.json',
                     'changes': copy.deepcopy(list(changes))}]
    raw['steps'] += steps if steps is not None else [
        {'id': 'fit_c', 'model': {'from': 'add_c'}, 'obs_data': 'data/chain_obs.json',
         'params_for_id': 'data/chain_params.json'}]
    return raw


def _parse(library, raw):
    return parse_workflow(raw, path=library['chain_array'])


def _records(model):
    return {r['name']: (r['module_type'], r['inp_instances'], r['out_instances'])
            for r in model.records}


# --------------------------------------------------------------------------------------------
# the changes
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_an_added_record_is_linked_on_both_sides_and_later_steps_take_the_model(library):
    workflow = _parse(library, _raw(library))
    model = workflow.step('add_c').model
    assert isinstance(model, ModifiedArray)
    assert _records(model) == {'A': ('lin_a', [], ['C']), 'C': ('lin_c', ['A'], [])}
    # the calibration and the target run on the step's model, after it
    assert workflow.step('fit_c').model is model and workflow.target is model
    assert workflow.step('fit_c').depends_on == ['add_c']
    plan = {s['id']: s for s in plan_workflow(workflow)['steps']}
    assert plan['add_c']['model'] == {'module_array': os.path.join(library['arrays'],
                                                                   'a_only_module_array.json'),
                                      'parameters': os.path.join(library['arrays'],
                                                                 'a_only_parameters.json')}
    assert plan['add_c']['records'] == ['A', 'C']
    assert plan['fit_c']['model'] == {'from': 'add_c'}
    assert plan['fit_c']['calibrates'][0]['target_parameterisation_name'] == 'C/q'


@pytest.mark.unit
def test_changes_remove_change_connect_and_disconnect_records(library):
    changes = [ADD_C,
               {'add': {'name': 'B', 'module_type': 'lin_b', 'module_subtype': 'v1'},
                'reason': 'a second source'},
               {'connect': {'from': 'B', 'to': 'C'}, 'reason': 'C also reads B'},
               {'disconnect': {'from': 'A', 'to': 'C'}, 'reason': 'C reads B instead'},
               {'change': {'name': 'B', 'parameterisation': 'default'}, 'reason': 'its values'},
               {'remove': 'A', 'reason': 'A is replaced by B'}]
    model = _parse(library, _raw(library, changes)).step('add_c').model
    assert _records(model) == {'C': ('lin_c', ['B'], []), 'B': ('lin_b', [], ['C'])}
    assert model.removed == ('A',)
    # a removed record's values go with it
    assert 'A' not in model.parameters_document({'A': {'p': {}}, 'C': {'q': {}}})
    # the modules brought in, changed or removed, one per record, each with its reasons
    assert [(m['name'], m['action'], m['module_type'], m['reasons']) for m in model.modules] == [
        ('C', 'added', 'lin_c', ['C reads the output of A']),
        ('B', 'added', 'lin_b', ['a second source', 'its values']),
        ('A', 'removed', 'lin_a', ['A is replaced by B'])]


@pytest.mark.unit
def test_a_changed_module_says_what_it_was_and_where_phlynx_can_open_it(library):
    # the library keeps a version's PhLynx export with its results
    archive = os.path.join(library['modules'], 'lin_b', 'versions', 'v1', 'results',
                           'lin_b_v1_phlynx.omex')
    os.makedirs(os.path.dirname(archive))
    open(archive, 'wb').close()
    changes = [ADD_C, {'change': {'name': 'A', 'module_type': 'lin_b'}, 'reason': 'B for A'}]
    plan = plan_workflow(_parse(library, _raw(library, changes)))['steps'][0]
    added, changed = plan['modules']
    assert (added['name'], added['action'], added['version']) == ('C', 'added', 'v1')
    assert added['config_path'].endswith('lin_c_v1_modules_config.json')
    assert added['phlynx_archive'] is None and 'no PhLynx export' in added['phlynx_note']
    assert (changed['name'], changed['action'], changed['module_type']) == ('A', 'changed', 'lin_b')
    assert changed['previous'] == {'module_type': 'lin_a', 'version': 'v1',
                                   'parameterisation': 'default'}
    assert changed['phlynx_archive'] == archive and changed['phlynx_note'] is None


@pytest.mark.unit
def test_a_modify_step_may_change_the_model_of_another(library):
    second = {'id': 'drop_link', 'kind': 'modify', 'model': {'from': 'add_c'},
              'changes': [{'disconnect': {'from': 'A', 'to': 'C'}, 'reason': 'test'}]}
    raw = _raw(library, target={'from': 'drop_link'})
    raw['steps'].insert(0, second)      # listed first, run second
    workflow = _parse(library, raw)
    assert [s.id for s in workflow.topological_order()][:2] == ['add_c', 'drop_link']
    model = workflow.target
    assert model.step == 'drop_link' and model.base is workflow.step('add_c').model
    assert _records(model) == {'A': ('lin_a', [], []), 'C': ('lin_c', [], [])}
    # the files are still the first base's
    assert model.parameters == os.path.join(library['arrays'], 'a_only_parameters.json')


@pytest.mark.unit
@pytest.mark.parametrize('change, message', [
    ({'connect': {'from': 'A', 'to': 'D'}, 'reason': 'x'}, r'change 2: there is no record "D"'),
    ({'connect': {'from': 'A', 'to': 'C'}, 'reason': 'x'}, 'change 2: "A" already outputs to "C"'),
    ({'disconnect': {'from': 'C', 'to': 'A'}, 'reason': 'x'}, '"C" does not output to "A"'),
    ({'add': {'name': 'A', 'module_type': 'lin_a', 'module_subtype': 'v1'}, 'reason': 'x'},
     'there is already a record "A"'),
    ({'remove': 'D', 'reason': 'x'}, 'there is no record "D"'),
])
def test_a_change_that_cannot_be_made_says_why(library, change, message):
    with pytest.raises(WorkflowError, match=message):
        _parse(library, _raw(library, [ADD_C, change]))


@pytest.mark.unit
def test_a_model_not_in_the_library_is_refused_when_resolved(library):
    change = {'change': {'name': 'C', 'module_subtype': 'v9'}, 'reason': 'a version not there'}
    workflow = _parse(library, _raw(library, [ADD_C, change]))
    with pytest.raises(WorkflowError, match='record "C" .* lin_c version v9, which is not in '
                                            'the module library'):
        resolve_workflow(workflow)


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda raw: raw['steps'][1].update(model={'from': 'fit_c'}),
     '"fit_c" is a calibration step; "from" names a modify step'),
    (lambda raw: raw.update(target={'from': 'nowhere'}), '"nowhere" is not a step'),
    (lambda raw: raw['steps'][0].update(model={'module_type': 'lin_a', 'version': 'v1',
                                               'parameterisation': 'default'}),
     'a modify step changes a module array'),
    (lambda raw: raw['steps'][0].update(obs_data='data/chain_obs.json'),
     "runs nothing, so it has no \\['obs_data'\\]"),
    (lambda raw: raw['steps'][0].update(model='missing_module_array.json'),
     'missing_module_array.json does not exist'),
])
def test_invalid_modify_steps_say_why(library, edit, message):
    raw = _raw(library)
    edit(raw)
    with pytest.raises(WorkflowError, match=message):
        _parse(library, raw)


@pytest.mark.unit
@pytest.mark.parametrize('edit', [
    lambda step: step['changes'][0].pop('reason'),                       # every change says why
    lambda step: step['changes'][0].update(remove='A'),                  # one operation each
    lambda step: step.pop('changes'),
    lambda step: step['changes'].append({'change': {'name': 'A'}, 'reason': 'nothing'}),
    lambda step: step.update(kind='calibration', obs_data='o', params_for_id='p'),  # not modify
])
def test_the_schema_holds_a_modify_step_to_its_shape(library, edit):
    raw = _raw(library)
    validate(CALIBRATION_WORKFLOW_SCHEMA, raw, what='calibration workflow')
    edit(raw['steps'][0])
    with pytest.raises(SchemaValidationError):
        validate(CALIBRATION_WORKFLOW_SCHEMA, raw, what='calibration workflow')


# --------------------------------------------------------------------------------------------
# a run
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_modified_model_keeps_its_values_and_its_new_module_gets_the_library_s(library,
                                                                                tmp_path):
    path = os.path.join(library['arrays'], 'modified_workflow.json')
    with open(path, 'w') as f:
        json.dump(_raw(library), f)
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(path, output_dir=out, write_calibrated=True)
    assert result['complete']

    # the step writes the model it makes; A keeps the value it was calibrated to
    modify = _read_json(os.path.join(out, 'add_c', 'step_result.json'))
    assert modify['model']['module_array'].endswith('a_only_module_array.json')
    assert [c['reason'] for c in modify['changes']] == ['C reads the output of A']
    assert [r['name'] for r in _read_json(modify['module_array'])] == ['A', 'C']
    assert {r['variable_name']: float(r['value']) for r in read_parameters(
        modify['parameters'])} == {'A/p': pytest.approx(P_A)}

    # q = sqrt(Z_C - 2 p_A^2) = 3 needs p_A = 2 (with the default 1 it would be sqrt(15))
    merged = {r['parameterisation_name']: r['value'] for r in result['merged_parameters']}
    assert merged == {'C/q': pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)}
    fit = _read_json(os.path.join(out, 'fit_c', 'step_result.json'))
    assert fit['model'] == {'from': 'add_c'}
    assert fit['inputs']['modified_model']['step'] == 'add_c'

    # the calibrated parameters go beside the base's, with the module array they belong to
    written = {r['variable_name']: r for r in read_parameters(
        result['calibrated_parameters_file'])}
    assert result['calibrated_parameters_file'] == os.path.join(
        library['arrays'], 'add_c_calibrated_parameters.json')
    assert float(written['A/p']['value']) == pytest.approx(P_A)
    assert float(written['C/q']['value']) == pytest.approx(3.0, rel=1e-4)
    assert written['C/k']['value'] == '1.0'          # the library's default, kept
    assert [r['name'] for r in _read_json(os.path.join(library['arrays'],
                                                       'add_c_module_array.json'))] == ['A', 'C']

    # the provenance diagram shows the change, made by hand, and the model passed on
    graph = _read_json(os.path.join(out, 'provenance_graph.json'))
    nodes = {n['id']: n for n in graph['nodes']}
    assert nodes['add_c']['kind'] == 'values'
    assert any('C reads the output of A' in line for line in nodes['add_c']['lines'])
    # the block lists the module brought in, as the step's result and its view do
    assert [(m['name'], m['action'], m['module_type']) for m in nodes['add_c']['modules']] == [
        ('C', 'added', 'lin_c')]
    assert modify['modules'] == nodes['add_c']['modules']
    from libcuflynx.calibration_workflow import workflow_model
    view = workflow_model(path, 'add_c', output_dir=out, work_dir=str(tmp_path / 'view'))
    assert view['modules'] == modify['modules'] and os.path.isfile(view['model_path'])
    assert view['model']['module_array'].endswith('a_only_module_array.json')
    assert {'from': 'add_c', 'to': 'fit_c', 'label': 'model'}.items() <= next(
        e for e in graph['edges'] if (e['from'], e['to']) == ('add_c', 'fit_c')).items()

    # the methods text says what was changed, and why
    from libcuflynx.reporting.methods import workflow_methods_latex
    text = workflow_methods_latex(path, run_dir=out)
    assert 'This step changes the model' in text and 'C reads the output of A' in text
    assert 'This step calibrates the model made in step add' in text

    # a change to the changes makes the steps on the model stale
    raw = _raw(library)
    raw['steps'][0]['changes'][0]['add']['parameterisation'] = 'other'
    with open(path, 'w') as f:
        json.dump(raw, f)
    status = workflow_status(load_workflow(path), out)['steps']
    assert status['add_c']['stale'] and status['fit_c']['stale']


@pytest.mark.unit
def test_a_modified_target_is_not_written_over_another_module_array(library, tmp_path):
    path = os.path.join(library['arrays'], 'modified_workflow.json')
    raw = _raw(library)
    raw['steps'][0]['id'] = 'a_only'     # a_only_module_array.json (A alone) is already there
    raw['steps'][1]['model'] = raw['target'] = {'from': 'a_only'}
    with open(path, 'w') as f:
        json.dump(raw, f)
    with pytest.raises(WorkflowError, match='a_only_module_array.json exists and is not the '
                                            'module array of modify step "a_only"'):
        run_calibration_workflow(path, output_dir=str(tmp_path / 'run'), write_calibrated=True)


@pytest.mark.unit
def test_a_packed_modify_step_runs_without_its_files(library, tmp_path):
    from libcuflynx.calibration_workflow import pack_workflow
    path = os.path.join(library['arrays'], 'modified_workflow.json')
    with open(path, 'w') as f:
        json.dump(_raw(library), f)
    packed = pack_workflow(path, str(tmp_path / 'pack'))
    raw = _read_json(packed['workflow'])
    assert raw['target'] == {'from': 'add_c'}
    assert raw['steps'][0]['model'] == {
        'module_array': 'inputs/add_c/a_only_module_array.json',
        'parameters': 'inputs/add_c/a_only_parameters.json'}
    assert raw['steps'][0]['changes'] == [ADD_C]
    assert raw['steps'][1]['model'] == {'from': 'add_c'}
