"""
A reduction chosen by hand: ``"method": "manual"`` keeps the parameters named, of the step's own
params_for_id, and reads no sensitivity step.

On ``chain_analysed`` of the workflow test library with its screen taken out: ``fit_a``
calibrates p of A, ``reduce`` keeps q of C from the screened params_for_id (p of A and q of C),
``fit`` calibrates what was kept with p fixed from ``fit_a``, and ``ia`` analyses that fit.
"""

import json
import os

import numpy as np
import pytest

from _params_for_id import params_for_id_entries
from libcuflynx.calibration_workflow import (WorkflowBuilder, WorkflowError, plan_workflow,
                                             resolve_workflow, run_calibration_workflow)
from libcuflynx.calibration_workflow.builder import library_module
from libcuflynx.calibration_workflow.runner import workflow_status
from libcuflynx.calibration_workflow.spec import parse_workflow
from libcuflynx.external_testing.workflow_library import (OPTIMISER_SETTINGS, Y_A, Z_C,
                                                          build_workflow_library)
from libcuflynx.parsers.PrimitiveParsers import analysis_options
from libcuflynx.reporting.methods import workflow_methods_latex
from libcuflynx.reporting.provenance import provenance_graph
from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA, load_schema
from libcuflynx.sensitivity_analysis.parameter_reduction import (ManualReduction,
                                                                 ParamReduction,
                                                                 run_param_reduction)

REASON = 'p of A is known from the first step'


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _by_hand(library, parameters=('mod__C/q',)):
    '''chain_analysed with ``reduce`` chosen by hand and no screen.'''
    raw = _read_json(library['chain_analysed'])
    steps = {s['id']: s for s in raw['steps']}
    reduce = {'id': 'reduce', 'kind': 'reduction',
              'params_for_id': steps['screen']['params_for_id'],
              'settings': {'param_reduction_options': {
                  'method': 'manual', 'parameters': list(parameters), 'reason': REASON}}}
    raw['steps'] = [steps['fit_a'], reduce, steps['fit'], steps['ia']]
    return raw


def _path(library):
    return os.path.join(library['chain/screened'], 'by_hand.json')


def _write(library, raw):
    with open(_path(library), 'w') as f:
        json.dump(raw, f, indent=1)
    return _path(library)


def _parse(library, raw):
    return parse_workflow(raw, path=_path(library))


def _reduce(raw):
    return next(s for s in raw['steps'] if s['id'] == 'reduce')


def _options(raw):
    return _reduce(raw)['settings']['param_reduction_options']


# --------------------------------------------------------------------------------------------
# the method
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_schemas_say_a_reduction_may_be_chosen_by_hand():
    options = load_schema(CALIBRATION_WORKFLOW_SCHEMA)['$defs']['param_reduction_options']
    assert 'manual' in options['properties']['method']['enum']
    assert options['properties']['parameters']['type'] == 'array'
    assert options['properties']['reason']['type'] == 'string'
    descriptors = {o['name']: o for o in analysis_options('param_reduction')}
    assert 'manual' in descriptors['method']['choices']
    assert (descriptors['parameters']['type'], descriptors['reason']['type']) == ('list', 'str')


@pytest.mark.unit
def test_a_manual_reduction_keeps_the_entries_named_and_reads_no_sobol_run(library, tmp_path):
    source = os.path.join(library['chain/screened'], 'screened_params_for_id.json')
    labels = [e['targets'] for e in params_for_id_entries(source)]
    assert len(labels) == 2
    options = {'method': 'manual', 'parameters': ['mod__C/q'], 'reason': REASON}
    reduction = ParamReduction.from_options(options)
    assert isinstance(reduction, ManualReduction) and reduction.as_dict() == options
    out = tmp_path / 'kept' / 'reduced.json'
    assert reduction.reduce(None, source, str(out)) == [{'parameter': 'mod__C/q'}]
    assert [e['targets'] for e in params_for_id_entries(str(out))] == [['mod/C/q']]
    report = _read_json(tmp_path / 'kept' / 'param_reduction.json')
    assert report['reduction'] == options and report['params_for_id'] == source

    # a plain run (do_param_reduction) takes it too
    kept = run_param_reduction({'sa_options': {'output_dir': str(tmp_path / 'sa')},
                                'params_for_id_path': source,
                                'param_reduction_options': options})
    assert kept == [{'parameter': 'mod__C/q'}]

    with pytest.raises(ValueError, match="has no parameter \\['mod__C/k'\\]"):
        ManualReduction(['mod__C/k'], REASON).reduce(None, source, str(out))


@pytest.mark.unit
@pytest.mark.parametrize('options, message', [
    ({'reason': REASON}, 'needs parameters'),
    ({'parameters': [], 'reason': REASON}, 'needs parameters'),
    ({'parameters': 'mod__C/q', 'reason': REASON}, 'needs parameters'),
    ({'parameters': ['a/b', 'a/b'], 'reason': REASON}, 'more than once'),
    ({'parameters': ['a/b']}, 'needs a reason'),
    ({'parameters': ['a/b'], 'reason': ' '}, 'needs a reason'),
    ({'parameters': ['a/b'], 'reason': REASON, 'n_params': 1},
     "n_params is not read by method 'manual'"),
])
def test_a_manual_reduction_needs_its_names_and_its_reason(options, message):
    with pytest.raises(ValueError, match=message):
        ParamReduction.from_options(dict(options, method='manual'))


@pytest.mark.unit
def test_parameters_and_reason_are_for_a_manual_reduction():
    with pytest.raises(ValueError, match="parameters is not read by method 'pca'"):
        ParamReduction.from_options({'method': 'pca', 'parameters': ['a/b']})


# --------------------------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_reduction_chosen_by_hand_is_read(library):
    workflow = _parse(library, _by_hand(library))
    reduce = workflow.step('reduce')
    assert reduce.chosen_by_hand and reduce.model is None and reduce.depends_on == []
    assert reduce.sensitivity_from is None
    assert reduce.params_for_id == os.path.join(library['chain/screened'],
                                                'screened_params_for_id.json')
    assert reduce.reduction == {'method': 'manual', 'parameters': ['mod__C/q'],
                                'reason': REASON}
    assert not workflow.chosen_at_run(workflow.step('fit'))
    assert [s.id for s in workflow.topological_order()] == ['fit_a', 'reduce', 'fit', 'ia']
    # a reduction that reads a sensitivity step is not chosen by hand
    analysed = parse_workflow(_read_json(library['chain_analysed']),
                              path=library['chain_analysed'])
    assert not analysed.step('reduce').chosen_by_hand
    assert analysed.chosen_at_run(analysed.step('fit'))


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda w: _reduce(w).update(sensitivity_from='fit_a'),
     'chosen by hand .* has no sensitivity_from'),
    (lambda w: _reduce(w).pop('params_for_id'), 'chosen by hand .* needs params_for_id'),
    (lambda w: _options(w).pop('reason'), 'needs parameters .* and reason'),
    (lambda w: _options(w).pop('parameters'), 'needs parameters .* and reason'),
    (lambda w: _options(w).update(parameters=[]), 'parameters'),
    (lambda w: _options(w).update(parameters=['mod__C/q', 'mod__C/q']), 'parameters'),
    (lambda w: _options(w).update(method='sobol', n_params=1),
     'parameters and reason are for a manual reduction'),
    (lambda w: _reduce(w).update(model=w['steps'][0]['model']),
     r"keeps entries of its params_for_id and has no \['model'\]"),
])
def test_an_invalid_reduction_chosen_by_hand_says_why(library, edit, message):
    raw = _by_hand(library)
    edit(raw)
    with pytest.raises(WorkflowError, match=message):
        _parse(library, raw)


@pytest.mark.unit
def test_a_reduction_that_reads_a_sensitivity_step_names_no_params_for_id(library):
    raw = _read_json(library['chain_analysed'])
    _reduce(raw)['params_for_id'] = raw['steps'][1]['params_for_id']
    with pytest.raises(WorkflowError, match='needs sensitivity_from and has no params_for_id'):
        parse_workflow(raw, path=library['chain_analysed'])


@pytest.mark.unit
def test_the_names_chosen_are_checked_against_the_params_for_id(library):
    raw = _by_hand(library, parameters=['mod__C/q', 'mod__C/k'])
    with pytest.raises(WorkflowError, match=r'step "reduce": .*screened_params_for_id.json has '
                                            r"no entry for \['mod__C/k'\] \(its parameters: "):
        resolve_workflow(_parse(library, raw))
    raw = _by_hand(library)
    _reduce(raw)['params_for_id'] = 'missing_params_for_id.json'
    with pytest.raises(WorkflowError, match='step "reduce": its params_for_id file .* does not '
                                            'exist'):
        resolve_workflow(_parse(library, raw))


@pytest.mark.unit
def test_what_a_choice_by_hand_calibrates_is_known_before_the_run(library):
    """A step fitting a choice made by hand claims the parameters named, so the plan lists them
    and an unrelated calibration of one is refused before anything runs. (A step fitting a
    sensitivity-based reduction's choice is checked when it has run.)"""
    raw = _by_hand(library)
    plan = {s['id']: s for s in plan_workflow(_write(library, raw))['steps']}
    assert [c['model_name'] for c in plan['fit']['calibrates']] == ['q_mod__C']
    assert plan['reduce']['reduction']['method'] == 'manual'
    assert plan['reduce']['params_for_id'].endswith('screened_params_for_id.json')
    assert 'sensitivity_from' not in plan['reduce']

    steps = {s['id']: s for s in raw['steps']}
    other = {k: v for k, v in steps['fit'].items() if k != 'params_for_id_from'}
    raw['steps'].append(dict(other, id='other', params_for_id=steps['ia']['params_for_id']))
    with pytest.raises(WorkflowError, match=r'"q_mod__C" by \'fit\' and \'other\''):
        resolve_workflow(_parse(library, raw), target_overlap='error')


@pytest.mark.integration
def test_a_workflow_calibrates_what_was_chosen_by_hand(library, tmp_path):
    path = _write(library, _by_hand(library))
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(path, output_dir=out)
    assert result['complete'] and result['order'] == ['fit_a', 'reduce', 'fit', 'ia']
    step = lambda sid: _read_json(os.path.join(out, sid, 'step_result.json'))  # noqa: E731

    reduce = step('reduce')
    assert reduce['selected'] == [{'parameter': 'mod__C/q'}]
    assert reduce['reduction'] == {'method': 'manual', 'parameters': ['mod__C/q'],
                                   'reason': REASON}
    assert [e['targets'] for e in params_for_id_entries(reduce['params_for_id'])] == [
        ['mod/C/q']]
    assert step('fit')['params_for_id_used'] == reduce['params_for_id']
    merged = {r['parameterisation_name']: (r['value'], r['from_step'])
              for r in result['merged_parameters']}
    assert merged['A/p'] == (pytest.approx(np.sqrt(Y_A / 2), rel=1e-5), 'fit_a')
    assert merged['C/q'] == (pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-5), 'fit')

    # provenance: the choice is marked set by hand, and nothing generates its box
    graph = provenance_graph(out)
    node = next(n for n in graph['nodes'] if n['id'] == 'reduce')
    assert (node['kind'], node['mark'], node['status']) == ('analysis', 'set_by_hand', 'done')
    assert node['lines'][:2] == ['chosen by hand: `mod__C/q`', REASON]
    edges = {(e['from'], e['to']): e for e in graph['edges']}
    assert edges[('reduce', 'fit')]['label'] == 'parameters'
    assert not [e for e in graph['edges'] if e['to'] == 'reduce']

    latex = workflow_methods_latex(path, run_dir=out)
    assert 'This step keeps parameters chosen by hand' in latex and REASON in latex

    # the params_for_id it keeps entries of is its input
    assert not any(s['stale'] for s in workflow_status(path, out)['steps'].values())
    with open(os.path.join(library['chain/screened'], 'screened_params_for_id.json'), 'a') as f:
        f.write('\n')
    assert {i: s['stale'] for i, s in workflow_status(path, out)['steps'].items()} == {
        'fit_a': False, 'reduce': True, 'fit': True, 'ia': True}


@pytest.mark.unit
def test_a_packed_workflow_has_the_params_for_id_chosen_from(library, tmp_path):
    from libcuflynx.calibration_workflow import pack_workflow
    packed = pack_workflow(_write(library, _by_hand(library)), str(tmp_path / 'pack'))
    raw = _read_json(packed['workflow'])
    assert _reduce(raw)['params_for_id'] == 'inputs/reduce/screened_params_for_id.json'
    assert os.path.isfile(os.path.join(str(tmp_path / 'pack'), _reduce(raw)['params_for_id']))
    plan = {s['id']: s for s in plan_workflow(packed['workflow'])['steps']}
    assert [c['model_name'] for c in plan['fit']['calibrates']] == ['q_mod__C']


@pytest.mark.unit
def test_the_builder_adds_a_reduction_chosen_by_hand(library, tmp_path):
    screened = library['chain/screened']
    wf = WorkflowBuilder('by_hand', target=library_module('chain', 'v1', 'screened'),
                         directory=tmp_path / 'wf', module_library_dirs=[library['modules']],
                         settings=OPTIMISER_SETTINGS)
    wf.add_reduction('keep', method='manual', parameters=['mod__C/q'], reason=REASON,
                     params_for_id=os.path.join(screened, 'screened_params_for_id.json'))
    wf.add_calibration('fit', obs_data=os.path.join(screened, 'screened_obs_data.json'),
                       params_for_id_from='keep')
    keep = wf.to_dict()['steps'][0]
    assert keep['settings'] == {'param_reduction_options': {
        'method': 'manual', 'parameters': ['mod__C/q'], 'reason': REASON}}
    assert 'sensitivity_from' not in keep and 'params_for_id' in keep
    plan = {s['id']: s for s in plan_workflow(wf.write())['steps']}
    assert [c['model_name'] for c in plan['fit']['calibrates']] == ['q_mod__C']
    with pytest.raises(WorkflowError, match='needs sensitivity_from'):
        wf.add_reduction('ranked', method='sobol', n_params=1)
