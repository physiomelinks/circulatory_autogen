"""
A combine step, and the rule it settles: a parameter given to the target, or to a step, by two
calibrations that do not take from each other is refused.

On ``chain_rest`` of the workflow test library: ``fit_a`` calibrates p of A (2), and ``rest``
calibrates q of C (3) in the chain with p fixed from ``fit_a``.
"""

import json
import os
import warnings

import numpy as np
import pytest

from _params_for_id import write_params_for_id
from libcuflynx.calibration_workflow import (WorkflowBuilder, WorkflowError, plan_workflow,
                                             resolve_workflow, run_calibration_workflow)
from libcuflynx.calibration_workflow.runner import workflow_status
from libcuflynx.calibration_workflow.spec import parse_workflow
from libcuflynx.external_testing.workflow_library import Y_A, Z_C, build_workflow_library
from libcuflynx.reporting.methods import workflow_methods_latex
from libcuflynx.reporting.provenance import provenance_graph
from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA, load_schema

Q_C = np.sqrt(Z_C - Y_A)


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


def _parse(library, raw):
    return parse_workflow(raw, path=library['chain_rest'])


def _resolve(library, raw):
    '''Resolved as a run of every step resolves it: a parameter the target is given by
    unrelated steps is refused.'''
    return resolve_workflow(_parse(library, raw), target_overlap='error')


def _combine(step_id, combine_from, method='best_fit'):
    return {'id': step_id, 'kind': 'combine', 'combine_from': list(combine_from),
            'combine': {'method': method}}


def _two_fits(library):
    '''chain_rest with q calibrated twice, both with p fixed from fit_a: ``poor`` between bounds
    that leave the answer out, then ``good``. Returns (fit_a, poor, good).'''
    fit_a, rest = _read_json(library['chain_rest'])['steps']
    narrow = write_params_for_id(
        os.path.join(library['chain/rest'], 'narrow_params_for_id.json'),
        [{'targets': ['mod/C/q'], 'min': 0.1, 'max': 1.0, 'name_for_plotting': 'q_mod__C'}])
    return fit_a, dict(rest, id='poor', params_for_id=narrow), dict(rest, id='good')


def _raw(library, steps):
    return dict(_read_json(library['chain_rest']), steps=list(steps))


# --------------------------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_schema_has_the_combine_step():
    step = load_schema(CALIBRATION_WORKFLOW_SCHEMA)['$defs']['step']['properties']
    assert 'combine' in step['kind']['enum']
    assert step['combine']['properties']['method']['enum'] == ['best_fit', 'mean']
    assert step['combine_from']['minItems'] == 2


@pytest.mark.unit
def test_a_combine_step_is_read_and_runs_after_its_inputs(library):
    fit_a, poor, good = _two_fits(library)
    workflow = _parse(library, _raw(library, [_combine('best', ['poor', 'good']), fit_a, poor,
                                              good]))
    best = workflow.step('best')
    assert (best.kind, best.model, best.combine_from, best.combine) == (
        'combine', None, ['poor', 'good'], {'method': 'best_fit'})
    assert best.sources() == {'fixed_from': [], 'priors_from': [],
                              'combine_from': ['poor', 'good']}
    assert [s.id for s in workflow.topological_order()] == ['fit_a', 'poor', 'good', 'best']
    assert workflow.ancestors()['best'] == {'fit_a', 'poor', 'good'}


@pytest.mark.unit
@pytest.mark.parametrize('change, match', [
    (lambda s: s.pop('combine_from'), 'combine_from'),
    (lambda s: s.pop('combine'), 'combine'),
    (lambda s: s.update(combine_from=['good']), 'combine_from'),
    (lambda s: s.update(combine_from=['good', 'good']), 'combine_from'),
    (lambda s: s.update(combine={'method': 'median'}), 'method'),
    (lambda s: s.update(settings={'dt': 0.1}), 'runs nothing and has no "settings"'),
    (lambda s: s.update(model={'select': ['C']}), r"combines the values of other steps and has "
                                                  r"no \['model'\]"),
    (lambda s: s.update(combine_from=['good', 'missing']), 'unknown step "missing"'),
    (lambda s: s.update(combine_from=['good', 'scan']), '"combine_from" names "scan", a sweep '
                                                        'step; it takes a calibration or set or '
                                                        'combine step'),
])
def test_a_combine_step_that_cannot_be_read_is_refused(library, change, match):
    fit_a, poor, good = _two_fits(library)
    best = _combine('best', ['poor', 'good'])
    change(best)
    scan = {'id': 'scan', 'kind': 'sweep', 'model': good['model'],
            'settings': {'sim_time': 1.0},
            'sweep': {'outputs': ['mod__C/z'], 'parameters': {}}}
    with pytest.raises(WorkflowError, match=match):
        _parse(library, _raw(library, [fit_a, poor, good, best, scan]))


@pytest.mark.unit
def test_combine_keys_are_for_a_combine_step(library):
    fit_a, poor, good = _two_fits(library)
    for key, value in (('combine_from', ['fit_a', 'poor']), ('combine', {'method': 'mean'})):
        with pytest.raises(WorkflowError, match='combine'):
            _parse(library, _raw(library, [fit_a, poor, dict(good, **{key: value})]))


@pytest.mark.unit
def test_best_fit_compares_calibrations_of_the_same_data(library):
    """A cost is comparable only between calibrations of the same obs_data with the same cost
    function, of the same parameters; the mean has no such limit."""
    fit_a, poor, good = _two_fits(library)
    hand = {'id': 'hand', 'kind': 'set', 'point': True, 'set': [
        {'parameter': 'mod__C/q', 'value': 2.0, 'reason': 'a guess'}]}

    def raw(inputs, method='best_fit', **changes):
        return _raw(library, [fit_a, poor, dict(good, **changes), hand,
                              _combine('best', inputs, method)])

    with pytest.raises(WorkflowError, match='"hand" \\(a set step\\) has none'):
        _parse(library, raw(['poor', 'good', 'hand']))
    _resolve(library, raw(['poor', 'good', 'hand'], 'mean'))

    with pytest.raises(WorkflowError, match='their obs_data differ'):
        _parse(library, raw(['fit_a', 'good']))
    with pytest.raises(WorkflowError, match="their cost_type differ .*'poor': 'gaussian_MLE', "
                                            "'good': 'MSE'"):
        _parse(library, raw(['poor', 'good'],
                            settings={'optimiser_options': {'cost_type': 'MSE'}}))

    # the same obs_data, but good calibrates p of A too
    both = write_params_for_id(
        os.path.join(library['chain/rest'], 'both_params_for_id.json'),
        [{'targets': ['mod/A/p'], 'min': 0.1, 'max': 5.0, 'name_for_plotting': 'p_mod__A'},
         {'targets': ['mod/C/q'], 'min': 0.1, 'max': 10.0, 'name_for_plotting': 'q_mod__C'}])
    with pytest.raises(WorkflowError, match=r"must calibrate the same parameters, but only some "
                                            r"of them calibrate \{'p_mod__A': \['good'\]\}"):
        _resolve(library, raw(['poor', 'good'], params_for_id=both))
    _resolve(library, raw(['poor', 'good'], 'mean', params_for_id=both))


# --------------------------------------------------------------------------------------------
# one value per parameter
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_unrelated_calibrations_of_a_parameter_need_a_combine_step(library):
    fit_a, poor, good = _two_fits(library)
    with pytest.raises(WorkflowError) as refused:
        _resolve(library, _raw(library, [fit_a, poor, good]))
    message = str(refused.value)
    assert 'the target chain/v1/rest is given 1 parameter(s) by steps that do not take from ' \
           'each other: "q_mod__C" by \'poor\' and \'good\'' in message
    assert '"kind": "combine", "combine_from": ["poor", "good"]' in message

    with warnings.catch_warnings():
        warnings.simplefilter('error', category=UserWarning)
        # the combine step is what is left; one that takes from the other replaces it; and a
        # set step replaces a calibrated value, whatever it takes from
        _resolve(library, _raw(library, [fit_a, poor, good, _combine('best', ['poor', 'good'])]))
        _resolve(library, _raw(library, [fit_a, poor, dict(good, start_from='poor')]))
        _resolve(library, _raw(library, [fit_a, poor, {'id': 'hand', 'kind': 'set', 'set': [
            {'parameter': 'mod__C/q', 'value': 2.0, 'reason': 'a guess'}]}]))

    # two combine steps of the same steps are two values again
    with pytest.raises(WorkflowError, match="\"q_mod__C\" by 'best' and 'mean'"):
        _resolve(library, _raw(library, [fit_a, poor, good, _combine('best', ['poor', 'good']),
                                         _combine('mean', ['poor', 'good'], 'mean')]))


@pytest.mark.unit
def test_reading_and_planning_warn_where_a_run_of_every_step_refuses(library):
    """The target is merged for use by a run of more than one step, so that is where two
    values of one parameter are refused. Reading, the plan and the status only warn, and the
    plan carries the message."""
    fit_a, poor, good = _two_fits(library)
    workflow = _parse(library, _raw(library, [fit_a, poor, good]))
    with pytest.warns(UserWarning, match="\"q_mod__C\" by 'poor' and 'good'.*Until then the "
                                         "value of the later step in run order is used"):
        resolved = resolve_workflow(workflow)
    assert 'the target chain/v1/rest is given 1 parameter(s)' in resolved.target_overlap
    with pytest.warns(UserWarning, match='do not take from each other'):
        plan = plan_workflow(workflow)
    assert plan['target_overlap'] == resolved.target_overlap
    settled = _raw(library, [fit_a, poor, good, _combine('best', ['poor', 'good'])])
    assert plan_workflow(_parse(library, settled))['target_overlap'] is None


@pytest.mark.unit
def test_unrelated_set_steps_of_one_parameter_warn(library):
    """A value set by hand never clashes, but two set steps that do not take from each other
    setting one parameter is likely a mistake: a warning names the one that is used."""
    fit_a, rest = _read_json(library['chain_rest'])['steps']

    def hand(step_id, **entry):
        return {'id': step_id, 'kind': 'set', 'set': [
            dict({'parameter': 'mod__A/p', 'reason': 'a guess'}, **entry)]}

    raw = _raw(library, [hand('first', value=1.0), hand('second', value=2.0),
                         dict(rest, fixed_from=['first', 'second'])])
    with pytest.warns(UserWarning) as caught:
        _resolve(library, raw)
    messages = [str(w.message) for w in caught if 'set steps' in str(w.message)]
    assert len(messages) == 2      # the target, and rest
    assert 'the target chain/v1/rest is given 1 parameter(s) by set steps that do not take ' \
           'from each other: "p_mod__A" by \'first\' and \'second\' (the value of \'second\', ' \
           'the later in run order, is used)' in messages[0]
    assert 'step "rest" is given 1 parameter(s) by set steps' in messages[1]

    with warnings.catch_warnings():
        warnings.simplefilter('error', category=UserWarning)
        # one scales the other's value, or one is all there is: no warning
        _resolve(library, _raw(library, [hand('first', value=1.0),
                                         hand('second', scale=2.0, of='first'),
                                         dict(rest, fixed_from=['second'])]))
        _resolve(library, _raw(library, [fit_a, hand('second', value=2.0),
                                         dict(rest, fixed_from=['second'])]))


@pytest.mark.integration
def test_a_second_calibration_runs_alone_before_its_combine_step_is_added(library, tmp_path):
    """As a front end adds steps: the second fit of q is run alone, which goes through with a
    warning and merges the later value; a run of every step is refused before anything runs;
    the combine step settles it."""
    fit_a, poor, good = _two_fits(library)
    out = str(tmp_path / 'out')
    run_calibration_workflow(_write(library, _raw(library, [fit_a, poor])), output_dir=out)

    path = _write(library, _raw(library, [fit_a, poor, good]))
    with pytest.warns(UserWarning, match='do not take from each other'):
        result = run_calibration_workflow(path, output_dir=out, only='good')
    merged = {r['parameterisation_name']: r for r in result['merged_parameters']}
    assert merged['C/q']['from_step'] == 'good'
    assert merged['C/q']['value'] == pytest.approx(Q_C, rel=1e-3)

    finished = _read_json(os.path.join(out, 'good', 'step_result.json'))['finished']
    for keys in ({}, {'stale_only': True}, {'only': 'good', 'write_calibrated': True}):
        with pytest.raises(WorkflowError, match="\"q_mod__C\" by 'poor' and 'good'"):
            run_calibration_workflow(path, output_dir=out, **keys)
    assert _read_json(os.path.join(out, 'good', 'step_result.json'))['finished'] == finished

    path = _write(library, _raw(library, [fit_a, poor, good,
                                          _combine('best', ['poor', 'good'])]))
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=UserWarning)
        resolve_workflow(_parse(library, _read_json(path)))
    result = run_calibration_workflow(path, output_dir=out, stale_only=True)
    merged = {r['parameterisation_name']: r for r in result['merged_parameters']}
    assert merged['C/q']['from_step'] == 'best' and result['complete']


@pytest.mark.unit
def test_a_step_given_a_parameter_by_unrelated_steps_is_refused(library):
    """The same rule for what a step is given: by fixed_from and result_from, through the steps
    those were given values by."""
    fit_a, poor, good = _two_fits(library)
    best = _combine('best', ['poor', 'good'])
    scan = {'id': 'scan', 'kind': 'sweep', 'model': good['model'],
            'settings': {'sim_time': 1.0},
            'sweep': {'outputs': ['mod__C/z'], 'parameters': {'mod__A/p': {'values': [1.0]}}}}

    def raw(**keys):
        return _raw(library, [fit_a, poor, good, best, dict(scan, **keys)])

    with pytest.raises(WorkflowError, match="step \"scan\" is given 1 parameter\\(s\\) by steps "
                                            "that do not take from each other: \"q_mod__C\" by "
                                            "'poor' and 'good'"):
        _resolve(library, raw(fixed_from=['poor', 'good']))
    with pytest.raises(WorkflowError, match="step \"scan\" is given"):
        _resolve(library, raw(fixed_from=['poor'], result_from='good'))
    # p of A reaches scan from fit_a through both, which is one giver
    plan = {s['id']: s for s in plan_workflow(_parse(library, raw(fixed_from=['best', 'poor'],
                                                                   result_from='good')))['steps']}
    assert plan['best']['combine'] == {'method': 'best_fit', 'gives': [
        {'target_model_name': 'q_mod__C', 'target_parameterisation_name': 'C/q'}]}
    assert plan['best']['combine_from'] == ['poor', 'good']
    assert {f['from_step'] for f in plan['scan']['fixed_from']} == {'best', 'poor'}
    # a subset leaves q of poor out
    _resolve(library, raw(fixed_from=['good', {'step': 'fit_a', 'parameters': ['p_mod']}]))


# --------------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_the_best_fit_replaces_its_inputs_in_later_steps_and_the_target(library, tmp_path):
    fit_a, poor, good = _two_fits(library)
    scan = {'id': 'scan', 'kind': 'sweep', 'model': good['model'], 'fixed_from': ['best'],
            'settings': {'sim_time': 1.0},
            'sweep': {'outputs': ['mod__C/z'], 'parameters': {'mod__A/p': {'values': [2.0]}}}}
    raw = _raw(library, [fit_a, poor, good, _combine('best', ['poor', 'good']), scan])
    path, out = _write(library, raw), str(tmp_path / 'out')
    result = run_calibration_workflow(path, output_dir=out)
    assert result['order'] == ['fit_a', 'poor', 'good', 'best', 'scan']
    step = {s: _read_json(os.path.join(out, s, 'step_result.json')) for s in result['order']}

    assert step['poor']['calibrated'][0]['value'] == pytest.approx(1.0, rel=1e-3)
    assert step['poor']['best_cost'] > step['good']['best_cost']
    best = step['best']
    assert (best['method'], best['chosen_from'], best['best_cost']) == (
        'best_fit', 'good', step['good']['best_cost'])
    assert best['costs'] == {s: step[s]['best_cost'] for s in ('poor', 'good')}
    assert best['fixed'] == []
    [record] = best['calibrated']
    assert record == {
        'vessel': 'mod__C', 'param': 'q', 'model_name': 'q_mod__C',
        'target_model_name': 'q_mod__C', 'target_host_name': 'mod__C/q',
        'target_parameterisation_name': 'C/q',
        'value': step['good']['calibrated'][0]['value'], 'units': record['units'],
        'combined_from': ['poor', 'good'], 'chosen_from': 'good'}
    assert record['value'] == pytest.approx(Q_C, rel=1e-3)

    merged = {r['parameterisation_name']: r for r in result['merged_parameters']}
    assert (merged['C/q']['value'], merged['C/q']['from_step']) == (record['value'], 'best')
    assert merged['A/p']['from_step'] == 'fit_a'
    # scan is given q by the combine step, and p as good was (the chosen fit's fixed values
    # are not the combine step's to pass on)
    assert [(f['model_name'], f['from_step'], f['value']) for f in step['scan']['fixed']] == [
        ('q_mod__C', 'best', record['value'])]

    graph = provenance_graph(out)
    node = next(n for n in graph['nodes'] if n['id'] == 'best')
    assert (node['kind'], node['mark'], node['status']) == ('step', 'calibrated', 'done')
    assert node['lines'][0] == 'best fit of poor, good'
    assert [p['name'] for p in node['parameters']] == ['q_mod__C']
    edges = {(e['from'], e['to']): e for e in graph['edges']}
    assert edges[('poor', 'best')]['label'] == edges[('good', 'best')]['label'] == 'combined'
    assert ('best', '__target__') in edges and ('best', 'scan') in edges
    target = next(n for n in graph['nodes'] if n['id'] == '__target__')
    assert {p['name']: p['from_step'] for p in target['parameters']} == {
        'p_mod__A': 'fit_a', 'q_mod__C': 'best'}

    latex = workflow_methods_latex(path, run_dir=out)
    assert 'takes the values of the calibration with the lowest cost among steps poor and good' \
        in latex
    assert 'It took those of step good' in latex

    # the combine step is stale once an input has run again
    assert not workflow_status(path, out)['steps']['best']['stale']
    run_calibration_workflow(path, output_dir=out, only='poor')
    status = workflow_status(path, out)['steps']
    assert (status['best']['stale'], status['scan']['stale']) == (True, True)


@pytest.mark.integration
def test_the_mean_of_values_set_by_hand(library, tmp_path):
    """A mean is taken per parameter over the steps that give it: p of A by both, k of C by one.
    Values that were all set by hand stay set by hand."""
    fit_a, rest = _read_json(library['chain_rest'])['steps']

    def hand(step_id, p, **more):
        entries = [{'parameter': 'mod__A/p', 'value': p, 'reason': f'guess {step_id}'}]
        return {'id': step_id, 'kind': 'set', 'point': True, 'set': entries + [
            {'parameter': name, 'value': value, 'reason': 'the gain'}
            for name, value in more.items()]}

    raw = _raw(library, [hand('low', 1.0), hand('high', 2.0, **{'mod__C/k': 1.0}),
                         _combine('middle', ['low', 'high'], 'mean'),
                         dict(rest, fixed_from=['middle'])])
    out = str(tmp_path / 'out')
    result = run_calibration_workflow(_write(library, raw), output_dir=out)
    middle = _read_json(os.path.join(out, 'middle', 'step_result.json'))
    values = {r['model_name']: r for r in middle['calibrated']}
    assert values['p_mod__A']['value'] == 1.5
    assert values['p_mod__A']['combined_from'] == ['low', 'high']
    assert (values['k_mod__C']['value'], values['k_mod__C']['combined_from']) == (1.0, ['high'])
    assert values['p_mod__A']['origin'] == 'set_by_hand'
    assert values['p_mod__A']['reason'] == 'the mean of the values set by hand in steps low, high'
    assert 'best_cost' not in middle and 'chosen_from' not in values['p_mod__A']

    rest_result = _read_json(os.path.join(out, 'rest', 'step_result.json'))
    assert {f['model_name']: (f['value'], f['from_step'], f['origin'])
            for f in rest_result['fixed']} == {'p_mod__A': (1.5, 'middle', 'set_by_hand'),
                                               'k_mod__C': (1.0, 'middle', 'set_by_hand')}
    assert rest_result['calibrated'][0]['value'] == pytest.approx(np.sqrt(Z_C - 2 * 1.5 ** 2),
                                                                  rel=1e-3)
    # the point steps are not merged, the combine step's values are
    merged = {r['parameterisation_name']: r for r in result['merged_parameters']}
    assert (merged['A/p']['value'], merged['A/p']['from_step'], merged['A/p']['origin']) == (
        1.5, 'middle', 'set_by_hand')


@pytest.mark.integration
def test_a_step_is_given_the_value_of_the_step_that_took_from_the_other(library, tmp_path):
    """rest calibrates p of A again, with fit_a's p fixed elsewhere: a step fixed from both is
    given rest's p, not fit_a's, and no clash is raised."""
    fit_a, rest = _read_json(library['chain_rest'])['steps']
    both = write_params_for_id(
        os.path.join(library['chain/rest'], 'both_params_for_id.json'),
        [{'targets': ['mod/A/p'], 'min': 2.2, 'max': 5.0, 'name_for_plotting': 'p_mod__A'},
         {'targets': ['mod/C/q'], 'min': 0.1, 'max': 10.0, 'name_for_plotting': 'q_mod__C'}])
    scan = {'id': 'scan', 'kind': 'sweep', 'model': rest['model'],
            'fixed_from': ['rest', 'fit_a'], 'settings': {'sim_time': 1.0},
            'sweep': {'outputs': ['mod__C/z'], 'parameters': {'mod__C/k': {'values': [1.0]}}}}
    raw = _raw(library, [fit_a, dict(rest, params_for_id=both), scan])
    out = str(tmp_path / 'out')
    run_calibration_workflow(_write(library, raw), output_dir=out)
    p_fit_a = _read_json(os.path.join(out, 'fit_a', 'step_result.json'))['calibrated'][0]['value']
    fixed = {f['model_name']: f for f in
             _read_json(os.path.join(out, 'scan', 'step_result.json'))['fixed']}
    assert fixed['p_mod__A']['from_step'] == 'rest'
    assert fixed['p_mod__A']['value'] >= 2.2 > p_fit_a


# --------------------------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_builder_adds_a_combine_step(library, tmp_path):
    wf = WorkflowBuilder.load(library['chain_rest'])
    fit_a, poor, good = _two_fits(library)
    wf.add_step(**poor)
    wf.add_step(**good)
    assert wf.add_combine('best', ['poor', 'good'], description='the better of two') == 'best'
    assert wf.to_dict()['steps'][-1] == {
        'id': 'best', 'kind': 'combine', 'description': 'the better of two',
        'combine_from': ['poor', 'good'], 'combine': {'method': 'best_fit'}}
    wf.add_combine('best', ['poor', 'good'], method='mean')
    assert wf.to_dict()['steps'][-1]['combine'] == {'method': 'mean'}
    with pytest.raises(WorkflowError):
        wf.add_combine('one', ['good'])
