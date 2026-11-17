"""What a calibration workflow can express so that a GUI run is one of its steps.

* a CellML model as a step's model and the target (``{"cellml_model_path", "parameters"}``);
* a calibration that starts at an earlier step's values (``start_from``);
* a ``uq`` step: MCMC or a Laplace approximation at a point, with no fit of its own, whose
  distribution a later calibration takes with ``priors_from``;
* an emulator trained outside the workflow (``emulator_dir``);
* what a step writes after it runs: a calibration's error vectors and calibrated CellML, a
  local sensitivity's nominal point.

The runs use a Lotka-Volterra CellML (``tests/data/workflow_parity/``) whose obs_data was made
at alpha = 4, gamma = 2.5 (the CellML has 5 and 3).
"""
import copy
import json
import os
import shutil

import numpy as np
import pytest

from libcuflynx.calibration_workflow import (CellMLModel, StoredDistribution, WorkflowBuilder,
                                             WorkflowError, cellml_model, load_workflow,
                                             pack_workflow, plan_workflow, resolve_workflow,
                                             run_calibration_workflow)
from libcuflynx.calibration_workflow.builder import set_theta, set_value
from libcuflynx.calibration_workflow.spec import parse_workflow
from libcuflynx.calibration_workflow.generate import read_parameter_rows

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'workflow_parity')
_MODEL, _OBS, _PARAMS = 'lotka_volterra.cellml', 'lv_obs_data.json', 'lv_params_for_id.json'
_ALPHA, _GAMMA = 'Lotka_Volterra_module/alpha', 'Lotka_Volterra_module/gamma'
_TRUE = {_ALPHA: 4.0, _GAMMA: 2.5}
_OWN = {_ALPHA: 5.0, _GAMMA: 3.0}
_FAST = {'dt': 0.01, 'param_id_method': 'CMA-ES',
         'solver_info': {'MaximumStep': 0.01},
         'optimiser_options': {'num_calls_to_function': 300, 'max_patience': 60, 'seed': 1,
                               'cost_type': 'gaussian_MLE'},
         'UQ_options': {'num_steps': 60, 'num_walkers': 6, 'burn_in': 0.5,
                        'posterior_predictive_samples': 12}}


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
    return path


@pytest.fixture
def lv(tmp_path):
    '''A copy of the fixture directory, so runs write nothing into the repository.'''
    directory = str(tmp_path / 'lv')
    shutil.copytree(_DATA, directory)
    return directory


def _model(parameters=None):
    return cellml_model(_MODEL, parameters)


def _fit(id='fit', **extra):
    return dict({'id': id, 'model': _model(), 'obs_data': _OBS, 'params_for_id': _PARAMS},
                **extra)


def _raw(steps, settings=None, target=None):
    return {'schema_version': 1, 'workflow_name': 'lv', 'target': target or _model(),
            'settings': copy.deepcopy(_FAST if settings is None else settings),
            'steps': steps}


def _write(lv, raw, name='workflow.json'):
    return _write_json(os.path.join(lv, name), raw)


def _step(out, sid):
    return _read_json(os.path.join(out, sid, 'step_result.json'))


def _values(path):
    return {r['variable_name']: float(r['value']) for r in read_parameter_rows(path)}


def _values_file(lv, **values):
    '''A parameters file beside the model setting ``Lotka_Volterra_module/<name>``.'''
    from libcuflynx.utilities import parameters_file as pf
    doc = pf.new_document()
    doc['Lotka_Volterra_module'] = {name: {'value': repr(float(value)), 'units': 'dimensionless'}
                                    for name, value in values.items()}
    return pf.write_parameters_file(os.path.join(lv, 'values.json'), doc)


def _constants(path):
    from libcuflynx.calibration_workflow.cellml_files import read_cellml_constants
    return {name: c['value'] for name, c in read_cellml_constants(path).items()}


# --------------------------------------------------------------------------------------------
# 1. the file: a CellML model
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_cellml_model_is_a_step_model_and_the_target(lv):
    workflow = load_workflow(_write(lv, _raw([_fit()])))
    model = CellMLModel(os.path.join(lv, _MODEL))
    assert workflow.target == model and workflow.step('fit').model == model
    assert model.as_dict() == {'cellml_model_path': os.path.join(lv, _MODEL)}
    assert model.label() == 'lotka_volterra'
    with_values = load_workflow(_write(lv, _raw([_fit(model=_model('values.json'))],
                                                target=_model('values.json'))))
    assert with_values.target.parameters == os.path.join(lv, 'values.json')


@pytest.mark.unit
@pytest.mark.parametrize('change, match', [
    (lambda raw: raw['steps'][0].update(model={'module_array': 'x_module_array.csv'}),
     'the target is the CellML model .* every step\'s model is that file'),
    (lambda raw: raw.update(target={'module_array': 'x_module_array.csv'}),
     'run a CellML model, so the target must be that file'),
    (lambda raw: raw['settings'].update(model_type='cpp'),
     'runs as model_type cellml, python, casadi_python, aadc_python; not \'cpp\''),
    (lambda raw: raw['steps'][0].update(settings={'model_type': 'external_python'}),
     'step "fit": .* not \'external_python\''),
    (lambda raw: raw['steps'][0].update(model={'cellml_model_path': 'lv.py'}),
     'cellml_model_path'),
    (lambda raw: raw['steps'][0].update(model={'cellml_model_path': _MODEL,
                                                'parameters': 'values.csv'}),
     'parameters'),
])
def test_what_a_cellml_model_cannot_be_given_is_refused(lv, change, match):
    raw = _raw([_fit()])
    change(raw)
    with pytest.raises(WorkflowError, match=match):
        load_workflow(_write(lv, raw))


@pytest.mark.unit
def test_the_components_are_the_vessels_and_the_constants_its_parameters(lv):
    resolved = resolve_workflow(load_workflow(_write(lv, _raw([_fit()]))))
    target = resolved.target
    assert target.kind == 'cellml' and target.vessels == {'Lotka_Volterra_module'}
    # constants only: the states x and y, and the time, are not parameters
    assert target.parameters == {
        'Lotka_Volterra_module/' + n: v for n, v in
        dict(alpha=5.0, beta=0.2, delta=0.2, gamma=3.0, u_alpha=0.0, u_gamma=0.0).items()}
    step = plan_workflow(os.path.join(lv, 'workflow.json'))['steps'][0]
    assert [c['target_model_name'] for c in step['calibrates']] == [
        'alpha_Lotka_Volterra_module', 'gamma_Lotka_Volterra_module']


@pytest.mark.unit
def test_a_name_that_is_not_a_constant_is_refused(lv):
    _write_json(os.path.join(lv, 'state.json'), {'version': 1, 'params': [
        {'targets': ['Lotka_Volterra_module/x'], 'min': 1, 'max': 30}]})
    with pytest.raises(WorkflowError, match='"Lotka_Volterra_module/x", which is not a '
                                            'constant of the CellML model'):
        resolve_workflow(load_workflow(_write(lv, _raw([_fit(params_for_id='state.json')]))))
    raw = _raw([{'id': 'hand', 'kind': 'set', 'set': [
        {'parameter': 'Lotka_Volterra_module/y', 'value': 1, 'reason': 'a test'}]}])
    with pytest.raises(WorkflowError, match='is not a constant of the CellML model'):
        resolve_workflow(load_workflow(_write(lv, raw)))
    _values_file(lv, x=3)
    with pytest.raises(WorkflowError, match=r'sets \[\'Lotka_Volterra_module/x\'\], which are '
                                            'not constants'):
        resolve_workflow(load_workflow(_write(lv, _raw([_fit(model=_model('values.json'))],
                                                       target=_model('values.json')))))


@pytest.mark.unit
def test_generation_writes_the_values_into_a_flattened_copy(lv, tmp_path):
    from libcuflynx.calibration_workflow.generate import generate_model
    _values_file(lv, beta=0.25)
    model = CellMLModel(os.path.join(lv, _MODEL), os.path.join(lv, 'values.json'))
    generated = generate_model(model, str(tmp_path / 'gen'), 'step', overrides=[
        {'variable_name': _ALPHA, 'units': 'dimensionless', 'value': '4.5',
         'data_reference': 'a test'}])
    assert generated['model_path'].endswith(os.path.join('step', 'step.cellml'))
    constants = _constants(generated['model_path'])
    assert constants[_ALPHA] == 4.5 and constants['Lotka_Volterra_module/beta'] == 0.25
    assert constants[_GAMMA] == 3.0
    values = _values(generated['parameters_path'])
    assert values['alpha_Lotka_Volterra_module'] == 4.5
    assert values['beta_Lotka_Volterra_module'] == 0.25
    # the user's file is only read
    assert _constants(os.path.join(lv, _MODEL))[_ALPHA] == 5.0


# --------------------------------------------------------------------------------------------
# 2. the file: start_from, uq, emulator_dir, an identifiability step's result_from
# --------------------------------------------------------------------------------------------

def _uq(id='uq', method='mcmc', **extra):
    return dict({'id': id, 'kind': 'uq', 'model': _model(), 'obs_data': _OBS,
                 'params_for_id': _PARAMS,
                 'settings': {'UQ_options': dict(_FAST['UQ_options'], method=method)}}, **extra)


@pytest.mark.unit
def test_the_new_keys_are_read(lv):
    hand = {'id': 'hand', 'kind': 'set', 'set': [
        {'parameter': _ALPHA, 'value': 4.2, 'reason': 'a guess'}]}
    os.makedirs(os.path.join(lv, 'trained'))
    raw = _raw([hand, _fit(start_from='hand'), _uq(result_from='fit'),
                _uq('uq-own', method='laplace', emulator_dir='trained'),
                _fit('refit', priors_from=['uq'], store_distribution=True),
                {'id': 'ia', 'kind': 'identifiability', 'model': _model(), 'obs_data': _OBS,
                 'params_for_id': _PARAMS, 'settings': {'ia_options': {'method': 'Laplace'}}}])
    workflow = load_workflow(_write(lv, raw))
    fit = workflow.step('fit')
    assert fit.start_from == 'hand' and fit.depends_on == ['hand']
    assert fit.sources() == {'fixed_from': [], 'priors_from': [], 'start_from': ['hand']}
    assert workflow.step('uq').result_from == 'fit'
    assert workflow.step('uq-own').emulator_dir == os.path.join(lv, 'trained')
    assert workflow.step('uq-own').depends_on == []
    assert [p.step for p in workflow.step('refit').priors_from] == ['uq']
    assert workflow.step('ia').result_from is None
    order = [s.id for s in workflow.topological_order()]
    assert order.index('hand') < order.index('fit') < order.index('uq') < order.index('refit')
    plan = {s['id']: s for s in plan_workflow(os.path.join(lv, 'workflow.json'))['steps']}
    assert plan['fit']['start_from'] == 'hand'
    assert plan['uq-own']['emulator_dir'] == os.path.join(lv, 'trained')


@pytest.mark.unit
@pytest.mark.parametrize('steps, match', [
    ([_fit('a'), _fit(start_from='a', emulator_from='a')], 'emulator_from'),
    ([_uq(start_from='x')], 'a uq step has no "start_from"'),
    ([_fit('a'), {'id': 'sa', 'kind': 'sensitivity', 'model': _model(), 'obs_data': _OBS,
                  'params_for_id': _PARAMS, 'start_from': 'a',
                  'settings': {'sa_options': {'method': 'local'}}}],
     'a sensitivity step has no "start_from"'),
    ([_fit('a'), _uq(result_from='a', emulator_dir='e', emulator_from='a')],
     '"emulator_dir" and "emulator_from" both'),
    ([_fit(emulator_dir='e', fixed_from=[]),
      {'id': 'em', 'kind': 'emulation', 'model': _model(), 'obs_data': _OBS,
       'params_for_id': _PARAMS, 'emulator_dir': 'e',
       'settings': {'emulator_settings': {}}}], 'emulation step has no "emulator_dir"'),
    ([_uq(method='smc')], "UQ_options.method is 'smc'"),
    ([_fit('a'), _fit(priors_from=['a'], store_distribution=True)],
     'does not store its distribution'),
    ([dict(_fit(store_distribution=True),
           settings={'UQ_options': {'method': 'laplace'}})],
     'add a uq step with "result_from": "fit"'),
    ([{'id': 'sw', 'kind': 'sweep', 'model': _model(), 'settings': {'sim_time': 1},
       'sweep': {'outputs': ['Lotka_Volterra_module/x'], 'parameters': {}}},
      _fit(start_from='sw')], 'start_from" names "sw", a sweep step'),
])
def test_what_the_new_keys_cannot_be_given_is_refused(lv, steps, match):
    with pytest.raises(WorkflowError, match=match):
        load_workflow(_write(lv, _raw(steps)))


@pytest.mark.unit
def test_a_uq_step_needs_its_options(lv):
    settings = {k: v for k, v in _FAST.items() if k != 'UQ_options'}
    with pytest.raises(WorkflowError, match='uq step "uq" needs "UQ_options"'):
        load_workflow(_write(lv, _raw([dict(_uq(), settings={})], settings)))
    raw = _raw([dict(_uq(), settings={})], dict(settings, DEBUG=True,
                                                debug_UQ_options={'method': 'laplace'}))
    from libcuflynx.calibration_workflow.spec import uq_method
    workflow = load_workflow(_write(lv, raw))
    assert uq_method(workflow.step_settings(workflow.step('uq'))) == 'laplace'


@pytest.mark.unit
def test_add_uq_with_no_options_is_mcmc_with_the_defaults(lv, tmp_path):
    from libcuflynx.calibration_workflow.spec import uq_method
    settings = {k: v for k, v in _FAST.items() if k != 'UQ_options'}
    wf = WorkflowBuilder('lv', target=cellml_model(os.path.join(lv, _MODEL)),
                         directory=str(tmp_path / 'wf'), settings=settings)
    wf.add_uq('uq', obs_data=os.path.join(lv, _OBS), params_for_id=os.path.join(lv, _PARAMS))
    workflow = load_workflow(wf.write())
    merged = workflow.step_settings(workflow.step('uq'))
    assert merged['UQ_options'] == {} and uq_method(merged) == 'mcmc'


@pytest.mark.unit
def test_the_builder_writes_the_new_keys(lv, tmp_path):
    os.makedirs(os.path.join(lv, 'trained'))
    wf = WorkflowBuilder('lv', target=cellml_model(os.path.join(lv, _MODEL)),
                         directory=str(tmp_path / 'wf'), settings=_FAST)
    wf.add_set('hand', [{'parameter': _ALPHA, 'value': 4.2, 'reason': 'a guess'}])
    wf.add_calibration('fit', obs_data=os.path.join(lv, _OBS),
                       params_for_id=os.path.join(lv, _PARAMS), start_from='hand')
    wf.add_uq('uq', obs_data=os.path.join(lv, _OBS), params_for_id=os.path.join(lv, _PARAMS),
              uq_options={'method': 'laplace'}, result_from='fit',
              emulator_dir=os.path.join(lv, 'trained'))
    wf.add_identifiability('ia', obs_data=os.path.join(lv, _OBS),
                           params_for_id=os.path.join(lv, _PARAMS),
                           ia_options={'method': 'Laplace'})
    written = _read_json(wf.write())
    rel = os.path.relpath(lv, str(tmp_path / 'wf'))
    assert written['target'] == {'cellml_model_path': os.path.join(rel, _MODEL)}
    steps = {s['id']: s for s in written['steps']}
    assert steps['fit']['start_from'] == 'hand'
    assert steps['fit']['model'] == written['target']
    assert steps['uq']['kind'] == 'uq' and steps['uq']['result_from'] == 'fit'
    assert steps['uq']['emulator_dir'] == os.path.join(rel, 'trained')
    assert steps['uq']['settings'] == {'UQ_options': {'method': 'laplace'}}
    # with no uq_options: an empty UQ_options, MCMC with the defaults
    wf.add_uq('uq-default', obs_data=os.path.join(lv, _OBS),
              params_for_id=os.path.join(lv, _PARAMS), result_from='fit')
    written = _read_json(wf.write())
    default = next(s for s in written['steps'] if s['id'] == 'uq-default')
    assert default['settings'] == {'UQ_options': {}}
    assert 'result_from' not in steps['ia']
    assert WorkflowBuilder.load(wf.write()).to_dict() == written


# --------------------------------------------------------------------------------------------
# 3. runs
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_a_cellml_calibration_recovers_the_values_and_writes_the_calibrated_cellml(lv, tmp_path):
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(_write(lv, _raw([_fit()])), output_dir=out,
                                      write_calibrated=True)
    assert result['complete']
    merged = {r['host_name']: r['value'] for r in result['merged_parameters']}
    for name, value in _TRUE.items():
        assert merged[name] == pytest.approx(value, rel=0.02)

    step = _step(out, 'fit')
    assert step['model'] == {'cellml_model_path': os.path.join(lv, _MODEL)}
    assert 'cellml_model' in step['inputs']
    # the step's own calibrated CellML, and its error vectors (CA#341), in its run directory
    calibrated = step['calibrated_model']
    assert calibrated == os.path.join(out, 'fit', 'fit_calibrated.cellml')
    best = dict(zip([_ALPHA, _GAMMA], step['best_fit']))
    assert _constants(calibrated)[_ALPHA] == pytest.approx(best[_ALPHA], rel=1e-9)
    percent = np.load(step['percent_error_vec'])
    names = list(np.load(step['error_vec_names']))
    assert len(percent) == len(names) == 4 and np.max(np.abs(percent)) < 2.0
    # no held-out data in the obs_data: no validation is recorded
    assert step['validation_results'] is None and step['validation'] == []

    # the target: the merged CellML in _target/, and beside the source the calibrated
    # parameters file and CellML
    target = result['target_model']['model_path']
    assert target.endswith('.cellml') and _constants(target)[_GAMMA] == pytest.approx(
        merged[_GAMMA])
    assert result['calibrated_parameters_file'] == os.path.join(
        lv, 'lotka_volterra_calibrated_parameters.json')
    assert result['calibrated_model_file'] == os.path.join(lv, 'lotka_volterra_calibrated.cellml')
    written = _constants(result['calibrated_model_file'])
    # every value at full precision, as the step's own calibrated CellML has it
    assert written[_ALPHA] == merged[_ALPHA] == _constants(calibrated)[_ALPHA]
    values = _read_json(result['calibrated_parameters_file'])['Lotka_Volterra_module']
    assert values['gamma']['value'] == merged[_GAMMA]
    assert written['Lotka_Volterra_module/beta'] == 0.2
    # one writer (param_id.calibrated_model): each CellML written keeps the source's form,
    # only the calibrated values changed
    with open(os.path.join(lv, _MODEL)) as f:
        source = f.read().splitlines()
    for path in (calibrated, result['calibrated_model_file'], target):
        with open(path) as f:
            lines = f.read().splitlines()
        assert len(lines) == len(source)
        assert sum(a != b for a, b in zip(source, lines)) == 2, path

    # the provenance diagram and the run command's label name the CellML
    assert os.path.isfile(os.path.join(out, 'provenance.svg'))
    with open(os.path.join(out, 'provenance_graph.json')) as f:
        assert 'lotka_volterra.cellml' in f.read()
    # a freshly finished run is current: the graph checks the CellML input with the digest
    # the runner recorded (the file and its imports), as workflow_status does
    from libcuflynx.reporting.provenance import provenance_graph
    graph = provenance_graph(out)
    assert not graph['workflow']['stale']
    assert [n['id'] for n in graph['nodes'] if n['stale']] == []
    cellml_input = [e for n in graph['nodes'] for e in n['inputs'] if e['key'] == 'cellml_model']
    assert cellml_input and {e['status'] for e in cellml_input} == {'current'}
    from libcuflynx.reporting.flowchart import model_line
    assert model_line(step['model']) == 'CellML model `lotka_volterra.cellml`'
    from libcuflynx.scripts.calibration_workflow_run_script import _model_label
    assert _model_label(step['model']) == os.path.join(lv, _MODEL)


@pytest.mark.integration
@pytest.mark.parametrize('model_type, solver', [('python', 'solve_ivp'),
                                                ('casadi_python', 'casadi_integrator')])
def test_a_cellml_model_runs_as_its_generated_python(lv, tmp_path, model_type, solver):
    '''A CellML target with a Python model_type: each step runs the Python generated from its
    copy of the CellML, and the values still go into the CellML.'''
    out = str(tmp_path / 'run')
    settings = dict(copy.deepcopy(_FAST), model_type=model_type)
    del settings['solver_info']
    result = run_calibration_workflow(_write(lv, _raw([_fit()], settings=settings)),
                                      output_dir=out, write_calibrated=True)
    assert result['complete']
    merged = {r['host_name']: r['value'] for r in result['merged_parameters']}
    for name, value in _TRUE.items():
        assert merged[name] == pytest.approx(value, rel=0.02)

    step = _step(out, 'fit')
    assert step['settings']['model_type'] == model_type
    generated = step['generated']
    assert step['model_path'] == os.path.join(out, 'fit', 'model', 'generated_models', 'fit',
                                              'fit.py')
    assert generated['inp_data_dict']['model_type'] == model_type
    # the CellML the Python was generated from has the step's values
    assert generated['flat_model_path'] == step['model_path'][:-len('.py')] + '.cellml'
    assert _constants(generated['flat_model_path'])[_ALPHA] == _OWN[_ALPHA]
    # the calibrated model is the CellML with the best fit
    best = dict(zip([_ALPHA, _GAMMA], step['best_fit']))
    assert step['calibrated_model'] == os.path.join(out, 'fit', 'fit_calibrated.cellml')
    assert _constants(step['calibrated_model'])[_ALPHA] == pytest.approx(best[_ALPHA], rel=1e-9)
    # the target: its Python, generated from the merged CellML beside it
    target = result['target_model']['model_path']
    assert target.endswith('.py')
    assert _constants(target[:-len('.py')] + '.cellml')[_GAMMA] == pytest.approx(merged[_GAMMA])
    assert _constants(result['calibrated_model_file'])[_GAMMA] == pytest.approx(merged[_GAMMA])


@pytest.mark.integration
def test_a_sweep_of_a_cellml_model_runs_its_generated_python(lv, tmp_path):
    '''A sweep with model_type python and no solver named runs solve_ivp on the Python.'''
    out = str(tmp_path / 'run')
    wf = WorkflowBuilder('lv', target=cellml_model(os.path.join(lv, _MODEL)),
                         directory=str(tmp_path / 'wf'),
                         settings={'model_type': 'python', 'dt': 0.01, 'sim_time': 1.0})
    wf.add_sweep('sweep', parameters={_ALPHA: {'values': [4.0, 5.0]}},
                 outputs=['Lotka_Volterra_module/x'])
    result = wf.run(output_dir=out)
    assert result['complete']
    step = _step(out, 'sweep')
    assert step['model_path'].endswith('sweep.py')


@pytest.mark.integration
def test_a_calibration_validates_its_best_fit_against_held_out_data(lv, tmp_path):
    '''A prediction item with a value is held-out data: the step writes
    validation_results.json, as cuflynx-plot does, and records each item's statistics.'''
    obs = _read_json(os.path.join(lv, _OBS))
    held = {key: value for key, value in obs['data_items'][1].items()
            if key not in ('cost_type', 'plot_type', 'weight')}
    held['data_item_name'] = 'y_max_held'
    obs['prediction_items'] = [held]
    _write_json(os.path.join(lv, _OBS), obs)
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(_write(lv, _raw([_fit()])), output_dir=out)
    assert result['complete']

    step = _step(out, 'fit')
    assert step['validation_results'] == os.path.join(step['param_id_output_dir'],
                                                      'validation_results.json')
    (item,) = step['validation']
    assert item['data_item_name'] == 'y_max_held' and item['n_points'] == 1
    assert item['operation'] == 'max' and item['data_type'] == 'constant'
    # the fit recovers the values the data was made at, so the held-out value is met
    assert item['rmse'] < 0.01 * held['value'] and item['mean_abs_z'] < 1.0
    assert 't' not in item
    (written,) = _read_json(step['validation_results'])['items']
    assert written['model'][0] == pytest.approx(held['value'], rel=0.01)
    assert written['rmse'] == item['rmse']


@pytest.mark.integration
def test_a_modifier_entry_gives_its_parameters_values_not_theta(lv, tmp_path):
    '''A params_for_id modifier entry is calibrated as theta. The step records, merges and
    writes its parameters' values, modifier(theta, baseline), and records theta beside them.'''
    params = _read_json(os.path.join(lv, _PARAMS))
    params['params'][0] = {'name': 'alpha_scale', 'modifies': [_ALPHA], 'modifier': 'scale',
                           'min': 0.2, 'max': 1.4}
    _write_json(os.path.join(lv, _PARAMS), params)
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(_write(lv, _raw([_fit()])), output_dir=out,
                                      write_calibrated=True)
    assert result['complete']

    step = _step(out, 'fit')
    theta = step['best_fit'][0]
    # alpha is 5.0 in the CellML and the data was made at 4.0
    assert theta == pytest.approx(_TRUE[_ALPHA] / _OWN[_ALPHA], rel=0.02)
    calibrated = {f"{c['vessel']}/{c['param']}": c['value'] for c in step['calibrated']}
    assert calibrated[_ALPHA] == pytest.approx(theta * _OWN[_ALPHA], rel=1e-12)
    assert calibrated[_GAMMA] == step['best_fit'][1]
    assert step['modifiers'] == [{'name': 'alpha_scale', 'modifier': 'scale',
                                  'targets': [_ALPHA], 'theta': theta,
                                  'baselines': [_OWN[_ALPHA]],
                                  'values': [calibrated[_ALPHA]]}]
    # the merged target and both calibrated CellML files have alpha, not theta
    merged = {r['host_name']: r['value'] for r in result['merged_parameters']}
    assert merged[_ALPHA] == calibrated[_ALPHA]
    assert _constants(step['calibrated_model'])[_ALPHA] == pytest.approx(calibrated[_ALPHA])
    assert _constants(result['calibrated_model_file'])[_ALPHA] == pytest.approx(
        calibrated[_ALPHA])


@pytest.mark.integration
def test_start_from_is_the_first_point_the_optimiser_evaluates(lv, tmp_path):
    hand = {'id': 'hand', 'kind': 'set', 'set': [
        {'parameter': _ALPHA, 'value': 4.2, 'reason': 'near the fit'},
        {'parameter': _GAMMA, 'value': 2.7, 'reason': 'near the fit'}]}
    # a gradient method: its x0 is the model's values (CMA-ES takes them too; the genetic
    # algorithm draws its first population at random)
    settings = dict(_FAST, param_id_method='sp_minimize', do_ad=True,
                    optimiser_options={'num_calls_to_function': 3, 'cost_type': 'gaussian_MLE'})
    out = str(tmp_path / 'run')
    run_calibration_workflow(_write(lv, _raw([hand, _fit(start_from='hand')], settings)),
                             output_dir=out)
    step = _step(out, 'fit')
    assert step['start_from'] == {'step': 'hand', 'values': [
        dict(r, from_step='hand') for r in step['start_from']['values']]}
    assert [r['value'] for r in step['start_from']['values']] == [4.2, 2.7]
    # started there, not fixed there
    assert step['fixed'] == []
    assert _values(step['generated']['parameters_path'])['alpha_Lotka_Volterra_module'] == 4.2
    from libcuflynx.param_id.run_history import read_run_history
    history = read_run_history(step['param_id_output_dir'])
    first = np.asarray(history['param_history'])[0]
    assert first == pytest.approx([4.2, 2.7], rel=1e-6)

    # without start_from the optimiser starts at the CellML's own values
    out = str(tmp_path / 'own')
    run_calibration_workflow(_write(lv, _raw([_fit()], settings)), output_dir=out)
    history = read_run_history(_step(out, 'fit')['param_id_output_dir'])
    assert np.asarray(history['param_history'])[0] == pytest.approx([5.0, 3.0], rel=1e-6)


def _modifier_params(lv):
    """The fixture's params_for_id with alpha calibrated through a scale modifier."""
    params = _read_json(os.path.join(lv, _PARAMS))
    params['params'][0] = {'name': 'alpha_scale', 'modifies': [_ALPHA], 'modifier': 'scale',
                           'min': 0.2, 'max': 1.4}
    return _write_json(os.path.join(lv, _PARAMS), params)


@pytest.mark.unit
def test_a_theta_entry_is_for_a_point_set_step(lv, tmp_path):
    theta = set_theta('alpha_scale', 0.9, reason='slider position')
    assert theta == {'entry': 'alpha_scale', 'theta': 0.9, 'reason': 'slider position'}
    wf = WorkflowBuilder('lv', target=cellml_model(os.path.join(lv, _MODEL)),
                         directory=str(tmp_path / 'wf'), settings=copy.deepcopy(_FAST))
    wf.add_set('sliders', [theta, set_value(_GAMMA, 2.8, reason='slider position')],
               point=True)
    step = load_workflow(wf.write()).step('sliders')
    assert step.set_thetas == [theta]
    assert [entry['parameter'] for entry in step.set_values] == [_GAMMA]

    # a theta is not a value of the target: the step must be a point
    with pytest.raises(WorkflowError, match='needs "point": true'):
        load_workflow(_write(lv, _raw([{'id': 'sliders', 'kind': 'set', 'set': [theta]},
                                       _fit(start_from='sliders')])))
    for entry, match in [
            (dict(theta, parameter=_ALPHA), 'entry'),
            (dict(theta, value=1.0), 'entry'),
            ({'entry': 'alpha_scale', 'reason': 'no theta'}, 'entry')]:
        with pytest.raises(Exception, match=match):
            load_workflow(_write(lv, _raw([{'id': 'sliders', 'kind': 'set', 'point': True,
                                            'set': [entry]}, _fit(start_from='sliders')])))


@pytest.mark.integration
def test_a_modifier_entry_starts_and_is_analysed_at_a_theta(lv, tmp_path):
    """A modifier entry's point is its theta. A step given one (a point set step's
    set_theta, or a calibration's fitted theta) starts or analyses there, and the parameters
    the entry modifies keep their values in the step's model, so theta means the same."""
    _modifier_params(lv)
    sliders = {'id': 'sliders', 'kind': 'set', 'point': True, 'set': [
        set_theta('alpha_scale', 0.9, reason='slider position'),
        set_value(_GAMMA, 2.8, reason='slider position')]}
    quick = {'param_id_method': 'sp_minimize', 'do_ad': True,
             'optimiser_options': {'num_calls_to_function': 3, 'cost_type': 'gaussian_MLE'}}
    analysis = {'model': _model(), 'obs_data': _OBS, 'params_for_id': _PARAMS}
    raw = _raw([
        sliders,
        _fit('quick', start_from='sliders', settings=quick),
        _fit(),
        dict(analysis, id='ia-sliders', kind='identifiability', result_from='sliders'),
        dict(analysis, id='ia-fit', kind='identifiability', result_from='fit'),
        dict(analysis, id='local', kind='sensitivity', result_from='sliders',
             settings={'sa_options': {'method': 'local'}}),
        dict(analysis, id='ia-own', kind='identifiability')],
        dict(_FAST, ia_options={'method': 'Laplace'}))
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(_write(lv, raw), output_dir=out)
    assert result['complete']
    alpha = 'alpha_Lotka_Volterra_module'

    def model_values(sid):
        return _values(_step(out, sid)['generated']['parameters_path'])

    assert _step(out, 'sliders')['thetas'] == [sliders['set'][0]]

    # start_from: the first point is the theta, about alpha as the CellML has it
    step = _step(out, 'quick')
    assert step['start_from']['thetas'] == {'alpha_scale': 0.9}
    assert [r['value'] for r in step['start_from']['values']] == [2.8]
    assert model_values('quick')[alpha] == _OWN[_ALPHA]
    from libcuflynx.param_id.run_history import read_run_history
    history = read_run_history(step['param_id_output_dir'])
    assert np.asarray(history['param_history'])[0] == pytest.approx([0.9, 2.8], rel=1e-6)

    # result_from a set step: analysed at the theta
    step = _step(out, 'ia-sliders')
    assert step['at'] == [0.9, 2.8] and model_values('ia-sliders')[alpha] == _OWN[_ALPHA]
    local = _step(out, 'local')
    assert local['nominal']['values'] == [0.9, 2.8]
    assert model_values('local')[alpha] == _OWN[_ALPHA]

    # result_from a calibration: at the theta it fitted, not at theta 1 about the fitted alpha
    fit = _step(out, 'fit')
    step = _step(out, 'ia-fit')
    assert step['at'] == pytest.approx(fit['best_fit'], rel=1e-12)
    assert step['at'][0] == pytest.approx(_TRUE[_ALPHA] / _OWN[_ALPHA], rel=0.02)
    assert model_values('ia-fit')[alpha] == _OWN[_ALPHA]
    # the Laplace mean is the point it was given
    assert step['laplace']['mean'] == pytest.approx(step['at'], rel=1e-9)

    # no point given: the theta that leaves alpha as the model has it
    assert _step(out, 'ia-own')['at'] == [1.0, _OWN[_GAMMA]]


@pytest.mark.integration
def test_a_theta_for_an_entry_the_step_does_not_have_is_refused(lv, tmp_path):
    sliders = {'id': 'sliders', 'kind': 'set', 'point': True, 'set': [
        set_theta('alpha_scale', 0.9, reason='slider position')]}
    out = str(tmp_path / 'run')
    with pytest.raises(Exception, match='no modifier entry of that name'):
        run_calibration_workflow(_write(lv, _raw([sliders, _fit(start_from='sliders')])),
                                 output_dir=out)


@pytest.mark.integration
def test_mcmc_at_a_result_runs_no_optimiser_and_stores_a_chain(lv, tmp_path, capfd):
    out = str(tmp_path / 'run')
    hand = {'id': 'hand', 'kind': 'set', 'set': [
        {'parameter': _ALPHA, 'value': 4.0, 'reason': 'the truth'},
        {'parameter': _GAMMA, 'value': 2.5, 'reason': 'the truth'}]}
    off = _uq('uq-off', result_from='hand')
    off['settings']['UQ_options']['posterior_predictive'] = False
    run_calibration_workflow(_write(lv, _raw([hand, _uq(result_from='hand'), off])),
                             output_dir=out)
    # an MCMC optimises nothing, so it announces no optimiser defaults
    assert 'No optimiser options provided' not in capfd.readouterr().out
    uq = _step(out, 'uq')
    assert uq['method'] == 'mcmc' and uq['at'] == [4.0, 2.5] and uq['at_source'] == 'hand'
    # the values it analyses are its point, not fixed values
    assert uq['parameters'] == [_ALPHA, _GAMMA] and uq['fixed'] == []
    # no calibration ran: no best fit, no optimiser's run directory
    assert 'best_fit' not in uq and uq['calibrated'] == []
    runs = [name for name in os.listdir(uq['uq_output_dir'])
            if os.path.isdir(os.path.join(uq['uq_output_dir'], name)) and name != 'inputs']
    assert runs == ['mcmc_uq_lv_obs_data']
    posterior = StoredDistribution.load(os.path.join(out, 'uq'))
    assert posterior.names == ['alpha_Lotka_Volterra_module', 'gamma_Lotka_Volterra_module']
    assert np.median(posterior.samples, axis=0) == pytest.approx([4.0, 2.5], rel=0.05)
    assert posterior.chain is not None
    # the posterior predictive check, from the chain, in the step directory
    predictive = uq['posterior_predictive_summary']
    assert predictive['drawn_from'] == 'chain' and predictive['num_samples'] == 12
    assert uq['posterior_predictive_dir'] == os.path.join(out, 'uq', 'posterior_predictive')
    draws = np.load(uq['posterior_predictive_draws'], allow_pickle=True)
    assert draws['thetas'].shape == (12, 2) and draws['predictions'].shape[0] == 12
    assert os.path.isfile(uq['posterior_predictive_coverage'])
    assert set(predictive['levels']) == {'0.8', '0.95'}
    off = _step(out, 'uq-off')
    assert 'posterior_predictive_dir' not in off
    assert not os.path.exists(os.path.join(out, 'uq-off', 'posterior_predictive'))


@pytest.mark.integration
def test_laplace_samples_are_the_prior_of_a_later_calibration(lv, tmp_path):
    out = str(tmp_path / 'run')
    settings = dict(_FAST, ia_options={'method': 'Laplace'})
    # the refit calibrates beta, with alpha and gamma again under the Laplace prior
    _write_json(os.path.join(lv, 'beta.json'), {'version': 1, 'params': [
        {'targets': ['Lotka_Volterra_module/beta'], 'min': 0.1, 'max': 0.4}]})
    raw = _raw([_fit(), _uq('laplace', method='laplace', result_from='fit'),
                _fit('refit', params_for_id='beta.json', priors_from=['laplace'],
                     store_distribution=True)], settings)
    raw['steps'][1]['settings']['UQ_options']['num_samples'] = 500
    result = run_calibration_workflow(_write(lv, raw), output_dir=out)
    assert result['complete']
    laplace = _step(out, 'laplace')
    fit = _step(out, 'fit')
    assert laplace['at'] == pytest.approx(fit['best_fit'])
    assert laplace['laplace']['mean'] == pytest.approx(fit['best_fit'])
    assert all(0 < s < 0.5 for s in laplace['laplace']['std'])
    samples = np.load(os.path.join(out, 'laplace', 'distribution', 'samples.npy'))
    assert samples.shape == (500 - laplace['laplace']['samples_outside_bounds'], 2)
    # the posterior predictive check, from those samples
    assert laplace['posterior_predictive_summary']['drawn_from'] == 'samples'
    draws = np.load(laplace['posterior_predictive_draws'], allow_pickle=True)
    assert draws['thetas'].shape == (12, 2)
    assert all(any(np.array_equal(t, s) for s in samples) for t in draws['thetas'])
    assert np.mean(samples, axis=0) == pytest.approx(fit['best_fit'], rel=0.02)
    refit = _step(out, 'refit')
    assert refit['priors'][0]['step'] == 'laplace'
    assert refit['priors'][0]['parameters'] == ['alpha_Lotka_Volterra_module',
                                                'gamma_Lotka_Volterra_module']


@pytest.mark.integration
def test_do_uq_with_the_laplace_method_runs_it_after_the_fit(lv, tmp_path):
    from libcuflynx.calibration_workflow.runner import _get_comm, _Run
    from libcuflynx.scripts.param_id_run_script import run_param_id
    settings = dict(_FAST, optimiser_options=dict(_FAST['optimiser_options'],
                                                  num_calls_to_function=60))
    path = _write(lv, _raw([_fit()], settings))
    out = str(tmp_path / 'run')
    run_calibration_workflow(path, output_dir=out)
    # the step's own run, again, with do_uq and the Laplace method
    workflow = load_workflow(path)
    step = _step(out, 'fit')
    run = _Run(resolve_workflow(workflow), out, _get_comm(None), None)
    inp = run._inp(workflow.step('fit'), step['settings'], step['generated'],
                   step['params_for_id_used'])
    again = tmp_path / 'again' / 'param_id'
    again.parent.mkdir()
    inp.update(do_uq=True, UQ_options={'method': 'laplace'}, write_cuflynx_study=False,
               param_id_output_dir=str(again))
    outcome = run_param_id(inp)
    assert outcome['mcmc'] is None
    mean = np.load(str(tmp_path / 'again' / 'fit_laplace_mean.npy'))
    covariance = np.load(str(tmp_path / 'again' / 'fit_laplace_covariance.npy'))
    assert mean == pytest.approx(outcome['best_param_vals'])
    assert covariance.shape == (2, 2) and np.all(np.diag(covariance) > 0)


@pytest.mark.integration
def test_local_sensitivity_and_identifiability_record_their_point(lv, tmp_path, capfd):
    out = str(tmp_path / 'run')
    settings = dict(_FAST, ia_options={'method': 'Laplace'})
    raw = _raw([_fit(), {'id': 'local', 'kind': 'sensitivity', 'model': _model(),
                         'obs_data': _OBS, 'params_for_id': _PARAMS, 'result_from': 'fit',
                         'settings': {'sa_options': {'method': 'local'}}},
                {'id': 'local-own', 'kind': 'sensitivity', 'model': _model(),
                 'obs_data': _OBS, 'params_for_id': _PARAMS,
                 'settings': {'sa_options': {'method': 'local'}}},
                {'id': 'ia-own', 'kind': 'identifiability', 'model': _model(),
                 'obs_data': _OBS, 'params_for_id': _PARAMS}], settings)
    run_calibration_workflow(_write(lv, raw), output_dir=out)
    # num_samples and sample_type are Sobol's: a local analysis announces no defaults of them
    printed = capfd.readouterr().out
    assert 'No num_samples specified' not in printed and 'No sample_type' not in printed
    best = _step(out, 'fit')['best_fit']
    local = _step(out, 'local')['nominal']
    # fit's values of what local analyses are its nominal point, not fixed values
    assert _step(out, 'local')['fixed'] == []
    assert local['source'] == 'fit' and local['parameters'] == [_ALPHA, _GAMMA]
    assert local['values'] == pytest.approx(best)
    own = _step(out, 'local-own')['nominal']
    assert own == {'source': 'model', 'parameters': [_ALPHA, _GAMMA], 'values': [5.0, 3.0]}
    # the signed coefficients, rows labelled as the Sobol outputs are
    import pandas as pd
    signed = _step(out, 'local')['local_sensitivity_signed']
    assert signed == os.path.join(out, 'local', 'sensitivity', 'local_sensitivity_signed.csv')
    table = pd.read_csv(signed, index_col='output')
    assert list(table.columns) == [_ALPHA, _GAMMA]
    assert all(' (Exp' in label for label in table.index)
    ia = _step(out, 'ia-own')
    assert ia['at'] == [5.0, 3.0] and ia['at_source'] == 'model'


@pytest.mark.integration
def test_an_identifiability_step_runs_a_profile_likelihood_and_records_it(lv, tmp_path):
    '''ia_options.method profile_likelihood: the step profiles each parameter about its
    result_from point and records each profile's file and confidence interval.'''
    out = str(tmp_path / 'run')
    wf = WorkflowBuilder('lv', target=cellml_model(os.path.join(lv, _MODEL)),
                         directory=str(tmp_path / 'wf'), settings=copy.deepcopy(_FAST))
    hand = wf.add_set('hand', [set_value(_ALPHA, 4.0, reason='the true value'),
                               set_value(_GAMMA, 2.5, reason='the true value')], point=True)
    wf.add_identifiability(
        'profile', obs_data=os.path.join(lv, _OBS), params_for_id=os.path.join(lv, _PARAMS),
        result_from=hand,
        ia_options={'method': 'profile_likelihood', 'num_points': 5, 'range_factor': 0.05,
                    'optimiser_options': {'num_calls_to_function': 40, 'max_patience': 10}})
    assert wf.run(output_dir=out)['complete']

    step = _step(out, 'profile')
    assert step['method'] == 'profile_likelihood'
    assert step['at'] == [4.0, 2.5] and step['at_source'] == 'hand'
    assert 'laplace' not in step
    record = step['profile_likelihood']
    assert record['num_points'] == 5 and record['range_factor'] == 0.05
    assert record['dir'].startswith(os.path.join(out, 'profile'))
    assert os.path.isfile(record['summary'])
    assert [p['parameter'] for p in record['profiles']] == [_ALPHA, _GAMMA]
    for profile, best in zip(record['profiles'], [4.0, 2.5]):
        written = _read_json(profile['file'])
        assert profile['best_val'] == written['best_val'] == best
        assert len(written['param_values']) == len(written['costs']) == 5
        assert profile['ci_95'] == written['ci_95']
    # the manifest lists the profiles' directory as the step's output
    assert step['profile_likelihood_dir'] == record['dir']


@pytest.mark.integration
def test_a_step_runs_on_an_emulator_trained_outside_the_workflow(lv, tmp_path):
    pytest.importorskip('autoemulate')
    from libcuflynx.external_testing.workflow_library import EMULATOR_SETTINGS
    settings = dict(_FAST, emulator_settings=EMULATOR_SETTINGS)
    first = str(tmp_path / 'first')
    emulate = {'id': 'emulate', 'kind': 'emulation', 'model': _model(), 'obs_data': _OBS,
               'params_for_id': _PARAMS}
    run_calibration_workflow(_write(lv, _raw([emulate], settings), 'train.json'),
                             output_dir=first)
    trained = _step(first, 'emulate')['emulator_dir']
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(
        _write(lv, _raw([_fit(emulator_dir=trained)], settings)), output_dir=out)
    step = _step(out, 'fit')
    assert step['inputs']['emulator']['path'] == trained
    merged = {r['host_name']: r['value'] for r in result['merged_parameters']}
    assert merged[_ALPHA] == pytest.approx(_TRUE[_ALPHA], rel=0.1)


@pytest.mark.unit
def test_an_emulator_dir_is_an_input_and_is_packed(lv, tmp_path):
    from libcuflynx.calibration_workflow.runner import _get_comm, _Run
    trained = os.path.join(lv, 'trained')
    os.makedirs(trained)
    _write_json(os.path.join(trained, 'emulator_metadata.json'), {'a': 1})
    path = _write(lv, _raw([_fit(emulator_dir='trained')]))
    run = _Run(resolve_workflow(load_workflow(path)), str(tmp_path / 'run'), _get_comm(None),
               None)
    before = run.inputs('fit')['emulator']
    assert before['path'] == trained
    inp = run._inp(load_workflow(path).step('fit'), _FAST,
                   {'resources_dir': 'r', 'generated_models_dir': 'g'}, 'p.json')
    assert inp['use_emulator'] is True and inp['emulator_settings']['emulator_dir'] == trained
    _write_json(os.path.join(trained, 'emulator_metadata.json'), {'a': 2})
    assert run.inputs('fit')['emulator']['sha256'] != before['sha256']

    packed = pack_workflow(path, str(tmp_path / 'pack'))
    raw = _read_json(packed['workflow'])
    assert raw['steps'][0]['emulator_dir'] == os.path.join('inputs', 'fit', 'trained')
    assert os.path.isfile(os.path.join(str(tmp_path / 'pack'), 'inputs', 'fit', 'trained',
                                       'emulator_metadata.json'))


@pytest.mark.unit
def test_a_cellml_model_is_packed_with_its_imports_and_parameters(lv, tmp_path):
    # the model imports its parameters from a file in a subdirectory
    with open(os.path.join(lv, _MODEL)) as f:
        text = f.read()
    os.makedirs(os.path.join(lv, 'units'))
    with open(os.path.join(lv, 'units', 'units.cellml'), 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<model xmlns="http://www.cellml.org/'
                'cellml/2.0#" name="units"><units name="per_second"><unit units="second" '
                'exponent="-1"/></units></model>\n')
    text = text.replace('<component name="environment">',
                        '<import xmlns:xlink="http://www.w3.org/1999/xlink" '
                        'xlink:href="units/units.cellml"><units units_ref="per_second" '
                        'name="per_second"/></import>\n  <component name="environment">')
    with open(os.path.join(lv, _MODEL), 'w') as f:
        f.write(text)
    _values_file(lv, beta=0.25)
    path = _write(lv, _raw([dict(_fit(), model=_model('values.json'))],
                           target=_model('values.json')))
    dest = str(tmp_path / 'pack')
    packed = pack_workflow(path, dest)
    raw = _read_json(packed['workflow'])
    # in inputs/_target itself: the name of the folder it came from means nothing in the pack
    home = os.path.join('inputs', '_target')
    assert raw['target'] == {'cellml_model_path': os.path.join(home, _MODEL),
                             'parameters': os.path.join('inputs', '_target', 'values.json')}
    assert raw['steps'][0]['model'] == raw['target']
    assert os.path.isfile(os.path.join(dest, home, 'units', 'units.cellml'))
    # the pack resolves alone: the imports are where the CellML names them
    reloaded = resolve_workflow(load_workflow(packed['workflow']))
    assert reloaded.target.parameters['Lotka_Volterra_module/beta'] == 0.25


@pytest.mark.unit
def test_the_command_names_every_step_kind_and_every_file_it_writes(monkeypatch, capsys):
    import libcuflynx.calibration_workflow as cw
    from libcuflynx.scripts.calibration_workflow_run_script import build_parser, run
    parser = build_parser()
    assert all(word in parser.description for word in (
        'calibration', 'sensitivity', 'reduction', 'emulator', 'identifiability', '(uq)',
        'set step', 'sweep'))
    assert '<model>_calibrated.cellml' in ' '.join(parser.format_help().split())
    result = {'workflow_name': 'lv', 'complete': True, 'merged_parameters': [],
              'calibrated_parameters_file': '/x/lv_calibrated_parameters.json',
              'calibrated_model_file': '/x/lotka_volterra_calibrated.cellml'}
    monkeypatch.setattr(cw, 'run_calibration_workflow', lambda *a, **k: result)
    run(parser.parse_args(['workflow.json', '--write-calibrated']))
    printed = capsys.readouterr().out
    assert '[workflow] wrote /x/lv_calibrated_parameters.json' in printed
    assert '[workflow] wrote /x/lotka_volterra_calibrated.cellml' in printed


@pytest.mark.unit
def test_every_step_kind_has_its_outputs_and_reporting_words():
    from libcuflynx.calibration_workflow.runner import STEP_OUTPUTS
    from libcuflynx.calibration_workflow.spec import STEP_KINDS
    from libcuflynx.reporting.methods import KIND_NAMES
    assert set(STEP_OUTPUTS) == set(STEP_KINDS) == set(KIND_NAMES)
