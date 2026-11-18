"""External Python models in calibration workflows (``{"external_model_path": ...}``).

A workflow step, and its target, may be an ``external_python`` model: a ``.py`` defining
``SIM_HELPER``. It is never generated: each step runs the file with its values -- the class
defaults with the step's fixed values over them -- in an ``external_parameters_file``, and the
merged result is ``<prefix>_calibrated_parameters.json`` beside the ``.py``.

The runs use the shipped 1D heat equation (``funcs_user/example_model_external/``, numpy only,
about 1 ms a run), whose obs_data was made at k = 0.25, u_D = 0.1 (the defaults are 0.4 and 0.0),
and its shipped workflow ``heat1d_calibration_workflow.json``.
"""
import copy
import glob
import json
import os
import shutil
import textwrap

import numpy as np
import pytest

from libcuflynx.calibration_workflow import (ExternalModel, StoredDistribution, WorkflowBuilder,
                                             WorkflowError, external_model, load_workflow,
                                             pack_workflow, plan_workflow, resolve_workflow,
                                             run_calibration_workflow)
from libcuflynx.calibration_workflow.spec import parse_workflow
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.utilities.parameters_file import read_parameter_rows

_EXAMPLE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'funcs_user', 'example_model_external'))
_WORKFLOW = 'heat1d_calibration_workflow.json'
_TRUE_K, _TRUE_U_D = 0.25, 0.1
_DT = 0.005
#: the probe means of heat1d_obs_data.json, and their std
_OBS, _STD = np.array([0.25338190, 0.23883291, 0.14911117]), 0.005
_OUTPUTS = ['heat/T_p1', 'heat/T_p2', 'heat/T_p3']
_SOLVER_INFO = {'solver': 'external', 'method': 'external'}
#: a calibration of k small enough for a test (the shipped workflow's, with fewer calls)
_FAST = {'dt': _DT, 'param_id_method': 'CMA-ES',
         'optimiser_options': {'num_calls_to_function': 200, 'max_patience': 40, 'seed': 1,
                               'cost_type': 'gaussian_MLE'},
         'UQ_options': {'num_steps': 150, 'num_walkers': 8, 'burn_in': 0.5}}


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
    return path


@pytest.fixture
def example(tmp_path):
    '''A copy of the example directory, so runs write nothing into the repository.'''
    directory = str(tmp_path / 'heat1d')
    shutil.copytree(_EXAMPLE_DIR, directory,
                    ignore=shutil.ignore_patterns('__pycache__', '*_calibrated_parameters.json'))
    return directory


def _raw(example, steps=None, settings=None):
    '''The shipped workflow, with only ``steps`` (ids, in order) and ``settings``.'''
    raw = _read_json(os.path.join(example, _WORKFLOW))
    raw['settings'] = copy.deepcopy(_FAST if settings is None else settings)
    if steps is not None:
        by_id = {s['id']: s for s in raw['steps']}
        raw['steps'] = [by_id[i] for i in steps]
    return raw


def _write(example, raw, name='workflow.json'):
    return _write_json(os.path.join(example, name), raw)


def _fit_k(example, method='CMA-ES', store=False, **settings):
    '''The shipped workflow's boundary and fit-k steps, fit-k without MCMC unless ``store``.'''
    raw = _raw(example, ['boundary', 'fit-k'], dict(_FAST, param_id_method=method, **settings))
    raw['steps'][1]['store_distribution'] = store
    return raw


def _helper(model_path, parameters_file=None, sim_time=0.5):
    solver_info = dict(_SOLVER_INFO)
    if parameters_file:
        solver_info['external_parameters_file'] = parameters_file
    return get_simulation_helper(model_path=model_path, solver='external',
                                 model_type='external_python', dt=_DT, sim_time=sim_time,
                                 solver_info=solver_info, pre_time=0.0)


def _probe_means(helper):
    assert helper.run()
    return np.array([np.mean(trace) for trace in helper.get_results(_OUTPUTS, flatten=True)])


def _values(path):
    return {r['variable_name']: float(r['value']) for r in read_parameter_rows(path)}


# --------------------------------------------------------------------------------------------
# 1. the file
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_an_external_model_is_a_step_model_and_the_target(example):
    workflow = load_workflow(os.path.join(example, _WORKFLOW))
    model = ExternalModel(os.path.join(example, 'heat1d_model.py'))
    assert workflow.target == model
    assert workflow.target.kind == 'external' and workflow.target.prefix == 'heat1d'
    assert [s.model for s in workflow.steps] == [None, model, model]
    assert model.as_dict() == {'external_model_path': model.path}


@pytest.mark.unit
@pytest.mark.parametrize('change, match', [
    # an external step needs the same file as its target
    (lambda raw: raw.update(target='other_module_array.csv'), 'the target must be that file'),
    (lambda raw: raw['steps'][1].update(model='other_module_array.csv'),
     'name another model'),
    (lambda raw: raw['steps'][1].update(vessel_map={'heat': 'other'}), 'vessel_map'),
    (lambda raw: raw['steps'][1].update(submodule='heat'), 'submodule'),
    (lambda raw: raw['steps'][1].update(vessel='heat'), 'vessel'),
    (lambda raw: raw['steps'][1].update(model={'select': ['heat']}), 'module-array target'),
    # the schema: only the path, and a .py
    (lambda raw: raw['steps'][1].update(model={'external_model_path': 'heat1d_model.py',
                                               'parameters': 'p.json'}), 'parameters'),
    (lambda raw: raw.update(target={'external_model_path': 'heat1d_model.json'}),
     'external_model_path'),
])
def test_what_an_external_model_cannot_be_given_is_refused(example, change, match):
    raw = _raw(example)
    change(raw)
    with pytest.raises(WorkflowError, match=match):
        parse_workflow(raw, path=os.path.join(example, 'workflow.json'))


# --------------------------------------------------------------------------------------------
# 2. resolving
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_components_are_the_vessels_and_the_names_its_parameters(example):
    resolved = resolve_workflow(load_workflow(os.path.join(example, _WORKFLOW)))
    assert resolved.target.kind == 'external'
    assert resolved.target.vessels == {'heat'}
    assert resolved.target.parameters == {'heat/k': 0.4, 'heat/u_D': 0.0}
    fit = resolved.steps['fit-k']
    assert [(p.vessel, p.param) for p in fit.parameters] == [('heat', 'k')]
    assert fit.target_model_name(fit.parameters[0]) == fit.parameters[0].model_name
    assert fit.target_parameterisation_name(fit.parameters[0]) == 'heat/k'
    assert resolved.target.file('calibrated_parameters.json') == os.path.join(
        example, 'heat1d_calibrated_parameters.json')


@pytest.mark.unit
def test_a_name_the_model_does_not_declare_is_refused(example):
    _write_json(os.path.join(example, 'bad_params_for_id.json'), {'version': 1, 'params': [
        {'targets': ['heat/kappa'], 'min': 0.1, 'max': 1.0}]})
    raw = _raw(example)
    raw['steps'][1]['params_for_id'] = 'bad_params_for_id.json'
    with pytest.raises(WorkflowError, match='"heat/kappa", which the external model .* does '
                                            'not declare'):
        resolve_workflow(load_workflow(_write(example, raw)))
    raw = _raw(example)
    raw['steps'][0]['set'][0]['parameter'] = 'heat/u_E'
    with pytest.raises(WorkflowError, match='not declared by the external model'):
        resolve_workflow(load_workflow(_write(example, raw)))


# --------------------------------------------------------------------------------------------
# 3. the helper's external_parameters_file
# --------------------------------------------------------------------------------------------

_TINY = '''
    import numpy as np

    class Tiny:
        parameters = {"tiny/a": 2.0, "tiny/b": 3.0}
        output_names = ["tiny/y"]

        def init_solver(self, config):
            self.vals = dict(self.parameters)

        def update_times(self, dt, start_time, sim_time, pre_time):
            self.t = start_time + np.arange(int(pre_time / dt) + int(sim_time / dt) + 1) * dt

        def set_param_vals(self, param_dict):
            self.vals.update(param_dict)

        def run(self):
            self.y = self.vals["tiny/a"] * self.t + self.vals["tiny/b"]
            return True

        def get_results(self):
            return {"tiny/y": self.y}

    SIM_HELPER = Tiny
'''


@pytest.mark.unit
def test_the_parameters_file_replaces_the_defaults_and_survives_a_reset(tmp_path):
    model = tmp_path / 'tiny_model.py'
    model.write_text(textwrap.dedent(_TINY))
    values = _write_json(str(tmp_path / 'values.json'), {
        'schema_version': 1, 'tiny': {'b': {'value': 7.5, 'units': '', 'data_reference': ''}}})
    helper = get_simulation_helper(model_path=str(model), solver='external',
                                   model_type='external_python', dt=0.1, sim_time=1.0,
                                   solver_info=dict(_SOLVER_INFO,
                                                    external_parameters_file=values))
    # the user class runs with the file's value from the start
    assert helper.run()
    assert helper.get_results(['tiny/y'], flatten=True)[0][0] == pytest.approx(7.5)
    assert helper.get_init_param_vals(['tiny/a', 'tiny/b']) == [2.0, 7.5]
    assert helper.get_default_param_vals(['tiny/b']) == [7.5]
    helper.set_param_vals(['tiny/b'], [1.0])
    helper.reset_and_clear()
    assert helper.get_init_param_vals(['tiny/b']) == [7.5]
    assert helper.run()
    assert helper.get_results(['tiny/y'], flatten=True)[0][0] == pytest.approx(7.5)

    unknown = _write_json(str(tmp_path / 'unknown.json'), {
        'schema_version': 1, 'tiny': {'c': {'value': 1, 'units': '', 'data_reference': ''}}})
    with pytest.raises(ValueError, match="'tiny/c' is not declared"):
        get_simulation_helper(model_path=str(model), solver='external',
                              model_type='external_python', dt=0.1, sim_time=1.0,
                              solver_info=dict(_SOLVER_INFO, external_parameters_file=unknown))


@pytest.mark.unit
def test_the_user_input_reaches_the_helper_through_solver_info(base_user_inputs, tmp_path):
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser, solver_info_fields
    assert 'external_parameters_file' in {f['name'] for f in solver_info_fields('external')}
    values = _write_json(str(tmp_path / 'values.json'), {
        'schema_version': 1, 'heat': {'k': {'value': 0.3, 'units': '', 'data_reference': ''}}})
    config = dict(base_user_inputs, file_prefix='heat1d', model_type='external_python',
                  solver='external', solver_info=dict(_SOLVER_INFO),
                  external_model_path=os.path.join(_EXAMPLE_DIR, 'heat1d_model.py'),
                  external_parameters_file=values, resources_dir=str(tmp_path),
                  generated_models_dir=str(tmp_path / 'generated'),
                  param_id_output_dir=str(tmp_path / 'out'))
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)
    assert parsed['solver_info']['external_parameters_file'] == values
    from libcuflynx.solver_wrappers import get_simulation_helper_from_inp_data_dict
    helper = get_simulation_helper_from_inp_data_dict(parsed)
    assert helper.get_init_param_vals(['heat/k', 'heat/u_D']) == [0.3, 0.0]


# --------------------------------------------------------------------------------------------
# 4-5. calibration, with a value set by hand
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_a_set_value_is_held_while_k_is_calibrated(example, tmp_path):
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(_write(example, _fit_k(example)), output_dir=out)
    assert result['complete'] and result['order'] == ['boundary', 'fit-k']
    merged = {r['parameterisation_name']: r for r in result['merged_parameters']}
    assert merged['heat/k']['value'] == pytest.approx(_TRUE_K, abs=0.01)
    assert merged['heat/u_D']['value'] == _TRUE_U_D
    assert merged['heat/u_D']['origin'] == 'set_by_hand'

    step = _read_json(os.path.join(out, 'fit-k', 'step_result.json'))
    assert step['model'] == {'external_model_path': os.path.join(example, 'heat1d_model.py')}
    assert step['model_path'] == os.path.join(example, 'heat1d_model.py')
    assert [(r['model_name'], r['value']) for r in step['fixed']] == [('u_D_heat', _TRUE_U_D)]
    # the step ran with u_D from its generated parameters file ...
    generated = step['generated']['parameters_path']
    assert _values(generated) == {'heat/k': 0.4, 'heat/u_D': _TRUE_U_D}
    assert 'external_model' in step['inputs']
    # ... which is what the best fit matches the data with: at k alone, from the class
    # defaults (u_D = 0), it is far off
    helper = _helper(step['model_path'], generated)
    helper.set_param_vals(['heat/k'], [step['best_fit'][0]])
    assert np.abs(_probe_means(helper) - _OBS).max() < 2 * _STD
    plain = _helper(step['model_path'])
    plain.set_param_vals(['heat/k'], [step['best_fit'][0]])
    assert np.abs(_probe_means(plain) - _OBS).max() > 5 * _STD

    # the provenance diagram names the model; the methods text, written from a CellML, says
    # why it has none
    with open(os.path.join(out, 'provenance_graph.json')) as f:
        assert 'heat1d_model.py' in f.read()
    from libcuflynx.reporting.flowchart import model_line
    assert model_line(step['model']) == 'external Python model `heat1d_model.py`'
    from libcuflynx.reporting.methods import workflow_methods_latex
    with pytest.raises(ValueError, match='its target is the external Python model'):
        workflow_methods_latex(os.path.join(example, 'workflow.json'), run_dir=out)


@pytest.mark.integration
@pytest.mark.parametrize('method', ['genetic_algorithm', 'CMA-ES'])
def test_each_optimiser_recovers_k(example, tmp_path, method):
    raw = _fit_k(example, method)
    if method == 'genetic_algorithm':
        # its calls must outnumber its default population (744)
        raw['settings']['optimiser_options']['num_calls_to_function'] = 1500
    result = run_calibration_workflow(_write(example, raw),
                                      output_dir=str(tmp_path / method))
    merged = {r['parameterisation_name']: r['value'] for r in result['merged_parameters']}
    assert merged['heat/k'] == pytest.approx(_TRUE_K, abs=0.02)


# --------------------------------------------------------------------------------------------
# 6. UQ
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_a_stored_posterior_is_the_prior_of_the_refinement(example, tmp_path):
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(_write(example, _raw(example)), output_dir=out)
    assert result['order'] == ['boundary', 'fit-k', 'refine']

    # the MCMC of fit-k has its own run directory, beside the calibration's
    chains = glob.glob(os.path.join(out, 'fit-k', 'param_id', 'mcmc_*', 'mcmc_chain.npy'))
    assert len(chains) == 1
    fit_k = StoredDistribution.load(os.path.join(out, 'fit-k'))
    assert fit_k.names == ['k_heat']
    assert np.median(fit_k.samples) == pytest.approx(_TRUE_K, abs=0.01)

    refine = _read_json(os.path.join(out, 'refine', 'step_result.json'))
    assert refine['priors'][0]['parameters'] == ['k_heat']
    assert refine['point_estimate'] == 'posterior_median'
    posterior = StoredDistribution.load(os.path.join(out, 'refine'))
    # u_D's prior is uniform on [-0.5, 0.5]; the data pin it far tighter
    assert np.std(posterior.marginal('u_D_heat')) < (1.0 / np.sqrt(12)) / 5
    merged = {r['parameterisation_name']: r for r in result['merged_parameters']}
    assert merged['heat/k']['value'] == pytest.approx(_TRUE_K, abs=0.02)
    assert merged['heat/u_D']['value'] == pytest.approx(_TRUE_U_D, abs=0.03)
    assert merged['heat/u_D']['from_step'] == 'refine'


# --------------------------------------------------------------------------------------------
# 7. the analyses around a calibration
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_sensitivity_reduction_identifiability_and_a_target_sweep(example, tmp_path):
    from libcuflynx.external_testing.workflow_library import IA_OPTIONS, SA_OPTIONS
    # u_D varies over a narrow band, so k is what the Sobol screen ranks first
    _write_json(os.path.join(example, 'screen_params_for_id.json'), {'version': 1, 'params': [
        {'targets': ['heat/k'], 'min': 0.05, 'max': 1.0},
        {'targets': ['heat/u_D'], 'min': 0.095, 'max': 0.105}]})
    raw = _fit_k(example)
    raw['settings'].update(sa_options=SA_OPTIONS, ia_options=IA_OPTIONS)
    obs = 'heat1d_obs_data.json'
    raw['steps'] += [
        {'id': 'screen', 'kind': 'sensitivity', 'model': external_model('heat1d_model.py'),
         'obs_data': obs, 'params_for_id': 'screen_params_for_id.json', 'result_from': 'fit-k'},
        {'id': 'reduce', 'kind': 'reduction', 'sensitivity_from': 'screen',
         'settings': {'param_reduction_options': {'method': 'sobol', 'n_params': 1}}},
        {'id': 'fit-kept', 'model': external_model('heat1d_model.py'), 'obs_data': obs,
         'params_for_id_from': 'reduce', 'fixed_from': ['boundary']},
        {'id': 'ia', 'kind': 'identifiability', 'model': external_model('heat1d_model.py'),
         'obs_data': obs, 'params_for_id': 'heat1d_k_params_for_id.json',
         'result_from': 'fit-k'},
        {'id': 'dose', 'kind': 'sweep', 'on_target': True, 'obs_data': obs,
         'sweep': {'outputs': ['heat/T_p1'], 'parameters': {'heat/k': {'values': [0.2, 0.25]}}}},
    ]
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(_write(example, raw), output_dir=out)
    assert result['complete']
    step = lambda sid: _read_json(os.path.join(out, sid, 'step_result.json'))  # noqa: E731

    assert os.path.isfile(step('screen')['sobol_indices'])
    assert [s['parameter'] for s in step('reduce')['selected']] == ['heat/k']
    merged = {r['parameterisation_name']: r for r in result['merged_parameters']}
    assert merged['heat/k']['from_step'] == 'fit-kept'
    assert merged['heat/k']['value'] == pytest.approx(_TRUE_K, abs=0.01)

    ia = step('ia')
    assert ia['at'] == pytest.approx(step('fit-k')['best_fit'])
    assert 0 < ia['laplace']['std'][0] < 0.05

    dose = step('dose')
    assert dose['n_failed'] == 0
    # the sweep ran on the target with u_D = 0.1: at k = 0.25 it reproduces probe 1's mean
    results = np.load(dose['sweep_results'])
    at_truth = list(results['points'][:, 0]).index(0.25)
    assert np.mean(results['trace_0'][at_truth, 0]) == pytest.approx(_OBS[0], abs=2 * _STD)


# --------------------------------------------------------------------------------------------
# 8. emulation
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_a_calibration_on_an_emulator_of_the_external_model(example, tmp_path):
    pytest.importorskip('autoemulate')
    from libcuflynx.external_testing.workflow_library import EMULATOR_SETTINGS
    # without fit-k: two calibrations of k that do not take from each other are refused
    raw = _raw(example, ['boundary'], dict(_FAST, param_id_method='CMA-ES'))
    raw['settings']['emulator_settings'] = EMULATOR_SETTINGS
    obs, k_only = 'heat1d_obs_data.json', 'heat1d_k_params_for_id.json'
    raw['steps'] += [
        {'id': 'emulate', 'kind': 'emulation', 'model': external_model('heat1d_model.py'),
         'obs_data': obs, 'params_for_id': k_only, 'fixed_from': ['boundary']},
        {'id': 'fit-emulated', 'model': external_model('heat1d_model.py'), 'obs_data': obs,
         'params_for_id': k_only, 'emulator_from': 'emulate'},
    ]
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(_write(example, raw), output_dir=out)
    assert os.path.isdir(_read_json(os.path.join(out, 'emulate', 'step_result.json'))
                         ['emulator_dir'])
    merged = {r['parameterisation_name']: r for r in result['merged_parameters']}
    assert merged['heat/k']['from_step'] == 'fit-emulated'
    assert merged['heat/k']['value'] == pytest.approx(_TRUE_K, abs=0.03)


_PROFILE = {'method': 'profile_likelihood', 'num_points': 5, 'range_factor': 0.05,
            'optimiser_options': {'num_calls_to_function': 40, 'max_patience': 10}}


def _ia_steps(**keys):
    '''A Laplace and a profile-likelihood identifiability step on k, at the fit-k values.'''
    step = dict(kind='identifiability', model=external_model('heat1d_model.py'),
                obs_data='heat1d_obs_data.json', params_for_id='heat1d_k_params_for_id.json',
                result_from='fit-k', **keys)
    return [dict(step, id='laplace', settings={'ia_options': {'method': 'Laplace'}}),
            dict(step, id='profile', settings={'ia_options': _PROFILE})]


@pytest.mark.integration
def test_a_profile_likelihood_of_the_external_model(example, tmp_path):
    '''An external model (how a reduced-order model runs) takes an identifiability step's
    profile likelihood as a CellML model does.'''
    raw = _fit_k(example)
    raw['steps'] += _ia_steps()[1:]
    out = str(tmp_path / 'run')
    assert run_calibration_workflow(_write(example, raw), output_dir=out)['complete']

    step = _read_json(os.path.join(out, 'profile', 'step_result.json'))
    best_k = _read_json(os.path.join(out, 'fit-k', 'step_result.json'))['best_fit']
    assert step['method'] == 'profile_likelihood' and step['at'] == pytest.approx(best_k)
    (profile,) = step['profile_likelihood']['profiles']
    assert profile['parameter'] == 'heat/k'
    assert profile['best_val'] == pytest.approx(best_k[0])
    assert profile['ci_95'] == _read_json(profile['file'])['ci_95']


@pytest.mark.integration
def test_the_identifiability_steps_run_on_an_emulator_of_the_external_model(
        example, tmp_path, monkeypatch):
    '''emulator_from on an identifiability step of an external model: the Laplace analysis and
    the profile likelihood evaluate the emulator, not the model.'''
    pytest.importorskip('autoemulate')
    from libcuflynx.external_testing.workflow_library import EMULATOR_SETTINGS
    from libcuflynx.solver_wrappers import external_simulation_helper
    raw = _fit_k(example)
    # An interpolating RBF from 64 samples: the probe means are not quadratic in k over
    # [0.05, 1], so the library's quadratic from 32 samples fails min_r2 on probe 3.
    raw['settings']['emulator_settings'] = {**EMULATOR_SETTINGS, 'num_train_samples': 64,
                                            'n_iter': 8, 'models': 'RadialBasisFunctions'}
    raw['steps'].append(
        {'id': 'emulate', 'kind': 'emulation', 'model': external_model('heat1d_model.py'),
         'obs_data': 'heat1d_obs_data.json', 'params_for_id': 'heat1d_k_params_for_id.json',
         'fixed_from': ['boundary']})
    first = str(tmp_path / 'first')
    assert run_calibration_workflow(_write(example, raw, 'train.json'),
                                    output_dir=first)['complete']
    emulator = _read_json(os.path.join(first, 'emulate', 'step_result.json'))['emulator_dir']

    # a second workflow analyses that emulator at a point set by hand
    analyses = _ia_steps(emulator_dir=emulator)
    for step in analyses:
        step['result_from'] = 'hand'
    raw['steps'] = [{'id': 'hand', 'kind': 'set', 'point': True, 'set': [
        {'parameter': 'heat/k', 'value': _TRUE_K, 'reason': 'the true value'}]}] + analyses

    def no_model_run(self, *args, **kwargs):
        raise AssertionError('the step simulated the model, not the emulator')
    monkeypatch.setattr(external_simulation_helper.SimulationHelper, 'run', no_model_run)
    out = str(tmp_path / 'run')
    assert run_calibration_workflow(_write(example, raw), output_dir=out)['complete']

    laplace = _read_json(os.path.join(out, 'laplace', 'step_result.json'))
    profile = _read_json(os.path.join(out, 'profile', 'step_result.json'))
    assert laplace['inputs']['emulator']['path'] == emulator
    assert profile['inputs']['emulator']['path'] == emulator
    assert laplace['laplace']['std'][0] > 0
    assert [p['parameter'] for p in profile['profile_likelihood']['profiles']] == ['heat/k']


# --------------------------------------------------------------------------------------------
# 9. the target: the calibrated parameters, a pack and the manifest
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_the_calibrated_parameters_run_the_model_standalone(example, tmp_path,
                                                            base_user_inputs):
    out = str(tmp_path / 'run')
    path = _write(example, _fit_k(example))
    result = run_calibration_workflow(path, output_dir=out, write_calibrated=True)
    calibrated = result['calibrated_parameters_file']
    assert calibrated == os.path.join(example, 'heat1d_calibrated_parameters.json')
    values = _values(calibrated)
    assert values['heat/k'] == pytest.approx(_TRUE_K, abs=0.01)
    assert values['heat/u_D'] == pytest.approx(_TRUE_U_D)
    rows = {r['variable_name']: r for r in read_parameter_rows(calibrated)}
    assert rows['heat/k']['origin'] == 'calibrated'
    assert rows['heat/u_D']['origin'] == 'set_by_hand'

    # an ordinary run of the .py with that file: the parser hands it to the helper
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.solver_wrappers import get_simulation_helper_from_inp_data_dict
    config = dict(base_user_inputs, file_prefix='heat1d', model_type='external_python',
                  solver='external', solver_info=dict(_SOLVER_INFO), dt=_DT, sim_time=0.5,
                  pre_time=0.0, external_model_path=os.path.join(example, 'heat1d_model.py'),
                  external_parameters_file=calibrated, resources_dir=str(tmp_path),
                  generated_models_dir=str(tmp_path / 'generated'),
                  param_id_output_dir=str(tmp_path / 'standalone'))
    helper = get_simulation_helper_from_inp_data_dict(
        YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False))
    assert helper.get_init_param_vals(['heat/k', 'heat/u_D']) == pytest.approx(
        [values['heat/k'], values['heat/u_D']])
    assert np.abs(_probe_means(helper) - _OBS).max() < 2 * _STD

    # the study manifest names the .py as the target and each step's model
    from libcuflynx.utilities.cuflynx_study import read_study
    study = read_study(os.path.join(out, 'heat1d_cuflynx_study.json'))['workflow']
    assert os.path.basename(study['target']['model']['external_model_path']) == \
        'heat1d_model.py'
    assert os.path.basename(study['steps'][1]['model']['external_model_path']) == \
        'heat1d_model.py'

    # a pack holds the .py and the files beside it, and runs as the original does
    packed = pack_workflow(path, str(tmp_path / 'pack'))
    workflow = load_workflow(packed['workflow'])
    assert workflow.target.path.startswith(str(tmp_path / 'pack'))
    assert os.path.isfile(workflow.target.path)
    assert os.path.dirname(workflow.step('fit-k').obs_data) == \
        os.path.dirname(workflow.target.path)
    plan = plan_workflow(workflow)
    assert [s['id'] for s in plan['steps']] == ['boundary', 'fit-k']


@pytest.mark.unit
def test_the_builder_writes_an_external_model_relative_to_the_workflow(example, tmp_path):
    model = os.path.join(example, 'heat1d_model.py')
    wf = WorkflowBuilder('built', target=external_model(model), directory=example,
                         settings=_FAST)
    wf.add_set('boundary', [{'parameter': 'heat/u_D', 'value': 0.1, 'reason': 'measured'}])
    wf.add_calibration('fit-k', model=external_model(model),
                       obs_data=os.path.join(example, 'heat1d_obs_data.json'),
                       params_for_id=os.path.join(example, 'heat1d_k_params_for_id.json'),
                       fixed_from=['boundary'])
    raw = wf.to_dict()
    assert raw['target'] == {'external_model_path': 'heat1d_model.py'}
    assert raw['steps'][1]['model'] == {'external_model_path': 'heat1d_model.py'}
    wf.write()
    assert load_workflow(wf.path).target == ExternalModel(model)


# --------------------------------------------------------------------------------------------
# 10-11. the shipped example, and the command
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_shipped_example_parses_and_resolves():
    path = os.path.join(_EXAMPLE_DIR, _WORKFLOW)
    plan = plan_workflow(path)
    assert [s['id'] for s in plan['steps']] == ['boundary', 'fit-k', 'refine']
    steps = {s['id']: s for s in plan['steps']}
    assert [c['target_parameterisation_name'] for c in steps['fit-k']['calibrates']] == \
        ['heat/k']
    assert [c['target_parameterisation_name'] for c in steps['refine']['calibrates']] == \
        ['heat/u_D']
    assert steps['refine']['priors_from'][0]['step'] == 'fit-k'


@pytest.mark.unit
def test_the_command_labels_an_external_model():
    from libcuflynx.scripts.calibration_workflow_run_script import _model_label
    assert _model_label({'external_model_path': '/x/heat1d_model.py'}) == '/x/heat1d_model.py'


@pytest.mark.integration
@pytest.mark.slow
def test_the_command_runs_the_shipped_example(example, tmp_path, capsys):
    # the command's own path, without main()'s MPI.Finalize, which would end the test session
    from libcuflynx.scripts.calibration_workflow_run_script import build_parser, run
    raw = _raw(example)
    raw['settings']['UQ_options'] = {'num_steps': 60, 'num_walkers': 8, 'burn_in': 0.5}
    path = _write(example, raw, _WORKFLOW)
    out = str(tmp_path / 'run')
    run(build_parser().parse_args([path, '--output-dir', out, '--write-calibrated']))
    printed = capsys.readouterr().out
    assert f'fit-k (calibration, {os.path.join(example, "heat1d_model.py")})' in printed
    values = _values(os.path.join(example, 'heat1d_calibrated_parameters.json'))
    assert values['heat/k'] == pytest.approx(_TRUE_K, abs=0.03)
    assert glob.glob(os.path.join(out, 'fit-k', 'param_id', 'mcmc_*', 'mcmc_chain.npy'))
