"""The four ways to run a calibration or a Sobol analysis give the same numbers.

The routes are: the dict route the notebooks use (``init_from_dict`` and then the setters), the
file route (``CVS0DParamID`` / ``SensitivityAnalysis`` given the obs_data and params_for_id
paths) and a calibration workflow (``run_calibration_workflow``, which runs the
``user_inputs`` scripts). CUFLynx runs the workflow and file routes. On 3compartment with a seed,
all three must give the same calibrated parameters, cost, best-fit traces and Sobol indices.

The obs_data sets ``offline_pre_time``, which every route must apply, and a params_for_id with
``q_lv_init`` (a state's initial value) must be refused with it on every route.
"""
import copy
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from libcuflynx.calibration_workflow import run_calibration_workflow
from libcuflynx.param_id.paramID import CVS0DParamID
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = 1e-12

SOLVER_INFO = {'solver': 'CVODE_myokit', 'MaximumStep': 0.001, 'rtol': 1e-8, 'atol': 1e-8}
OPTIMISER_OPTIONS = {'num_calls_to_function': 40, 'cost_convergence': 0.001, 'max_patience': 10,
                     'cost_type': 'MSE', 'seed': 1234}
SA_OPTIONS = {'method': 'sobol', 'sample_type': 'saltelli', 'num_samples': 8}

OBS_DATA = {
    'protocol_info': {'pre_times': [2.0], 'sim_times': [[1.0]], 'params_to_change': {},
                      'offline_pre_time': 4.0},
    'prediction_items': [],
    'data_items': [
        {'data_item_name': 'P aortic root mean', 'trace_name_for_plotting': 'P_{aoMean}',
         'operands': ['aortic_root/u'], 'operation': 'mean', 'unit': 'Pa',
         'value': 12700.0, 'std': 200, 'obs_dt': 0.01, 'weight': 1.0, 'data_type': 'constant',
         'experiment_idx': 0, 'subexperiment_idx': 0},
        {'data_item_name': 'P aortic root max', 'trace_name_for_plotting': 'P_{aoMax}',
         'operands': ['aortic_root/u'], 'operation': 'max', 'unit': 'Pa',
         'value': 15500.0, 'std': 200, 'obs_dt': 0.01, 'weight': 1.0, 'data_type': 'constant',
         'experiment_idx': 0, 'subexperiment_idx': 0},
    ],
}
PARAMS = [{'vessel_name': 'aortic_root', 'param_name': 'C', 'min': 1e-9, 'max': 5e-8,
           'name_for_plotting': 'C_{ao}'},
          {'vessel_name': 'systemic_T', 'param_name': 'R_T', 'min': 5e7, 'max': 5e8,
           'name_for_plotting': 'R_{T}'}]
Q_LV_INIT = {'vessel_name': 'global', 'param_name': 'q_lv_init', 'min': 2e-4, 'max': 1.5e-3,
             'name_for_plotting': 'q_{lv,0}'}


def _params_file(entries):
    return {'version': 1, 'params': [
        {'targets': [f"{e['vessel_name']}/{e['param_name']}"], 'min': e['min'], 'max': e['max'],
         'name_for_plotting': e['name_for_plotting']} for e in entries]}


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def _write_inputs(root, params):
    """The model, the obs_data, the params_for_id and a workflow with a Sobol step and a
    calibration step, all under ``root``."""
    os.makedirs(os.path.join(root, 'model'), exist_ok=True)
    os.makedirs(os.path.join(root, 'data'), exist_ok=True)
    for name in ('3compartment_module_array.csv', '3compartment_parameters.json'):
        shutil.copy(os.path.join(ROOT, 'resources', name), os.path.join(root, 'model', name))
    _write_json(os.path.join(root, 'data', 'obs_data.json'), OBS_DATA)
    _write_json(os.path.join(root, 'data', 'params_for_id.json'), _params_file(params))
    model = 'model/3compartment_module_array.csv'
    workflow = {
        'schema_version': 1, 'workflow_name': 'route_consistency', 'target': model,
        # DEBUG without debug_UQ_options: run_param_id falls back to UQ_options
        'settings': {'dt': 0.01, 'DEBUG': True, 'solver_info': dict(SOLVER_INFO)},
        'steps': [
            {'id': 'screen', 'kind': 'sensitivity', 'model': model,
             'obs_data': 'data/obs_data.json', 'params_for_id': 'data/params_for_id.json',
             'settings': {'sa_options': dict(SA_OPTIONS)}},
            {'id': 'fit', 'model': model,
             'obs_data': 'data/obs_data.json', 'params_for_id': 'data/params_for_id.json',
             'settings': {'param_id_method': 'genetic_algorithm',
                          'optimiser_options': dict(OPTIMISER_OPTIONS)}},
        ],
    }
    path = os.path.join(root, 'calibration_workflow.json')
    _write_json(path, workflow)
    return path


def _dict_inp(model_path, prefix, out_dir):
    """The notebook's config: the user_inputs parser over a dict, then edited in code."""
    inp = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.json',
           'model_type': 'cellml', 'generated_model_path': model_path,
           'resources_dir': os.path.dirname(model_path),
           'generated_models_dir': os.path.join(out_dir, 'generated_models'),
           'param_id_output_dir': os.path.join(out_dir, 'param_id'),
           'solver_info': dict(SOLVER_INFO), 'dt': 0.01, 'DEBUG': True,
           'param_id_method': 'genetic_algorithm', 'optimiser_options': dict(OPTIMISER_OPTIONS),
           'sa_options': dict(SA_OPTIONS, output_dir=os.path.join(out_dir, 'sa'))}
    return YamlFileParser().parse_user_inputs_file(inp)


def _calibration_result(param_id):
    output_dir = param_id.output_dir
    return {'best': np.asarray(param_id.get_best_param_vals(), dtype=float),
            'cost': float(param_id.param_id.best_cost),
            'traces': _traces(output_dir)}


def _traces(case_dir):
    with np.load(os.path.join(case_dir, 'all_outputs_with_best_param_vals_exp_0.npz'),
                 allow_pickle=True) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files if np.asarray(npz[k]).dtype.kind in 'fi'}


def _sobol(sa_dir):
    found = sorted(glob.glob(os.path.join(sa_dir, '*Sobol_indices.csv')))
    assert len(found) == 1, f'expected one Sobol indices file in {sa_dir}, found {found}'
    frame = pd.read_csv(found[0])
    return frame[[c for c in frame.columns if c in ('S1', 'ST')]].to_numpy(dtype=float)


def _warm_state(sa):
    """The state the Sobol runs start from: the offline warm-up's end, or None."""
    return sa.SA_manager.sim_helper._offline_default_state


def _assert_same(a, b, what):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, f'{what}: shapes {a.shape} and {b.shape}'
    np.testing.assert_allclose(a, b, rtol=REL, atol=0, err_msg=what)


@pytest.fixture(scope='module')
def routes(tmp_path_factory):
    """Each route's calibration and Sobol results, from the same generated models."""
    root = str(tmp_path_factory.mktemp('routes'))
    mp = pytest.MonkeyPatch()
    mp.setenv('CUFLYNX_USER_DIR', os.path.join(root, 'user_dir'))
    try:
        workflow_path = _write_inputs(os.path.join(root, 'inputs'), PARAMS)
        obs_path = os.path.join(root, 'inputs', 'data', 'obs_data.json')
        pfi_path = os.path.join(root, 'inputs', 'data', 'params_for_id.json')

        # the workflow route
        result = run_calibration_workflow(workflow_path, output_dir=os.path.join(root, 'wf'))
        assert result['complete']
        steps = {}
        for sid in ('screen', 'fit'):
            with open(os.path.join(root, 'wf', sid, 'step_result.json')) as f:
                steps[sid] = json.load(f)
        fit, screen = steps['fit'], steps['screen']
        out = {'workflow': {
            'calibration': {'best': np.asarray(fit['best_fit'], dtype=float),
                            'cost': float(fit['best_cost']),
                            'traces': _traces(fit['param_id_output_dir'])},
            'sobol': _sobol(screen['sensitivity_dir']),
            'solver_info': {sid: steps[sid]['solver_info'] for sid in steps}}}

        # the dict route: init_from_dict, then the setters
        d = os.path.join(root, 'dict')
        inp = _dict_inp(fit['model_path'], 'fit', d)
        param_id = CVS0DParamID.init_from_dict(inp)
        # set twice, as the notebooks do when they add a data item: the warm-up must start
        # from the model's initial values both times, not from the first warm-up's end
        param_id.set_ground_truth_data(copy.deepcopy(OBS_DATA))
        param_id.set_params_for_id(copy.deepcopy(PARAMS))
        param_id.set_ground_truth_data(copy.deepcopy(OBS_DATA))
        param_id.set_param_id_method('genetic_algorithm')
        param_id.set_optimiser_options(dict(OPTIMISER_OPTIONS))
        param_id.run()
        param_id.simulate_with_best_param_vals()
        calibration = _calibration_result(param_id)

        sa_inp = _dict_inp(screen['model_path'], 'screen', d)
        before = copy.deepcopy(sa_inp)
        sa = SensitivityAnalysis.init_from_dict(sa_inp)
        assert sa_inp == before, 'SensitivityAnalysis.init_from_dict changed the caller\'s dict'
        sa.set_ground_truth_data(copy.deepcopy(OBS_DATA))
        sa.set_params_for_id(copy.deepcopy(PARAMS))
        sa_options = dict(SA_OPTIONS, output_dir=os.path.join(d, 'sa'))
        sa.set_sa_options(sa_options)
        sa.run_sensitivity_analysis(sa_options)
        out['dict'] = {'calibration': calibration, 'sobol': _sobol(sa_options['output_dir']),
                       'sobol_warm_state': _warm_state(sa)}

        # the file route: the constructors given the paths
        f = os.path.join(root, 'file')
        os.makedirs(f)
        param_id = CVS0DParamID(fit['model_path'], 'cellml', 'genetic_algorithm',
                                file_name_prefix='fit', params_for_id_path=pfi_path,
                                param_id_obs_path=obs_path, dt=0.01,
                                solver_info=dict(SOLVER_INFO),
                                optimiser_options=dict(OPTIMISER_OPTIONS), DEBUG=True,
                                param_id_output_dir=os.path.join(f, 'param_id'),
                                resources_dir=os.path.dirname(fit['model_path']))
        param_id.run()
        param_id.simulate_with_best_param_vals()
        calibration = _calibration_result(param_id)
        sa_options = dict(SA_OPTIONS, output_dir=os.path.join(f, 'sa'))
        sa = SensitivityAnalysis(screen['model_path'], 'cellml', 'screen', sa_options, DEBUG=True,
                                 param_id_output_dir=os.path.join(f, 'param_id'),
                                 resources_dir=os.path.dirname(screen['model_path']),
                                 solver_info=dict(SOLVER_INFO), dt=0.01,
                                 param_id_obs_path=obs_path, params_for_id_path=pfi_path)
        sa.run_sensitivity_analysis(sa_options)
        out['file'] = {'calibration': calibration, 'sobol': _sobol(sa_options['output_dir']),
                       'sobol_warm_state': _warm_state(sa)}
        out['root'] = root
        yield out
    finally:
        mp.undo()


@pytest.mark.integration
@pytest.mark.parametrize('route', ['dict', 'file'])
def test_a_calibration_gives_the_workflow_result_on_every_route(routes, route):
    expected, got = routes['workflow']['calibration'], routes[route]['calibration']
    _assert_same(got['best'], expected['best'], f'{route}: calibrated parameters')
    _assert_same(got['cost'], expected['cost'], f'{route}: cost')
    assert got['traces'].keys() == expected['traces'].keys()
    for name in expected['traces']:
        _assert_same(got['traces'][name], expected['traces'][name], f'{route}: trace {name}')


@pytest.mark.integration
@pytest.mark.parametrize('route', ['dict', 'file'])
def test_a_sobol_analysis_gives_the_workflow_indices_on_every_route(routes, route):
    _assert_same(routes[route]['sobol'], routes['workflow']['sobol'], f'{route}: Sobol S1/ST')


@pytest.mark.integration
def test_the_offline_warm_up_is_applied(routes, tmp_path, monkeypatch):
    """Without offline_pre_time the best-fit traces differ: the routes agree because each
    applies the warm-up, not because none does."""
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path / 'user_dir'))
    with open(os.path.join(routes['root'], 'wf', 'fit', 'step_result.json')) as f:
        fit = json.load(f)
    obs = copy.deepcopy(OBS_DATA)
    del obs['protocol_info']['offline_pre_time']
    param_id = CVS0DParamID.init_from_dict(_dict_inp(fit['model_path'], 'fit', str(tmp_path)))
    param_id.set_ground_truth_data(obs)
    param_id.set_params_for_id(copy.deepcopy(PARAMS))
    best = routes['workflow']['calibration']['best']
    with_warm_up = routes['workflow']['calibration']['cost']
    param_id.set_best_param_vals(best)
    param_id.param_id.set_optimiser_options(dict(OPTIMISER_OPTIONS))
    without = float(param_id.param_id.get_cost_from_params(best))
    assert abs(without - with_warm_up) > 1e-6 * abs(with_warm_up)


@pytest.mark.integration
@pytest.mark.parametrize('route', ['dict', 'file'])
def test_sobol_starts_from_the_offline_warm_up(routes, route):
    assert routes[route]['sobol_warm_state'] is not None


@pytest.mark.integration
def test_the_step_result_records_the_solver_info_the_step_ran_with(routes):
    for sid, used in routes['workflow']['solver_info'].items():
        for key, value in SOLVER_INFO.items():
            assert used[key] == value, (sid, key)
        assert 'dt_solver' not in used, sid


# --------------------------------------------------------------------------------------------
# q_lv_init with offline_pre_time is refused on every route
# --------------------------------------------------------------------------------------------

OFFLINE_REFUSAL = "offline_pre_time can't be used"


@pytest.fixture(scope='module')
def q_lv_init_inputs(tmp_path_factory, routes):
    root = str(tmp_path_factory.mktemp('q_lv_init'))
    workflow_path = _write_inputs(os.path.join(root, 'inputs'), PARAMS + [Q_LV_INIT])
    with open(os.path.join(routes['root'], 'wf', 'fit', 'step_result.json')) as f:
        fit = json.load(f)
    return {'root': root, 'workflow': workflow_path, 'model_path': fit['model_path'],
            'obs': os.path.join(root, 'inputs', 'data', 'obs_data.json'),
            'params_for_id': os.path.join(root, 'inputs', 'data', 'params_for_id.json')}


@pytest.mark.integration
@pytest.mark.parametrize('params_first', [False, True])
def test_q_lv_init_with_offline_pre_time_is_refused_on_the_dict_route(
        q_lv_init_inputs, tmp_path, monkeypatch, params_first):
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path / 'user_dir'))
    param_id = CVS0DParamID.init_from_dict(
        _dict_inp(q_lv_init_inputs['model_path'], 'fit', str(tmp_path)))
    steps = [lambda: param_id.set_ground_truth_data(copy.deepcopy(OBS_DATA)),
             lambda: param_id.set_params_for_id(copy.deepcopy(PARAMS + [Q_LV_INIT]))]
    if params_first:
        steps.reverse()
    steps[0]()
    with pytest.raises(ValueError, match=OFFLINE_REFUSAL):
        steps[1]()


@pytest.mark.integration
def test_q_lv_init_with_offline_pre_time_is_refused_on_the_file_route(
        q_lv_init_inputs, tmp_path, monkeypatch):
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path / 'user_dir'))
    with pytest.raises(ValueError, match=OFFLINE_REFUSAL):
        CVS0DParamID(q_lv_init_inputs['model_path'], 'cellml', 'genetic_algorithm',
                     file_name_prefix='fit', params_for_id_path=q_lv_init_inputs['params_for_id'],
                     param_id_obs_path=q_lv_init_inputs['obs'], dt=0.01,
                     solver_info=dict(SOLVER_INFO), optimiser_options=dict(OPTIMISER_OPTIONS),
                     param_id_output_dir=str(tmp_path / 'param_id'),
                     resources_dir=os.path.dirname(q_lv_init_inputs['model_path']))


@pytest.mark.integration
@pytest.mark.parametrize('route', ['dict', 'file'])
def test_q_lv_init_with_offline_pre_time_is_refused_by_sobol(
        q_lv_init_inputs, tmp_path, monkeypatch, route):
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path / 'user_dir'))
    model_path = q_lv_init_inputs['model_path']
    sa_options = dict(SA_OPTIONS, output_dir=str(tmp_path / 'sa'))
    if route == 'dict':
        sa = SensitivityAnalysis.init_from_dict(_dict_inp(model_path, 'fit', str(tmp_path)))
        sa.set_ground_truth_data(copy.deepcopy(OBS_DATA))
        sa.set_params_for_id(copy.deepcopy(PARAMS + [Q_LV_INIT]))
        sa.set_sa_options(sa_options)
    else:
        sa = SensitivityAnalysis(model_path, 'cellml', 'fit', sa_options,
                                 param_id_output_dir=str(tmp_path / 'param_id'),
                                 resources_dir=os.path.dirname(model_path),
                                 solver_info=dict(SOLVER_INFO), dt=0.01,
                                 param_id_obs_path=q_lv_init_inputs['obs'],
                                 params_for_id_path=q_lv_init_inputs['params_for_id'])
    with pytest.raises(ValueError, match=OFFLINE_REFUSAL):
        sa.run_sensitivity_analysis(sa_options)


@pytest.mark.integration
def test_q_lv_init_with_offline_pre_time_is_refused_by_a_workflow(q_lv_init_inputs, monkeypatch):
    monkeypatch.setenv('CUFLYNX_USER_DIR', os.path.join(q_lv_init_inputs['root'], 'user_dir'))
    with pytest.raises(ValueError, match=OFFLINE_REFUSAL):
        run_calibration_workflow(q_lv_init_inputs['workflow'],
                                 output_dir=os.path.join(q_lv_init_inputs['root'], 'wf'))


# --------------------------------------------------------------------------------------------
# dt_solver and MaximumStep, and the shared CVODE_myokit defaults
# --------------------------------------------------------------------------------------------

CONFLICT = {'solver': 'CVODE_myokit', 'MaximumStep': 0.001, 'dt_solver': 0.01}
DT_SOLVER_REFUSAL = 'dt_solver=0.01 and MaximumStep=0.001'


@pytest.mark.unit
def test_dt_solver_and_a_different_maximum_step_are_refused_by_the_parser(tmp_path, monkeypatch):
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path / 'user_dir'))
    with pytest.raises(ValueError, match=DT_SOLVER_REFUSAL):
        YamlFileParser().parse_user_inputs_file(
            {'file_prefix': 'm', 'input_param_file': 'm_parameters.json', 'model_type': 'cellml', 'solver_info': dict(CONFLICT),
             'resources_dir': str(tmp_path),
             'generated_models_dir': str(tmp_path / 'generated_models'),
             'param_id_output_dir': str(tmp_path / 'param_id')})


@pytest.mark.integration
def test_dt_solver_and_a_different_maximum_step_are_refused_by_the_constructors(
        routes, tmp_path, monkeypatch):
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path / 'user_dir'))
    with open(os.path.join(routes['root'], 'wf', 'fit', 'step_result.json')) as f:
        model_path = json.load(f)['model_path']
    with pytest.raises(ValueError, match=DT_SOLVER_REFUSAL):
        CVS0DParamID.init_from_dict({'model_path': model_path, 'model_type': 'cellml',
                                     'param_id_method': 'genetic_algorithm',
                                     'file_prefix': 'fit', 'solver_info': dict(CONFLICT),
                                     'param_id_output_dir': str(tmp_path / 'param_id')})
    with pytest.raises(ValueError, match=DT_SOLVER_REFUSAL):
        SensitivityAnalysis(model_path, 'cellml', 'fit', dict(SA_OPTIONS,
                                                              output_dir=str(tmp_path / 'sa')),
                            solver_info=dict(CONFLICT))


@pytest.mark.unit
@pytest.mark.parametrize('solver_info', [
    {'solver': 'CVODE_myokit', 'dt_solver': 0.01},
    {'solver': 'CVODE_myokit', 'MaximumStep': 0.001, 'dt_solver': 0.001},
    {'solver': 'CVODE_myokit', 'MaximumStep': 0.001},
])
def test_the_parser_leaves_a_cellml_maximum_step_as_given(tmp_path, monkeypatch, solver_info):
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path / 'user_dir'))
    inp = YamlFileParser().parse_user_inputs_file(
        {'file_prefix': 'm', 'input_param_file': 'm_parameters.json', 'model_type': 'cellml', 'solver_info': dict(solver_info),
         'resources_dir': str(tmp_path), 'generated_models_dir': str(tmp_path / 'gm'),
         'param_id_output_dir': str(tmp_path / 'param_id')})
    parsed = inp['solver_info']
    assert parsed.get('MaximumStep') == solver_info.get('MaximumStep')
    assert parsed.get('dt_solver') == solver_info.get('dt_solver')


@pytest.mark.unit
def test_cvode_myokit_has_one_set_of_default_tolerances():
    from libcuflynx.parsers.PrimitiveParsers import get_solver_info_default, solver_field_defaults
    from libcuflynx.solver_wrappers.myokit_helper import CA_DEFAULT_ABS_TOL, CA_DEFAULT_REL_TOL
    declared = solver_field_defaults('CVODE_myokit')
    defaults = get_solver_info_default('cellml')
    assert defaults['rtol'] == declared['rtol'] == CA_DEFAULT_REL_TOL == 1e-6
    assert defaults['atol'] == declared['atol'] == CA_DEFAULT_ABS_TOL
    assert defaults['MaximumStep'] == declared['MaximumStep']
