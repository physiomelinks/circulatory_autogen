"""The CVODE_libopencor backend (solver_wrappers/libopencor_helper.py) against CVODE_myokit.

Both backends integrate the same flattened CellML with CVODE, so at tight tolerances their
trajectories agree to well within 0.1%. Skipped unless libopencor is installed (the
`libopencor` extra, Python 3.12 or later).
"""
import os
import shutil

import numpy as np
import pytest

from libcuflynx.protocol_runners.protocol_runner import ProtocolRunner
from libcuflynx.solver_wrappers import get_simulation_helper

pytestmark = [pytest.mark.need_libopencor, pytest.mark.solver]

_TESTS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
_LOTKA = os.path.join(_TESTS_DIR, 'test_inputs', 'Lotka_Volterra_forced.cellml')
_GOODWIN = os.path.join(_REPO_ROOT, 'resources', 'Goodwin.cellml')
_TIGHT = {'MaximumStep': 0.001, 'rtol': 1e-9, 'atol': 1e-11}
# Relative to each trace's largest magnitude. Two CVODE runs at rtol 1e-9 differ by ~1e-6; the
# bound leaves room for different internal step choices without hiding a wrong model.
_REL_TOL = 1e-4


def _helper(solver, path, sim_time, pre_time=0.0, dt=0.01, solver_info=_TIGHT):
    info = dict(solver_info)
    if solver == 'CVODE_libopencor':
        info.setdefault('MaximumNumberOfSteps', 100000)
    return get_simulation_helper(model_path=path, solver=solver, model_type='cellml', dt=dt,
                                 sim_time=sim_time, solver_info=info, pre_time=pre_time)


def _state_names(libopencor_helper):
    return list(libopencor_helper._states)


def _assert_close(a, b, names, tol=_REL_TOL):
    for name, x, y in zip(names, a, b):
        x, y = np.asarray(x, float), np.asarray(y, float)
        assert x.shape == y.shape, (name, x.shape, y.shape)
        scale = max(np.max(np.abs(y)), 1e-12)
        err = np.max(np.abs(x - y)) / scale
        assert err < tol, f'{name}: max relative difference {err:.2e} >= {tol:.0e}'


def _compare_with_myokit(path, sim_time, pre_time, extra_outputs=()):
    lo = _helper('CVODE_libopencor', path, sim_time, pre_time)
    mk = _helper('CVODE_myokit', path, sim_time, pre_time)
    assert lo.run() and mk.run()
    names = _state_names(lo) + list(extra_outputs)
    _assert_close(lo.get_results(names, flatten=True), mk.get_results(names, flatten=True), names)
    np.testing.assert_allclose(lo.get_time(), mk.get_time())
    return lo, mk


@pytest.fixture
def three_compartment(generated_cellml_model_factory):
    return generated_cellml_model_factory('3compartment', '3compartment_parameters.json')


def test_lotka_volterra_matches_myokit():
    _compare_with_myokit(_LOTKA, sim_time=5.0, pre_time=0.0)


def test_goodwin_pmr_model_matches_myokit(tmp_path):
    """An external CellML 1.0 model from the Physiome Model Repository."""
    path = tmp_path / 'Goodwin.cellml'
    shutil.copy(_GOODWIN, path)
    _compare_with_myokit(str(path), sim_time=20.0, pre_time=5.0)


def test_3compartment_matches_myokit_after_a_warm_up(three_compartment):
    """Stiff, with imported units and state initial values set by *_init constants; pre_time
    is integrated without being stored."""
    lo, _ = _compare_with_myokit(three_compartment, sim_time=1.0, pre_time=2.0,
                                 extra_outputs=['aortic_root/u'])
    assert len(lo.get_time()) == 101 and lo.get_time()[0] == 0.0


def test_parameter_changes_between_runs():
    lo = _helper('CVODE_libopencor', _LOTKA, 5.0)
    mk = _helper('CVODE_myokit', _LOTKA, 5.0)
    names = ['Lotka_Volterra_module/x', 'Lotka_Volterra_module/y']

    assert lo.run()
    baseline = lo.get_results(names, flatten=True)
    lo.reset_and_clear()

    for alpha in (4.0, 6.0):
        for helper in (lo, mk):
            helper.set_param_vals(['Lotka_Volterra_module/alpha'], [alpha])
            assert helper.run()
        assert lo.get_init_param_vals(['Lotka_Volterra_module/alpha']) == [[alpha]]
        _assert_close(lo.get_results(names, flatten=True), mk.get_results(names, flatten=True),
                      names)
        lo.reset_and_clear()
        mk.reset_and_clear()

    lo.set_param_vals(['Lotka_Volterra_module/alpha'], [5.0])
    assert lo.run()
    _assert_close(lo.get_results(names, flatten=True), baseline, names, tol=1e-12)


def test_setting_a_state_directly_and_through_its_init_constant(three_compartment):
    lo = _helper('CVODE_libopencor', three_compartment, 0.1)
    [[q_lv0]] = lo.get_init_param_vals(['heart/q_lv'])
    lo.set_param_vals(['heart/q_lv_init'], [1.5 * q_lv0])
    assert lo.get_init_param_vals(['heart/q_lv']) == [[pytest.approx(1.5 * q_lv0)]]
    assert lo.run()
    assert lo.get_results(['heart/q_lv'], flatten=True)[0][0] == pytest.approx(1.5 * q_lv0)

    lo.reset_and_clear()
    lo.set_param_vals(['heart/q_lv'], [0.5 * q_lv0])
    assert lo.run()
    assert lo.get_results(['heart/q_lv'], flatten=True)[0][0] == pytest.approx(0.5 * q_lv0)
    # A state set with set_param_vals holds for every experiment until reset_and_clear, as
    # ProtocolExecutor sets the calibrated values and then calls reset_states per experiment.
    lo.reset_states()
    assert lo.run()
    assert lo.get_results(['heart/q_lv'], flatten=True)[0][0] == pytest.approx(0.5 * q_lv0)
    with pytest.raises(ValueError, match='change_states=False'):
        lo.set_param_vals(['heart/q_lv'], [q_lv0], change_states=False)


def test_a_protocol_with_constant_steps_matches_myokit():
    """params_to_change steps a constant between sub-experiments; the state carries over."""
    protocol_info = {
        'pre_times': [1.0, 0.0],
        'sim_times': [[2.0, 3.0], [4.0]],
        'params_to_change': {'Lotka_Volterra_module/u_alpha': [[0.0, 0.5], [0.25]]},
    }
    outputs = {}
    for solver in ('CVODE_libopencor', 'CVODE_myokit'):
        info = dict(_TIGHT, **({'MaximumNumberOfSteps': 100000}
                               if solver == 'CVODE_libopencor' else {}))
        runner = ProtocolRunner(_LOTKA, inp_data_dict={'dt': 0.01, 'solver_info': info},
                                solver=solver, model_type='cellml')
        _, res_all, t_all = runner.run_protocols(_LOTKA, protocol_info)
        var2idx = runner.get_var2idx_dict()
        x = next(i for name, i in var2idx.items() if name.split('.')[-1].split('/')[-1] == 'x')
        outputs[solver] = ([np.asarray(r[x]) for r in res_all], t_all)
    for exp in range(2):
        _assert_close([outputs['CVODE_libopencor'][0][exp]], [outputs['CVODE_myokit'][0][exp]],
                      [f'x, experiment {exp}'])


def test_protocol_traces_are_refused():
    lo = _helper('CVODE_libopencor', _LOTKA, 1.0)
    with pytest.raises(NotImplementedError, match='CVODE_myokit'):
        lo.set_protocol_info({'pre_times': [0.0], 'sim_times': [[1.0]],
                              'params_to_change': {'Lotka_Volterra_module/u_alpha': [['trace']]}})
    with pytest.raises(NotImplementedError, match='protocol trace'):
        lo.set_param_vals(['Lotka_Volterra_module/u_alpha'], ['trace'])


def test_offline_warm_up_is_the_start_of_every_run():
    """run_offline_pre_and_set_default_state(T) then a run of S equals one run of T + S."""
    names = ['Lotka_Volterra_module/x', 'Lotka_Volterra_module/y']
    warm = _helper('CVODE_libopencor', _LOTKA, 2.0)
    warm.run_offline_pre_and_set_default_state(1.5)
    for _ in range(2):  # every run starts from the warm state
        warm.reset_states()
        assert warm.run()
        offline = warm.get_results(names, flatten=True)
    whole = _helper('CVODE_libopencor', _LOTKA, 2.0, pre_time=1.5)
    assert whole.run()
    _assert_close(offline, whole.get_results(names, flatten=True), names, tol=1e-5)


def test_an_unknown_solver_info_key_is_reported_and_ignored():
    with pytest.warns(UserWarning, match='not settings of CVODE_libopencor'):
        helper = _helper('CVODE_libopencor', _LOTKA, 1.0, solver_info={'max_step': 0.01})
    assert helper.run()
    assert helper.solver_info['MaximumStep'] == 0.001


def test_defaults_match_the_schema():
    from libcuflynx.parsers.PrimitiveParsers import solver_field_defaults
    from libcuflynx.solver_wrappers.libopencor_helper import DEFAULT_SOLVER_INFO
    assert solver_field_defaults('CVODE_libopencor') == DEFAULT_SOLVER_INFO


@pytest.mark.integration
def test_a_calibration_cost_matches_myokit(temp_output_dir):
    """The calibration path (CVS0DParamID, two experiments with a constant forcing, parameters
    set per evaluation) gives the same cost on both backends."""
    from libcuflynx.param_id.paramID import CVS0DParamID
    inputs = os.path.join(_TESTS_DIR, 'test_inputs')
    costs = {}
    for solver in ('CVODE_libopencor', 'CVODE_myokit'):
        info = {'solver': solver, 'MaximumStep': 0.001, 'rtol': 1e-9, 'atol': 1e-11}
        if solver == 'CVODE_libopencor':
            info['MaximumNumberOfSteps'] = 100000
        engine = CVS0DParamID(
            _LOTKA, 'cellml', 'genetic_algorithm', False, 'Lotka_Volterra_forced',
            params_for_id_path=os.path.join(inputs, 'Lotka_Volterra_forced_params_for_id.json'),
            param_id_obs_path=os.path.join(inputs, 'Lotka_Volterra_forced_cross_exp_obs_data.json'),
            sim_time=5.0, pre_time=0.0, solver_info=info, dt=0.01, optimiser_options=None,
            do_ad=False, DEBUG=False, param_id_output_dir=os.path.join(temp_output_dir, solver),
            resources_dir=inputs).param_id
        mins = np.asarray(engine.param_id_info['param_mins'], dtype=float)
        maxs = np.asarray(engine.param_id_info['param_maxs'], dtype=float)
        costs[solver] = [engine.get_cost_obs_and_pred_from_params(mins + f * (maxs - mins))[0]
                         for f in (0.3, 0.6)]
    assert costs['CVODE_libopencor'][0] != pytest.approx(costs['CVODE_libopencor'][1])
    np.testing.assert_allclose(costs['CVODE_libopencor'], costs['CVODE_myokit'], rtol=1e-4)
