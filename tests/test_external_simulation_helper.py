"""Tests for the ``model_type: external_python`` backend.

The user supplies a whole solver *class* -- one that owns its own time stepping. These verify
that such a model reaches the calibration, sensitivity and local-sensitivity pipelines unchanged,
and that the wrapper enforces the parts of the contract a user class can get wrong quietly.

The end-to-end runs use the shipped 1D heat-equation example in
``funcs_user/example_model_external/``; the contract-enforcement tests write throwaway model
files, because what they need is a class that is *wrong* in one specific way. The other shipped
example, the scipy ODE in ``funcs_user/example_model_scipy/``, has its own file
(``tests/test_scipy_ode_example.py``).
"""
import json
import os
import shutil
import textwrap

import numpy as np
import pytest
from mpi4py import MPI

from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser, YamlFileParser
from libcuflynx.solver_wrappers import get_simulation_helper, get_simulation_helper_from_inp_data_dict
from libcuflynx.solver_wrappers.external_simulation_helper import SimulationHelper as ExternalSimulationHelper
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.scripts.sensitivity_analysis_run_script import run_SA
from libcuflynx.scripts.param_id_run_script import run_param_id


_EXAMPLE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'funcs_user', 'example_model_external')
)
_MODEL_PATH = os.path.join(_EXAMPLE_DIR, 'heat1d_model.py')
_OBS_DATA_PATH = os.path.join(_EXAMPLE_DIR, 'heat1d_obs_data.json')
# Ground truth used to build heat1d_obs_data.json (the defaults are k=0.4, u_D=0.0).
_TRUE_K, _TRUE_U_D = 0.25, 0.1
# The window the three probe means in heat1d_obs_data.json were computed on.
_DT, _SIM_TIME = 0.005, 0.5
_SOLVER_INFO = {'solver': 'external', 'method': 'external'}


def _make_resources(temp_output_dir):
    """Copy the example resource files into an isolated per-test resources dir, so a run never
    writes its dated config back into the repo's example directory."""
    comm = MPI.COMM_WORLD
    resources_dir = os.path.join(temp_output_dir, 'resources')
    if comm.Get_rank() == 0:
        os.makedirs(resources_dir, exist_ok=True)
        for name in ('heat1d_params_for_id.csv', 'heat1d_parameters.csv', 'heat1d_obs_data.json'):
            shutil.copy(os.path.join(_EXAMPLE_DIR, name), os.path.join(resources_dir, name))
    if comm.Get_size() > 1:
        comm.Barrier()
    return resources_dir


def _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    resources_dir = _make_resources(temp_output_dir)
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'heat1d',
        'input_param_file': 'heat1d_parameters.csv',
        'model_type': 'external_python',
        'solver': 'external',
        # Override the base solver_info, which carries CVODE-only keys the external solver
        # rejects (it declares exactly one setting of its own).
        'solver_info': dict(_SOLVER_INFO),
        'external_model_path': _MODEL_PATH,
        'resources_dir': resources_dir,
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.0,
        'sim_time': _SIM_TIME,
        'dt': _DT,
        'DEBUG': True,
        'do_uq': False,
        'do_ad': False,
        'plot_predictions': False,
        'model_out_names': ['heat/T_p2'],
        'param_id_obs_path': os.path.join(resources_dir, 'heat1d_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 160, 'cost_type': 'gaussian_MLE'},
    })
    return config


def _write_model(tmp_path, body, name='tmp_model.py'):
    """Write a throwaway external model file and return its path."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return str(path)


# A minimal, correct class the contract tests vary one piece of at a time.
_MINIMAL_MODEL = '''
    import numpy as np

    class Tiny:
        parameters = {"tiny/a": 2.0, "tiny/b": 3.0}
        output_names = ["tiny/y"]

        def init_solver(self, config):
            self.vals = dict(self.parameters)
            self.config_seen = config
            self.update_times(config['dt'], config['start_time'],
                              config['sim_time'], config['pre_time'])

        def update_times(self, dt, start_time, sim_time, pre_time):
            self.n = int(pre_time / dt) + int(sim_time / dt) + 1
            self.t = start_time + np.arange(self.n) * dt

        def set_param_vals(self, param_dict):
            self.vals.update(param_dict)

        def run(self):
            self.y = self.vals["tiny/a"] * self.t + self.vals["tiny/b"]
            return True

        def get_results(self):
            return {"tiny/y": self.y}

    SIM_HELPER = Tiny
'''


# ---------------------------------------------------------------------------
# The shipped obs_data
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_shipped_obs_data_carries_the_window_its_values_were_computed_on():
    """``heat1d_obs_data.json`` must name its own run window in ``protocol_info``.

    The three probe means below it are means over the 0.5 s window and nothing else, so the
    window belongs in the file rather than in whatever ``user_inputs.yaml`` happens to say. It
    is also the only place the CUFLynx GUI looks when it sizes a run: an example with no
    ``protocol_info`` cannot be run from it at all.

    (This is about what the *example* ships, not about what the parser allows. CA still
    accepts a bare list of data_items and synthesises a protocol from the yaml's window;
    ``tests/test_operation_kwargs.py::test_shipped_extra_ops_obs_data_json_still_validates``
    is where that support is pinned.)
    """
    with open(_OBS_DATA_PATH) as handle:
        shipped = json.load(handle)

    protocol = shipped['protocol_info']
    # pre_times is per experiment; sim_times is per experiment, per subexperiment. One
    # experiment of one 0.5 s stretch, from t = 0, with no spin-up to discard.
    assert protocol['pre_times'] == [0.0]
    assert protocol['sim_times'] == [[_SIM_TIME]]

    # And it still parses -- with no pre_time/sim_time offered, so the file has to be
    # self-sufficient rather than falling back on the yaml's window.
    parsed = ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=_OBS_DATA_PATH)
    assert parsed['protocol_info']['pre_times'] == [0.0]
    assert parsed['protocol_info']['sim_times'] == [[_SIM_TIME]]
    assert len(parsed['gt_df']) == len(shipped['data_items']) == 3


# ---------------------------------------------------------------------------
# Generation, config plumbing, factory dispatch
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_external_generation_is_noop(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """Generation is a no-op success when the model file exists, and fails clearly when it
    doesn't -- there is nothing to generate, only something to point at."""
    config = _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    assert generate_with_new_architecture(False, config) is True

    missing = _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    missing['external_model_path'] = os.path.join(_EXAMPLE_DIR, 'does_not_exist_model.py')
    assert generate_with_new_architecture(False, missing) is False


@pytest.mark.unit
def test_the_config_resolves_to_the_external_model_and_solver(
        base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """external_model_path becomes model_path; the solver and its placeholder method survive
    validation."""
    config = _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)

    assert parsed['model_path'] == _MODEL_PATH
    assert parsed['uncalibrated_model_path'] == _MODEL_PATH
    assert parsed['solver_info']['solver'] == 'external'
    assert parsed['solver_info']['method'] == 'external'


@pytest.mark.unit
def test_the_default_model_path_follows_the_file_prefix(
        base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """Without an explicit external_model_path, the model is funcs_user/{prefix}_model.py."""
    config = _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    del config['external_model_path']
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)

    assert parsed['model_path'].endswith(os.path.join('funcs_user', 'heat1d_model.py'))


@pytest.mark.unit
def test_the_factory_refuses_the_external_solver_for_other_model_types():
    with pytest.raises(ValueError, match="external_python"):
        get_simulation_helper(model_path=_MODEL_PATH, solver='external', model_type='python',
                              dt=0.005, sim_time=0.5, solver_info=dict(_SOLVER_INFO))


# ---------------------------------------------------------------------------
# The backend itself
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.solver
def test_external_backend_roundtrip(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """The factory routes external_python to a working SimulationHelper that drives the user's
    own solver and returns named results."""
    config = _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)

    sim = get_simulation_helper_from_inp_data_dict(parsed)
    assert isinstance(sim, ExternalSimulationHelper)
    assert sim.get_all_variable_names() == ['heat/T_p1', 'heat/T_p2', 'heat/T_p3',
                                            'heat/k', 'heat/u_D', 'time']

    # Defaults come from the class's `parameters` attribute.
    assert sim.get_init_param_vals(['heat/k', 'heat/u_D']) == [0.4, 0.0]

    sim.set_param_vals(['heat/k', 'heat/u_D'], [_TRUE_K, _TRUE_U_D])
    assert sim.run() is True

    t = sim.get_time()
    p1, p2, p3 = sim.get_results(['heat/T_p1', 'heat/T_p2', 'heat/T_p3'], flatten=True)
    assert len(t) == len(p1) == len(p2) == len(p3) == int(0.5 / 0.005) + 1
    assert abs(t[0]) < 1e-9
    # The targets baked into heat1d_obs_data.json, computed at the true parameters.
    assert np.mean(p1) == pytest.approx(0.25338190, abs=1e-6)
    assert np.mean(p2) == pytest.approx(0.23883291, abs=1e-6)
    assert np.mean(p3) == pytest.approx(0.14911117, abs=1e-6)
    # Repeatable: run() restarts from the initial condition every time.
    assert sim.run() is True
    assert np.mean(sim.get_results(['heat/T_p2'], flatten=True)[0]) == pytest.approx(0.23883291,
                                                                                    abs=1e-6)


@pytest.mark.unit
def test_the_first_logged_time_is_pre_time_not_zero(tmp_path):
    """The double-shift guard.

    ``tSim`` keeps the pre-time offset, because ``protocol_executor`` subtracts
    ``pre_times[exp]`` from it itself. Removing it here as well would make every logged trace
    start at ``-pre_time``, which reads as a plausible time vector and silently misaligns every
    series comparison.
    """
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO), pre_time=0.5)

    assert sim.pre_steps == 5
    assert sim.n_steps == 10
    assert sim.tSim[0] == pytest.approx(0.5)
    assert sim.tSim[-1] == pytest.approx(1.5)
    assert len(sim.tSim) == 11
    # get_time() is the pre-time-relative view, and only it starts at zero.
    assert sim.get_time()[0] == pytest.approx(0.0)
    assert sim.get_time(include_pre_time=True)[0] == pytest.approx(0.5)

    assert sim.run() is True
    # The pre-time samples are dropped from the results, so a result lines up with tSim.
    y = sim.get_results(['tiny/y'], flatten=True)[0]
    assert len(y) == len(sim.tSim)
    # y = a*t + b over the FULL grid; the first logged sample is the one at t = pre_time.
    assert y[0] == pytest.approx(2.0 * 0.5 + 3.0)


@pytest.mark.unit
def test_time_is_special_cased_and_a_flat_name_list_is_promoted(tmp_path):
    """param_id passes a list of lists (one per observable's operands) and gets a list of lists
    back; a flat list of names is promoted so plain callers do not have to wrap theirs."""
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    assert sim.run() is True

    grouped = sim.get_results([['tiny/y', 'time'], ['tiny/y']])
    assert [len(row) for row in grouped] == [2, 1]
    assert np.allclose(grouped[0][1], sim.tSim)

    flat_input = sim.get_results(['tiny/y', 'time'])
    assert [len(row) for row in flat_input] == [1, 1]
    assert np.allclose(flat_input[1][0], sim.tSim)
    assert np.allclose(sim.get_results(['time'], flatten=True)[0], sim.tSim)


@pytest.mark.unit
def test_a_parameter_can_be_read_back_as_a_series(tmp_path):
    """get_all_results has to answer for parameters too, and a parameter is constant over the
    run rather than absent from it."""
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    assert sim.run() is True

    all_results = sim.get_all_results_dict()
    assert set(all_results) == {'tiny/y', 'tiny/a', 'tiny/b', 'time'}
    assert np.allclose(all_results['tiny/a'], 2.0)
    assert len(all_results['tiny/a']) == len(sim.tSim)


@pytest.mark.unit
def test_results_before_a_run_say_so(tmp_path):
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    with pytest.raises(RuntimeError, match=r"run\(\)"):
        sim.get_results(['tiny/y'])
    with pytest.raises(RuntimeError, match="not been run"):
        sim.get_all_results_dict()


@pytest.mark.unit
def test_a_wrong_length_result_names_the_grid_that_produced_it(tmp_path):
    """A short array is a bug in the user's class, not a diverged solve, so it raises -- and the
    message has to carry enough to find it: what was expected, what arrived, and the dt/sim_time
    that set the expectation."""
    body = _MINIMAL_MODEL.replace('return {"tiny/y": self.y}',
                                  'return {"tiny/y": self.y[:-2]}')
    path = _write_model(tmp_path, body, name='short_model.py')
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))

    with pytest.raises(ValueError) as excinfo:
        sim.run()
    message = str(excinfo.value)
    assert '9 samples' in message and 'expected 11' in message
    assert 'dt=0.1' in message and 'sim_time=1.0' in message
    assert 'tiny/y' in message


@pytest.mark.unit
def test_a_missing_output_is_named(tmp_path):
    body = _MINIMAL_MODEL.replace('return {"tiny/y": self.y}', 'return {}')
    path = _write_model(tmp_path, body, name='empty_model.py')
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    with pytest.raises(ValueError, match=r"missing output\(s\) \['tiny/y'\]"):
        sim.run()


@pytest.mark.unit
def test_a_raising_run_becomes_false_not_a_crash(tmp_path, capsys):
    """A user solver blowing up on a bad candidate is an ordinary event during calibration: it
    must return False (which the cost turns into inf) while still printing the traceback, so the
    failure is visible rather than swallowed."""
    body = _MINIMAL_MODEL.replace('self.y = self.vals["tiny/a"] * self.t + self.vals["tiny/b"]',
                                  'raise ArithmeticError("solver exploded")')
    path = _write_model(tmp_path, body, name='raising_model.py')
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))

    assert sim.run() is False
    assert 'solver exploded' in capsys.readouterr().out


@pytest.mark.unit
def test_a_run_returning_false_is_propagated(tmp_path):
    body = _MINIMAL_MODEL.replace('            return True\n', '            return False\n', 1)
    path = _write_model(tmp_path, body, name='diverging_model.py')
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    assert sim.run() is False


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_grouped_params_for_id_row_sets_every_name_in_it(tmp_path):
    """One calibrated value driving several model parameters. ``zip`` would set only the first,
    which is the bug pair_names_with_values exists to prevent."""
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))

    sim.set_param_vals([['tiny/a', 'tiny/b']], [7.0])
    assert sim.get_init_param_vals([['tiny/a', 'tiny/b']]) == [[7.0, 7.0]]
    assert sim.user.vals == {'tiny/a': 7.0, 'tiny/b': 7.0}

    # And a genuine length mismatch raises rather than truncating.
    with pytest.raises(ValueError, match='grouped entry'):
        sim.set_param_vals([['tiny/a', 'tiny/b']], [[1.0, 2.0, 3.0]])


@pytest.mark.unit
def test_get_default_param_vals_serves_the_declared_defaults(tmp_path):
    """A modifier reads its baseline from here, so it must be the declared default and not the
    live value -- otherwise theta*baseline compounds every calibration iteration."""
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))

    sim.set_param_vals(['tiny/a'], [99.0])
    assert sim.get_init_param_vals(['tiny/a']) == [99.0]
    assert sim.get_default_param_vals(['tiny/a']) == [2.0]
    assert sim.get_default_param_vals([['tiny/a', 'tiny/b']]) == [[2.0, 3.0]]


@pytest.mark.unit
def test_the_wrapper_tracks_values_when_the_user_class_does_not(tmp_path):
    """``get_init_param_vals`` is optional on the user class; without it the wrapper answers from
    what it has been told to set, and with it the user's answer is used."""
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    assert not hasattr(sim.user, 'get_init_param_vals')
    sim.set_param_vals(['tiny/b'], [11.0])
    assert sim.get_init_param_vals(['tiny/a', 'tiny/b']) == [2.0, 11.0]

    delegating = _MINIMAL_MODEL + '''
    def _user_get_init(self, names):
        return [-1.0 for _ in names]

    Tiny.get_init_param_vals = _user_get_init
'''
    path2 = _write_model(tmp_path, delegating, name='delegating_model.py')
    sim2 = ExternalSimulationHelper(path2, 0.1, 1.0, dict(_SOLVER_INFO))
    assert sim2.get_init_param_vals(['tiny/a', 'tiny/b']) == [-1.0, -1.0]


@pytest.mark.unit
def test_a_protocol_trace_key_is_refused_clearly(tmp_path):
    """protocol_traces drive a variable from a time series; only CVODE_myokit implements that.
    A string value here would otherwise reach float() and fail as a type error, hiding what was
    actually asked for."""
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    with pytest.raises(NotImplementedError, match='CVODE_myokit'):
        sim.set_param_vals(['tiny/a'], ['some_trace_key'])


@pytest.mark.unit
def test_an_undeclared_parameter_is_refused_with_the_declared_ones(tmp_path):
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    with pytest.raises(ValueError, match=r"tiny/c.*not declared"):
        sim.set_param_vals(['tiny/c'], [1.0])


@pytest.mark.unit
def test_reset_and_clear_puts_the_parameters_back_to_their_defaults(tmp_path):
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    sim.set_param_vals(['tiny/a', 'tiny/b'], [50.0, 60.0])
    assert sim.run() is True

    sim.reset_and_clear()
    assert sim.get_init_param_vals(['tiny/a', 'tiny/b']) == [2.0, 3.0]
    # The last results survive as a cached dict, exactly as the scipy backend does.
    assert np.allclose(sim.get_all_results_dict()['tiny/a'], 50.0)


@pytest.mark.unit
def test_offline_pre_time_is_refused_with_a_reason(tmp_path):
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))
    with pytest.raises(NotImplementedError, match='state carry-over'):
        sim.run_offline_pre_and_set_default_state(1.0)


# ---------------------------------------------------------------------------
# The contract, as enforced at load time
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_file_without_sim_helper_says_which_name_is_missing(tmp_path):
    path = _write_model(tmp_path, _MINIMAL_MODEL.replace('SIM_HELPER = Tiny', ''),
                        name='no_registration.py')
    with pytest.raises(ValueError, match='does not define SIM_HELPER'):
        ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))


@pytest.mark.unit
def test_sim_helper_must_be_the_class_not_an_instance(tmp_path):
    path = _write_model(tmp_path, _MINIMAL_MODEL.replace('SIM_HELPER = Tiny',
                                                         'SIM_HELPER = Tiny()'),
                        name='instance_registration.py')
    with pytest.raises(ValueError, match='must be the solver class itself'):
        ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))


@pytest.mark.unit
def test_a_missing_required_method_is_reported_up_front(tmp_path):
    body = _MINIMAL_MODEL.replace('        def run(self):', '        def run_it(self):')
    path = _write_model(tmp_path, body, name='no_run.py')
    with pytest.raises(ValueError, match=r"missing required method\(s\) \['run'\]"):
        ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))


@pytest.mark.unit
@pytest.mark.parametrize('bad, match', [
    ('parameters = ["tiny/a"]', 'must be a literal dict'),
    ('parameters = {"a": 1.0}', "must be of the form 'component/variable'"),
    ('parameters = {"tiny/a": "one"}', 'must be a number'),
])
def test_a_malformed_parameters_attribute_is_named(tmp_path, bad, match):
    body = _MINIMAL_MODEL.replace('parameters = {"tiny/a": 2.0, "tiny/b": 3.0}', bad)
    path = _write_model(tmp_path, body, name='bad_params.py')
    with pytest.raises(ValueError, match=match):
        ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))


@pytest.mark.unit
@pytest.mark.parametrize('bad, match', [
    ('output_names = "tiny/y"', 'must be a literal list'),
    ('output_names = ["y"]', r"output name 'y' lacks a '/'"),
    ('output_names = []', 'is empty'),
])
def test_a_malformed_output_names_attribute_is_named(tmp_path, bad, match):
    body = _MINIMAL_MODEL.replace('output_names = ["tiny/y"]', bad)
    path = _write_model(tmp_path, body, name='bad_outputs.py')
    with pytest.raises(ValueError, match=match):
        ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))


@pytest.mark.unit
def test_two_external_models_can_be_alive_at_once(tmp_path):
    """The module name is derived from the file path, not a fixed literal: under a shared name
    the second import would rebind the first's module."""
    first = _write_model(tmp_path, _MINIMAL_MODEL, name='first_model.py')
    second = _write_model(tmp_path,
                          _MINIMAL_MODEL.replace('"tiny/a": 2.0', '"tiny/a": 20.0'),
                          name='second_model.py')
    sim_a = ExternalSimulationHelper(first, 0.1, 1.0, dict(_SOLVER_INFO))
    sim_b = ExternalSimulationHelper(second, 0.1, 1.0, dict(_SOLVER_INFO))
    assert sim_a.get_default_param_vals(['tiny/a']) == [2.0]
    assert sim_b.get_default_param_vals(['tiny/a']) == [20.0]


@pytest.mark.unit
def test_user_config_reaches_the_user_class(tmp_path):
    """The one solver_info setting this backend declares, and the only channel a user class has
    for options CA knows nothing about."""
    path = _write_model(tmp_path, _MINIMAL_MODEL)
    solver_info = dict(_SOLVER_INFO, user_config={'mesh': 'fine', 'threads': 4})
    sim = ExternalSimulationHelper(path, 0.1, 1.0, solver_info)

    config = sim.user.config_seen
    assert config['solver_info']['user_config'] == {'mesh': 'fine', 'threads': 4}
    assert config['user_config'] == {'mesh': 'fine', 'threads': 4}
    assert (config['dt'], config['sim_time'], config['pre_time'], config['start_time']) == \
        (0.1, 1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# extra_plots
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_get_extra_figures_returns_the_users_figure_and_nothing_when_absent(tmp_path):
    """The optional hook is a hook, not a requirement: a class without it contributes an empty
    list rather than failing."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')

    sim = ExternalSimulationHelper(_MODEL_PATH, 0.005, 0.1, dict(_SOLVER_INFO))
    assert sim.run() is True
    figures = sim.get_extra_figures()
    assert len(figures) == 1
    assert isinstance(figures[0], matplotlib.figure.Figure)

    # The colorbar's ticks are bounded to 2 significant figures: the default
    # formatter prints float artefacts (0.30000000000000004) as tick labels.
    colorbar_axes = [ax for ax in figures[0].axes if ax.get_label() == '<colorbar>']
    assert len(colorbar_axes) == 1
    formatter = colorbar_axes[0].yaxis.get_major_formatter()
    assert formatter(0.30000000000000004, None) == '0.3'
    matplotlib.pyplot.close(figures[0])

    plain = ExternalSimulationHelper(_write_model(tmp_path, _MINIMAL_MODEL), 0.1, 1.0,
                                     dict(_SOLVER_INFO))
    assert plain.get_extra_figures() == []


# ---------------------------------------------------------------------------
# End to end: calibration, global SA, local (FD) sensitivities
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_external_recovers_params(base_user_inputs, temp_output_dir,
                                           temp_generated_models_dir):
    """A genetic-algorithm calibration runs end-to-end against a user-owned solver and recovers
    the diffusivity used to build the observed data (starting from the k=0.4 default)."""
    rank = MPI.COMM_WORLD.Get_rank()
    config = _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)

    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(temp_output_dir,
                                  'genetic_algorithm_heat1d_heat1d_obs_data')
        best_path = os.path.join(output_dir, 'best_param_vals.npy')
        assert os.path.exists(best_path), f"expected calibration output at {best_path}"

        best = np.load(best_path)
        names = np.loadtxt(os.path.join(output_dir, 'param_names.csv'), dtype=str, delimiter=',')
        vals = {str(n): float(v) for n, v in zip(np.atleast_1d(names), np.atleast_1d(best))}

        k = vals.get('heat/k')
        u_D = vals.get('heat/u_D')
        assert k is not None and u_D is not None, f"unexpected param names: {vals}"
        assert 0.05 <= k <= 1.0 and -0.5 <= u_D <= 0.5
        # Small GA budget, so the tolerance is generous -- but it must have moved off the
        # k=0.4 default and towards the truth.
        assert k == pytest.approx(_TRUE_K, abs=0.1)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_sensitivity_analysis_external_succeeds(base_user_inputs, temp_output_dir,
                                                temp_generated_models_dir):
    """Sobol sensitivity analysis runs end-to-end over both parameters of an external model."""
    rank = MPI.COMM_WORLD.Get_rank()
    config = _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    config['sa_options'] = {
        'method': 'sobol',
        'num_samples': 16,
        'sample_type': 'saltelli',
        'output_dir': os.path.join(temp_output_dir, 'heat1d_SA_results'),
    }

    run_SA(config)

    if rank == 0:
        assert os.path.exists(config['sa_options']['output_dir'])


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_fd_observable_sensitivities_on_an_external_model(base_user_inputs, temp_output_dir,
                                                          temp_generated_models_dir):
    """Local sensitivities via finite differences.

    There is no analytic arm for an external model -- CA cannot differentiate a solver it does
    not own -- so FD is the whole story here, and it has to work through the ordinary param-id
    object rather than a special path.
    """
    from libcuflynx.param_id.paramID import CVS0DParamID

    if MPI.COMM_WORLD.Get_rank() != 0:
        pytest.skip('single-rank check')

    config = _heat1d_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    # init_from_dict wants the *resolved* config: model_path, params_for_id_path and
    # param_id_obs_path are all derived by the parser.
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=True)
    pid = CVS0DParamID.init_from_dict(parsed)

    engine = pid.param_id
    sens = engine.get_observable_sensitivities(
        np.array([_TRUE_K, _TRUE_U_D]), gradient_method='FD')

    assert sens, 'no observable sensitivities were produced'
    for obs_label, by_param in sens.items():
        assert set(by_param) == {'heat/k', 'heat/u_D'}, (obs_label, by_param)
        for value in by_param.values():
            assert value is not None and np.isfinite(value)

    # Compare against a central difference taken straight through the wrapper. Asserting only
    # "finite" would pass on any number at all, including one differenced against the wrong
    # parameter -- the mistake worth catching here.
    reference = _reference_derivatives()
    labels = [engine._observable_label(i) for i in engine.obs_info['const_idx_to_obs_idx']]
    for label, (d_dk, d_du_d) in zip(labels, reference):
        assert sens[label]['heat/k'] == pytest.approx(d_dk, rel=2e-2, abs=1e-3)
        assert sens[label]['heat/u_D'] == pytest.approx(d_du_d, rel=2e-2, abs=1e-3)
    # The three probes are not interchangeable: p1 and p2 sit under the initial bump and cool
    # faster as k rises, while p3 is far enough away that the spreading bump warms it.
    assert reference[0][0] < 0 and reference[1][0] < 0 and reference[2][0] > 0


def _reference_derivatives():
    """d(probe mean)/d(k, u_D) at the true parameters, by central difference on the wrapper."""
    sim = ExternalSimulationHelper(_MODEL_PATH, 0.005, 0.5, dict(_SOLVER_INFO))

    def features(k, u_D):
        sim.set_param_vals(['heat/k', 'heat/u_D'], [k, u_D])
        assert sim.run() is True
        return np.array([series.mean() for series in
                         sim.get_results(['heat/T_p1', 'heat/T_p2', 'heat/T_p3'], flatten=True)])

    h_k, h_u = 1e-4, 1e-4
    d_dk = (features(_TRUE_K + h_k, _TRUE_U_D) - features(_TRUE_K - h_k, _TRUE_U_D)) / (2 * h_k)
    d_du = (features(_TRUE_K, _TRUE_U_D + h_u) - features(_TRUE_K, _TRUE_U_D - h_u)) / (2 * h_u)
    return list(zip(d_dk, d_du))


# A class whose extra_plots() refuses, the way a field solver's does when its last run
# diverged or never happened: the snapshots it draws from are simply not there.
_REFUSING_PLOTS_MODEL = _MINIMAL_MODEL.replace(
    '    SIM_HELPER = Tiny',
    '''        def extra_plots(self):
            raise RuntimeError('extra_plots() was called before a successful run(); '
                               'there are no fields to draw yet')

    SIM_HELPER = Tiny''')


@pytest.mark.unit
def test_a_hook_that_declines_to_draw_does_not_fail_the_simulation(tmp_path, capsys):
    """Decorative output must not decide whether the simulation succeeded.

    ``get_extra_figures`` already tolerates a *missing* ``extra_plots``; it has to tolerate
    one that raises for the same reason. A field solver draws from state its last run built,
    so a diverged run -- an ordinary event during a calibration, reported by ``run()``
    returning False -- leaves it with nothing to draw. Letting that propagate turned a
    legitimate "no fit at these parameters" into ``Simulation failed:``, under a banner
    blaming solver tolerances that pointed nowhere near the cause (the shipped
    ``funcs_user/heat_fenics`` model did exactly this).

    The reason is printed rather than swallowed, so a hook that is genuinely broken is still
    discoverable.
    """
    path = _write_model(tmp_path, _REFUSING_PLOTS_MODEL)
    sim = ExternalSimulationHelper(path, 0.1, 1.0, dict(_SOLVER_INFO))

    assert sim.get_extra_figures() == []
    assert 'extra_plots' in capsys.readouterr().out

    # And the run itself is unaffected -- the point is that plotting cannot break it.
    assert sim.run() is True
