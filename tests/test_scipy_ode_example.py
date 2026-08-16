"""Tests for the scipy-ODE example of the ``model_type: external_python`` contract.

``funcs_user/example_model_scipy/`` is the damped oscillator that used to be the
example of an ODE solved by a class that calls ``scipy.integrate.solve_ivp``
inside its own ``run()``. It is the simple end of the external contract, and it is the shape most
users arriving with "I have an ODE" need, so it gets its own coverage rather than riding on
``tests/test_external_simulation_helper.py`` (which exercises the wrapper's contract enforcement
against a hand-marched PDE).
"""
import os
import shutil

import numpy as np
import pytest
from mpi4py import MPI

from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
from libcuflynx.solver_wrappers import get_simulation_helper, get_simulation_helper_from_inp_data_dict
from libcuflynx.solver_wrappers.external_simulation_helper import SimulationHelper as ExternalSimulationHelper
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.scripts.param_id_run_script import run_param_id


_EXAMPLE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'funcs_user', 'example_model_scipy')
)
_MODEL_PATH = os.path.join(_EXAMPLE_DIR, 'oscillator_model.py')
# Ground truth used to build oscillator_obs_data.json (the defaults are c=0.5, k=4.0).
_TRUE_C, _TRUE_K = 0.7, 5.0
_SOLVER_INFO = {'solver': 'external', 'method': 'external'}
# The observable targets baked into oscillator_obs_data.json, produced by running the model's
# own __main__ block at the true parameters.
_TRUE_MEAN_X = 0.01663962
_TRUE_MIN_X = -0.60704870
_TRUE_RANGE_V = 2.87274454

_DT, _SIM_TIME = 0.05, 10.0


def _make_resources(temp_output_dir):
    """Copy the example resource files into an isolated per-test resources dir, so a run never
    writes its dated config back into the repo's example directory."""
    comm = MPI.COMM_WORLD
    resources_dir = os.path.join(temp_output_dir, 'resources')
    if comm.Get_rank() == 0:
        os.makedirs(resources_dir, exist_ok=True)
        for name in ('oscillator_params_for_id.csv', 'oscillator_parameters.csv',
                     'oscillator_obs_data.json'):
            shutil.copy(os.path.join(_EXAMPLE_DIR, name), os.path.join(resources_dir, name))
    if comm.Get_size() > 1:
        comm.Barrier()
    return resources_dir


def _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    resources_dir = _make_resources(temp_output_dir)
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'oscillator',
        'input_param_file': 'oscillator_parameters.csv',
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
        'model_out_names': ['oscillator/x'],
        'param_id_obs_path': os.path.join(resources_dir, 'oscillator_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 160, 'cost_type': 'gaussian_MLE'},
    })
    return config


# ---------------------------------------------------------------------------
# Loading the example by path
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_example_loads_by_path_and_declares_its_surface():
    """The class is found through SIM_HELPER and its two literal attributes become the
    parameters and outputs CA addresses it by."""
    sim = ExternalSimulationHelper(_MODEL_PATH, _DT, _SIM_TIME, dict(_SOLVER_INFO))

    assert sim.get_all_variable_names() == ['oscillator/x', 'oscillator/v', 'oscillator/energy',
                                            'oscillator/c', 'oscillator/k', 'time']
    assert sim.get_default_param_vals(['oscillator/c', 'oscillator/k']) == [0.5, 4.0]


@pytest.mark.unit
def test_the_config_resolves_to_the_example_and_the_external_solver(
        base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """external_model_path becomes model_path, generation is a no-op success, and the factory
    hands back the external backend rather than the scipy libCellML one."""
    config = _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    assert generate_with_new_architecture(False, config) is True

    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)
    assert parsed['model_path'] == _MODEL_PATH
    assert parsed['solver_info']['solver'] == 'external'

    sim = get_simulation_helper_from_inp_data_dict(parsed)
    assert isinstance(sim, ExternalSimulationHelper)


# ---------------------------------------------------------------------------
# The grid, the physics, and repeatability
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.solver
def test_the_grid_length_is_the_documented_arithmetic():
    """N = int(pre_time/dt) + int(sim_time/dt) samples plus the endpoint, and the pre_time
    samples are produced by the model and dropped by CA -- the bookkeeping the retired
    framework does not do for you, which this example shows."""
    sim = ExternalSimulationHelper(_MODEL_PATH, _DT, _SIM_TIME, dict(_SOLVER_INFO), pre_time=1.0)

    assert sim.pre_steps == int(1.0 / _DT)
    assert sim.n_steps == int(_SIM_TIME / _DT)
    assert sim.run() is True

    # The model returned the whole grid, pre_time included...
    assert len(sim._results['oscillator/x']) == sim.pre_steps + sim.n_steps + 1
    # ...and only the logged portion comes back out.
    x = sim.get_results(['oscillator/x'], flatten=True)[0]
    assert len(x) == len(sim.get_time()) == sim.n_steps + 1
    assert sim.get_time()[0] == pytest.approx(0.0)
    assert sim.get_time(include_pre_time=True)[0] == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.solver
def test_the_example_reproduces_the_obs_data_targets():
    """At the parameters the fixtures were built from, the three observables are the values
    baked into oscillator_obs_data.json. This is the check that the port of the physics from the
    retired wrapper is faithful, not just runnable."""
    sim = ExternalSimulationHelper(_MODEL_PATH, _DT, _SIM_TIME, dict(_SOLVER_INFO))
    sim.set_param_vals(['oscillator/c', 'oscillator/k'], [_TRUE_C, _TRUE_K])
    assert sim.run() is True

    x, v, energy = sim.get_results(['oscillator/x', 'oscillator/v', 'oscillator/energy'],
                                   flatten=True)
    assert len(x) == int(_SIM_TIME / _DT) + 1
    assert np.mean(x) == pytest.approx(_TRUE_MEAN_X, abs=1e-7)
    assert np.min(x) == pytest.approx(_TRUE_MIN_X, abs=1e-7)
    assert np.max(v) - np.min(v) == pytest.approx(_TRUE_RANGE_V, abs=1e-7)
    # The algebraic output needs no separate hook under this contract: energy = (v^2 + k x^2)/2,
    # decaying because the oscillator is damped.
    assert np.allclose(energy, 0.5 * (v ** 2 + _TRUE_K * x ** 2))
    assert energy[0] == pytest.approx(0.5 * _TRUE_K)
    assert energy[-1] < energy[0]


@pytest.mark.unit
@pytest.mark.solver
def test_run_is_repeatable_and_a_parameter_change_takes_effect():
    """run() restarts from the initial condition every time, so A, B, A gives the same trace
    twice -- the rule most easily broken by a class that carries state -- and B differs from A,
    so a calibration is actually moving the model."""
    sim = ExternalSimulationHelper(_MODEL_PATH, _DT, _SIM_TIME, dict(_SOLVER_INFO))

    sim.set_param_vals(['oscillator/c', 'oscillator/k'], [_TRUE_C, _TRUE_K])
    assert sim.run() is True
    first = sim.get_results(['oscillator/x'], flatten=True)[0].copy()

    sim.set_param_vals(['oscillator/c', 'oscillator/k'], [1.5, 9.0])
    assert sim.run() is True
    other = sim.get_results(['oscillator/x'], flatten=True)[0].copy()
    assert sim.get_init_param_vals(['oscillator/c', 'oscillator/k']) == [1.5, 9.0]
    # Heavier damping and a stiffer spring: a visibly different trajectory.
    assert not np.allclose(first, other, atol=1e-3)
    assert np.min(other) > np.min(first)

    sim.set_param_vals(['oscillator/c', 'oscillator/k'], [_TRUE_C, _TRUE_K])
    assert sim.run() is True
    again = sim.get_results(['oscillator/x'], flatten=True)[0]
    assert np.array_equal(first, again)


@pytest.mark.unit
def test_user_config_carries_the_solve_ivp_settings():
    """The retired backend took method/rtol/atol as solver_info keys CA read. They now belong to
    the model, through the free-form user_config -- the example reads them in init_solver."""
    solver_info = dict(_SOLVER_INFO, user_config={'method': 'LSODA', 'rtol': 1e-6, 'atol': 1e-9})
    sim = ExternalSimulationHelper(_MODEL_PATH, _DT, _SIM_TIME, solver_info)

    assert (sim.user.method, sim.user.rtol, sim.user.atol) == ('LSODA', 1e-6, 1e-9)
    assert sim.run() is True
    x = sim.get_results(['oscillator/x'], flatten=True)[0]

    # And the defaults apply when nothing is configured.
    plain = ExternalSimulationHelper(_MODEL_PATH, _DT, _SIM_TIME, dict(_SOLVER_INFO))
    assert (plain.user.method, plain.user.rtol, plain.user.atol) == ('RK45', 1e-8, 1e-8)
    assert plain.run() is True
    # A different integrator at looser tolerances, but the same physics: the setting reaches the
    # solver (it is not silently ignored) without changing the answer.
    assert np.allclose(x, plain.get_results(['oscillator/x'], flatten=True)[0], atol=1e-5)


# ---------------------------------------------------------------------------
# End to end: a DEBUG-scale calibration
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_scipy_example_recovers_the_stiffness(base_user_inputs, temp_output_dir,
                                                       temp_generated_models_dir):
    """A DEBUG-scale genetic-algorithm calibration runs end-to-end against the example and
    recovers the stiffness used to build the observed data, starting from the k=4.0 default."""
    rank = MPI.COMM_WORLD.Get_rank()
    config = _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)

    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(temp_output_dir,
                                  'genetic_algorithm_oscillator_oscillator_obs_data')
        best_path = os.path.join(output_dir, 'best_param_vals.npy')
        assert os.path.exists(best_path), f"expected calibration output at {best_path}"

        best = np.load(best_path)
        names = np.loadtxt(os.path.join(output_dir, 'param_names.csv'), dtype=str, delimiter=',')
        vals = {str(n): float(v) for n, v in zip(np.atleast_1d(names), np.atleast_1d(best))}

        c = vals.get('oscillator/c')
        k = vals.get('oscillator/k')
        assert c is not None and k is not None, f"unexpected param names: {vals}"
        assert 0.05 <= c <= 2.0 and 1.0 <= k <= 10.0
        # Small GA budget, so the tolerances are generous -- but it must have moved off the
        # defaults and towards the truth.
        assert k == pytest.approx(_TRUE_K, abs=1.0)
        assert c == pytest.approx(_TRUE_C, abs=0.4)
