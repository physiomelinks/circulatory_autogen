"""
Tests for the ``model_type: python_user_defined`` backend.

These verify that an arbitrary user-supplied Python ODE model (wrapped in
funcs_user/) can be driven through the calibration, sensitivity-analysis and
identifiability pipelines, using the shipped damped-oscillator example in
funcs_user/example_model/.
"""
import os
import shutil

import numpy as np
import pytest
from mpi4py import MPI

from parsers.PrimitiveParsers import YamlFileParser
from solver_wrappers import get_simulation_helper_from_inp_data_dict
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from scripts.sensitivity_analysis_run_script import run_SA
from scripts.param_id_run_script import run_param_id


_EXAMPLE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'funcs_user', 'example_model')
)
_WRAPPER_PATH = os.path.join(_EXAMPLE_DIR, 'oscillator_wrapper.py')
# Ground truth used to build oscillator_obs_data.json.
_TRUE_C, _TRUE_K = 0.7, 5.0


def _make_resources(temp_output_dir):
    """Copy the example resource files into an isolated per-test resources dir
    (so the run never writes dated configs back into the repo example dir)."""
    comm = MPI.COMM_WORLD
    resources_dir = os.path.join(temp_output_dir, 'resources')
    if comm.Get_rank() == 0:
        os.makedirs(resources_dir, exist_ok=True)
        for name in ('oscillator_params_for_id.csv', 'oscillator_parameters.csv',
                     'oscillator_obs_data.json'):
            shutil.copy(os.path.join(_EXAMPLE_DIR, name),
                        os.path.join(resources_dir, name))
    if comm.Get_size() > 1:
        comm.Barrier()
    return resources_dir


def _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    resources_dir = _make_resources(temp_output_dir)
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'oscillator',
        'input_param_file': 'oscillator_parameters.csv',
        'model_type': 'python_user_defined',
        'solver': 'user_defined',
        # The wrapper is integrated by solve_ivp; override the base solver_info
        # (which carries CVODE-only keys) and let the method default to RK45.
        'solver_info': {'solver': 'user_defined'},
        'model_wrapper_path': _WRAPPER_PATH,
        'resources_dir': resources_dir,
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.0,
        'sim_time': 10.0,
        'dt': 0.05,
        'DEBUG': True,
        'do_mcmc': False,
        'do_ad': False,
        'plot_predictions': False,
        'model_out_names': ['oscillator/x'],
        'param_id_obs_path': os.path.join(resources_dir, 'oscillator_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 150, 'cost_type': 'gaussian_MLE'},
    })
    return config


@pytest.mark.unit
def test_python_user_defined_generation_is_noop(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """Generation is a no-op success when the wrapper exists, and fails clearly when it doesn't."""
    config = _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    assert generate_with_new_architecture(False, config) is True

    missing = _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    missing['model_wrapper_path'] = os.path.join(_EXAMPLE_DIR, 'does_not_exist_wrapper.py')
    assert generate_with_new_architecture(False, missing) is False


@pytest.mark.unit
@pytest.mark.solver
def test_python_user_defined_backend_roundtrip(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """The factory routes python_user_defined to a working SimulationHelper that
    integrates the user's rhs and returns named results."""
    config = _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)

    assert parsed['model_path'] == _WRAPPER_PATH
    assert parsed['solver_info']['solver'] == 'user_defined'

    sim = get_simulation_helper_from_inp_data_dict(parsed)
    assert 'oscillator/x' in sim.get_all_variable_names()

    # Defaults come from the wrapper PARAMETERS.
    assert sim.get_init_param_vals(['oscillator/c', 'oscillator/k']) == [0.5, 4.0]

    sim.set_param_vals(['oscillator/c', 'oscillator/k'], [_TRUE_C, _TRUE_K])
    assert sim.run() is True

    t = sim.get_time()
    x = sim.get_results(['oscillator/x'], flatten=True)[0]
    assert len(t) == len(x)
    assert abs(t[0]) < 1e-9
    # Targets baked into oscillator_obs_data.json (computed at the true params).
    assert np.mean(x) == pytest.approx(0.01663962, abs=2e-3)
    assert np.min(x) == pytest.approx(-0.60704870, abs=3e-2)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_sensitivity_analysis_python_user_defined_succeeds(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """Sobol sensitivity analysis runs end-to-end on a python_user_defined model."""
    rank = MPI.COMM_WORLD.Get_rank()
    config = _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    config['sa_options'] = {
        'method': 'sobol',
        'num_samples': 16,
        'sample_type': 'saltelli',
        'output_dir': os.path.join(temp_output_dir, 'oscillator_SA_results'),
    }

    run_SA(config)

    if rank == 0:
        assert os.path.exists(config['sa_options']['output_dir'])


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_python_user_defined_recovers_params(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """Calibration runs end-to-end and recovers the oscillator parameters used to
    build the observed data (starting from the c=0.5, k=4.0 defaults)."""
    rank = MPI.COMM_WORLD.Get_rank()
    config = _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)

    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(temp_output_dir, 'genetic_algorithm_oscillator_oscillator_obs_data')
        best_path = os.path.join(output_dir, 'best_param_vals.npy')
        assert os.path.exists(best_path), f"expected calibration output at {best_path}"

        best = np.load(best_path)
        names = np.loadtxt(os.path.join(output_dir, 'param_names.csv'), dtype=str, delimiter=',')
        vals = {str(n): float(v) for n, v in zip(np.atleast_1d(names), np.atleast_1d(best))}

        # Within bounds and in the right ballpark of the ground truth (genetic
        # algorithm with a small budget, so tolerances are generous).
        c = vals.get('oscillator/c')
        k = vals.get('oscillator/k')
        assert c is not None and k is not None, f"unexpected param names: {vals}"
        assert 0.05 <= c <= 2.0 and 1.0 <= k <= 10.0
        assert c == pytest.approx(_TRUE_C, abs=0.4)
        assert k == pytest.approx(_TRUE_K, abs=2.0)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_identifiability_python_user_defined_laplace_disabled(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """Laplace identifiability/UQ is temporarily disabled (issue #293): the run reaches the
    identifiability step after calibration and raises NotImplementedError rather than emitting a
    numerically meaningless covariance.

    When #293 is fixed and Laplace is re-enabled, restore this to assert the covariance is written
    (and validate its values, per #293's acceptance criteria).
    """
    config = _oscillator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    config['do_ia'] = True
    config['ia_options'] = {'method': 'Laplace'}

    with pytest.raises(NotImplementedError, match="#293"):
        run_param_id(config)

    # No covariance should have been written.
    parent_dir = os.path.dirname(temp_output_dir)
    cov_path = os.path.join(parent_dir, 'oscillator_laplace_covariance.npy')
    assert not os.path.exists(cov_path)
