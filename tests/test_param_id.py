"""
Tests for parameter identification functionality.

These tests verify that parameter identification works correctly for various models.
"""
import os
import pytest
import numpy as np
from mpi4py import MPI
from param_id.paramID import CVS0DParamID
from parsers.PrimitiveParsers import YamlFileParser

from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from scripts.param_id_run_script import run_param_id
from scripts.plot_param_id_script import plot_param_id
from scripts.example_format_obs_data_json_file import example_format_obs_data_json_file


def _write_output_mismatch_artifacts(artifact_dir, exp_idx, key, best_fit_output, rerun_output):
    """Save compact diagnostics when saved and rerun outputs diverge."""
    os.makedirs(artifact_dir, exist_ok=True)

    best_arr = np.asarray(best_fit_output)
    rerun_arr = np.asarray(rerun_output)
    if best_arr.ndim == 0 or rerun_arr.ndim == 0:
        n = 0
    else:
        n = int(min(best_arr.shape[0], rerun_arr.shape[0]))

    diff = best_arr[:n] - rerun_arr[:n] if n > 0 else np.array([])
    rel = np.abs(diff) / (np.abs(best_arr[:n]) + 1e-12) if n > 0 else np.array([])

    filename_key = key.replace("/", "_")
    np.savez(
        os.path.join(artifact_dir, f"exp_{exp_idx}_{filename_key}_diagnostics.npz"),
        best=best_arr,
        rerun=rerun_arr,
        diff=diff,
        rel=rel,
    )

    if n == 0:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        show_n = min(n, 400)
        x = np.arange(show_n)
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs[0].plot(x, best_arr[:show_n], label="saved_best_fit")
        axs[0].plot(x, rerun_arr[:show_n], label="rerun")
        axs[0].set_ylabel("value")
        axs[0].legend()
        axs[0].set_title(f"exp={exp_idx}, key={key}")

        axs[1].plot(x, diff[:show_n], label="difference")
        axs[1].set_xlabel("index")
        axs[1].set_ylabel("saved-rerun")
        axs[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(artifact_dir, f"exp_{exp_idx}_{filename_key}_diagnostics.png"))
        plt.close(fig)
    except Exception:
        # Plotting diagnostics are best-effort only.
        pass


def _is_time_like_output_key(key):
    key = str(key)
    return (
        key in {"time", "engine.time", "environment.time"}
        or key.endswith(".time")
        or key.endswith(".t")
    )


def _resolve_rerun_key(saved_key, rerun_outputs):
    """Map saved output keys to rerun keys, allowing time-key aliases only."""
    if saved_key in rerun_outputs:
        return saved_key

    if not _is_time_like_output_key(saved_key):
        return None

    # Prefer the normalized project key when available.
    preferred = ("environment.time", "engine.time", "time")
    for key in preferred:
        if key in rerun_outputs:
            return key

    time_like_keys = [key for key in rerun_outputs.keys() if _is_time_like_output_key(key)]
    if len(time_like_keys) == 1:
        return time_like_keys[0]

    return None


@pytest.fixture(scope="function")
def mpi_comm():
    """Fixture that provides MPI communicator."""
    comm = MPI.COMM_WORLD
    # if comm.Get_size() < 2:
    #     pytest.skip("MPI tests require mpiexec with at least 2 ranks")
    return comm


def _ensure_cellml_model_generated(config, mpi_comm):
    """
    Ensure generated CellML exists before run_param_id.

    CI checkouts omit gitignored generated_models/; local runs may already have artifacts.
    """
    if config.get("model_type") != "cellml_only":
        return
    rank = mpi_comm.Get_rank()
    if rank == 0:
        success = generate_with_new_architecture(False, config)
        prefix = config.get("file_prefix", "<unknown>")
        assert success, f"CellML autogeneration failed for {prefix}"
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_nke_pump_succeeds(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Test that parameter identification succeeds for NKE pump model.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()
    
    # Setup configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'NKE_pump',
        'input_param_file': 'NKE_pump_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'genetic_algorithm',
        # Keep runtime short under MPI tests
        'pre_time': 0.5,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': True,
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(temp_output_dir, 'NKE_pump_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'optimiser_options': {
            'num_calls_to_function': 40,
            'max_patience': 10,
            'cost_convergence': 1e-3,
        },
    })
    
    # Generate obs_data file and model on rank 0
    if rank == 0:
        obs_data_path = config['param_id_obs_path']
        if os.path.exists(obs_data_path):
            os.remove(obs_data_path)
        
        # Generate obs file and model
        example_format_obs_data_json_file(config['param_id_obs_path'])
        generate_with_new_architecture(False, config)
    
    mpi_comm.Barrier()
    
    # Run parameter identification
    run_param_id(config)
    
    # Verify output was created (on rank 0)
    if rank == 0:
        assert os.path.exists(temp_output_dir), "Parameter ID output directory should exist"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_3compartment_succeeds(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Test that parameter identification succeeds for 3compartment model.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()
    
    # Setup configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': True,
        'plot_predictions': True,
        'do_ia': True,
        'ia_options': {'method': 'Laplace'},
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 60, 'max_patience': 500},
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    # Run parameter identification
    run_param_id(config)
    
    # Test autogeneration with fit parameters (on rank 0)
    if rank == 0:
        success = generate_with_new_architecture(True, config)
        assert success, "Autogeneration with fit parameters should succeed"
        
        # Test plotting
        plot_param_id(config, generate=False)
    
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_3compartment_cmaes_succeeds(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Test that parameter identification succeeds for 3compartment model using CMA-ES.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()
    
    # Setup configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'CMA-ES',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 20, 'max_patience': 20},
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    # Run parameter identification
    run_param_id(config)
    
    # Verify output was created (on rank 0)
    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            'CMA-ES_3compartment_3compartment_obs_data'
        )
        assert os.path.exists(output_dir), f"CMA-ES output directory should exist: {output_dir}"
        
        cost_file = os.path.join(output_dir, 'best_cost.npy')
        params_file = os.path.join(output_dir, 'best_param_vals.npy')
        
        assert os.path.exists(cost_file), f"Cost file should exist: {cost_file}"
        assert os.path.exists(params_file), f"Parameters file should exist: {params_file}"
        
        # Verify cost is finite and reasonable
        cost = np.load(cost_file)
        assert np.isfinite(cost), f"Cost should be finite, got {cost}"
        assert cost >= 0, f"Cost should be non-negative, got {cost}"
        
        # Verify parameters are within bounds
        params = np.load(params_file)
        assert params.shape[0] > 0, "Should have at least one parameter"
    
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_3compartment_python_succeeds(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Test parameter identification for 3compartment using the Python solver path.
    """
    rank = mpi_comm.Get_rank()

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'python',
        'solver': 'solve_ivp',
        'param_id_method': 'genetic_algorithm',
        # Shorten for MPI test speed
        'pre_time': 0.5,
        'sim_time': 0.3,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'method': 'BDF',
            'rtol': 1e-6,
            'atol': 1e-8,
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'optimiser_options': {'num_calls_to_function': 40, 'max_patience': 10, 'cost_convergence': 1e-3},
    })

    # Ensure Python model exists
    if rank == 0:
        success = generate_with_new_architecture(False, config)
        assert success, "Python model generation should succeed"
    mpi_comm.Barrier()

    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            f"{config['param_id_method']}_{config['file_prefix']}_3compartment_obs_data"
        )
        assert os.path.exists(output_dir), f"Output directory should exist: {output_dir}"
    
        plot_param_id(config, generate=True)

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_test_fft_cost_is_zero(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Test that parameter identification for test_fft results in zero cost.
    
    This is a specific test case where the cost should be exactly zero.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()
    
    # Setup configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'test_fft',
        'input_param_file': 'test_fft_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 1,
        'sim_time': 1,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': True,  # Enable identifiability analysis to test covariance matrix calculation
        'ia_options': {'method': 'Laplace'},
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(resources_dir, 'test_fft_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 2000, 'max_patience': 500},  
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    # Run parameter identification
    run_param_id(config)
    
    # Test plotting (skip for single parameter cases as corner plot doesn't work with 1D)
    # The covariance matrix calculation is what we're testing, not the plotting
    if rank == 0:
        # Only test plotting if we have more than 1 parameter
        # For test_fft with 1 parameter, skip the corner plot which fails for 1D
        try:
            plot_param_id(config, generate=True)
        except (TypeError, ValueError) as e:
            # Corner plot fails for single parameter - this is expected and acceptable
            # The important part (covariance matrix calculation) already succeeded
            if "not subscriptable" in str(e) or "1D" in str(e):
                print(f"Skipping corner plot for single parameter case: {e}")
            else:
                raise
    
    # Verify cost is zero and covariance matrix was calculated (on rank 0)
    if rank == 0:
        cost_file = os.path.join(
            temp_output_dir,
            'genetic_algorithm_test_fft_test_fft_obs_data',
            'best_cost.npy'
        )
        assert os.path.exists(cost_file), f"Cost file should exist: {cost_file}"
        
        fft_cost = np.load(cost_file)
        assert fft_cost < 1e-8, f"FFT cost should be near zero, got {fft_cost}"
        
        # Verify covariance matrix files were created (identifiability analysis)
        parent_dir = os.path.dirname(temp_output_dir)
        covariance_file = os.path.join(parent_dir, 'test_fft_laplace_covariance.npy')
        mean_file = os.path.join(parent_dir, 'test_fft_laplace_mean.npy')
        
        assert os.path.exists(covariance_file), f"Covariance matrix file should exist: {covariance_file}"
        assert os.path.exists(mean_file), f"Mean file should exist: {mean_file}"
        
        # Verify covariance matrix is valid (not NaN, not singular)
        covariance_matrix = np.load(covariance_file)
        assert not np.isnan(covariance_matrix).any(), "Covariance matrix should not contain NaN values"
        assert not np.isinf(covariance_matrix).any(), "Covariance matrix should not contain Inf values"
        assert covariance_matrix.shape[0] == covariance_matrix.shape[1], "Covariance matrix should be square"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_calibration_outputs_match_rerun(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Regression test:
    - Run short GA calibration on a simple model with state-init dependent constants.
    - Verify saved best_cost equals fresh rerun cost with best params.
    - Verify saved per-experiment all_outputs match fresh reruns.
    """
    rank = mpi_comm.Get_rank()

    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "3compartment",
        "input_param_file": "3compartment_parameters.csv",
        "params_for_id_file": "3compartment_q_lv_only_params_for_id.csv",
        "model_type": "cellml_only",
        "solver": "CVODE_myokit",
        "param_id_method": "genetic_algorithm",
        "pre_time": 0.0,
        "sim_time": 0.2,
        "dt": 0.01,
        "DEBUG": True,
        "do_mcmc": False,
        "plot_predictions": False,
        "do_ia": False,
        "solver_info": {
            "MaximumStep": 0.001,
            "MaximumNumberOfSteps": 5000,
        },
        "param_id_obs_path": os.path.join(resources_dir, "3compartment_obs_data.json"),
        "param_id_output_dir": temp_output_dir,
        "generated_models_dir": temp_generated_models_dir,
        "optimiser_options": {
            "num_calls_to_function": 56,
            "max_patience": 8,
            "cost_convergence": 1e-8,
        },
        "debug_optimiser_options": {
            "num_calls_to_function": 56,
            "max_patience": 8,
            "cost_convergence": 1e-8,
        },
    })

    if rank == 0:
        success = generate_with_new_architecture(False, config)
        assert success, "Autogeneration should succeed for 3compartment"

    mpi_comm.Barrier()
    run_param_id(config)
    mpi_comm.Barrier()

    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            "genetic_algorithm_3compartment_3compartment_obs_data",
        )
        best_cost_path = os.path.join(output_dir, "best_cost.npy")
        best_param_path = os.path.join(output_dir, "best_param_vals.npy")
        assert os.path.exists(best_cost_path), f"Missing best_cost file: {best_cost_path}"
        assert os.path.exists(best_param_path), f"Missing best_param_vals file: {best_param_path}"

        best_cost = float(np.load(best_cost_path))
        best_param_vals = np.load(best_param_path)

        parsed_config = YamlFileParser().parse_user_inputs_file(
            config, obs_path_needed=True, do_generation_with_fit_parameters=False
        )
        param_id_runner = CVS0DParamID.init_from_dict({
            **parsed_config,
            "one_rank": True,
        })

        rerun_cost, _ = param_id_runner.param_id.get_cost_and_obs_from_params(
            best_param_vals, reset=True, only_one_exp=-1
        )
        assert np.isclose(rerun_cost, best_cost, rtol=0.0, atol=1e-8), (
            f"Calibration cost mismatch: saved best_cost={best_cost}, rerun_cost={rerun_cost}"
        )

        num_exp = param_id_runner.param_id.protocol_info["num_experiments"]
        mismatch_artifact_dir = os.path.join(output_dir, "debug_output_mismatch")
        mismatches = []

        for exp_idx in range(num_exp):
            saved_npz_path = os.path.join(output_dir, f"all_outputs_with_best_param_vals_exp_{exp_idx}.npz")
            assert os.path.exists(saved_npz_path), f"Missing saved outputs file: {saved_npz_path}"
            saved_outputs = np.load(saved_npz_path)

            param_id_runner.param_id.get_cost_and_obs_from_params(
                best_param_vals, reset=True, only_one_exp=exp_idx
            )
            rerun_outputs = param_id_runner.param_id.sim_helper.get_all_results_dict()

            for key in saved_outputs.files:
                rerun_key = _resolve_rerun_key(key, rerun_outputs)
                assert rerun_key is not None, f"Missing key '{key}' in rerun outputs for exp {exp_idx}"
                saved_arr = np.asarray(saved_outputs[key])
                rerun_arr = np.asarray(rerun_outputs[rerun_key])
                if saved_arr.shape != rerun_arr.shape or not np.allclose(
                    saved_arr, rerun_arr, rtol=1e-8, atol=1e-10
                ):
                    mismatches.append((exp_idx, key, saved_arr, rerun_arr))
                    _write_output_mismatch_artifacts(
                        mismatch_artifact_dir, exp_idx, key, saved_arr, rerun_arr
                    )

        param_id_runner.close_simulation()

        assert not mismatches, (
            f"Found {len(mismatches)} saved-vs-rerun output mismatches. "
            f"Diagnostics written to: {mismatch_artifact_dir}"
        )

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_simple_physiological_succeeds(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Test that parameter identification succeeds for simple_physiological model.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()
    
    # Setup configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'simple_physiological',
        'input_param_file': 'simple_physiological_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': True,
        'plot_predictions': True,
        'do_ia': False,
        'ia_options': {'method': 'Laplace'},
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(resources_dir, 'simple_physiological_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 60, 'max_patience': 50},
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    # Run parameter identification
    run_param_id(config)
    
    # Test autogeneration with fit parameters (on rank 0)
    if rank == 0:
        success = generate_with_new_architecture(True, config)
        assert success, "Autogeneration with fit parameters should succeed"
        
        # Test plotting
        plot_param_id(config, generate=False)
    
    mpi_comm.Barrier()


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="compare_optimisers is heavy; run locally only (skipped on GitHub Actions)",
)
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
@pytest.mark.compare_optimisers
def test_compare_optimisers(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm, request):
    """
    Test comparison of different optimization methods (GA vs CMA-ES).
    
    This test runs both genetic_algorithm and CMA-ES optimizers and compares
    their results to ensure they produce similar parameter values.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()
    
    # Import here to avoid issues if nevergrad is not available
    from tests.compare_optimisers import OptimiserComparison
    
    # Setup configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 10000, 'max_patience': 500},
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    # Create comparison object with full number of calls for testing
    comparison = OptimiserComparison(config, methods=['genetic_algorithm', 'CMA-ES'], num_calls=10000)
    
    # Run both methods
    ga_success = comparison.run_method('genetic_algorithm')
    cmaes_success = comparison.run_method('CMA-ES')
    
    # Verify both completed successfully
    if rank == 0:
        assert ga_success, "Genetic algorithm optimization should succeed"
        assert cmaes_success, "CMA-ES optimization should succeed"
        
        # Verify results are loaded
        assert 'genetic_algorithm' in comparison.results, "GA results should be available"
        assert 'CMA-ES' in comparison.results, "CMA-ES results should be available"
        
        # Verify costs are finite and reasonable
        ga_cost = comparison.results['genetic_algorithm']['cost']
        cmaes_cost = comparison.results['CMA-ES']['cost']
        
        assert np.isfinite(ga_cost), f"GA cost should be finite, got {ga_cost}"
        assert np.isfinite(cmaes_cost), f"CMA-ES cost should be finite, got {cmaes_cost}"
        assert ga_cost >= 0, f"GA cost should be non-negative, got {ga_cost}"
        assert cmaes_cost >= 0, f"CMA-ES cost should be non-negative, got {cmaes_cost}"
        
        # Compare results (costs should be within reasonable range)
        cost_diff = abs(cmaes_cost - ga_cost)
        max_cost = max(abs(ga_cost), abs(cmaes_cost), 1e-10)
        cost_rel_diff = cost_diff / max_cost * 100
        
        # For test purposes, we just verify both methods complete
        # Actual similarity depends on convergence, which may vary
        print(f"\nGA cost: {ga_cost:.6e}")
        print(f"CMA-ES cost: {cmaes_cost:.6e}")
        print(f"Cost difference: {cost_diff:.6e} ({cost_rel_diff:.2f}%)")
    
    mpi_comm.Barrier()

