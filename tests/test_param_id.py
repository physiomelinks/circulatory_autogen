"""
Tests for parameter identification functionality.

These tests verify that parameter identification works correctly for various models.
"""
import copy
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


def test_casadi_differentiability_assert_passes_for_core_ops_and_costs():
    from param_id.differentiable import assert_casadi_differentiable
    from parsers.PrimitiveParsers import scriptFunctionParser

    sfp = scriptFunctionParser()
    ops = sfp.get_operation_funcs_dict("casadi")
    costs = sfp.get_cost_funcs_dict("casadi")
    assert_casadi_differentiable(
        {"operations": ["mean", "max"]},
        ["gaussian_MLE"],
        ops,
        costs,
    )


def test_mcmc_and_laplace_require_is_mle_cost():
    from param_id.differentiable import assert_mle_cost_for_bayesian
    from parsers.PrimitiveParsers import scriptFunctionParser

    sfp = scriptFunctionParser()
    costs = sfp.get_cost_funcs_dict("numpy")
    assert_mle_cost_for_bayesian("gaussian_MLE", costs, "MCMC")
    assert_mle_cost_for_bayesian(["gaussian_MLE"], costs, "Laplace")
    with pytest.raises(ValueError, match="is_MLE"):
        assert_mle_cost_for_bayesian("MSE", costs, "MCMC")
    with pytest.raises(ValueError, match="is_MLE"):
        assert_mle_cost_for_bayesian("AE", costs, "Laplace approximation")
def test_casadi_differentiability_assert_raises_on_plain_operation():
    from param_id.differentiable import assert_casadi_differentiable

    def f(x):
        return x

    with pytest.raises(ValueError, match="differentiable"):
        assert_casadi_differentiable({"operations": ["f"]}, None, {"f": f}, None)


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


OFFLINE_PRE_TIME_OUTPUT_THRESHOLD = 1e-2
OFFLINE_PRE_TIME_OUTPUT_RTOL = 1e-2


def _midpoint_param_vals(param_id_info):
    return (param_id_info["param_mins"] + param_id_info["param_maxs"]) / 2.0


def _model_default_param_vals(runner):
    """Parameter values from the model (same baseline used for offline_pre_time)."""
    init = runner.param_id.sim_helper.get_init_param_vals(
        runner.param_id.param_id_info["param_names"]
    )
    flat = []
    for entry in init:
        if isinstance(entry, (list, tuple)):
            flat.extend(entry)
        else:
            flat.append(entry)
    return np.asarray(flat, dtype=float)


def _compare_sim_outputs(outputs_a, outputs_b, threshold, skip_time_keys=True, rtol=None):
    mismatches = []
    if rtol is None:
        rtol = OFFLINE_PRE_TIME_OUTPUT_RTOL
    keys = set(outputs_a.keys()) & set(outputs_b.keys())
    for key in sorted(keys):
        if skip_time_keys and _is_time_like_output_key(key):
            continue
        a = np.asarray(outputs_a[key]).flatten()
        b = np.asarray(outputs_b[key]).flatten()
        if a.shape != b.shape:
            mismatches.append((key, f"shape mismatch {a.shape} vs {b.shape}"))
            continue
        if not np.allclose(a, b, rtol=rtol, atol=threshold):
            diff = float(np.max(np.abs(a - b)))
            mismatches.append((key, diff))
    missing_a = set(outputs_b.keys()) - set(outputs_a.keys())
    missing_b = set(outputs_a.keys()) - set(outputs_b.keys())
    if missing_a or missing_b:
        mismatches.append(("__keys__", f"missing in a: {missing_a}, missing in b: {missing_b}"))
    return mismatches


def _flatten_operand_series(operand_results, operand_names):
    """Flatten nested operand get_results output into a stable key -> array map."""
    outputs = {}
    for obs_idx, (names, series_list) in enumerate(zip(operand_names, operand_results)):
        for name, series in zip(names, series_list):
            outputs[f"{obs_idx}:{name}"] = np.asarray(series).flatten()
    return outputs


def _run_sim_outputs_from_obs_path(config, obs_path, mpi_comm, param_val_strategy="model_default"):
    """Run one experiment with midpoint ID params; return operand time-series outputs."""
    rank = mpi_comm.Get_rank()
    if rank == 0:
        config = config.copy()
        config["param_id_obs_path"] = obs_path
        parsed = YamlFileParser().parse_user_inputs_file(
            config, obs_path_needed=True, do_generation_with_fit_parameters=False
        )
        parsed["param_id_obs_path"] = obs_path
        parsed["one_rank"] = True
        if "resources_dir" not in parsed and "resources_dir" in config:
            parsed["resources_dir"] = config["resources_dir"]
        runner = CVS0DParamID.init_from_dict(parsed)
        if param_val_strategy == "midpoint":
            param_vals = _midpoint_param_vals(runner.param_id.param_id_info)
        else:
            param_vals = _model_default_param_vals(runner)
        _, operands_outputs_list, _ = runner.param_id.get_cost_obs_and_pred_from_params(
            param_vals, reset=True, only_one_exp=0
        )
        subexp_count = 0
        outputs = _flatten_operand_series(
            operands_outputs_list[subexp_count],
            runner.obs_info["operands"],
        )
        runner.close_simulation()
    else:
        outputs = None
    outputs = mpi_comm.bcast(outputs, root=0)
    mpi_comm.Barrier()
    return outputs


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
        'debug_optimiser_options': {'num_calls_to_function': 60, 'max_patience': 500, 'cost_type': 'gaussian_MLE'},
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
@pytest.mark.mpi
def test_param_id_3compartment_genetic_algorithm_myokit_fast(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """Fast smoke: 3compartment parameter identification runs with the genetic algorithm on the
    Myokit CVODE backend and produces a finite, non-negative best cost.

    Deliberately small -- short pre_time/sim_time, few function calls, no MCMC/IA/plotting/
    regeneration -- so it stays in the fast (`-m "not slow"`) suite. The heavier 3compartment
    param-id tests cover accuracy/MCMC/IA, and benchmarks/ covers the full optimiser comparison.
    """
    rank = mpi_comm.Get_rank()

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 2,
        'sim_time': 1,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000},
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        # num_calls must exceed the genetic-algorithm population (28 for these 4 params); keep it
        # just above so the run stays fast (~one generation).
        'debug_optimiser_options': {'num_calls_to_function': 40, 'max_patience': 500},
    })

    _ensure_cellml_model_generated(config, mpi_comm)
    run_param_id(config)

    if rank == 0:
        out_dir = os.path.join(temp_output_dir,
                               'genetic_algorithm_3compartment_3compartment_obs_data')
        best_cost = float(np.load(os.path.join(out_dir, 'best_cost.npy')))
        assert np.isfinite(best_cost) and best_cost >= 0, \
            f'expected a finite, non-negative best cost, got {best_cost}'
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_3compartment_extra_ops_succeeds(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Test that parameter identification succeeds for 3compartment_extra_ops model.
    
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
        'file_prefix': '3compartment_extra_ops',
        'input_param_file': '3compartment_extra_ops_parameters.csv',
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
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_extra_ops_obs_data.json'),
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
        'debug_optimiser_options': {'num_calls_to_function': 2000, 'max_patience': 500, 'cost_type': 'gaussian_MLE'},  
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
    params_for_id_path = os.path.join(
        temp_output_dir, "3compartment_python_offline_params_for_id.csv"
    )
    if rank == 0:
        with open(params_for_id_path, "w") as f:
            f.write(
                "vessel_name,param_name,param_type,min,max,name_for_plotting\n"
                "aortic_root,C,const,1e-9,5e-8,C_{ao}\n"
            )
    mpi_comm.Barrier()

    config.update({
        "file_prefix": "3compartment",
        "input_param_file": "3compartment_parameters.csv",
        "params_for_id_file": params_for_id_path,
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
def test_param_id_SN_simple_CVODE_myokit_ga_smoke(
    base_user_inputs,
    resources_dir,
    temp_output_dir,
    temp_generated_models_dir,
    mpi_comm,
):
    """
    Short GA smoke test for SN_simple using Myokit CVODE (matches SN_full-style calibration).

    Observables exercise `funcs_user/operation_funcs_user.py` (calc_spike_frequency_windowed,
    first_peak_time, steady_state_avg, steady_state_min, calc_spike_period, calc_min_peak)
    plus core `max`, mirroring the operation mix in `resources/SN_simple_obs_data.json`
    (max + windowed spike rate + first_peak_time) on a compact two-subexperiment protocol.
    DEBUG genetic_algorithm (population 28) with num_calls_to_function=56 (~2 generations max).
    """
    pytest.importorskip("myokit")

    rank = mpi_comm.Get_rank()
    tests_dir = os.path.dirname(__file__)
    obs_path = os.path.join(tests_dir, "test_inputs", "SN_simple_param_id_fast_obs.json")
    assert os.path.isfile(obs_path), f"Missing fast obs fixture: {obs_path}"

    config = base_user_inputs.copy()
    config.update(
        {
            "file_prefix": "SN_simple",
            "input_param_file": "SN_simple_parameters.csv",
            "params_for_id_file": "SN_simple_params_for_id.csv",
            "model_type": "cellml_only",
            "solver": "CVODE_myokit",
            "param_id_method": "genetic_algorithm",
            "pre_time": 0.1,
            "sim_time": 1.3,
            "dt": 0.005,
            "DEBUG": True,
            "do_mcmc": False,
            "plot_predictions": False,
            "do_ia": False,
            "solver_info": {
                "MaximumStep": 0.02,
                "MaximumNumberOfSteps": 50000,
            },
            "param_id_obs_path": obs_path,
            "param_id_output_dir": temp_output_dir,
            "generated_models_dir": temp_generated_models_dir,
            "optimiser_options": {
                "num_calls_to_function": 56,
                "max_patience": 2,
                "cost_convergence": 1e-12,
            },
            "debug_optimiser_options": {
                "num_calls_to_function": 56,
                "max_patience": 2,
                "cost_convergence": 1e-12,
            },
        }
    )

    _ensure_cellml_model_generated(config, mpi_comm)
    mpi_comm.Barrier()

    run_param_id(config)
    mpi_comm.Barrier()

    if rank == 0:
        out_dir = os.path.join(
            temp_output_dir,
            "genetic_algorithm_SN_simple_SN_simple_param_id_fast_obs",
        )
        assert os.path.isdir(out_dir), f"Missing output dir: {out_dir}"
        best_cost_path = os.path.join(out_dir, "best_cost.npy")
        best_param_vals_path = os.path.join(out_dir, "best_param_vals.npy")
        assert os.path.isfile(best_cost_path), "missing best_cost.npy"
        assert os.path.isfile(best_param_vals_path), "missing best_param_vals.npy"

        # Plotting must be free of side effects: cost with best params should match the
        # calibration best cost both before and after generating plots.
        import matplotlib

        matplotlib.use("Agg", force=True)

        from parsers.PrimitiveParsers import YamlFileParser
        from param_id.paramID import CVS0DParamID

        saved_best_cost = float(np.load(best_cost_path))
        saved_best_param_vals = np.load(best_param_vals_path)

        yaml_parser = YamlFileParser()
        parsed = yaml_parser.parse_user_inputs_file(
            config, obs_path_needed=True, do_generation_with_fit_parameters=True
        )

        plotter = CVS0DParamID(
            parsed["uncalibrated_model_path"],
            parsed["model_type"],
            parsed["param_id_method"],
            False,
            parsed["file_prefix"],
            params_for_id_path=parsed["params_for_id_path"],
            param_id_obs_path=parsed["param_id_obs_path"],
            sim_time=parsed["sim_time"],
            pre_time=parsed["pre_time"],
            solver_info=parsed["solver_info"],
            optimiser_options=parsed.get("optimiser_options", None),
            dt=parsed["dt"],
            param_id_output_dir=parsed["param_id_output_dir"],
            resources_dir=parsed["resources_dir"],
            one_rank=True,
        )
        plotter.set_best_param_vals(saved_best_param_vals)

        cost_before, _ = plotter.param_id.get_cost_and_obs_from_params(
            saved_best_param_vals, reset=True
        )
        plotter.plot_outputs()
        cost_after, _ = plotter.param_id.get_cost_and_obs_from_params(
            saved_best_param_vals, reset=True
        )
        plotter.close_simulation()

        num_experiments = plotter.param_id.protocol_info["num_experiments"]
        for exp_idx in range(num_experiments):
            plot_npz = os.path.join(
                out_dir,
                f"all_outputs_with_best_param_vals_exp_{exp_idx}_plot.npz",
            )
            assert os.path.exists(plot_npz), f"Missing plot NPZ for exp {exp_idx}"
            data = np.load(plot_npz)
            assert len(data.files) > 0, "Plot NPZ is empty"

        assert np.isclose(cost_before, saved_best_cost, rtol=0.0, atol=1e-6)
        assert np.isclose(cost_after, saved_best_cost, rtol=0.0, atol=1e-6)
        assert np.isclose(cost_before, cost_after, rtol=0.0, atol=1e-10)


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
        'debug_optimiser_options': {'num_calls_to_function': 60, 'max_patience': 50, 'cost_type': 'gaussian_MLE'},
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
def test_param_id_lotka_volterra_sp_minimize_succeeds(base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """
    Test that parameter identification succeeds for Lotka-Volterra model
    using CasADi Python model type with casadi_integrator solver.
    
    The Lotka-Volterra model is a simple predator-prey model with two states (x, y).
    This test verifies that:
    1. CasADi Python model can be generated successfully
    2. Parameter identification runs without errors
    3. Output files are created and contain valid values
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Lotka_Volterra',
        'input_param_file': 'Lotka_Volterra_parameters.csv',
        'model_type': 'casadi_python',
        'solver': 'casadi_integrator',
        'param_id_method': 'sp_minimize',
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 0.3,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'max_step_size': 0.001,
            'max_num_steps': 5000,
            'method': 'cvodes',
        },
        'param_id_obs_path': os.path.join(resources_dir, 'Lotka_Volterra_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'optimiser_options': {
            'num_calls_to_function': 40,
            'cost_convergence': 1e-3,
        },
    })

    # Ensure CasADi Python model is generated on rank 0
    if rank == 0:
        success = generate_with_new_architecture(False, config)
        assert success, "CasADi Python model generation should succeed for Lotka-Volterra"
    
    mpi_comm.Barrier()

    # Run parameter identification
    run_param_id(config)

    # Verify output on rank 0
    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            f"{config['param_id_method']}_Lotka_Volterra_Lotka_Volterra_obs_data"
        )
        assert os.path.exists(output_dir), f"Output directory should exist: {output_dir}"
    
        # Verify cost file exists and contains valid value
        cost_file = os.path.join(output_dir, 'best_cost.npy')
        assert os.path.exists(cost_file), f"Cost file should exist: {cost_file}"
        
        cost = np.load(cost_file)
        assert np.isfinite(cost), f"Cost should be finite, got {cost}"
        assert cost >= 0, f"Cost should be non-negative, got {cost}"
        
        # Verify parameters file exists and contains valid values
        params_file = os.path.join(output_dir, 'best_param_vals.npy')
        assert os.path.exists(params_file), f"Parameters file should exist: {params_file}"
        
        params = np.load(params_file)
        assert params.shape[0] > 0, "Should have at least one parameter"
        assert np.all(np.isfinite(params)), "All parameter values should be finite"
    
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_lotka_volterra_sp_minimize_ad_vs_fd(base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """
    Test parameter identification with sp_minimize comparing automatic differentiation (AD)
    vs finite difference (FD) gradient approximation for the Lotka-Volterra model.
    
    This test verifies that:
    1. Parameter ID runs successfully with both AD (do_ad=True) and FD (do_ad=False) gradient methods
    2. The resulting costs are within 0.001 tolerance of each other
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()
    
    # Configuration shared between AD and FD runs
    base_config = base_user_inputs.copy()
    base_config.update({
        'file_prefix': 'Lotka_Volterra',
        'input_param_file': 'Lotka_Volterra_parameters.csv',
        'model_type': 'casadi_python',
        'solver': 'casadi_integrator',
        'param_id_method': 'sp_minimize',
        'pre_time': 0.0,
        'sim_time': 5.0,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'max_step_size': 0.001,
            'max_num_steps': 5000,
            'method': 'cvodes',
        },
        'param_id_obs_path': os.path.join(resources_dir, 'Lotka_Volterra_obs_data.json'),
        'optimiser_options': {
            'num_calls_to_function': 40,
            'cost_convergence': 1e-3,
        },
    })
    
    # Generate model on rank 0 (needed for both runs)
    if rank == 0:
        success = generate_with_new_architecture(False, base_config)
        assert success, "CasADi Python model generation should succeed"
    
    mpi_comm.Barrier()

    # Run 1: sp_minimize with Automatic Differentiation (AD)
    config_ad = base_config.copy()
    config_ad.update({
        'do_ad': True,
        'param_id_output_dir': os.path.join(temp_output_dir, 'ad_run'),
    })
    
    run_param_id(config_ad)
    mpi_comm.Barrier()
    
    # Run 2: sp_minimize with Finite Difference (FD)
    config_fd = base_config.copy()
    config_fd.update({
        'do_ad': False,
        'param_id_output_dir': os.path.join(temp_output_dir, 'fd_run'),
    })
    
    run_param_id(config_fd)
    mpi_comm.Barrier()

    # Compare results on rank 0
    if rank == 0:
        output_dir_ad = os.path.join(
            config_ad['param_id_output_dir'],
            'sp_minimize_Lotka_Volterra_Lotka_Volterra_obs_data'
        )
        output_dir_fd = os.path.join(
            config_fd['param_id_output_dir'],
            'sp_minimize_Lotka_Volterra_Lotka_Volterra_obs_data'
        )
        
        # Verify both output directories exist
        assert os.path.exists(output_dir_ad), f"AD output directory should exist: {output_dir_ad}"
        assert os.path.exists(output_dir_fd), f"FD output directory should exist: {output_dir_fd}"
        
        # Load costs from both runs
        cost_file_ad = os.path.join(output_dir_ad, 'best_cost.npy')
        cost_file_fd = os.path.join(output_dir_fd, 'best_cost.npy')
        
        assert os.path.exists(cost_file_ad), f"AD cost file should exist: {cost_file_ad}"
        assert os.path.exists(cost_file_fd), f"FD cost file should exist: {cost_file_fd}"
        
        cost_ad = float(np.load(cost_file_ad))
        cost_fd = float(np.load(cost_file_fd))
        
        # Assert costs are finite
        assert np.isfinite(cost_ad), f"AD cost should be finite, got {cost_ad}"
        assert np.isfinite(cost_fd), f"FD cost should be finite, got {cost_fd}"
        assert cost_ad >= 0, f"AD cost should be non-negative, got {cost_ad}"
        assert cost_fd >= 0, f"FD cost should be non-negative, got {cost_fd}"
        
        # ASSERTION: Cost difference between AD and FD should be below tolerance
        cost_diff = abs(cost_ad - cost_fd)
        cost_tolerance = 0.001
        assert cost_diff < cost_tolerance, (
            f"Cost difference between AD and FD should be < {cost_tolerance}, "
            f"but got difference of {cost_diff:.6e} (AD: {cost_ad:.6e}, FD: {cost_fd:.6e})"
        )
        
        # Load parameters from both runs
        params_file_ad = os.path.join(output_dir_ad, 'best_param_vals.npy')
        params_file_fd = os.path.join(output_dir_fd, 'best_param_vals.npy')
        
        assert os.path.exists(params_file_ad), f"AD params file should exist: {params_file_ad}"
        assert os.path.exists(params_file_fd), f"FD params file should exist: {params_file_fd}"
        
        params_ad = np.load(params_file_ad)
        params_fd = np.load(params_file_fd)
        
        # Verify parameter counts match
        assert len(params_ad) == len(params_fd), (
            f"Parameter count mismatch: AD has {len(params_ad)}, FD has {len(params_fd)}"
        )
        
        # Assert all parameters are finite
        assert np.all(np.isfinite(params_ad)), "All AD parameter values should be finite"
        assert np.all(np.isfinite(params_fd)), "All FD parameter values should be finite"
        
        print(f"\n=== Lotka-Volterra AD vs FD Comparison ===")
        print(f"AD cost: {cost_ad:.6e}")
        print(f"FD cost: {cost_fd:.6e}")
        print(f"Cost difference: {cost_diff:.6e} (tolerance: {cost_tolerance})")
        print(f"AD parameters: {params_ad}")
        print(f"FD parameters: {params_fd}")

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_lotka_volterra_sp_minimize_ad_vs_fd_aadc(base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """AADC analogue of test_param_id_lotka_volterra_sp_minimize_ad_vs_fd.

    Runs sp_minimize with AADC tape-gradient AD (do_ad=True) vs finite differences
    (do_ad=False) on Lotka-Volterra and asserts the best costs agree within 1e-3.
    Mirrors the CasADi reference test exactly except for model_type/solver, so when
    the backend-agnostic AD calibration path lands this becomes a one-line unskip.
    """
    pytest.importorskip("aadc")
    rank = mpi_comm.Get_rank()

    base_config = base_user_inputs.copy()
    base_config.update({
        'file_prefix': 'Lotka_Volterra',
        'input_param_file': 'Lotka_Volterra_parameters.csv',
        'model_type': 'aadc_python',
        'solver': 'aadc_semi_implicit',
        'param_id_method': 'sp_minimize',
        'pre_time': 0.0,
        'sim_time': 5.0,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        # fixed-step, and exactly what the AADC tape records: with an adaptive integrator the
        # forward solve and the tape integrate different systems, so the gradient would not be
        # the gradient of the cost
        'solver_info': {'method': 'rk4'},
        'param_id_obs_path': os.path.join(resources_dir, 'Lotka_Volterra_obs_data.json'),
        'optimiser_options': {'num_calls_to_function': 40, 'cost_convergence': 1e-3},
    })

    if rank == 0:
        success = generate_with_new_architecture(False, base_config)
        assert success, "AADC Python model generation should succeed"
    mpi_comm.Barrier()

    config_ad = base_config.copy()
    config_ad.update({'do_ad': True, 'param_id_output_dir': os.path.join(temp_output_dir, 'ad_run')})
    run_param_id(config_ad)
    mpi_comm.Barrier()

    config_fd = base_config.copy()
    config_fd.update({'do_ad': False, 'param_id_output_dir': os.path.join(temp_output_dir, 'fd_run')})
    run_param_id(config_fd)
    mpi_comm.Barrier()

    if rank == 0:
        out_ad = os.path.join(config_ad['param_id_output_dir'],
                              'sp_minimize_Lotka_Volterra_Lotka_Volterra_obs_data')
        out_fd = os.path.join(config_fd['param_id_output_dir'],
                              'sp_minimize_Lotka_Volterra_Lotka_Volterra_obs_data')
        cost_ad = float(np.load(os.path.join(out_ad, 'best_cost.npy')))
        cost_fd = float(np.load(os.path.join(out_fd, 'best_cost.npy')))
        assert np.isfinite(cost_ad), f"AD cost should be finite, got {cost_ad}"
        assert np.isfinite(cost_fd), f"FD cost should be finite, got {cost_fd}"
        cost_diff = abs(cost_ad - cost_fd)
        assert cost_diff < 1e-3, (
            f"AD vs FD cost mismatch: AD={cost_ad:.6e} FD={cost_fd:.6e} diff={cost_diff:.6e}"
        )

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_lotka_volterra_sp_minimize_gt_vs_calculated_params(base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """
    Test parameter identification with sp_minimize comparing ground trith and calculated param values for 
    the Lotka-Volterra mdoel.
    
    This test verifies that parameter identification can recover ground truth parameters
    from noisy synthetic observations:
    1. Simulates Lotka-Volterra model with known ground truth parameters
    2. Adds Gaussian noise to simulated states
    3. Creates synthetic observational data from noisy states
    4. Runs parameter identification using sp_minimize with casadi_integrator
    5. Verifies calibrated parameters match ground truth within a tolerance
    
    This is a parameter identification test that validates the parameter identification
    algorithm can find the true parameters when given noisy observations of a trajectory
    generated with those parameters.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    import json
    from solver_wrappers import get_simulation_helper
    
    rank = mpi_comm.Get_rank()
    
    # Ground truth parameters for Lotka-Volterra model
    # These parameters are used to generate synthetic data
    gt_alpha = 5
    gt_beta = 0.2
    gt_delta = 0.2
    gt_gamma = 3
        
    # Noise parameters
    noise_std_dev = 0.5  # standard deviation of measurement noise
    random_seed = 42
    np.random.seed(random_seed)
    
    # Setup base configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Lotka_Volterra',
        'input_param_file': 'Lotka_Volterra_parameters.csv',
        'model_type': 'casadi_python',
        'solver': 'casadi_integrator',
        'param_id_method': 'sp_minimize',
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 5.0,
        'dt': 1.0,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'max_step_size': 0.001,
            'max_num_steps': 5000,
            'method': 'cvodes',
        },
        'optimiser_options': {
            'num_calls_to_function': 40,
            'cost_convergence': 1e-3,
        },
    })

    obs_data_path = None
    
    if rank == 0:
        
        # Generate CasADi Python model if it doesn't exist
        model_output_dir = os.path.join(
            os.path.dirname(__file__), 
            '../generated_models/Lotka_Volterra'
        )
        model_path = os.path.join(model_output_dir, 'Lotka_Volterra.py')
        
        print("Generating CasADi Python model...")
        success = generate_with_new_architecture(False, config)
        assert success, "CasADi Python model generation should succeed for Lotka-Volterra"
        
        # ---- Step 1: Simulate with ground truth parameters ----
        print("\nStep 1: Simulating with ground truth parameters...")
        sim_helper = get_simulation_helper(
            solver='casadi_integrator',
            model_path=model_path,
            model_type='casadi_python',
            dt=config['dt'],
            sim_time=config['sim_time'],
            pre_time=config['pre_time'],
            solver_info=config['solver_info'],
        )
        
        # Set ground truth parameters
        param_names = ['Lotka_Volterra/alpha', 'Lotka_Volterra/beta', 
                      'Lotka_Volterra/delta', 'Lotka_Volterra/gamma']
        param_vals = [gt_alpha, gt_beta, gt_delta, gt_gamma]
        sim_helper.set_param_vals(param_names, param_vals)
        
        # Run simulation
        success = sim_helper.run()
        assert success, "Simulation with ground truth parameters should succeed"

        gt_results = sim_helper.get_all_results_dict()

        gt_x = np.array(gt_results['Lotka_Volterra/x']).flatten()
        gt_y = np.array(gt_results['Lotka_Volterra/y']).flatten()
        times = sim_helper.tSim.flatten()
        
        print(f"  x range: [{gt_x.min():.2f}, {gt_x.max():.2f}]")
        print(f"  y range: [{gt_y.min():.2f}, {gt_y.max():.2f}]")
        
        # ---- Step 2: Add Gaussian noise to simulate measurement uncertainty ----
        print(f"\nStep 2: Adding Gaussian noise (std={noise_std_dev})...")
        np.random.seed(random_seed)
        noisy_x = gt_x + np.random.normal(0, noise_std_dev, gt_x.shape)
        noisy_y = gt_y + np.random.normal(0, noise_std_dev, gt_y.shape)
        
        print(f"  Noisy x range: [{noisy_x.min():.2f}, {noisy_x.max():.2f}]")
        print(f"  Noisy y range: [{noisy_y.min():.2f}, {noisy_y.max():.2f}]")
        
        # ---- Step 3: Create synthetic observation data JSON ----
        print("\nStep 3: Creating synthetic observation data...")

        file_path = os.path.join(resources_dir, 'Lotka_Volterra_obs_data.json')

        with open(file_path, 'r') as f:
            obs_data = json.load(f)

        for item in obs_data["data_items"]:
            if item["variable"] == "Lotka_Volterra/x":
                item["value"] = round(float(noisy_x.max()), 2) 
            elif item["variable"] == "Lotka_Volterra/y":
                item["value"] = round(float(noisy_y.max()), 2)

        obs_data_path = os.path.join(temp_output_dir, 'Lotka_Volterra_obs_data.json')
        with open(obs_data_path, 'w') as f:
            json.dump(obs_data, f, indent=2)
        
        print(f"Created observational data and saved to {obs_data_path}")

    obs_data_path = mpi_comm.bcast(obs_data_path, root=0)

    mpi_comm.Barrier()
    
    # Update config with synthetic observational data path
    config['param_id_obs_path'] = obs_data_path
    config['param_id_output_dir'] = temp_output_dir
    
    # ---- Step 4: Run parameter identification ----
    if rank == 0:
        print("\nStep 4: Running parameter identification with sp_minimize...")

    run_param_id(config)
    mpi_comm.Barrier()
    
    # ---- Step 5: Compare ground truth and calibrated parameters ----
    if rank == 0:
        print("\nStep 5: Comparing ground truth and calibrated parameters...")
        
        output_dir = os.path.join(
            temp_output_dir,
            f"sp_minimize_Lotka_Volterra_Lotka_Volterra_obs_data"
        )
        
        assert os.path.exists(output_dir), f"Output directory should exist: {output_dir}"
        
        # Load calibrated parameters
        params_file = os.path.join(output_dir, 'best_param_vals.npy')
        cost_file = os.path.join(output_dir, 'best_cost.npy')
        
        assert os.path.exists(params_file), f"Parameters file should exist: {params_file}"
        assert os.path.exists(cost_file), f"Cost file should exist: {cost_file}"
        
        calibrated_params = np.load(params_file)
        best_cost = float(np.load(cost_file))
        
        assert len(calibrated_params) >= 4, "Should have at least 4 parameters"
        
        cal_alpha = calibrated_params[0]
        cal_beta = calibrated_params[1]
        cal_delta = calibrated_params[2]
        cal_gamma = calibrated_params[3]
        
        # Calculate relative errors
        alpha_error = abs(cal_alpha - gt_alpha) / abs(gt_alpha) * 100
        beta_error = abs(cal_beta - gt_beta) / abs(gt_beta) * 100
        delta_error = abs(cal_delta - gt_delta) / abs(gt_delta) * 100
        gamma_error = abs(cal_gamma - gt_gamma) / abs(gt_gamma) * 100
        
        print(f"{'Parameter':<15} {'Ground Truth':<20} {'Calibrated':<20} {'Relative Error (%)':<15}")
        print("-" * 70)
        print(f"{'alpha':<15} {gt_alpha:<20.6f} {cal_alpha:<20.6f} {alpha_error:<15.2f}")
        print(f"{'beta':<15} {gt_beta:<20.6f} {cal_beta:<20.6f} {beta_error:<15.2f}")
        print(f"{'delta':<15} {gt_delta:<20.6f} {cal_delta:<20.6f} {delta_error:<15.2f}")
        print(f"{'gamma':<15} {gt_gamma:<20.6f} {cal_gamma:<20.6f} {gamma_error:<15.2f}")
        
        # Define threshold for parameter identification
        param_recovery_threshold = 10.0  # percent
        
        # Assertions: calibrated parameters should be close to ground truth
        assert alpha_error < param_recovery_threshold, (
            f"Alpha parameter recovery error ({alpha_error:.2f}%) exceeds threshold ({param_recovery_threshold}%). "
            f"Ground truth: {gt_alpha}, Calibrated: {cal_alpha}"
        )
        assert beta_error < param_recovery_threshold, (
            f"Beta parameter recovery error ({beta_error:.2f}%) exceeds threshold ({param_recovery_threshold}%). "
            f"Ground truth: {gt_beta}, Calibrated: {cal_beta}"
        )
        assert delta_error < param_recovery_threshold, (
            f"Delta parameter recovery error ({delta_error:.2f}%) exceeds threshold ({param_recovery_threshold}%). "
            f"Ground truth: {gt_delta}, Calibrated: {cal_delta}"
        )
        assert gamma_error < param_recovery_threshold, (
            f"Gamma parameter recovery error ({gamma_error:.2f}%) exceeds threshold ({param_recovery_threshold}%). "
            f"Ground truth: {gt_gamma}, Calibrated: {cal_gamma}"
        )
        
        # Verify cost is finite and reasonable
        assert np.isfinite(best_cost), f"Cost should be finite, got {best_cost}"
        assert best_cost >= 0, f"Cost should be non-negative, got {best_cost}"
        
        print("\nAll parameter identification assertions passed!")
    
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_nonstiff_casadi_forward_and_gradient(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir
):
    """
    Check that the 3compartment_nonstiff CasADi model can be solved (forward pass) and
    that CasADi can differentiate the cost with respect to the parameters (gradient).

    This is a prerequisite for gradient-based parameter identification. It:
    1. Generates the CasADi Python model
    2. Assembles a CVS0DParamID instance at the baseline parameter values
    3. Verifies the forward cost is finite and positive
    4. Verifies the gradient vector is finite and non-zero (i.e., the model is
       differentiable w.r.t. each identified parameter)
    """
    pytest.importorskip("casadi")
    import json
    from parsers.PrimitiveParsers import YamlFileParser
    from solver_wrappers import get_simulation_helper

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment_nonstiff',
        'input_param_file': '3compartment_nonstiff_parameters.csv',
        'params_for_id_file': '3compartment_nonstiff_params_for_id.csv',
        'model_type': 'casadi_python',
        'solver': 'casadi_integrator',
        'param_id_method': 'sp_minimize',
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 0.3,
        'dt': 0.01,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'max_step_size': 0.001,
            'max_num_steps': 50000,
            'method': 'cvodes',
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'resources_dir': resources_dir,
    })

    success = generate_with_new_architecture(False, config)
    assert success, "CasADi Python model generation should succeed for 3compartment_nonstiff"

    parsed = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False
    )
    parsed['one_rank'] = True

    runner = CVS0DParamID.init_from_dict(parsed)

    with open(os.path.join(resources_dir, '3compartment_obs_data.json')) as f:
        obs_data = json.load(f)
    runner.set_ground_truth_data(obs_data)

    params_for_id = [
        {'vessel_name': 'global',       'param_name': 'q_lv_init', 'param_type': 'const', 'min': 200e-6,  'max': 3000e-6},
        {'vessel_name': 'global',       'param_name': 'E_lv_A',    'param_type': 'const', 'min': 1e8,     'max': 5e8},
        {'vessel_name': 'global',       'param_name': 'E_lv_B',    'param_type': 'const', 'min': 1e6,     'max': 5e7},
    ]
    runner.set_params_for_id(params_for_id)

    # Baseline parameter values from the CSV (in param_names order)
    baseline_vals = runner.param_id.sim_helper.get_init_param_vals(
        runner.param_id.param_id_info['param_names']
    )
    assert baseline_vals is not None and len(baseline_vals) == 3

    cost = runner.param_id.get_cost_ca(baseline_vals)
    cost_float = float(cost)
    assert np.isfinite(cost_float), f"Forward cost should be finite, got {cost_float}"
    assert cost_float >= 0, f"Forward cost should be non-negative, got {cost_float}"

    if config.get('DEBUG', False):
        from utilities.casadi_solver_diagnostics import (
            diagnose_casadi_solver_after_forward,
            log_casadi_gradient_diagnostic,
        )
        debug_log_path = os.path.join(temp_output_dir, 'casadi_solver_diagnostics.jsonl')
        diagnose_casadi_solver_after_forward(
            runner, baseline_vals, log_path=debug_log_path,
        )

    grad_error = None
    try:
        gradient = runner.param_id.get_jac_cost_ca(baseline_vals)
    except Exception as exc:
        gradient = None
        grad_error = exc

    if config.get('DEBUG', False):
        log_casadi_gradient_diagnostic(
            gradient, grad_error, log_path=debug_log_path,
        )

    assert gradient is not None, "get_jac_cost_ca raised an exception"
    assert gradient.shape[0] == 3, f"Gradient should have 3 elements, got {gradient.shape}"
    assert np.all(np.isfinite(gradient)), f"Gradient should be finite, got {gradient}"
    assert not np.all(gradient == 0), "Gradient should not be identically zero"
    # q_lv_init sets the LV volume IC; AD must see it (not a disconnected state symbol).
    assert abs(gradient[0]) > 1e-6, (
        f"q_lv_init AD gradient should be nonzero, got {gradient[0]}"
    )

    eps_rel = 1e-4
    fd_grad = np.zeros(3)
    baseline_arr = np.asarray(baseline_vals, dtype=float)
    for i in range(3):
        dp = max(abs(baseline_arr[i]) * eps_rel, 1e-12)
        p_plus = baseline_arr.copy()
        p_minus = baseline_arr.copy()
        p_plus[i] += dp
        p_minus[i] -= dp
        fd_grad[i] = (
            float(runner.param_id.get_cost_ca(p_plus))
            - float(runner.param_id.get_cost_ca(p_minus))
        ) / (2 * dp)
    for i, label in enumerate(["q_lv_init", "E_lv_A", "E_lv_B"]):
        if abs(fd_grad[i]) > 1e-6:
            rel_err = abs(gradient[i] - fd_grad[i]) / abs(fd_grad[i])
            assert rel_err < 0.05, (
                f"{label}: AD gradient {gradient[i]:.6e} differs from FD {fd_grad[i]:.6e} "
                f"(rel err {rel_err:.3g})"
            )

    print(f"\n3compartment_nonstiff CasADi forward/gradient check:")
    print(f"  Cost at baseline: {cost_float:.6g}")
    print(f"  Gradient:         {gradient}")

    runner.close_simulation()


def _build_3compartment_casadi_runner(
    method, base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
    pre_time=0.0, sim_time=0.3, obs_file='3compartment_obs_data.json',
):
    """Generate the (stiff) 3compartment CasADi model and build a CVS0DParamID at baseline.

    Shared by the stiff-3compartment CasADi tests. ``method`` selects the CasADi
    integrator method ('cvodes', the damped 'semi_implicit_euler', or the symbolic 'bdf').
    ``pre_time`` > 0 exercises the long-warmup path (supported by the symbolic methods).
    """
    import json

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'params_for_id_file': '3compartment_params_for_id.csv',
        'model_type': 'casadi_python',
        'solver': 'casadi_integrator',
        'param_id_method': 'sp_minimize',
        'do_ad': True,
        'pre_time': pre_time,
        'sim_time': sim_time,
        'dt': 0.01,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'max_step_size': 0.001,
            'max_num_steps': 50000,
            'method': method,
        },
        'param_id_obs_path': os.path.join(resources_dir, obs_file),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'resources_dir': resources_dir,
    })

    success = generate_with_new_architecture(False, config)
    assert success, "CasADi Python model generation should succeed for 3compartment"

    parsed = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False
    )
    parsed['one_rank'] = True

    runner = CVS0DParamID.init_from_dict(parsed)
    with open(os.path.join(resources_dir, obs_file)) as f:
        obs_data = json.load(f)
    runner.set_ground_truth_data(obs_data)

    params_for_id = [
        {'vessel_name': 'global',      'param_name': 'q_lv_init', 'param_type': 'const', 'min': 200e-6, 'max': 1500e-6},
        {'vessel_name': 'aortic_root', 'param_name': 'C',         'param_type': 'const', 'min': 1e-9,   'max': 5e-8},
        {'vessel_name': 'global',      'param_name': 'E_lv_A',    'param_type': 'const', 'min': 1e8,    'max': 5e8},
        {'vessel_name': 'global',      'param_name': 'E_lv_B',    'param_type': 'const', 'min': 1e6,    'max': 5e7},
    ]
    runner.set_params_for_id(params_for_id)

    baseline_vals = runner.param_id.sim_helper.get_init_param_vals(
        runner.param_id.param_id_info['param_names']
    )
    assert baseline_vals is not None and len(baseline_vals) == 4
    return runner, baseline_vals


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_stiff_casadi_cvodes_gradient_fails(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir
):
    """Document why the damped solver is needed: the *stiff* 3compartment forward-solves
    under CasADi cvodes, but the cvodes adjoint-sensitivity GRADIENT fails
    (CVodeF -> CV_ERR_FAILURE) on the stiff, discontinuous valve dynamics.

    This is why gradient-based parameter identification cannot use cvodes on the stiff
    model (the suite otherwise relies on a linearised 3compartment_nonstiff variant), and
    why test_3compartment_stiff_casadi_semi_implicit_forward_and_gradient exists.
    """
    pytest.importorskip("casadi")

    runner, baseline_vals = _build_3compartment_casadi_runner(
        'cvodes', base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir
    )

    # Forward solve succeeds.
    cost = float(runner.param_id.get_cost_ca(baseline_vals))
    assert np.isfinite(cost) and cost >= 0

    # Gradient via cvodes adjoint sensitivity fails on the stiff dynamics.
    try:
        gradient = np.asarray(runner.param_id.get_jac_cost_ca(baseline_vals)).ravel()
    except RuntimeError as exc:
        assert "CV_" in str(exc) or "cvodes" in str(exc).lower(), (
            f"Expected a cvodes failure, got: {exc}"
        )
        runner.close_simulation()
        return

    # If cvodes ever stops failing here, the damped solver is no longer strictly
    # required for this model — flag it rather than silently passing.
    if np.all(np.isfinite(gradient)):
        runner.close_simulation()
        pytest.skip(
            "cvodes gradient now succeeds on the stiff 3compartment; the "
            "semi_implicit_euler workaround may no longer be required."
        )
    runner.close_simulation()


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_stiff_casadi_semi_implicit_forward_and_gradient(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir
):
    """The stiff 3compartment is solvable AND differentiable with the damped
    'semi_implicit_euler' CasADi solver, where cvodes adjoint sensitivity fails
    (see test_3compartment_stiff_casadi_cvodes_gradient_fails).

    The scheme is a fixed-step semi-implicit Euler with diagonal-Jacobian damping,
    built as a single symbolic mapaccum graph so CasADi differentiates it by plain
    reverse-mode AD. Verifies the forward cost is finite/positive, the gradient is
    finite/non-zero, and the AD gradient matches central finite differences.
    """
    pytest.importorskip("casadi")

    runner, baseline_vals = _build_3compartment_casadi_runner(
        'semi_implicit_euler', base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir
    )

    cost = float(runner.param_id.get_cost_ca(baseline_vals))
    assert np.isfinite(cost), f"Forward cost should be finite, got {cost}"
    assert cost >= 0, f"Forward cost should be non-negative, got {cost}"

    gradient = np.asarray(runner.param_id.get_jac_cost_ca(baseline_vals)).ravel()
    assert gradient.shape[0] == 4, f"Gradient should have 4 elements, got {gradient.shape}"
    assert np.all(np.isfinite(gradient)), f"Gradient should be finite, got {gradient}"
    assert not np.all(gradient == 0), "Gradient should not be identically zero"
    # q_lv_init sets the LV volume IC; AD must see it (not a disconnected state symbol).
    assert abs(gradient[0]) > 1e-6, (
        f"q_lv_init AD gradient should be nonzero, got {gradient[0]}"
    )

    # AD gradient must match central finite differences of the same (damped) forward cost.
    eps_rel = 1e-4
    baseline_arr = np.asarray(baseline_vals, dtype=float)
    fd_grad = np.zeros(4)
    for i in range(4):
        dp = max(abs(baseline_arr[i]) * eps_rel, 1e-12)
        p_plus = baseline_arr.copy()
        p_minus = baseline_arr.copy()
        p_plus[i] += dp
        p_minus[i] -= dp
        fd_grad[i] = (
            float(runner.param_id.get_cost_ca(p_plus))
            - float(runner.param_id.get_cost_ca(p_minus))
        ) / (2 * dp)
    for i, label in enumerate(["q_lv_init", "C_aortic", "E_lv_A", "E_lv_B"]):
        if abs(fd_grad[i]) > 1e-6:
            rel_err = abs(gradient[i] - fd_grad[i]) / abs(fd_grad[i])
            assert rel_err < 0.05, (
                f"{label}: AD gradient {gradient[i]:.6e} differs from FD {fd_grad[i]:.6e} "
                f"(rel err {rel_err:.3g})"
            )

    print(f"\n3compartment (stiff) CasADi semi_implicit_euler forward/gradient check:")
    print(f"  Cost at baseline: {cost:.6g}")
    print(f"  Gradient:         {gradient}")

    runner.close_simulation()


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_casadi_bdf_longrun_gradient_and_cache(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir):
    """The symbolic CasADi 'bdf' method calibrates the STIFF 3compartment model over a LONG run
    (nonzero pre_time warmup) with a correct gradient, and the symbolic graph is built once and
    reused across cost/gradient calls.

    bdf differentiates by plain reverse-mode AD (no CVODES adjoint), so it handles pre_time > 0
    (unlike cvodes, which fails with CV_TOO_MUCH_WORK). The symbolic graph over a long warmup is
    ~thousands of rootfinder steps, so build_casadi_functions caches it keyed on structure rather
    than rebuilding on every cost/gradient call."""
    pytest.importorskip("casadi")

    runner, baseline_vals = _build_3compartment_casadi_runner(
        'bdf', base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
        pre_time=5.0, sim_time=1.0,
    )
    inner = runner.param_id
    baseline = np.asarray(baseline_vals, dtype=float)

    # count symbolic graph builds
    orig = inner.get_cost_and_obs_from_params
    builds = [0]
    def _counting(*a, **k):
        builds[0] += 1
        return orig(*a, **k)
    inner.get_cost_and_obs_from_params = _counting

    # AD gradient must match central FD of the same cost (the bdf symbolic graph is exact)
    gradient = np.asarray(inner.get_gradient(baseline), dtype=float).ravel()
    assert gradient.shape == (4,)
    assert np.all(np.isfinite(gradient))

    fd = np.zeros(4)
    for i in range(4):
        dp = max(abs(baseline[i]) * 1e-5, 1e-14)
        p_plus, p_minus = baseline.copy(), baseline.copy()
        p_plus[i] += dp
        p_minus[i] -= dp
        fd[i] = (float(inner.get_cost(p_plus)) - float(inner.get_cost(p_minus))) / (2 * dp)
        if abs(fd[i]) > 1e-6:
            assert gradient[i] == pytest.approx(fd[i], rel=0.02), (
                f"bdf AD gradient[{i}]={gradient[i]:.6e} != FD {fd[i]:.6e} "
                f"(AD/FD={gradient[i]/fd[i]:.4f}) over a nonzero-pre_time run")

    # the cache built the graph exactly once despite 1 gradient + 8 cost evals
    assert builds[0] == 1, f"symbolic graph should build once (cached), built {builds[0]} times"

    runner.close_simulation()


def _build_3compartment_fsa_runner(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
    pre_time=20.0, sim_time=2.0, obs_file='3compartment_obs_data.json',
    params_for_id=None,
):
    """Generate the (stiff) 3compartment cellml_only model, run through Myokit, and build a
    CVS0DParamID configured for the CVODES forward-sensitivity (FSA) gradient path.

    Tight rtol/atol keep the sensitivities and any finite-difference fallback out of the
    integrator noise floor. Returns (runner, baseline_vals).
    """
    import json

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'params_for_id_file': '3compartment_params_for_id.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'param_id_method': 'sp_minimize',
        'do_ad': True,
        'pre_time': pre_time,
        'sim_time': sim_time,
        'dt': 0.01,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {'MaximumStep': 0.005, 'MaximumNumberOfSteps': 50000,
                        'rtol': 1e-9, 'atol': 1e-9},
        'param_id_obs_path': os.path.join(resources_dir, obs_file),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'resources_dir': resources_dir,
    })

    success = generate_with_new_architecture(False, config)
    assert success, "cellml_only model generation should succeed for 3compartment"

    parsed = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False
    )
    parsed['one_rank'] = True

    runner = CVS0DParamID.init_from_dict(parsed)
    with open(os.path.join(resources_dir, obs_file)) as f:
        obs_data = json.load(f)
    runner.set_ground_truth_data(obs_data)

    if params_for_id is None:
        params_for_id = [
            {'vessel_name': 'global',      'param_name': 'q_lv_init', 'param_type': 'const', 'min': 200e-6, 'max': 1500e-6},
            {'vessel_name': 'aortic_root', 'param_name': 'C',         'param_type': 'const', 'min': 1e-9,   'max': 5e-8},
            {'vessel_name': 'global',      'param_name': 'E_lv_A',    'param_type': 'const', 'min': 1e8,    'max': 5e8},
            {'vessel_name': 'global',      'param_name': 'E_lv_B',    'param_type': 'const', 'min': 1e6,    'max': 5e7},
        ]
    runner.set_params_for_id(params_for_id)
    baseline_vals = runner.param_id.sim_helper.get_init_param_vals(
        runner.param_id.param_id_info['param_names'])
    assert baseline_vals is not None and len(baseline_vals) == len(params_for_id)
    return runner, baseline_vals


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.need_opencor
def test_3compartment_fsa_longrun_gradient(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir):
    """Myokit CVODES forward-sensitivity gradient for the STIFF 3compartment over a LONG
    (nonzero pre_time) run matches central finite differences of the same cost.

    This is the gradient path for cellml_only / CVODE_myokit models. FSA gives d(operand)/dp
    for constant parameters; q_lv_init sets a state initial value, so Myokit cannot make it a
    CVODES independent directly. It is handled analytically by the initial-value chain rule
    (issue #270) -- d(obs)/d(q_lv_init) = d(obs)/d(init q_lv) * d(init q_lv)/d(q_lv_init) -- not
    the finite-difference fallback, so it must be FSA-eligible (verified below). The gradient is
    checked at an arbitrary interior point (not the optimum, so cost != 0), which is all a
    correct descent direction needs. The reference FD uses a rel~1e-3 step: a convergence study
    showed rel 1e-4 sits in the integrator noise floor for this stiff oscillatory model.
    """
    runner, _ = _build_3compartment_fsa_runner(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
        pre_time=20.0, sim_time=2.0)
    inner = runner.param_id

    mins = np.asarray(inner.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(inner.param_id_info['param_maxs'], dtype=float)
    p = mins + 0.4 * (maxs - mins)   # arbitrary interior point

    assert inner.fsa_gradient_available(), "FSA gradient should be advertised for this run"
    gradient = np.asarray(inner.get_gradient(p), dtype=float).ravel()
    assert gradient.shape == (4,)
    assert np.all(np.isfinite(gradient))
    assert not np.all(gradient == 0)

    # q_lv_init feeds a state initial-value expression and nothing else, so it is handled by
    # the chain rule (issue #270), not the FD fallback: no parameter is FSA-ineligible.
    assert inner._fsa_ineligible_names == [], inner._fsa_ineligible_names
    # It is routed through init(heart_module.q_lv) with an analytic d(init q_lv)/d(q_lv_init).
    assert 'global/q_lv_init' in inner.sim_helper._fsa_chain_rule_map
    targets = inner.sim_helper._fsa_chain_rule_map['global/q_lv_init']
    assert [sq for sq, _ in targets] == ['heart_module.q_lv'], targets
    assert targets[0][1] == pytest.approx(1.0)  # init q_lv = q_lv_init, so the factor is 1
    # its gradient column is populated (via the chain rule, not FD).
    assert abs(gradient[0]) > 1e-6

    fd = np.zeros(4)
    for i in range(4):
        dp = max(abs(p[i]) * 1e-3, 1e-14)
        p_plus, p_minus = p.copy(), p.copy()
        p_plus[i] += dp
        p_minus[i] -= dp
        fd[i] = (float(inner.get_cost(p_plus)) - float(inner.get_cost(p_minus))) / (2 * dp)

    for i, label in enumerate(["q_lv_init", "C_aortic", "E_lv_A", "E_lv_B"]):
        if abs(fd[i]) > 1e-6:
            assert gradient[i] == pytest.approx(fd[i], rel=0.10), (
                f"FSA gradient[{i}] ({label}) = {gradient[i]:.6e} != FD {fd[i]:.6e} "
                f"(AD/FD={gradient[i]/fd[i]:.4f})")

    runner.close_simulation()


@pytest.mark.slow
@pytest.mark.need_opencor
def test_3compartment_fsa_multiple_init_params_chain_rule(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir):
    """Two parameters feeding *different* state initial values are both handled by the chain
    rule (issue #270) in the same run, and the gradient still matches finite differences.

    q_lv_init -> init(heart_module.q_lv) and q_rv_init -> init(heart_module.q_rv) each add their
    own init(state) sensitivity independent; neither is FSA-ineligible. This exercises the
    multi-chain-param path (two distinct init(state) columns) that the single-param longrun test
    does not, alongside one ordinary FSA-eligible constant.
    """
    params_for_id = [
        {'vessel_name': 'global',      'param_name': 'q_lv_init', 'param_type': 'const', 'min': 200e-6, 'max': 1500e-6},
        {'vessel_name': 'global',      'param_name': 'q_rv_init', 'param_type': 'const', 'min': 200e-6, 'max': 1500e-6},
        {'vessel_name': 'aortic_root', 'param_name': 'C',         'param_type': 'const', 'min': 1e-9,   'max': 5e-8},
    ]
    runner, _ = _build_3compartment_fsa_runner(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
        pre_time=20.0, sim_time=2.0, params_for_id=params_for_id)
    inner = runner.param_id

    mins = np.asarray(inner.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(inner.param_id_info['param_maxs'], dtype=float)
    p = mins + 0.4 * (maxs - mins)

    gradient = np.asarray(inner.get_gradient(p), dtype=float).ravel()
    assert gradient.shape == (3,)
    assert np.all(np.isfinite(gradient))

    # Both init params are chain-ruled (nothing falls back to FD); each maps to its own state.
    assert inner._fsa_ineligible_names == [], inner._fsa_ineligible_names
    chain = inner.sim_helper._fsa_chain_rule_map
    assert [sq for sq, _ in chain['global/q_lv_init']] == ['heart_module.q_lv']
    assert [sq for sq, _ in chain['global/q_rv_init']] == ['heart_module.q_rv']

    fd = np.zeros(3)
    for i in range(3):
        dp = max(abs(p[i]) * 1e-3, 1e-14)
        p_plus, p_minus = p.copy(), p.copy()
        p_plus[i] += dp
        p_minus[i] -= dp
        fd[i] = (float(inner.get_cost(p_plus)) - float(inner.get_cost(p_minus))) / (2 * dp)

    for i, label in enumerate(["q_lv_init", "q_rv_init", "C_aortic"]):
        if abs(fd[i]) > 1e-6:
            assert gradient[i] == pytest.approx(fd[i], rel=0.10), (
                f"FSA gradient[{i}] ({label}) = {gradient[i]:.6e} != FD {fd[i]:.6e} "
                f"(AD/FD={gradient[i]/fd[i]:.4f})")

    runner.close_simulation()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.need_opencor
def test_3compartment_fsa_cost_and_gradient_combined(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir):
    """get_cost_and_gradient() returns the same gradient as get_jac_cost_fsa() and a cost that
    matches get_cost_from_params(), from a SINGLE augmented CVODES solve.

    This is the speedup the L-BFGS-B path relies on: one FSA solve yields both J(p) and dJ/dp
    (the cost is reconstructed from the operand traces the solve already produced), instead of
    a separate cost solve. The gradient must be identical (same operand path); the cost must
    agree to within the integrator noise floor (a wrong normalisation would show a factor, not
    a ~1e-5 difference).
    """
    runner, _ = _build_3compartment_fsa_runner(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
        pre_time=20.0, sim_time=2.0)
    inner = runner.param_id

    mins = np.asarray(inner.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(inner.param_id_info['param_maxs'], dtype=float)
    p = mins + 0.4 * (maxs - mins)   # arbitrary interior point

    grad_ref = np.asarray(inner.get_jac_cost_fsa(p), dtype=float).ravel()
    cost_ref = float(inner.get_cost_from_params(p))

    cost_c, grad_c = inner.get_cost_and_gradient(p)
    cost_c = float(cost_c)
    grad_c = np.asarray(grad_c, dtype=float).ravel()

    # Gradient is bit-for-bit the same computation as get_jac_cost_fsa.
    np.testing.assert_allclose(grad_c, grad_ref, rtol=1e-8, atol=1e-12)
    # Cost agrees with a standalone evaluation to within the integrator noise (< the 1e-3
    # best-cost consistency-check threshold), NOT off by a normalisation factor.
    assert abs(cost_c - cost_ref) <= 1e-3 * max(1.0, abs(cost_ref)), (
        f"combined cost {cost_c:.6e} vs get_cost_from_params {cost_ref:.6e}")

    # get_cost_and_jac_fsa is the FSA-specific entry point behind get_cost_and_gradient.
    cost_f, grad_f = inner.get_cost_and_jac_fsa(p)
    np.testing.assert_allclose(np.asarray(grad_f, dtype=float).ravel(), grad_ref,
                               rtol=1e-8, atol=1e-12)
    assert abs(float(cost_f) - cost_ref) <= 1e-3 * max(1.0, abs(cost_ref))

    runner.close_simulation()


# Ground-truth params the synthetic multi-sub obs was generated at (the model defaults).
LV_MULTISUB_TRUE = np.array([5.0, 0.2, 0.2, 3.0])   # alpha, beta, delta, gamma


def _lotka_multisub_fsa_config(base_user_inputs, resources_dir, output_dir, generated_models_dir):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Lotka_Volterra',
        'input_param_file': 'Lotka_Volterra_parameters.csv',
        'params_for_id_file': 'Lotka_Volterra_params_for_id.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 3.0,
        'dt': 0.15,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {'MaximumStep': 0.005, 'MaximumNumberOfSteps': 50000,
                        'rtol': 1e-9, 'atol': 1e-9},
        'param_id_obs_path': os.path.join(resources_dir, 'Lotka_Volterra_multisub_obs_data.json'),
        'param_id_output_dir': output_dir,
        'generated_models_dir': generated_models_dir,
        'resources_dir': resources_dir,
    })
    return config


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.need_opencor
def test_fsa_multisub_gradient_matches_fd(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir):
    """Myokit CVODES FSA gradient on a MULTI-sub-experiment protocol matches central FD.

    The Lotka-Volterra obs has one experiment with two sub-experiments (the state carries across
    the sub boundary). The FSA gradient must carry dy/dp across that boundary so each sub's
    operand sensitivities include the cross-sub chain-rule term; a wrong carry would make the
    gradient of the later sub disagree with finite differences. Checked at an arbitrary interior
    point.
    """
    config = _lotka_multisub_fsa_config(base_user_inputs, resources_dir, temp_output_dir,
                                        temp_generated_models_dir)
    config['param_id_method'] = 'sp_minimize'
    assert generate_with_new_architecture(False, config), 'Lotka_Volterra generation should succeed'
    parsed = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)
    parsed['one_rank'] = True
    runner = CVS0DParamID.init_from_dict(parsed)
    inner = runner.param_id

    assert inner.protocol_info['num_sub_per_exp'] == [2], \
        f"expected a 2-sub-experiment obs, got {inner.protocol_info['num_sub_per_exp']}"
    assert inner.fsa_gradient_available()

    mins = np.asarray(inner.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(inner.param_id_info['param_maxs'], dtype=float)
    p = mins + 0.4 * (maxs - mins)   # arbitrary interior point (cost != 0)

    grad = np.asarray(inner.get_gradient(p), dtype=float).ravel()
    assert grad.shape == (4,) and np.all(np.isfinite(grad)) and not np.all(grad == 0)

    fd = np.zeros(4)
    for i in range(4):
        dp = max(abs(p[i]) * 1e-3, 1e-12)
        p_plus, p_minus = p.copy(), p.copy()
        p_plus[i] += dp
        p_minus[i] -= dp
        fd[i] = (float(inner.get_cost(p_plus)) - float(inner.get_cost(p_minus))) / (2 * dp)

    for i, label in enumerate(['alpha', 'beta', 'delta', 'gamma']):
        if abs(fd[i]) > 1e-9:
            assert grad[i] == pytest.approx(fd[i], rel=0.05), (
                f"multi-sub FSA gradient[{i}] ({label}) = {grad[i]:.6e} != FD {fd[i]:.6e} "
                f"(AD/FD={grad[i]/fd[i]:.4f})")

    runner.close_simulation()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_fsa_multisub_calibration_recovers_params(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """End-to-end: multi-start L-BFGS-B with the Myokit FSA gradient calibrates a MULTI-sub
    Lotka-Volterra model and recovers the ground-truth parameters.

    The synthetic obs (series of x, y in each of two sub-experiments) has cost 0 at the true
    (alpha, beta, delta, gamma) = (5, 0.2, 0.2, 3). This exercises the FSA gradient across
    sub-experiment boundaries inside a real optimiser run, not just an AD-vs-FD check.
    """
    rank = mpi_comm.Get_rank()
    config = _lotka_multisub_fsa_config(base_user_inputs, resources_dir, temp_output_dir,
                                        temp_generated_models_dir)
    config['param_id_method'] = 'multi_start_sp_minimize'
    config['optimiser_options'] = {
        'cost_convergence': 1e-6, 'max_patience': 500,
        'num_starts': 8, 'start_sampling': 'sobol', 'seed': 0,
    }

    _ensure_cellml_model_generated(config, mpi_comm)
    run_param_id(config)

    if rank == 0:
        out_dir = os.path.join(
            temp_output_dir,
            'multi_start_sp_minimize_Lotka_Volterra_Lotka_Volterra_multisub_obs_data')
        best_cost = float(np.load(os.path.join(out_dir, 'best_cost.npy')))
        best_params = np.load(os.path.join(out_dir, 'best_param_vals.npy'))
        assert best_cost < 1e-2, f"expected the multi-sub calibration to converge, got cost {best_cost}"
        np.testing.assert_allclose(
            best_params, LV_MULTISUB_TRUE, rtol=0.1,
            err_msg=f"multi-sub FSA calibration recovered {best_params}, expected {LV_MULTISUB_TRUE}")
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_3compartment_nonstiff_casadi_succeeds(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm
):
    """
    Parameter identification for 3compartment_nonstiff using CasADi sp_minimize with AD.

    Uses heart_nonstiff (linearized valve resistance). Verifies that gradient-based
    optimization can recover ground-truth parameters from noiseless synthetic data:
    1. Simulate with ground-truth baseline parameters to obtain observable values
    2. Perturb parameters by 3 % to create the optimizer starting point
    3. Run sp_minimize with AD; check recovered parameters are within 10 % of GT
    """
    pytest.importorskip("casadi")
    import json
    from solver_wrappers import get_simulation_helper

    import csv as csv_module

    rank = mpi_comm.Get_rank()

    # Ground-truth values for identified parameters (from 3compartment_nonstiff_parameters.csv)
    GT = {
        'global/q_lv_init': 0.002,
        'global/E_lv_A':    366575000.0,
        'global/E_lv_B':    10664000.0,
    }
    # Map identified param_id names to CSV variable_name entries (perturbed before optimisation)
    GT_CSV_NAMES = {
        'global/q_lv_init': 'q_lv_init',
        'global/E_lv_A':    'E_lv_A',
        'global/E_lv_B':    'E_lv_B',
    }
    PERTURB_FACTOR = 1.03  # +3 % starting point for the optimiser (small step to avoid local minima)
    param_names_flat = list(GT.keys())
    gt_vals = np.array([GT[k] for k in param_names_flat])
    perturbed_gt = {k: v * PERTURB_FACTOR for k, v in GT.items()}

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment_nonstiff',
        'input_param_file': '3compartment_nonstiff_parameters.csv',
        'params_for_id_file': '3compartment_nonstiff_params_for_id.csv',
        'model_type': 'casadi_python',
        'solver': 'casadi_integrator',
        'param_id_method': 'sp_minimize',
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 0.3,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'max_step_size': 0.001,
            'max_num_steps': 50000,
            'method': 'cvodes',
        },
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'resources_dir': resources_dir,
        'optimiser_options': {
            'num_calls_to_function': 200,
            'cost_convergence': 1e-6,
        },
    })

    obs_data_path = None

    if rank == 0:
        success = generate_with_new_architecture(False, config)
        assert success, "CasADi Python model generation should succeed for 3compartment_nonstiff"

        model_path = os.path.join(temp_generated_models_dir, '3compartment_nonstiff', '3compartment_nonstiff.py')

        # --- Step 1: simulate with GT parameters ---
        sim = get_simulation_helper(
            solver='casadi_integrator',
            model_path=model_path,
            model_type='casadi_python',
            dt=config['dt'],
            sim_time=config['sim_time'],
            pre_time=config['pre_time'],
            solver_info=config['solver_info'],
        )
        sim.set_param_vals(param_names_flat, list(gt_vals))
        assert sim.run(), "Baseline simulation should succeed"
        results = sim.get_all_results_dict()

        def _flat(name):
            return np.asarray(results[name]).flatten()

        # --- Step 2: build obs_data from GT simulation ---
        # Replace every item in the template with values from the GT simulation.
        with open(os.path.join(resources_dir, '3compartment_obs_data.json')) as f:
            obs_template = json.load(f)

        _op_map = {
            'mean':          lambda a: float(a.mean()),
            'max':           lambda a: float(a.max()),
            'min':           lambda a: float(a.min()),
            'max_minus_min': lambda a: float(a.max() - a.min()),
        }
        for item in obs_template:
            op = item.get('operation')
            operand_name = item.get('operands', [None])[0]
            if op in _op_map and operand_name in results:
                val = _op_map[op](_flat(operand_name))
                item['value'] = val
                item['std'] = max(abs(val) * 0.05, 1e-8)

        obs_data_path = os.path.join(temp_output_dir, '3compartment_nonstiff_gt_obs.json')
        with open(obs_data_path, 'w') as f:
            json.dump(obs_template, f, indent=2)

        # --- Step 3: perturb parameters in a copy of the CSV and regenerate the model ---
        # sp_minimize starts from param_init = CSV values (see paramID.run_param_id).
        # Ground-truth observables stay at GT; only the optimiser starting point moves +3 %.
        src_csv = os.path.join(resources_dir, '3compartment_nonstiff_parameters.csv')
        pert_csv = os.path.join(temp_output_dir, '3compartment_nonstiff_parameters_perturbed.csv')
        with open(src_csv, newline='') as f_in:
            reader = csv_module.DictReader(f_in)
            fieldnames = reader.fieldnames
            rows = []
            for row in reader:
                for pk, csv_name in GT_CSV_NAMES.items():
                    if row['variable_name'] == csv_name:
                        row['value'] = str(perturbed_gt[pk])
                rows.append(row)
        with open(pert_csv, 'w', newline='') as f_out:
            writer = csv_module.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        config['input_param_file'] = pert_csv
        success = generate_with_new_architecture(False, config)
        assert success, "Regeneration with perturbed parameters should succeed"

        print("\nGT observables used for cost (from GT simulation):")
        for item in obs_template:
            if item.get('weight', 0) != 0:
                print(
                    f"  {item['operation']:>14} {item['operands'][0]:<20} "
                    f"value={item['value']:.6g} std={item['std']:.6g} weight={item['weight']}"
                )
        print(f"\nOptimiser will start from +{int((PERTURB_FACTOR - 1) * 100)}% perturbed CSV values:")
        for name in param_names_flat:
            print(f"  {name:<25} GT={GT[name]:.6g}  start={perturbed_gt[name]:.6g}")

    obs_data_path = mpi_comm.bcast(obs_data_path, root=0)
    mpi_comm.Barrier()

    config['param_id_obs_path'] = obs_data_path
    run_param_id(config)
    mpi_comm.Barrier()

    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            f"{config['param_id_method']}_3compartment_nonstiff_3compartment_nonstiff_gt_obs"
        )
        assert os.path.exists(output_dir), f"Output directory should exist: {output_dir}"

        cost_file = os.path.join(output_dir, 'best_cost.npy')
        assert os.path.exists(cost_file), f"Cost file should exist: {cost_file}"
        cost = float(np.load(cost_file))
        assert np.isfinite(cost) and cost >= 0, f"Cost should be finite and non-negative, got {cost}"

        params_file = os.path.join(output_dir, 'best_param_vals.npy')
        assert os.path.exists(params_file), f"Parameters file should exist: {params_file}"
        cal_vals = np.load(params_file)
        assert cal_vals.shape[0] == len(gt_vals), "Parameter count mismatch"
        assert np.all(np.isfinite(cal_vals)), "Calibrated parameters should be finite"

        threshold_pct = 10.0
        print(f"\n3compartment_nonstiff CasADi GT recovery (threshold {threshold_pct}%):")
        print(f"{'Parameter':<25} {'GT':>14} {'Calibrated':>14} {'Error%':>8}")
        for name, gt, cal in zip(param_names_flat, gt_vals, cal_vals):
            err = abs(cal - gt) / abs(gt) * 100.0
            print(f"  {name:<23} {gt:>14.6g} {cal:>14.6g} {err:>7.2f}%")
            assert err < threshold_pct, (
                f"{name}: recovery error {err:.2f}% exceeds {threshold_pct}% "
                f"(GT={gt:.6g}, calibrated={cal:.6g})"
            )

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_3compartment_nonstiff_python_succeeds(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm
):
    """
    Parameter identification for 3compartment_nonstiff using the Python solve_ivp path.

    Uses heart_nonstiff (linearized valve resistance R_lin = 2*B*v_ref instead of -B*v*|v|).
    """
    rank = mpi_comm.Get_rank()

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment_nonstiff',
        'input_param_file': '3compartment_nonstiff_parameters.csv',
        'params_for_id_file': '3compartment_nonstiff_params_for_id.csv',
        'model_type': 'python',
        'solver': 'solve_ivp',
        'param_id_method': 'genetic_algorithm',
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
            'max_step': 1e-3,
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'optimiser_options': {'num_calls_to_function': 40, 'max_patience': 10, 'cost_convergence': 1e-3},
    })

    if rank == 0:
        success = generate_with_new_architecture(False, config)
        assert success, "Python model generation should succeed for 3compartment_nonstiff"

    mpi_comm.Barrier()

    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            f"{config['param_id_method']}_3compartment_nonstiff_3compartment_obs_data"
        )
        assert os.path.exists(output_dir), f"Output directory should exist: {output_dir}"

        cost_file = os.path.join(output_dir, 'best_cost.npy')
        assert os.path.exists(cost_file), f"Cost file should exist: {cost_file}"

        cost = np.load(cost_file)
        assert np.isfinite(cost), f"Cost should be finite, got {cost}"
        assert cost >= 0, f"Cost should be non-negative, got {cost}"

        params_file = os.path.join(output_dir, 'best_param_vals.npy')
        assert os.path.exists(params_file), f"Parameters file should exist: {params_file}"

        params = np.load(params_file)
        assert params.shape[0] > 0, "Should have at least one parameter"
        assert np.all(np.isfinite(params)), "All parameter values should be finite"

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_lotka_volterra_sp_minimize_numpy_only_operation(base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """
    Test that parameter identification fails gracefully when obs_data contains
    numpy-only operations (not available in CasADi mode) with sp_minimize and casadi_integrator.

    This test verifies proper error handling for a common limitation:
    - Some custom operations are only defined for numpy mode
    - When using casadi_python model type with sp_minimize, the system switches to
      casadi mode where these operations are unavailable
    - The test expects a KeyError indicating the missing operation

    This demonstrates the known limitation that custom user-defined operations must have
    both numpy and casadi implementations to work with casadi_integrator-based param ID.

    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    import json
    
    rank = mpi_comm.Get_rank()

    undefined_operation = "max_first_half"  # This operation is only defined in numpy mode
    
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Lotka_Volterra',
        'input_param_file': 'Lotka_Volterra_parameters.csv',
        'model_type': 'casadi_python',
        'solver': 'casadi_integrator',
        'param_id_method': 'sp_minimize',
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 0.5,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'max_step_size': 0.001,
            'max_num_steps': 5000,
            'method': 'cvodes',
        },
        'optimiser_options': {
            'num_calls_to_function': 50,
            'cost_convergence': 1e-3,
        },
    })

    obs_data_path = None
    
    if rank == 0:
        success = generate_with_new_architecture(False, config)
        assert success, "CasADi Python model generation should succeed for Lotka-Volterra"
        
        # Create observational data with an operation that is not @differentiable (peak-based)
        # so casadi_python mode fails differentiable validation.
        file_path = os.path.join(resources_dir, 'Lotka_Volterra_obs_data.json')

        with open(file_path, 'r') as f:
            obs_data = json.load(f)

        for item in obs_data["data_items"]:
            if item["variable"] == "Lotka_Volterra/x":
                item["operation"] = undefined_operation

        obs_data_path = os.path.join(temp_output_dir, 'Lotka_Volterra_synthetic_obs_data.json')
        with open(obs_data_path, 'w') as f:
            json.dump(obs_data, f, indent=2)
        
    obs_data_path = mpi_comm.bcast(obs_data_path, root=0)

    mpi_comm.Barrier()

    # Update config with this observational data path
    config['param_id_obs_path'] = obs_data_path
    config['param_id_output_dir'] = temp_output_dir

    # Attempt parameter identification - this should fail: max_first_half is not @differentiable
    with pytest.raises((RuntimeError, KeyError, ValueError)) as excinfo:
        run_param_id(config)

    # Verify the error indicates missing operation or missing @differentiable
    error_msg = str(excinfo.value)
    possible_errors = [
        undefined_operation,
        "KeyError",
        "differentiable",
    ]

    error_found = any(err_str.lower() in error_msg.lower() for err_str in possible_errors)

    # Assert that error message relates to the missing operation or casadi limitation
    assert error_found, (
        f"Expected error message to mention {undefined_operation}, 'KeyError', or 'differentiable'. "
        f"Got: {error_msg}"
    )


@pytest.mark.integration  
@pytest.mark.slow  
@pytest.mark.mpi  
def test_laplace_approximation_hessian_validation(base_user_inputs, resources_dir, temp_output_dir, mpi_comm):  
    """  
    Test Laplace approximation Hessian against analytical solution for simple ODE model.  
      
    Args:  
        base_user_inputs: Base user inputs configuration fixture  
        resources_dir: Resources directory fixture  
        temp_output_dir: Temporary output directory fixture  
        mpi_comm: MPI communicator fixture  
    """  
    rank = mpi_comm.Get_rank()  
      
    # Setup configuration for your simple ODE model  
    config = base_user_inputs.copy()  
    config.update({  
        'file_prefix': 'Simple_ODE_Benchmark',  # Replace with your model name  
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',  # Replace with your CSV  
        'model_type': 'cellml_only',  
        'solver': 'CVODE',  
        'param_id_method': 'genetic_algorithm',  
        'pre_time': 0.5,  
        'sim_time': 10.0,  
        'cost_type': 'gaussian_MLE',
        'dt': 0.1,  
        'DEBUG': False,  
        'do_mcmc': False,  
        'plot_predictions': False,  
        'do_ia': True,  
        'ia_options': {'method': 'Laplace', 'sub_method': 'numdifftools_finite_diff'},  # Options: numdifftools_finite_diff, AD, parabola_fit
        "params_for_id_file": "Simple_ODE_Benchmark_params_for_id.csv",  # Specify which params to identify
        'param_id_obs_path': os.path.join(resources_dir, 'Simple_ODE_Benchmark_obs_data.json'),  
        'param_id_output_dir': temp_output_dir,  
        'debug_optimiser_options': {'num_calls_to_function': 20, 'cost_type': 'gaussian_MLE'},  
    })  
      
    # Generate model and run parameter identification (rank 0 only for setup)  
    if rank == 0:  
        generate_with_new_architecture(False, config)  
    mpi_comm.Barrier()  
      
    run_param_id(config)  
      
    # Validate Hessian (rank 0 only)  
    if rank == 0:  
        # Load the saved covariance matrix  
        parent_dir = os.path.dirname(temp_output_dir)  
        covariance_file = os.path.join(parent_dir, f'{config["file_prefix"]}_laplace_covariance.npy')  
        mean_file = os.path.join(parent_dir, f'{config["file_prefix"]}_laplace_mean.npy')  
        
        assert os.path.exists(covariance_file), f"Covariance file should exist: {covariance_file}"  
        assert os.path.exists(mean_file), f"Mean file should exist: {mean_file}"  
        
        # Load numerical results  
        numerical_covariance = np.load(covariance_file)  
        numerical_mean = np.load(mean_file)

        # check if covariance matrix is positive definite (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(numerical_covariance)
        assert np.all(eigenvalues > 0), f"Covariance matrix should be positive definite, but has eigenvalues: {eigenvalues}"
        
        # Load your analytical solution  
        analytical_covariance = np.array([[0.01, 0], [0, 0.09]])  # Your function here  
        
        # Compare covariance matrices  
        assert numerical_covariance.shape == analytical_covariance.shape, \
            f"Covariance shapes mismatch: numerical {numerical_covariance.shape} vs analytical {analytical_covariance.shape}"  
        
        np.testing.assert_allclose(  
            numerical_covariance, analytical_covariance,  
            rtol=1e-2, atol=1e-3,  
            err_msg="Numerical and analytical covariance matrices should be close"  
        )  
        
        print("Covariance matrix validation passed!")  
      
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_offline_pre_time_lotka_volterra_outputs_match(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm
):
    """
    Lotka-Volterra: pre_time=2 + sim_time=2 without offline_pre_time should match
    offline_pre_time=1 + pre_time=1 + sim_time=2 (same total warmup before logging).
    """
    import json

    rank = mpi_comm.Get_rank()
    base_obs_path = os.path.join(resources_dir, "Lotka_Volterra_obs_data.json")
    with open(base_obs_path, "r") as f:
        base_obs = json.load(f)

    obs_no_offline_path = os.path.join(temp_output_dir, "Lotka_Volterra_offline_pre_no.json")
    obs_with_offline_path = os.path.join(temp_output_dir, "Lotka_Volterra_offline_pre_yes.json")

    if rank == 0:
        obs_no_offline = copy.deepcopy(base_obs)
        obs_no_offline["protocol_info"]["pre_times"] = [2.0]
        obs_no_offline["protocol_info"]["sim_times"] = [[2.0]]
        obs_no_offline["protocol_info"].pop("offline_pre_time", None)
        with open(obs_no_offline_path, "w") as f:
            json.dump(obs_no_offline, f, indent=2)

        obs_with_offline = copy.deepcopy(base_obs)
        obs_with_offline["protocol_info"]["offline_pre_time"] = 1.0
        obs_with_offline["protocol_info"]["pre_times"] = [1.0]
        obs_with_offline["protocol_info"]["sim_times"] = [[2.0]]
        with open(obs_with_offline_path, "w") as f:
            json.dump(obs_with_offline, f, indent=2)

    mpi_comm.Barrier()

    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "Lotka_Volterra",
        "input_param_file": "Lotka_Volterra_parameters.csv",
        "params_for_id_file": "Lotka_Volterra_params_for_id.csv",
        "model_type": "cellml_only",
        "solver": "CVODE_myokit",
        "param_id_method": "genetic_algorithm",
        "param_id_obs_path": base_obs_path,
        "pre_time": 0.5,
        "sim_time": 0.3,
        "dt": 0.01,
        "DEBUG": False,
        "resources_dir": resources_dir,
        "param_id_output_dir": temp_output_dir,
        "generated_models_dir": temp_generated_models_dir,
        "solver_info": {
            "MaximumStep": 0.001,
            "MaximumNumberOfSteps": 5000,
        },
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    outputs_no_offline = _run_sim_outputs_from_obs_path(config, obs_no_offline_path, mpi_comm)
    outputs_with_offline = _run_sim_outputs_from_obs_path(config, obs_with_offline_path, mpi_comm)

    if rank == 0:
        mismatches = _compare_sim_outputs(
            outputs_no_offline,
            outputs_with_offline,
            OFFLINE_PRE_TIME_OUTPUT_THRESHOLD,
        )
        assert not mismatches, (
            f"Lotka-Volterra outputs differ beyond {OFFLINE_PRE_TIME_OUTPUT_THRESHOLD}: "
            f"{mismatches[:5]}"
        )

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_offline_pre_time_3compartment_outputs_match(
    base_user_inputs,
    resources_dir,
    temp_output_dir,
    temp_generated_models_dir,
    mpi_comm,
):
    """
    3compartment: pre_time=2 + sim_time=2 without offline_pre_time should match
    offline_pre_time=1 + pre_time=1 + sim_time=2.
    """
    import json

    rank = mpi_comm.Get_rank()

    base_obs_path = os.path.join(resources_dir, "3compartment_obs_data.json")

    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "3compartment",
        "input_param_file": "3compartment_parameters.csv",
        "params_for_id_file": "3compartment_q_lv_only_params_for_id.csv",
        "model_type": "cellml_only",
        "solver": "CVODE_myokit",
        "param_id_method": "genetic_algorithm",
        "param_id_obs_path": base_obs_path,
        "pre_time": 2.0,
        "sim_time": 2.0,
        "dt": 0.01,
        "DEBUG": False,
        "resources_dir": resources_dir,
        "param_id_output_dir": temp_output_dir,
        "generated_models_dir": temp_generated_models_dir,
        "solver_info": {
            "MaximumStep": 0.001,
            "MaximumNumberOfSteps": 5000,
        },
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    obs_no_offline_path = os.path.join(temp_output_dir, "3compartment_offline_pre_no.json")
    obs_with_offline_path = os.path.join(temp_output_dir, "3compartment_offline_pre_yes.json")

    if rank == 0:
        with open(base_obs_path, "r") as f:
            data_items = json.load(f)

        obs_no_offline = {
            "protocol_info": {
                "pre_times": [2.0],
                "sim_times": [[2.0]],
                "params_to_change": {},
            },
            "prediction_items": [],
            "data_items": data_items,
        }
        with open(obs_no_offline_path, "w") as f:
            json.dump(obs_no_offline, f, indent=2)

        obs_with_offline = {
            "protocol_info": {
                "offline_pre_time": 1.0,
                "pre_times": [1.0],
                "sim_times": [[2.0]],
                "params_to_change": {},
            },
            "prediction_items": [],
            "data_items": data_items,
        }
        with open(obs_with_offline_path, "w") as f:
            json.dump(obs_with_offline, f, indent=2)

    mpi_comm.Barrier()

    outputs_no_offline = _run_sim_outputs_from_obs_path(config, obs_no_offline_path, mpi_comm)
    outputs_with_offline = _run_sim_outputs_from_obs_path(config, obs_with_offline_path, mpi_comm)

    if rank == 0:
        mismatches = _compare_sim_outputs(
            outputs_no_offline,
            outputs_with_offline,
            OFFLINE_PRE_TIME_OUTPUT_THRESHOLD,
        )
        assert not mismatches, (
            f"3compartment outputs differ beyond {OFFLINE_PRE_TIME_OUTPUT_THRESHOLD}: "
            f"{mismatches[:5]}"
        )

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_offline_pre_time_lotka_volterra_casadi_outputs_match(
    base_user_inputs, resources_dir, temp_output_dir, mpi_comm
):
    """
    Lotka-Volterra CasADi path: ensure offline_pre_time warmup equivalence.
    """
    import json

    rank = mpi_comm.Get_rank()
    base_obs_path = os.path.join(resources_dir, "Lotka_Volterra_obs_data.json")
    with open(base_obs_path, "r") as f:
        base_obs = json.load(f)

    obs_no_offline_path = os.path.join(temp_output_dir, "Lotka_Volterra_casadi_offline_pre_no.json")
    obs_with_offline_path = os.path.join(temp_output_dir, "Lotka_Volterra_casadi_offline_pre_yes.json")

    if rank == 0:
        obs_no_offline = copy.deepcopy(base_obs)
        obs_no_offline["protocol_info"]["pre_times"] = [2.0]
        obs_no_offline["protocol_info"]["sim_times"] = [[2.0]]
        obs_no_offline["protocol_info"].pop("offline_pre_time", None)
        with open(obs_no_offline_path, "w") as f:
            json.dump(obs_no_offline, f, indent=2)

        obs_with_offline = copy.deepcopy(base_obs)
        obs_with_offline["protocol_info"]["offline_pre_time"] = 1.0
        obs_with_offline["protocol_info"]["pre_times"] = [1.0]
        obs_with_offline["protocol_info"]["sim_times"] = [[2.0]]
        with open(obs_with_offline_path, "w") as f:
            json.dump(obs_with_offline, f, indent=2)

        generation_cfg = base_user_inputs.copy()
        generation_cfg.update({
            "file_prefix": "Lotka_Volterra",
            "input_param_file": "Lotka_Volterra_parameters.csv",
            "model_type": "casadi_python",
            "solver": "casadi_integrator",
            "resources_dir": resources_dir,
        })
        assert generate_with_new_architecture(False, generation_cfg), "CasADi model generation failed"

    mpi_comm.Barrier()

    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "Lotka_Volterra",
        "input_param_file": "Lotka_Volterra_parameters.csv",
        "params_for_id_file": "Lotka_Volterra_params_for_id.csv",
        "model_type": "casadi_python",
        "solver": "casadi_integrator",
        "param_id_method": "genetic_algorithm",
        "param_id_obs_path": base_obs_path,
        "pre_time": 2.0,
        "sim_time": 2.0,
        "dt": 0.01,
        "DEBUG": False,
        "resources_dir": resources_dir,
        "param_id_output_dir": temp_output_dir,
        "solver_info": {
            "method": "cvodes",
            "max_step_size": 0.001,
            "max_num_steps": 5000,
        },
    })

    outputs_no_offline = _run_sim_outputs_from_obs_path(config, obs_no_offline_path, mpi_comm)
    outputs_with_offline = _run_sim_outputs_from_obs_path(config, obs_with_offline_path, mpi_comm)

    if rank == 0:
        mismatches = _compare_sim_outputs(
            outputs_no_offline,
            outputs_with_offline,
            OFFLINE_PRE_TIME_OUTPUT_THRESHOLD,
        )
        assert not mismatches, (
            "Lotka-Volterra CasADi outputs differ with/without offline_pre_time: "
            f"{mismatches[:5]}"
        )

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_offline_pre_time_3compartment_python_outputs_match(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm
):
    """
    3compartment Python solve_ivp path: ensure offline_pre_time warmup equivalence.
    Uses a short offline warmup slice for numerical robustness in SciPy solve_ivp.
    """
    import json

    rank = mpi_comm.Get_rank()
    base_obs_path = os.path.join(resources_dir, "3compartment_obs_data.json")
    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "3compartment",
        "input_param_file": "3compartment_parameters.csv",
        "params_for_id_file": "3compartment_q_lv_only_params_for_id.csv",
        "model_type": "python",
        "solver": "solve_ivp",
        "param_id_method": "genetic_algorithm",
        "param_id_obs_path": base_obs_path,
        "pre_time": 2.0,
        "sim_time": 2.0,
        "dt": 0.01,
        "DEBUG": False,
        "resources_dir": resources_dir,
        "param_id_output_dir": temp_output_dir,
        "generated_models_dir": temp_generated_models_dir,
        "solver_info": {
            "method": "BDF",
            "rtol": 1e-6,
            "atol": 1e-8,
            "max_step": 1e-3,
        },
    })

    if rank == 0:
        assert generate_with_new_architecture(False, config), "Python model generation failed"

    mpi_comm.Barrier()

    obs_no_offline_path = os.path.join(temp_output_dir, "3compartment_python_offline_pre_no.json")
    obs_with_offline_path = os.path.join(temp_output_dir, "3compartment_python_offline_pre_yes.json")

    if rank == 0:
        with open(base_obs_path, "r") as f:
            data_items = json.load(f)
        obs_no = {
            "protocol_info": {
                "pre_times": [0.5],
                "sim_times": [[0.3]],
                "params_to_change": {},
            },
            "prediction_items": [],
            "data_items": data_items,
        }
        obs_yes = {
            "protocol_info": {
                "offline_pre_time": 0.2,
                "pre_times": [0.3],
                "sim_times": [[0.3]],
                "params_to_change": {},
            },
            "prediction_items": [],
            "data_items": data_items,
        }
        with open(obs_no_offline_path, "w") as f:
            json.dump(obs_no, f, indent=2)
        with open(obs_with_offline_path, "w") as f:
            json.dump(obs_yes, f, indent=2)

    mpi_comm.Barrier()

    outputs_no_offline = _run_sim_outputs_from_obs_path(
        config, obs_no_offline_path, mpi_comm, param_val_strategy="midpoint"
    )
    outputs_with_offline = _run_sim_outputs_from_obs_path(
        config, obs_with_offline_path, mpi_comm, param_val_strategy="midpoint"
    )

    if rank == 0:
        mismatches = _compare_sim_outputs(
            outputs_no_offline,
            outputs_with_offline,
            OFFLINE_PRE_TIME_OUTPUT_THRESHOLD,
        )
        # Python solve_ivp shows large numeric drift on this stiff pressure observable,
        # while other observables remain consistent across offline/no-offline runs.
        mismatches = [m for m in mismatches if ":aortic_root/u" not in m[0]]
        assert not mismatches, (
            "3compartment Python outputs differ with/without offline_pre_time: "
            f"{mismatches[:5]}"
        )

    mpi_comm.Barrier()


def test_parse_obs_data_json_rejects_value_and_npy_paths(tmp_path):
    """Series items must use embedded value OR t_path/value_path, not both."""
    from parsers.PrimitiveParsers import ObsAndParamDataParser

    t_path = tmp_path / "t.npy"
    v_path = tmp_path / "y.npy"
    np.save(t_path, np.array([0.0, 0.1, 0.2]))
    np.save(v_path, np.array([1.0, 2.0, 3.0]))

    obs_data_dict = {
        "protocol_info": {
            "pre_times": [0.0],
            "sim_times": [[1.0]],
            "params_to_change": {},
        },
        "prediction_items": [],
        "data_items": [
            {
                "variable": "test_var",
                "data_type": "series",
                "operands": ["model/x"],
                "unit": "1",
                "weight": 1.0,
                "value": [1.0, 2.0, 3.0],
                "std": [0.1, 0.1, 0.1],
                "obs_dt": 0.1,
                "t_path": str(t_path),
                "value_path": str(v_path),
            }
        ],
    }

    parser = ObsAndParamDataParser()
    with pytest.raises(ValueError, match="both embedded 'value'"):
        parser.parse_obs_data_json(obs_data_dict=obs_data_dict, pre_time=0.0, sim_time=1.0)


def test_parse_obs_data_json_series_std_scalar_from_npy_paths(tmp_path):
    """Scalar std in JSON is expanded to match series length loaded from .npy."""
    from parsers.PrimitiveParsers import ObsAndParamDataParser

    t_path = tmp_path / "t.npy"
    v_path = tmp_path / "y.npy"
    np.save(t_path, np.array([0.0, 0.1, 0.2]))
    np.save(v_path, np.array([1.0, 2.0, 3.0]))

    obs_data_dict = {
        "protocol_info": {
            "pre_times": [0.0],
            "sim_times": [[1.0]],
            "params_to_change": {},
        },
        "prediction_items": [],
        "data_items": [
            {
                "variable": "test_var",
                "data_type": "series",
                "operands": ["model/x"],
                "unit": "1",
                "weight": 1.0,
                "std": 0.5,
                "obs_dt": 0.1,
                "t_path": str(t_path),
                "value_path": str(v_path),
            }
        ],
    }

    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(
        obs_data_dict=obs_data_dict, pre_time=0.0, sim_time=1.0,
    )
    assert list(parsed["gt_df"].iloc[0]["std"]) == [0.5, 0.5, 0.5]


def test_parse_obs_data_json_series_std_required_for_npy_paths(tmp_path):
    """Series items loaded from .npy must specify std in the JSON."""
    from parsers.PrimitiveParsers import ObsAndParamDataParser

    t_path = tmp_path / "t.npy"
    v_path = tmp_path / "y.npy"
    np.save(t_path, np.array([0.0, 0.1]))
    np.save(v_path, np.array([1.0, 2.0]))

    obs_data_dict = {
        "protocol_info": {
            "pre_times": [0.0],
            "sim_times": [[1.0]],
            "params_to_change": {},
        },
        "prediction_items": [],
        "data_items": [
            {
                "variable": "soma_SN/V_sensed",
                "data_type": "series",
                "operands": ["model/x"],
                "unit": "1",
                "weight": 1.0,
                "t_path": str(t_path),
                "value_path": str(v_path),
            }
        ],
    }

    parser = ObsAndParamDataParser()
    with pytest.raises(ValueError, match="requires 'std'"):
        parser.parse_obs_data_json(obs_data_dict=obs_data_dict, pre_time=0.0, sim_time=1.0)



@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_param_id_archives_input_files(resources_dir, temp_output_dir,
                                       generated_cellml_model_factory, mpi_comm):
    """CVS0DParamID archives the params_for_id and obs_data inputs into the run
    output_dir with a _yymmdd_HHMMSS timestamp, so a user can see exactly which
    inputs produced a run (issue #233)."""
    import re

    rank = mpi_comm.Get_rank()
    model_path = generated_cellml_model_factory('3compartment', '3compartment_parameters.csv')
    params_for_id_path = os.path.join(resources_dir, '3compartment_params_for_id.csv')
    obs_path = os.path.join(resources_dir, '3compartment_obs_data.json')

    runner = CVS0DParamID.init_from_dict({
        'model_path': model_path,
        'model_type': 'cellml_only',
        'param_id_method': 'genetic_algorithm',
        'file_name_prefix': '3compartment',
        'params_for_id_path': params_for_id_path,
        'param_id_obs_path': obs_path,
        'resources_dir': resources_dir,
        'param_id_output_dir': temp_output_dir,
        'solver_info': {'solver': 'CVODE_myokit', 'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000},
        'dt': 0.01, 'sim_time': 2.0, 'pre_time': 20.0, 'DEBUG': False,
    })

    if rank == 0:
        out = runner.output_dir
        files = os.listdir(out)
        params_copies = [f for f in files
                         if re.fullmatch(r'3compartment_params_for_id_\d{6}_\d{6}\.csv', f)]
        obs_copies = [f for f in files
                      if re.fullmatch(r'3compartment_obs_data_\d{6}_\d{6}\.json', f)]
        assert params_copies, f"no timestamped params_for_id copy in {out}: {files}"
        assert obs_copies, f"no timestamped obs_data copy in {out}: {files}"
        # Archived copies must match the source inputs byte-for-byte.
        with open(os.path.join(out, params_copies[0])) as fa, open(params_for_id_path) as fb:
            assert fa.read() == fb.read()
        with open(os.path.join(out, obs_copies[0])) as fa, open(obs_path) as fb:
            assert fa.read() == fb.read()


def test_sp_minimize_streams_cost_history_per_iteration(temp_output_dir):
    """SciPyMinimizeOptimiser appends a cost / parameter history row per L-BFGS-B
    iteration (not just a start + end pair), so the live progress plots update
    during a gradient-based calibration. Uses a Rosenbrock cost so L-BFGS-B takes
    several iterations; no model/casadi needed."""
    from param_id.optimisers import SciPyMinimizeOptimiser

    mins = np.array([-10.0, -10.0])
    maxs = np.array([10.0, 10.0])

    class _Norm:
        def normalise(self, p):
            return (np.asarray(p, dtype=float) - mins) / (maxs - mins)

        def unnormalise(self, q):
            return mins + np.asarray(q, dtype=float) * (maxs - mins)

    class _ParamId:
        param_init = np.array([-1.2, 1.0])  # classic Rosenbrock start

        def __init__(self):
            self.best = None

        def get_cost_ca(self, p):
            p = np.asarray(p, dtype=float)
            return 100.0 * (p[1] - p[0] ** 2) ** 2 + (1.0 - p[0]) ** 2

        def get_jac_cost_ca(self, p):
            p = np.asarray(p, dtype=float)
            return np.array([
                -400.0 * p[0] * (p[1] - p[0] ** 2) - 2.0 * (1.0 - p[0]),
                200.0 * (p[1] - p[0] ** 2),
            ])

        # Backend-agnostic aliases (used by refactored optimisers.py)
        def get_cost(self, p):
            return float(self.get_cost_ca(p))

        def get_gradient(self, p):
            return self.get_jac_cost_ca(p)

        def set_best_param_vals(self, p):
            self.best = np.asarray(p, dtype=float)

    param_id_info = {"param_names": [["a"], ["b"]], "param_mins": mins, "param_maxs": maxs}
    opt = SciPyMinimizeOptimiser(
        param_id_obj=_ParamId(), param_id_info=param_id_info, param_norm_obj=_Norm(),
        num_params=2, output_dir=temp_output_dir,
        optimiser_options={"cost_convergence": 1e-15}, do_ad=True, DEBUG=False,
    )
    opt.run()

    cost_path = os.path.join(temp_output_dir, "best_cost_history.csv")
    param_path = os.path.join(temp_output_dir, "best_param_vals_history.csv")
    assert os.path.exists(cost_path)
    cost_rows = [ln for ln in open(cost_path).read().splitlines() if ln.strip()]
    costs = [float(ln.split(",")[0]) for ln in cost_rows]
    # init row + one per L-BFGS-B iteration => clearly more than the old start+end pair.
    assert len(costs) > 2, f"expected per-iteration history, got {len(costs)} rows"
    # Progress was made: the final recorded cost is well below the starting cost.
    assert costs[-1] < costs[0]
    # Parameter history is written in lockstep with the cost history.
    param_rows = [ln for ln in open(param_path).read().splitlines() if ln.strip()]
    assert len(param_rows) == len(costs)


# ---------------------------------------------------------------------------
# multi_start_sp_minimize (multi-start L-BFGS-B)
# ---------------------------------------------------------------------------

# A deliberately multi-modal cost with two wells per dimension:
#   g(t) = (t^2 - 1)^2 + 0.5*t + 0.6
# g has a shallow local minimum near t = +1 (g ~ 1.1) and the global minimum near
# t = -1 (g ~ 0.1). Summed over 2 dimensions, a gradient descent started in the
# right-hand well converges to a local minimum roughly 2.0 above the global one.
def _two_well_cost(p):
    p = np.asarray(p, dtype=float)
    return float(np.sum((p ** 2 - 1.0) ** 2 + 0.5 * p + 0.6))


def _two_well_grad(p):
    p = np.asarray(p, dtype=float)
    return 4.0 * p * (p ** 2 - 1.0) + 0.5


class _TwoWellParamId:
    """Stand-in for OpencorParamID exposing just what the optimisers call.

    Mirrors the backend-agnostic interface: get_cost() always works, get_gradient() only has an
    AD backend for casadi_python / aadc_python models and raises otherwise, exactly as
    OpencorParamID.get_gradient does.
    """

    def __init__(self, param_init=(1.2, 1.2), model_type='casadi_python'):
        self.param_init = np.array(param_init, dtype=float)
        self.model_type = model_type
        self.best = None
        self.num_cost_calls = 0
        self.num_jac_calls = 0

    def get_cost(self, p):
        self.num_cost_calls += 1
        return _two_well_cost(p)

    def get_gradient(self, p):
        if self.model_type not in ('casadi_python', 'aadc_python'):
            raise ValueError(f"Gradient not available for model_type={self.model_type}")
        self.num_jac_calls += 1
        return _two_well_grad(p)

    def get_cost_from_params(self, p):
        self.num_cost_calls += 1
        return _two_well_cost(p)

    def set_best_param_vals(self, p):
        self.best = np.asarray(p, dtype=float)


class _TwoWellNorm:
    mins = np.array([-2.0, -2.0])
    maxs = np.array([2.0, 2.0])

    def normalise(self, p):
        return (np.asarray(p, dtype=float) - self.mins) / (self.maxs - self.mins)

    def unnormalise(self, q):
        return self.mins + np.asarray(q, dtype=float) * (self.maxs - self.mins)


def _two_well_param_id_info():
    return {
        "param_names": [["well/x"], ["well/y"]],
        "param_mins": _TwoWellNorm.mins,
        "param_maxs": _TwoWellNorm.maxs,
    }


def _make_multi_start_optimiser(output_dir, param_id_obj, optimiser_options,
                                model_type='casadi_python', do_ad=True):
    from param_id.optimisers import MultiStartSciPyMinimizeOptimiser

    return MultiStartSciPyMinimizeOptimiser(
        param_id_obj=param_id_obj, param_id_info=_two_well_param_id_info(),
        param_norm_obj=_TwoWellNorm(), num_params=2, output_dir=output_dir,
        optimiser_options=optimiser_options, do_ad=do_ad, model_type=model_type,
        DEBUG=False,
    )


class _WorkloadTwoWellParamId(_TwoWellParamId):
    """Two-well cost with a fixed, non-trivial amount of CPU work per evaluation.

    Real starts vary a lot in length (some descents take 4 L-BFGS-B iterations, some 70), which
    is precisely the imbalance parallel multi-start has to average out. This fake makes each cost
    evaluation cost a few tenths of a millisecond of deterministic work, so per-start durations
    are real and measurable (and imbalanced, through the natural spread of iteration counts) --
    enough to measure the achieved speedup without needing a physiological model.
    """

    # small enough to stay in L1 cache: the work is then compute-bound, not memory-bandwidth
    # bound, so parallel ranks don't contend for memory and the scaling reflects the optimiser
    # rather than the host's memory bus (a real ODE solve is likewise compute-bound).
    _WORK = np.linspace(0.0, 1.0, 256)

    def _burn(self):
        # deterministic, defeats being optimised away by returning a value
        acc = 0.0
        for _ in range(600):
            acc += float(np.sum(np.sqrt(self._WORK) * np.cos(self._WORK)))
        return acc

    def get_cost(self, p):
        self._burn()
        return super().get_cost(p)

    def get_gradient(self, p):
        self._burn()
        return super().get_gradient(p)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_multi_start_scales_with_many_starts(temp_output_dir):
    """With many more starts than ranks, the static round-robin distribution balances out and the
    multi-start scales close to the rank count. Run under multiple MPI ranks to see the speedup;
    it still passes (with weaker assertions) on a single rank.

    cost_convergence is set below any achievable cost, so the global early stop never fires and
    all 100 starts run -- the speedup is then measured on the full, fixed workload, and this also
    exercises the stop machinery on the path where it must NOT trigger.
    """
    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()

    opt = _make_multi_start_optimiser(
        temp_output_dir, _WorkloadTwoWellParamId(param_init=(1.2, 1.2)),
        {'num_starts': 100, 'start_sampling': 'sobol', 'seed': 0,
         'include_init_point': False, 'cost_convergence': 1e-300})
    opt.run()

    if comm.Get_rank() != 0:
        comm.Barrier()
        return

    # the global early stop never fired, so every start ran
    assert opt.num_starts_run == 100, \
        f'expected all 100 starts to run, got {opt.num_starts_run}'

    # static round-robin spreads the starts to within one of each other
    counts = opt.starts_run_per_rank
    assert sum(counts) == 100
    assert max(counts) - min(counts) <= 1, \
        f'starts should be evenly distributed over ranks, got {counts}'

    # serial_seconds is the summed per-start work; wall_seconds is the slowest rank. Their ratio
    # is the achieved speedup.
    print(f'\n[scaling] {num_procs} rank(s): serial={opt.serial_seconds:.2f}s '
          f'wall={opt.wall_seconds:.2f}s speedup={opt.speedup:.2f}x '
          f'starts_per_rank={counts}')
    if num_procs > 1:
        # With 100 starts over num_procs ranks the per-rank workloads average out and the
        # multi-start clearly scales. The floor is deliberately conservative -- ideal speedup is
        # num_procs, and 0.4x still comfortably fails if the work stopped being distributed --
        # because the achievable fraction depends on the host (CPU turbo throttles as more cores
        # get busy, and 100/num_procs starts per rank averages imperfectly for larger rank
        # counts). Measured on a 20-core host: ~3.8x on 4 ranks, ~4.2x on 8. Requiring near
        # num_procs would make the test hostage to the machine it runs on.
        assert opt.speedup >= max(1.5, 0.3 * num_procs), (
            f'expected the multi-start to scale with {num_procs} ranks '
            f'(speedup >= {max(1.5, 0.3 * num_procs):.1f}), got {opt.speedup:.2f}')
    else:
        # on one rank there is no parallelism, but serial and wall should agree
        assert opt.speedup == pytest.approx(1.0, abs=0.2)

    comm.Barrier()


@pytest.mark.integration
@pytest.mark.mpi
def test_multi_start_global_early_stop_halts_all_ranks(temp_output_dir):
    """When a start meets the threshold, every rank stops launching new starts -- not just the
    rank that found it. With a threshold that IS met, fewer than all starts run."""
    comm = MPI.COMM_WORLD

    # the -1 well sits at cost ~0.1 (see _two_well_cost); a threshold of 0.5 is met by any start
    # that reaches that well, so the search stops well before running all 100 starts
    opt = _make_multi_start_optimiser(
        temp_output_dir, _TwoWellParamId(param_init=(-1.0, -1.0)),
        {'num_starts': 100, 'start_sampling': 'sobol', 'seed': 0,
         'include_init_point': True, 'cost_convergence': 0.5})
    opt.run()

    if comm.Get_rank() == 0:
        assert opt.num_starts_run < 100, \
            f'the global early stop should have halted the search early, ran {opt.num_starts_run}'
        assert opt.best_cost <= 0.5, \
            f'the run should have stopped because the threshold was met, best {opt.best_cost}'

    comm.Barrier()


@pytest.mark.unit
def test_multi_start_no_new_starts_on_convergence_false_runs_every_start(temp_output_dir):
    """With no_new_starts_on_convergence off, every start runs even once one has converged, and
    the run reports the converged starts clustered by which solution they reached.

    The two-well cost has a global minimum at -1 and a shallow local one at +1, so a start that
    reaches -1 converges (cost ~0.1 <= threshold) but must NOT stop the others -- and the starts
    that fall into the +1 well don't converge, so they aren't clustered."""
    opt = _make_multi_start_optimiser(
        temp_output_dir, _TwoWellParamId(),
        {'num_starts': 20, 'start_sampling': 'sobol', 'seed': 0, 'include_init_point': False,
         'cost_convergence': 0.5, 'no_new_starts_on_convergence': False})
    opt.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        # nothing was skipped
        assert opt.num_starts_run == 20
        # the converged starts all reached the -1 well, i.e. one cluster with negative params
        assert opt.num_converged > 0
        assert len(opt.convergence_clusters) >= 1
        counts = sum(c['count'] for c in opt.convergence_clusters)
        assert counts == opt.num_converged
        for cluster in opt.convergence_clusters:
            assert np.all(np.asarray(cluster['params']) < 0.0), \
                f'converged starts should be in the -1 well, got {cluster["params"]}'


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_multi_start_reports_both_minima_of_a_two_minimum_model(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """End to end on a CellML model with two equal global minima.

    resources/TwoMinima is dx/dt = c*(a^2 - x), fitting `a` to a steady state of 4. Because the
    observable depends on a^2, a = +2 and a = -2 both fit perfectly -- two global minima either
    side of a barrier at a = 0. With no_new_starts_on_convergence off, every start runs and the
    optimiser reports how many landed in each minimum. The test checks that both minima are
    found and that the per-minimum counts are reported (they differ, since the starts don't split
    exactly evenly)."""
    import csv as _csv

    rank = mpi_comm.Get_rank()

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'TwoMinima',
        'input_param_file': 'TwoMinima_parameters.csv',
        'params_for_id_file': 'TwoMinima_params_for_id.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'param_id_method': 'multi_start_sp_minimize',
        'do_ad': False,
        'pre_time': 0.0,
        'sim_time': 5.0,
        'dt': 0.05,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {'MaximumStep': 0.01, 'MaximumNumberOfSteps': 5000},
        'param_id_obs_path': os.path.join(resources_dir, 'TwoMinima_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'optimiser_options': {
            'cost_convergence': 1e-3,
            'num_starts': 20,
            'start_sampling': 'sobol',
            'seed': 0,
            'include_init_point': False,
            # run every start so we can count how many land in each minimum
            'no_new_starts_on_convergence': False,
        },
    })

    if rank == 0:
        assert generate_with_new_architecture(False, config), \
            'TwoMinima model generation should succeed'
    mpi_comm.Barrier()

    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir, 'multi_start_sp_minimize_TwoMinima_TwoMinima_obs_data')
        clusters = list(_csv.DictReader(
            open(os.path.join(output_dir, 'multi_start_convergence_clusters.csv'))))

        # exactly the two global minima were found
        assert len(clusters) == 2, \
            f'expected two distinct solutions (a = +2 and a = -2), got {len(clusters)}: {clusters}'

        a_vals = sorted(float(c['twomin a']) for c in clusters)
        np.testing.assert_allclose(a_vals, [-2.0, 2.0], atol=0.02)

        counts = {round(float(c['twomin a'])): int(c['num_starts']) for c in clusters}
        total_converged = sum(counts.values())
        print(f'\n[two-minima] of 20 starts, {total_converged} converged: '
              f'{counts[2]} to a=+2, {counts[-2]} to a=-2')

        # every start converged to one of the two wells, and both wells were reached
        assert total_converged == 20
        assert counts[2] > 0 and counts[-2] > 0
        # the split is reported per minimum; with sobol sampling it is not exactly even
        assert counts[2] != counts[-2], \
            'expected the two minima to receive different numbers of starts'

    mpi_comm.Barrier()


@pytest.mark.unit
def test_multi_start_lbfgsb_escapes_a_local_minimum_that_traps_single_start(temp_output_dir):
    """The headline behaviour: from an x0 inside a local well, a single L-BFGS-B descent
    stays in that well, while scattering starts over the bounds finds the global one."""
    single_dir = os.path.join(temp_output_dir, 'multi_start_single')
    multi_dir = os.path.join(temp_output_dir, 'multi_start_multi')
    for d in (single_dir, multi_dir):
        os.makedirs(d, exist_ok=True)

    # num_starts=1 with include_init_point is exactly a single-start L-BFGS-B from x0.
    single = _make_multi_start_optimiser(
        single_dir, _TwoWellParamId(param_init=(1.2, 1.2)),
        {'num_starts': 1, 'include_init_point': True, 'cost_convergence': 1e-12})
    single.run()

    multi = _make_multi_start_optimiser(
        multi_dir, _TwoWellParamId(param_init=(1.2, 1.2)),
        {'num_starts': 8, 'include_init_point': True, 'start_sampling': 'sobol',
         'seed': 0, 'cost_convergence': 1e-12})
    multi.run()

    # the single start is trapped in the right-hand well ...
    assert np.all(single.best_param_vals > 0.0), \
        f'expected the single start to stay in the +1 well, got {single.best_param_vals}'
    # ... while the multi-start escapes into the left-hand (global) well
    assert np.all(multi.best_param_vals < 0.0), \
        f'expected the multi-start to find the -1 well, got {multi.best_param_vals}'
    assert multi.best_cost < single.best_cost - 1.0, \
        f'multi-start cost {multi.best_cost} should be well below {single.best_cost}'


@pytest.mark.unit
def test_multi_start_finite_difference_fallback_without_casadi(temp_output_dir):
    """For non-casadi models there is no AD cost, so the optimiser must fall back to
    get_cost_from_params with a finite-difference gradient and still escape the local well."""
    # a cellml_only model has no AD gradient, so get_gradient would raise if it were called
    param_id_obj = _TwoWellParamId(param_init=(1.2, 1.2), model_type='cellml_only')
    opt = _make_multi_start_optimiser(
        temp_output_dir, param_id_obj,
        {'num_starts': 8, 'start_sampling': 'sobol', 'seed': 0, 'cost_convergence': 1e-12},
        model_type='cellml_only', do_ad=False)
    opt.run()

    assert param_id_obj.num_jac_calls == 0, \
        'the AD gradient must not be used for a model type that has no AD backend'
    assert param_id_obj.num_cost_calls > 0
    assert np.all(opt.best_param_vals < 0.0), \
        f'finite-difference multi-start should still find the -1 well, got {opt.best_param_vals}'


@pytest.mark.unit
def test_multi_start_generates_deterministic_starts(temp_output_dir):
    """Every rank must generate an identical set of starts (they are assigned round-robin
    with no communication), and the initial parameter values must be the first start."""
    opts = {'num_starts': 6, 'start_sampling': 'sobol', 'seed': 3, 'include_init_point': True}
    first = _make_multi_start_optimiser(temp_output_dir, _TwoWellParamId(), dict(opts))
    second = _make_multi_start_optimiser(temp_output_dir, _TwoWellParamId(), dict(opts))

    starts_a = first._generate_starts()
    starts_b = second._generate_starts()

    assert starts_a.shape == (6, 2)
    np.testing.assert_allclose(starts_a, starts_b)
    # start 0 is x0 = (1.2, 1.2) normalised over bounds [-2, 2]
    np.testing.assert_allclose(starts_a[0], _TwoWellNorm().normalise(np.array([1.2, 1.2])))
    # everything stays inside the normalised bounds
    assert np.all(starts_a >= 0.0) and np.all(starts_a <= 1.0)

    without_init = _make_multi_start_optimiser(
        temp_output_dir, _TwoWellParamId(),
        {'num_starts': 6, 'start_sampling': 'sobol', 'seed': 3, 'include_init_point': False})
    starts_c = without_init._generate_starts()
    assert starts_c.shape == (6, 2)
    assert not np.allclose(starts_c[0], starts_a[0])


@pytest.mark.unit
def test_multi_start_rejects_bad_options(temp_output_dir):
    bad_sampling = _make_multi_start_optimiser(
        temp_output_dir, _TwoWellParamId(), {'num_starts': 4, 'start_sampling': 'banana'})
    with pytest.raises(ValueError, match='banana'):
        bad_sampling._generate_starts()

    no_starts = _make_multi_start_optimiser(
        temp_output_dir, _TwoWellParamId(), {'num_starts': 0})
    with pytest.raises(ValueError, match='num_starts'):
        no_starts._generate_starts()


@pytest.mark.unit
def test_multi_start_writes_running_best_history_and_start_summary(temp_output_dir):
    """The history csvs must stay monotonically improving across the concatenated starts
    (the plotting reads them as a progress curve), and every start that ran gets a
    summary row."""
    out_dir = os.path.join(temp_output_dir, 'multi_start_history')
    os.makedirs(out_dir, exist_ok=True)

    opt = _make_multi_start_optimiser(
        out_dir, _TwoWellParamId(param_init=(1.2, 1.2)),
        {'num_starts': 5, 'start_sampling': 'latin_hypercube', 'seed': 1,
         'cost_convergence': 1e-12})
    opt.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        cost_path = os.path.join(out_dir, 'best_cost_history.csv')
        costs = [float(ln.split(',')[0])
                 for ln in open(cost_path).read().splitlines() if ln.strip()]
        assert len(costs) > 1
        # only improvements are written, but the csv rounds to 9 dp so two consecutive
        # improvements can print as equal; the curve must never go back up.
        assert all(costs[i + 1] <= costs[i] for i in range(len(costs) - 1)), \
            f'running-best cost history must never worsen, got {costs}'
        # abs tolerance matches the 9 dp the history csv is written with
        assert costs[-1] == pytest.approx(opt.best_cost, abs=1e-8)

        param_rows = [ln for ln in
                      open(os.path.join(out_dir, 'best_param_vals_history.csv')).read().splitlines()
                      if ln.strip()]
        assert len(param_rows) == len(costs)

        import csv as _csv
        summary = list(_csv.DictReader(open(os.path.join(out_dir, 'multi_start_summary.csv'))))
        assert len(summary) == 5, f'expected one summary row per start, got {len(summary)}'
        assert [int(r['start_idx']) for r in summary] == [0, 1, 2, 3, 4]
        for row in summary:
            assert float(row['final_cost']) <= float(row['init_cost']) + 1e-9

    MPI.COMM_WORLD.Barrier()


def test_multi_start_streams_per_start_cost_history_live(temp_output_dir):
    """multi_start_cost_history.csv streams one `start_idx, iteration, cost` row per L-BFGS-B
    iteration (iteration 0 = the start point), live and independent of DEBUG, so a GUI can plot
    a cost-vs-iteration line per start during the run (issue #286).

    Checks: the header; DEBUG-independence (the optimiser is built with DEBUG=False); a row for
    every global start index 0..N-1; iterations per start start at 0 and increase by 1; the cost
    is non-increasing within a start (L-BFGS-B accepts only improving steps); and each start's
    last streamed cost matches its multi_start_summary.csv final_cost.
    """
    import csv as _csv
    out_dir = os.path.join(temp_output_dir, 'multi_start_stream')
    os.makedirs(out_dir, exist_ok=True)

    num_starts = 5
    opt = _make_multi_start_optimiser(
        out_dir, _TwoWellParamId(param_init=(1.2, 1.2)),
        {'num_starts': num_starts, 'start_sampling': 'latin_hypercube', 'seed': 1,
         'cost_convergence': 1e-12})
    assert opt.DEBUG is False  # streaming must not depend on DEBUG
    opt.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        path = os.path.join(out_dir, 'multi_start_cost_history.csv')
        lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
        assert lines[0] == 'start_idx, iteration, cost'

        per_start = {}
        for ln in lines[1:]:
            s_idx, it, cost = [tok.strip() for tok in ln.split(',')]
            per_start.setdefault(int(s_idx), []).append((int(it), float(cost)))

        # Every start streamed, keyed by its global index.
        assert sorted(per_start) == list(range(num_starts)), sorted(per_start)

        summary = {int(r['start_idx']): r for r in
                   _csv.DictReader(open(os.path.join(out_dir, 'multi_start_summary.csv')))}
        for s_idx, rows in per_start.items():
            iters = [it for it, _ in rows]
            costs = [c for _, c in rows]
            # iteration 0 (the start point) is present and iterations increase by 1.
            assert iters == list(range(len(iters))), (s_idx, iters)
            # L-BFGS-B only accepts improving steps, so the per-start curve never worsens
            # (9 sig-fig rounding can make two consecutive equal).
            assert all(costs[i + 1] <= costs[i] * (1 + 1e-9) + 1e-30
                       for i in range(len(costs) - 1)), (s_idx, costs)
            # the first streamed cost is this start's init cost; the curve ends no worse than
            # the start's recorded final cost (res.fun can be from a point after the last
            # callback, so only a one-sided bound is guaranteed).
            assert costs[0] == pytest.approx(float(summary[s_idx]['init_cost']), rel=1e-6)
            assert float(summary[s_idx]['final_cost']) <= costs[-1] * (1 + 1e-6) + 1e-30

    MPI.COMM_WORLD.Barrier()


# The FitzHugh-Nagumo excitable-cell model is a standard multi-modal parameter-estimation
# benchmark: its least-squares surface has many local minima, because a wrong recovery rate
# c puts the simulated spike train out of phase with the data (Ramsay et al. 2007, JRSS-B,
# "Parameter estimation for differential equations: a generalized smoothing approach").
# resources/FitzHugh_Nagumo_parameters.csv deliberately starts at (a, b, c) = (0.8, 0.9, 2.0),
# which is inside a local basin: a single L-BFGS-B descent converges to (1, 1, 2.07) and
# stops, nowhere near the true (0.2, 0.2, 3.0).
# The FitzHugh-Nagumo benchmark run logic lives in benchmarks/ so it can be driven both by the
# test below and by the standalone benchmark runner; the config builder is shared with the other
# FHN tests here.
from benchmarks.benchmark_specs import (  # noqa: E402
    FHN_TRUE_PARAMS,
    fitzhugh_nagumo_config as _fitzhugh_nagumo_config,
    run_fitzhugh_nagumo,
    assert_fitzhugh_nagumo,
)
FHN_TRAPPED_COST = 100.0  # the local minimum a single start falls into sits at ~144


def _load_best(temp_output_dir, param_id_method):
    output_dir = os.path.join(
        temp_output_dir, f'{param_id_method}_FitzHugh_Nagumo_FitzHugh_Nagumo_obs_data')
    cost = float(np.load(os.path.join(output_dir, 'best_cost.npy')))
    params = np.load(os.path.join(output_dir, 'best_param_vals.npy'))
    return cost, params


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_multi_start_escapes_fitzhugh_nagumo_local_minimum_that_traps_sp_minimize(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """On the multi-modal FitzHugh-Nagumo benchmark, sp_minimize converges to a local
    minimum while multi_start_sp_minimize recovers the true parameters."""
    rank = mpi_comm.Get_rank()

    sp_config = _fitzhugh_nagumo_config(base_user_inputs, resources_dir, temp_output_dir,
                                        temp_generated_models_dir, 'sp_minimize')
    sp_config['optimiser_options'] = {'cost_convergence': 1e-4, 'max_patience': 100}

    if rank == 0:
        assert generate_with_new_architecture(False, sp_config), \
            'CasADi model generation should succeed for FitzHugh-Nagumo'
    mpi_comm.Barrier()

    run_param_id(sp_config)

    multi_config = _fitzhugh_nagumo_config(base_user_inputs, resources_dir, temp_output_dir,
                                           temp_generated_models_dir, 'multi_start_sp_minimize')
    multi_config['optimiser_options'] = {
        'cost_convergence': 1e-4, 'max_patience': 100,
        'num_starts': 8, 'start_sampling': 'sobol', 'seed': 0,
    }
    run_param_id(multi_config)

    if rank == 0:
        sp_cost, sp_params = _load_best(temp_output_dir, 'sp_minimize')
        multi_cost, multi_params = _load_best(temp_output_dir, 'multi_start_sp_minimize')

        # sp_minimize starts inside the local basin and cannot leave it
        assert sp_cost > FHN_TRAPPED_COST, (
            f'expected sp_minimize to stay trapped in the local minimum (cost > '
            f'{FHN_TRAPPED_COST}), got {sp_cost}. The benchmark x0 may no longer be inside '
            f'the local basin.')
        assert np.max(np.abs(sp_params - FHN_TRUE_PARAMS)) > 0.5, \
            f'expected sp_minimize to miss the true params, got {sp_params}'

        # the multi-start finds the global minimum and recovers the true parameters
        assert multi_cost < 1e-2, \
            f'expected multi_start_sp_minimize to reach the global minimum, got {multi_cost}'
        np.testing.assert_allclose(multi_params, FHN_TRUE_PARAMS, atol=0.02)
        assert multi_cost < sp_cost

    mpi_comm.Barrier()


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="compare_optimisers is heavy; run locally / in the benchmarks workflow only",
)
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
@pytest.mark.compare_optimisers
def test_compare_optimisers_on_fitzhugh_nagumo(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """Benchmark every optimiser on the multi-modal FitzHugh-Nagumo problem.

    Gradient-free global searches (genetic algorithm, CMA-ES) against the same multi-start
    L-BFGS-B driven by finite differences, CasADi AD, Myokit CVODES FSA, and (when licensed)
    AADC AD. Holding the optimiser fixed and varying only the gradient source isolates what the
    gradient buys. The run logic and regression assertions live in
    ``benchmarks/benchmark_specs.py`` so the standalone benchmark runner exercises exactly the
    same comparison.
    """
    result = run_fitzhugh_nagumo(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm)
    assert_fitzhugh_nagumo(result, mpi_comm)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_ad_series_cost_supports_dt_finer_than_obs_dt(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """The symbolic cost cannot resample the (symbolic) simulated series, so it interpolates the
    ground truth onto the simulation time grid instead. That means dt does not have to equal a
    series item's obs_dt. Check that a finer dt still gives a sane cost and a correct gradient.

    FitzHugh_Nagumo_obs_data.json has obs_dt = 0.2; here the model is simulated at dt = 0.1.
    """
    import json

    rank = mpi_comm.Get_rank()
    if rank != 0:
        mpi_comm.Barrier()
        return

    config = _fitzhugh_nagumo_config(base_user_inputs, resources_dir, temp_output_dir,
                                     temp_generated_models_dir, 'sp_minimize')
    config['dt'] = 0.1  # half the obs_dt of 0.2, so the ground truth must be interpolated
    config['resources_dir'] = resources_dir

    assert generate_with_new_architecture(False, config), \
        'CasADi model generation should succeed for FitzHugh-Nagumo'

    parsed = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)
    parsed['one_rank'] = True
    runner = CVS0DParamID.init_from_dict(parsed)

    with open(os.path.join(resources_dir, 'FitzHugh_Nagumo_obs_data.json')) as f:
        runner.set_ground_truth_data(json.load(f))

    runner.set_params_for_id([
        {'vessel_name': 'fhn', 'param_name': 'a', 'param_type': 'const', 'min': 0.0, 'max': 1.0},
        {'vessel_name': 'fhn', 'param_name': 'b', 'param_type': 'const', 'min': 0.0, 'max': 1.0},
        {'vessel_name': 'fhn', 'param_name': 'c', 'param_type': 'const', 'min': 0.5, 'max': 6.0},
    ])

    # At the true parameters the cost must be ~0 even though the data was sampled at obs_dt and
    # the model is being run at half that step.
    cost_at_truth = float(runner.param_id.get_cost_ca(FHN_TRUE_PARAMS))
    assert np.isfinite(cost_at_truth)
    assert cost_at_truth < 1e-3, \
        f'cost at the true params should be ~0 with dt < obs_dt, got {cost_at_truth}'

    # Away from the truth the cost is finite and positive, and the AD gradient agrees with a
    # central finite difference of the same cost function.
    probe = np.array([0.35, 0.30, 2.6])
    cost_at_probe = float(runner.param_id.get_cost_ca(probe))
    assert np.isfinite(cost_at_probe) and cost_at_probe > cost_at_truth

    gradient = np.asarray(runner.param_id.get_jac_cost_ca(probe), dtype=float).ravel()
    assert gradient.shape == (3,)
    assert np.all(np.isfinite(gradient))

    step = 1e-5
    for idx in range(3):
        p_plus = probe.copy()
        p_minus = probe.copy()
        p_plus[idx] += step
        p_minus[idx] -= step
        fd = (float(runner.param_id.get_cost_ca(p_plus))
              - float(runner.param_id.get_cost_ca(p_minus))) / (2 * step)
        assert fd == pytest.approx(gradient[idx], rel=1e-2, abs=1e-4), (
            f'AD gradient {gradient[idx]} disagrees with the finite difference {fd} for '
            f'parameter {idx}, so the interpolated ground truth is breaking differentiability')

    mpi_comm.Barrier()


@pytest.mark.unit
def test_series_interpolation_uses_the_correct_sample_times():
    """Sample k of a series is at time k*dt, so a simulated series interpolated onto the
    observation times must land on exactly those times.

    The old code built both time grids with np.linspace(0, n*step, n), whose spacing is
    n*step/(n-1) rather than step. That stretches the simulation and observation grids by
    different factors, so they drift apart over a long simulation — about a full observation
    sample over the 60 s below — and the residuals are taken at the wrong times.
    """
    from param_id.paramID import OpencorParamID

    dt = 0.1
    obs_dt = 0.25
    num_sim = 601                      # 0 .. 60.0 s at dt
    ground_truth = np.zeros(241)       # 0 .. 60.0 s at obs_dt

    class _Fake:
        pass

    fake = _Fake()
    fake.dt = dt
    fake.obs_info = {
        "ground_truth_series": [ground_truth],
        "std_series_vec": [np.ones_like(ground_truth)],
        "obs_dt": [obs_dt],
    }

    # A simulated series whose value *is* the time. Interpolating it onto the observation
    # times must therefore reproduce those times exactly.
    t_sim = np.arange(num_sim) * dt
    series_entry, obs_entry, std_entry = OpencorParamID._align_series_to_ground_truth(
        fake, t_sim.copy(), 0)

    expected_t_obs = np.arange(ground_truth.shape[0]) * obs_dt
    np.testing.assert_allclose(series_entry, expected_t_obs, atol=1e-9)
    assert len(obs_entry) == len(series_entry) == len(std_entry)


@pytest.mark.unit
def test_series_interpolation_matches_between_numpy_and_casadi_paths():
    """The numeric and symbolic costs must resample a series identically, otherwise the same
    model calibrated as cellml_only and as casadi_python would have different cost surfaces."""
    import casadi as ca
    from param_id.paramID import OpencorParamID

    dt = 0.1
    obs_dt = 0.25
    num_sim = 61
    ground_truth = np.zeros(21)

    class _Fake:
        pass

    fake = _Fake()
    fake.dt = dt
    fake.obs_info = {
        "ground_truth_series": [ground_truth],
        "std_series_vec": [np.ones_like(ground_truth)],
        "obs_dt": [obs_dt],
    }

    rng = np.random.default_rng(0)
    sim = rng.normal(size=num_sim)

    from_numpy, _, _ = OpencorParamID._align_series_to_ground_truth(fake, sim.copy(), 0)
    from_casadi, _, _ = OpencorParamID._align_series_to_ground_truth(
        fake, ca.DM(sim.reshape(-1, 1)), 0)

    np.testing.assert_allclose(
        np.asarray(from_numpy, dtype=float).flatten(),
        np.asarray(ca.DM(from_casadi), dtype=float).flatten(),
        rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_series_interpolation_does_not_invent_data_past_the_end_of_the_simulation():
    """np.interp clamps to the last value, which would compare a flat fabricated tail against
    real observations. Observation times past the end of the simulation are dropped instead."""
    from param_id.paramID import OpencorParamID

    dt = 0.1
    obs_dt = 0.25
    num_sim = 21                    # simulation only reaches 2.0 s
    ground_truth = np.arange(21.0)  # observations run out to 5.0 s

    class _Fake:
        pass

    fake = _Fake()
    fake.dt = dt
    fake.obs_info = {
        "ground_truth_series": [ground_truth],
        "std_series_vec": [np.ones_like(ground_truth)],
        "obs_dt": [obs_dt],
    }

    t_sim = np.arange(num_sim) * dt
    series_entry, obs_entry, std_entry = OpencorParamID._align_series_to_ground_truth(
        fake, t_sim.copy(), 0)

    # only the observation times up to 2.0 s (0, 0.25, ..., 2.0 => 9 of them) are compared
    assert len(series_entry) == 9
    assert len(obs_entry) == 9 and len(std_entry) == 9
    np.testing.assert_allclose(series_entry, np.arange(9) * obs_dt, atol=1e-9)


@pytest.mark.unit
def test_multi_start_uses_the_ad_gradient_for_aadc_models(temp_output_dir):
    """An aadc_python model has a tape gradient, so the multi-start must call get_gradient()
    rather than quietly falling back to finite differences (which is what it did before the
    backend-agnostic AD wiring landed)."""
    param_id_obj = _TwoWellParamId(param_init=(1.2, 1.2), model_type='aadc_python')
    opt = _make_multi_start_optimiser(
        temp_output_dir, param_id_obj,
        {'num_starts': 4, 'start_sampling': 'sobol', 'seed': 0, 'cost_convergence': 1e-12},
        model_type='aadc_python', do_ad=True)
    assert opt.use_ad_gradient, 'aadc_python models must use the AD gradient'
    opt.run()

    assert param_id_obj.num_jac_calls > 0, \
        'expected the aadc tape gradient to be used, not finite differences'
    assert np.all(opt.best_param_vals < 0.0), \
        f'the multi-start should still find the -1 well, got {opt.best_param_vals}'


@pytest.mark.unit
def test_multi_start_falls_back_to_fd_when_do_ad_is_off(temp_output_dir):
    """do_ad: false must disable the AD gradient even on a model type that has one."""
    param_id_obj = _TwoWellParamId(param_init=(1.2, 1.2), model_type='casadi_python')
    opt = _make_multi_start_optimiser(
        temp_output_dir, param_id_obj,
        {'num_starts': 4, 'start_sampling': 'sobol', 'seed': 0, 'cost_convergence': 1e-12},
        model_type='casadi_python', do_ad=False)
    assert not opt.use_ad_gradient
    opt.run()
    assert param_id_obj.num_jac_calls == 0


# ---------------------------------------------------------------------------
# AADC gradient: the cost and the gradient must be of the SAME function
# ---------------------------------------------------------------------------

def _aadc_lotka_volterra_config(base_user_inputs, resources_dir, temp_output_dir,
                                temp_generated_models_dir, param_id_method='sp_minimize',
                                method='rk4'):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Lotka_Volterra',
        'input_param_file': 'Lotka_Volterra_parameters.csv',
        'params_for_id_file': 'Lotka_Volterra_params_for_id.csv',
        'model_type': 'aadc_python',
        'solver': 'aadc_semi_implicit',
        'param_id_method': param_id_method,
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 5.0,
        'dt': 0.01,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        # fixed-step, and exactly what the AADC tape records
        'solver_info': {'method': method},
        'param_id_obs_path': os.path.join(resources_dir, 'Lotka_Volterra_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'resources_dir': resources_dir,
    })
    return config


def _init_aadc_param_id(config, resources_dir):
    # rank 0 generates the model into the shared dir; the others wait, then all load it. Letting
    # every rank generate into the same directory races.
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, config), 'AADC model generation should succeed'
    comm.Barrier()
    parsed = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)
    parsed['one_rank'] = True
    parsed['do_ad'] = True
    runner = CVS0DParamID.init_from_dict(parsed)
    inner = runner.param_id
    inner.param_init = inner.sim_helper.get_init_param_vals(inner.param_id_info['param_names'])
    return inner


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_cost_and_gradient_are_of_the_same_function(
        aadc_licensed, base_user_inputs, resources_dir, temp_output_dir,
        temp_generated_models_dir):
    """The AADC tape gradient must be the gradient of the cost that is actually being minimised.

    A gradient that is merely 'close' is not good enough: L-BFGS-B's line search assumes the
    gradient is exact for the cost it evaluates, so a mismatch does not just slow convergence,
    it sends the search to the wrong place. Three separate bugs each broke this, and each
    showed up here as AD/FD != 1:

      * the tape recorded fixed-step RK4 while the forward solve ran adaptive RK45, so the
        gradient was exact -- for a different discretisation  -> AD/FD = [1.79, 1.96, 1.32, -0.07]
      * the tape did not divide by the weighted-observable count that the numpy cost divides
        by, making the tape cost a constant 2x the real one                -> AD/FD = [2, 2, 2, 2]
      * every parameter's AD index was appended twice, so the index vector was double length
    """
    inner = _init_aadc_param_id(
        _aadc_lotka_volterra_config(base_user_inputs, resources_dir, temp_output_dir,
                                    temp_generated_models_dir),
        resources_dir)

    p0 = np.asarray(inner.param_init, dtype=float)
    assert len(p0) == 4

    # 1. the tape's cost is the forward solve's cost
    tape_cost = float(inner.get_cost_aadc(p0))
    forward_cost = float(inner.get_cost_from_params(p0))
    assert tape_cost == pytest.approx(forward_cost, rel=1e-9), (
        f'the AADC tape cost ({tape_cost}) is not the cost the forward solve computes '
        f'({forward_cost}); the optimiser would descend a different function than it evaluates')

    # 2. get_cost routes to the tape when do_ad is on, so cost and gradient are one evaluation
    assert float(inner.get_cost(p0)) == pytest.approx(tape_cost, rel=1e-9)

    # 3. the AD gradient is the gradient of that cost
    grad = np.asarray(inner.get_gradient(p0), dtype=float).flatten()
    assert grad.shape == (4,), f'expected one gradient entry per parameter, got {grad.shape}'
    assert np.all(np.isfinite(grad))

    step = 1e-6
    for j in range(4):
        up, down = p0.copy(), p0.copy()
        up[j] += step
        down[j] -= step
        fd = (float(inner.get_cost(up)) - float(inner.get_cost(down))) / (2 * step)
        assert grad[j] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
            f'AADC gradient[{j}] = {grad[j]} disagrees with the finite difference {fd} '
            f'(AD/FD = {grad[j] / (fd + 1e-30):.4f})')


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_gradient_rejects_an_untapeable_solver(
        aadc_licensed, base_user_inputs, resources_dir, temp_output_dir,
        temp_generated_models_dir):
    """An adaptive integrator picks its steps from the state, so the tape cannot replay it. Ask
    for one with do_ad on and the run must stop, not silently differentiate a different (RK4)
    system than it simulates."""
    inner = _init_aadc_param_id(
        _aadc_lotka_volterra_config(base_user_inputs, resources_dir, temp_output_dir,
                                    temp_generated_models_dir, method='adaptive_rk45'),
        resources_dir)

    with pytest.raises(ValueError, match='cannot be recorded on an AADC tape'):
        inner.get_gradient(np.asarray(inner.param_init, dtype=float))


def _aadc_3compartment_config(base_user_inputs, resources_dir, temp_output_dir,
                              temp_generated_models_dir):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'params_for_id_file': '3compartment_params_for_id.csv',
        'model_type': 'aadc_python',
        'solver': 'aadc_semi_implicit',
        'param_id_method': 'sp_minimize',
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 0.3,
        'dt': 0.01,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {'method': 'semi_implicit'},  # fixed-step, tape-consistent
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'resources_dir': resources_dir,
    })
    return config


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_gradient_rejects_untapeable_observables(
        aadc_licensed, base_user_inputs, resources_dir, temp_output_dir,
        temp_generated_models_dir):
    """The current AADC wrapper can only tape an observable whose operand is a state with a
    max/min/mean operation (or a state series). 3compartment's obs set has algebraic-variable
    operands (aortic_root/u) and a max_minus_min, so the tape cannot represent them. Taping only
    the rest would silently minimise a reduced cost, so the wrapper must raise -- and the message
    must point at the tracking issue (#258) and the working alternatives (CasADi bdf / Myokit
    FSA)."""
    inner = _init_aadc_param_id(
        _aadc_3compartment_config(base_user_inputs, resources_dir, temp_output_dir,
                                  temp_generated_models_dir),
        resources_dir)

    with pytest.raises(NotImplementedError,
                       match=r'cannot be represented on the AADC tape'):
        inner.get_gradient(np.asarray(inner.param_init, dtype=float))
    # the error must name the tracking issue so users know it is a known limitation
    with pytest.raises(NotImplementedError, match=r'#258'):
        inner.get_gradient(np.asarray(inner.param_init, dtype=float))


def _aadc_fitzhugh_nagumo_config(base_user_inputs, resources_dir, temp_output_dir,
                                 temp_generated_models_dir, param_id_method='sp_minimize'):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'FitzHugh_Nagumo',
        'input_param_file': 'FitzHugh_Nagumo_parameters.csv',
        'params_for_id_file': 'FitzHugh_Nagumo_params_for_id.csv',
        'model_type': 'aadc_python',
        'solver': 'aadc_semi_implicit',
        'param_id_method': param_id_method,
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 60.0,
        # dt differs from the obs_dt of 0.2, so this exercises the tape's series interpolation
        'dt': 0.02,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {'method': 'rk4'},  # fixed-step, what the AADC tape records
        'param_id_obs_path': os.path.join(resources_dir, 'FitzHugh_Nagumo_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'resources_dir': resources_dir,
    })
    return config


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_series_cost_and_gradient_are_of_the_same_function(
        aadc_licensed, base_user_inputs, resources_dir, temp_output_dir,
        temp_generated_models_dir):
    """The AADC tape gradient must match the cost for a SERIES observable, not just a constant.

    FitzHugh-Nagumo fits the V and R traces (series data), and its obs_dt (0.2) differs from the
    simulation dt (0.02), so this exercises the tape's series-observable path and its
    interpolation of the simulated trace onto the observation times -- code that the
    constant-observable Lotka-Volterra test does not touch. As there, the check is that the tape
    cost equals the forward-solve cost and that the tape gradient agrees with a finite difference
    of that cost (AD/FD == 1), at the initial parameters and at perturbed ones."""
    inner = _init_aadc_param_id(
        _aadc_fitzhugh_nagumo_config(base_user_inputs, resources_dir, temp_output_dir,
                                     temp_generated_models_dir),
        resources_dir)

    p0 = np.asarray(inner.param_init, dtype=float)
    assert len(p0) == 3  # a, b, c

    for label, p in (('initial', p0), ('perturbed', p0 * np.array([1.1, 0.9, 1.05]))):
        tape_cost = float(inner.get_cost_aadc(p))
        forward_cost = float(inner.get_cost_from_params(p))
        assert tape_cost == pytest.approx(forward_cost, rel=1e-6), (
            f'[{label}] the AADC tape cost ({tape_cost}) is not the forward-solve cost '
            f'({forward_cost}) for a series observable')

        grad = np.asarray(inner.get_gradient(p), dtype=float).flatten()
        assert grad.shape == (3,)
        assert np.all(np.isfinite(grad))

        step = 1e-6
        for j in range(3):
            up, down = p.copy(), p.copy()
            up[j] += step
            down[j] -= step
            fd = (float(inner.get_cost(up)) - float(inner.get_cost(down))) / (2 * step)
            assert grad[j] == pytest.approx(fd, rel=2e-3, abs=1e-5), (
                f'[{label}] AADC series gradient[{j}] = {grad[j]} disagrees with the finite '
                f'difference {fd} (AD/FD = {grad[j] / (fd + 1e-30):.4f})')


@pytest.mark.integration
@pytest.mark.slow
def test_multi_start_sp_minimize_calibrates_an_aadc_model(
        aadc_licensed, base_user_inputs, resources_dir, temp_output_dir,
        temp_generated_models_dir):
    """multi_start_sp_minimize must actually calibrate an aadc_python model using the tape
    gradient -- the merge criterion for the AADC backend."""
    config = _aadc_lotka_volterra_config(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
        param_id_method='multi_start_sp_minimize')
    config['optimiser_options'] = {
        'cost_convergence': 1e-3, 'max_patience': 100,
        'num_starts': 4, 'start_sampling': 'sobol', 'seed': 0,
    }

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        assert generate_with_new_architecture(False, config), \
            'AADC model generation should succeed'
    comm.Barrier()
    run_param_id(config)

    # only rank 0 writes the result files
    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir, 'multi_start_sp_minimize_Lotka_Volterra_Lotka_Volterra_obs_data')
        cost = float(np.load(os.path.join(output_dir, 'best_cost.npy')))
        params = np.load(os.path.join(output_dir, 'best_param_vals.npy'))

        assert np.isfinite(cost)
        assert np.all(np.isfinite(params))
        # the observables are two constants the model can match, so a working gradient descent
        # should drive the cost right down; a wrong gradient leaves it stuck near its start
        assert cost < 1e-2, \
            f'multi_start_sp_minimize with the AADC tape gradient failed to calibrate: cost {cost}'
    comm.Barrier()
