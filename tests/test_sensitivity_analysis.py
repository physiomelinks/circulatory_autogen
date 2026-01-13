"""
Tests for sensitivity analysis functionality.

These tests verify that sensitivity analysis works correctly for various models.
"""
import os
import pytest
from mpi4py import MPI

from scripts.sensitivity_analysis_run_script import run_SA


@pytest.fixture(scope="function")
def mpi_comm():
    """Fixture that provides MPI communicator."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("MPI tests require mpiexec with at least 2 ranks")
    return comm


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_sensitivity_analysis_3compartment_succeeds(base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """
    Test that sensitivity analysis succeeds for 3compartment model.
    
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
        'model_out_names': ['heart/u_lv'],
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'debug_optimiser_options': {'num_calls_to_function': 60},
        'sa_options': {
            'method': 'sobol',
            'num_samples': 16,
            'sample_type': 'saltelli',
            'output_dir': os.path.join(temp_output_dir, '3compartment_SA_results'),
        },
    })
    
    # Run sensitivity analysis
    run_SA(config)
    
    # Verify output was created (on rank 0)
    if rank == 0:
        output_dir = config['sa_options']['output_dir']
        assert os.path.exists(output_dir), f"Sensitivity analysis output directory should exist: {output_dir}"

