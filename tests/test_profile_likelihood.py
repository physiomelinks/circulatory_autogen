"""  
Tests for profile likelihood functionality.  
  
These tests verify that profile likelihood analysis works correctly for various models.  
"""  
import csv  
import os  
import json  
import tempfile  
  
import pytest  
import numpy as np  
from mpi4py import MPI  
  
from src.libcuflynx.identifiabilty_analysis.ProfileLikelihood import ProfileLikelihood  
from src.libcuflynx.scripts.param_id_run_script import run_param_id  
from src.libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture  
from src.libcuflynx.parsers.PrimitiveParsers import CSVFileParser  
  
  
@pytest.fixture(scope="function")  
def mpi_comm():  
    """Fixture that provides MPI communicator."""  
    comm = MPI.COMM_WORLD  
    return comm  
  
  
def _ensure_cellml_model_generated(config, mpi_comm):  
    """Ensure generated CellML exists before run_param_id (mirrors test_param_id.py)."""  
    if config.get("model_type") != "cellml_only":  
        return  
    rank = mpi_comm.Get_rank()  
    if rank == 0:  
        success = generate_with_new_architecture(False, config)  
        prefix = config.get("file_prefix", "<unknown>")  
        assert success, f"CellML autogeneration failed for {prefix}"  
    mpi_comm.Barrier()  
  
  
@pytest.mark.unit  
def test_profile_likelihood_initialization():  
    """Test ProfileLikelihood class initialization."""  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
  
    profile = ProfileLikelihood(  
        param_id=None,  
        param_id_info=param_id_info,  
        output_dir='/tmp/test',  
        num_points=20,  
        range_factor=0.1  
    )  
  
    assert profile.num_points == 20  
    assert profile.range_factor == 0.1  
    assert profile.output_dir == '/tmp/test'  
    assert profile.param_id_info == param_id_info  
  
  
@pytest.mark.unit  
def test_profile_likelihood_set_best_param_vals():  
    """Test setting best parameter values."""  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
  
    profile = ProfileLikelihood(  
        param_id=None,  
        param_id_info=param_id_info,  
        output_dir='/tmp/test'  
    )  
  
    best_params = np.array([0.5, 0.7])  
    profile.set_best_param_vals(best_params)  
  
    np.testing.assert_array_equal(profile.best_param_vals, best_params)  
  
  
@pytest.mark.unit  
def test_profile_likelihood_no_best_params():  
    """Test that run_profile_likelihood raises error without best params."""  
    param_id_info = {  
        'param_names': ['param1'],  
        'param_mins': np.array([0.0]),  
        'param_maxs': np.array([1.0])  
    }  
  
    profile = ProfileLikelihood(  
        param_id=None,  
        param_id_info=param_id_info,  
        output_dir='/tmp/test'  
    )  
  
    with pytest.raises(ValueError, match="Best parameter values must be set first"):  
        profile._run()  
  
@pytest.mark.integration  
@pytest.mark.slow  
@pytest.mark.mpi  
def test_profile_likelihood_3compartment_succeeds(  
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):  
    rank = mpi_comm.Get_rank()  
  
    # Dedicated params_for_id CSV for this test, written once on rank 0.  
    params_for_id_path = os.path.join(  
        temp_output_dir, "3compartment_profile_likelihood_params_for_id.csv"  
    )  
    if rank == 0:  
        with open(params_for_id_path, "w", newline="") as f:  
            writer = csv.writer(f)  
            writer.writerow(  
                ["vessel_name", "param_name", "param_type", "min", "max", "name_for_plotting"]  
            )  
            writer.writerow(["aortic_root", "C", "const", "1e-9", "5e-8", "C_{ao}"])  
            writer.writerow(["global", "q_lv_init", "const", "200e-6", "1500e-6", "q_{sbv}"])  
    mpi_comm.Barrier()  
  
    config = base_user_inputs.copy()  
    config.update({  
        'file_prefix': '3compartment',  
        'input_param_file': '3compartment_parameters.csv',  
        'params_for_id_file': params_for_id_path,   # <-- explicit, no reconstruction needed  
        'model_type': 'cellml_only',  
        'solver': 'CVODE_myokit',  
        'param_id_method': 'genetic_algorithm',  
        'pre_time': 20,  
        'sim_time': 2,  
        'dt': 0.01,  
        'DEBUG': True,  
        'plot_predictions': False,  
        'do_ia': False,  
        'solver_info': {'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000},  
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),  
        'param_id_output_dir': temp_output_dir,  
        'generated_models_dir': temp_generated_models_dir,  
        'optimiser_options': {  
            'num_calls_to_function': 100,  
            'max_patience': 10,  
            'cost_convergence': 1e-3,  
        },  
    })  
  
    _ensure_cellml_model_generated(config, mpi_comm)  
    run_param_id(config)  
  
    actual_output_dir = os.path.join(  
        temp_output_dir,  
        f"{config['param_id_method']}_{config['file_prefix']}_3compartment_obs_data"  
    )  
  
    if rank == 0:  
        csv_parser = CSVFileParser()  
        param_id_name_and_vals, _ = csv_parser.get_param_id_params_as_lists_of_tuples(  
            actual_output_dir  
        )  
        best_param_vals = np.array([val for name, val in param_id_name_and_vals])  
  
        from src.libcuflynx.param_id.paramID import CVS0DParamID  
  
        param_id = CVS0DParamID(  
            model_path=config['model_path'],  
            model_type=config['model_type'],  
            param_id_method=config['param_id_method'],  
            mcmc_instead=False,  
            file_name_prefix=config['file_prefix'],  
            params_for_id_path=config['params_for_id_path'],  
            param_id_obs_path=config['param_id_obs_path'],  
            sim_time=config['sim_time'],  
            pre_time=config['pre_time'],  
            solver_info=config['solver_info'],  
            dt=config['dt'],  
            optimiser_options=config['optimiser_options'],  
            DEBUG=config['DEBUG'],  
            param_id_output_dir=temp_output_dir,  
            resources_dir=config['resources_dir']  
        )  
  
        profile = ProfileLikelihood(  
            param_id=param_id.param_id,  
            param_id_info=param_id.param_id_info,  
            output_dir=temp_output_dir,  
            num_points=5,  
            range_factor=0.1,  
            optimiser_options={'num_calls_to_function': 50}  
        )  
        profile.set_best_param_vals(best_param_vals)  
        profile._run()  
  
        results_dir = os.path.join(temp_output_dir, 'profile_likelihood')  
        assert os.path.exists(results_dir), "Profile likelihood results directory should exist"  
        summary_file = os.path.join(results_dir, 'summary.json')  
        assert os.path.exists(summary_file), "Summary file should exist"  
        with open(summary_file, 'r') as f:  
            summary = json.load(f)  
        assert 'num_params' in summary and 'num_points' in summary  
        assert summary['num_points'] == 5
        assert len(os.listdir(results_dir)) > 0, "Parameter result files should exist"  
  
    mpi_comm.Barrier()  
  
@pytest.mark.unit  
def test_profile_likelihood_save_results():  
    """Test saving profile likelihood results."""  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
  
    with tempfile.TemporaryDirectory() as temp_dir:  
        profile = ProfileLikelihood(  
            param_id=None,  
            param_id_info=param_id_info,  
            output_dir=temp_dir  
        )  
  
        profile.profile_results = {  
            0: {  
                'param_values': np.array([0.1, 0.2, 0.3]),  
                'costs': np.array([1.0, 0.5, 1.0]),  
                'param_name': 'param1',  
                'best_val': 0.2  
            },  
            1: {  
                'param_values': np.array([0.4, 0.5, 0.6]),  
                'costs': np.array([2.0, 1.0, 2.0]),  
                'param_name': 'param2',  
                'best_val': 0.5  
            }  
        }  
  
        profile._save_results()  
  
        results_dir = os.path.join(temp_dir, 'profile_likelihood')  
        assert os.path.exists(results_dir)  
  
        summary_file = os.path.join(results_dir, 'summary.json')  
        assert os.path.exists(summary_file)  
  
        with open(summary_file, 'r') as f:  
            summary = json.load(f)  
  
        assert summary['num_params'] == 2  
        assert summary['parameters'] == ['param1', 'param2']  
  
        for param_idx in [0, 1]:  
            param_files = [f for f in os.listdir(results_dir) if f.startswith(f'param_{param_idx}_')]  
            assert len(param_files) > 0  
  
            with open(os.path.join(results_dir, param_files[0]), 'r') as f:  
                data = json.load(f)  
  
            assert 'param_values' in data  
            assert 'costs' in data  
            assert 'param_name' in data  
            assert 'best_val' in data  
  
  
@pytest.mark.unit  
def test_profile_likelihood_plotting():  
    """Test profile likelihood plotting functionality."""  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
      
    import tempfile  
    import matplotlib  
    matplotlib.use('Agg')  # Use non-interactive backend for testing  
      
    with tempfile.TemporaryDirectory() as temp_dir:  
        profile = ProfileLikelihood(  
            param_id=None,  
            param_id_info=param_id_info,  
            output_dir=temp_dir  
        )  
          
        # Mock results  
        profile.profile_results = {  
            0: {  
                'param_values': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  
                'costs': np.array([2.0, 1.0, 0.5, 1.0, 2.0]),  
                'param_name': 'param1',  
                'best_val': 0.3  
            },  
            1: {  
                'param_values': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  
                'costs': np.array([3.0, 1.5, 0.8, 1.5, 3.0]),  
                'param_name': 'param2',  
                'best_val': 0.3  
            }  
        }  
          
        profile._plot_profiles()  
          
        results_dir = os.path.join(temp_dir, 'profile_likelihood')  
          
        # Check that plot files were created  
        plot_files = [f for f in os.listdir(results_dir) if f.endswith('.pdf')]  
        assert len(plot_files) >= 1  # At least the combined plot  
          
        # Check individual parameter plots  
        for param_idx in [0, 1]:  
            param_plots = [f for f in os.listdir(results_dir)   
                          if f.startswith(f'param_{param_idx}_') and f.endswith('.pdf')]  
            assert len(param_plots) > 0