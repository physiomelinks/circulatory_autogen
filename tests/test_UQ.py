"""
Tests for parameter Uncertainty Quantification.

These tests verify that parameter Uncertainty Quantification works correctly for various models.
"""
import os
import pytest
import numpy as np
from mpi4py import MPI
from param_id.paramID import CVS0DParamID
from parsers.PrimitiveParsers import YamlFileParser
from pathlib import Path
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from scripts.param_id_run_script import run_param_id
from scripts.plot_param_id_script import plot_param_id
from scripts.example_format_obs_data_json_file import example_format_obs_data_json_file
from tests.test_param_id import _ensure_cellml_model_generated, mpi_comm
import pandas as pd

@pytest.mark.integration  
@pytest.mark.slow  
@pytest.mark.mpi  
def test_mcmc_unimodal_with_validation(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):  
    """  
    Test MCMC pipeline with unimodal distribution and validate results  
    using autocorrelation and effective sample size.  
    """  
    rank = mpi_comm.Get_rank()  
      
    config = base_user_inputs.copy()  
    config.update({  
        'file_prefix': 'Simple_ODE_Benchmark',  # Replace with your model name  
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',  # Replace with your CSV  
        'model_type': 'cellml_only',  
        'solver': 'CVODE',  
        'param_id_method': 'genetic_algorithm',  
        'pre_time': 0.5,  
        'sim_time': 10.0,  
        'dt': 0.1,  
        'DEBUG': False,  
        'do_mcmc': True,  
        'plot_predictions': False,  
        'do_ia': False,  
        "params_for_id_file": "Simple_ODE_Benchmark_params_for_id.csv",  # Specify which params to identify
        'param_id_obs_path': os.path.join(resources_dir, 'Simple_ODE_Benchmark_obs_data.json'),  
        'param_id_output_dir': temp_output_dir,
        'mcmc_options': {
            'mcmc_lib': 'pymc',  # 'emcee' or 'zeus' or 'pymc'
            'num_steps': 200,
            'num_walkers': 64,  # 2*num_params is a common choice
            'burn_in_percentage': 0.0,
            'method': 'mcmc'  # or 'smc' if using SMC
        },
        'optimiser_options': {
            "num_calls_to_function": 10000,  
            "cost_convergence": 0.01,        # Convergence tolerance  
            "max_patience": 5,              # Max generations without improvement  
            "cost_type": 'gaussian_MLE'
        }
    })  
      
    # Ensure model is generated  
    _ensure_cellml_model_generated(config, mpi_comm)  
      
    # Run parameter identification and MCMC  
    run_param_id(config)  
      
    # Validate results on rank 0  
    if rank == 0:  
        output_dir = os.path.join(  
            temp_output_dir,  
            f"{config['param_id_method']}_{config['file_prefix']}_Simple_ODE_Benchmark_obs_data"  
        )  
          
        # Check MCMC chain exists  
        chain_file = os.path.join(output_dir, 'mcmc_chain.npy')  
        assert os.path.exists(chain_file), "MCMC chain file should exist"  
          
        # Load and validate samples  
        samples = np.load(chain_file)
        burn_in_idx = int(samples.shape[0] * config['mcmc_options']['burn_in_percentage'])  
        samples = samples[burn_in_idx:, :, :]
        flat_samples = samples.reshape(-1, samples.shape[-1])  
          
        # Test plotting  
        mcmc_obj = CVS0DParamID(  
            config['model_path'], config['model_type'], config['param_id_method'], True,  
            config['file_prefix'],  
            params_for_id_path=config.get('params_for_id_path'),  
            param_id_obs_path=config['param_id_obs_path'],  
            sim_time=config['sim_time'], pre_time=config['pre_time'], dt=config['dt'],  
            param_id_output_dir=temp_output_dir, resources_dir=resources_dir,  
            mcmc_options=config['mcmc_options'], DEBUG=config['DEBUG'], one_rank=True  
        )  
        mcmc_obj.plot_mcmc()  

        plot_dir = Path(mcmc_obj.plot_dir)  
        matches = list(plot_dir.glob('mcmc_cornerplot_*.pdf'))
        assert len(matches) > 0

        posterior_stats = mcmc_obj.get_posterior_stats(samples)
        
        assert all(val > 50 for val in posterior_stats['ess_bulk']+posterior_stats['ess_tail']), f"Some parameters have low ESS: {posterior_stats['ess_bulk']+posterior_stats['ess_tail']}"
        # assert all(val > 50 for val in posterior_stats['ess_tail']), f"Some parameters have low tail ESS: {posterior_stats['ess_tail']}"
      
        for param, r_hat in posterior_stats['r_hat'].to_dict().items():
            assert r_hat < 1.1, f"R-hat value for {param} is too high: {r_hat}"

        # For unimodal case with known solution, check parameter recovery  
        best_params = np.load(os.path.join(output_dir, 'best_param_vals.npy'))  
        analytical_solution = np.array([1.0, 1.0])
        assert np.allclose(best_params, analytical_solution, rtol=0.1)  

        # check std of the solution
        std = posterior_stats['sd'].to_list()
        analytical_std = [0.1, 0.3]
        print(f"Posterior std: {std}, Analytical std: {analytical_std}")
        assert np.allclose(std, analytical_std, atol=0.2)  

    mpi_comm.Barrier()

@pytest.mark.integration  
@pytest.mark.slow  
@pytest.mark.mpi  
def test_mcmc_unimodal_with_validation_KDE_likelihood(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):  
    """  
    Test MCMC pipeline with unimodal distribution and validate results  
    using autocorrelation and effective sample size.  
    """  
    rank = mpi_comm.Get_rank()  
      
    tests_dir = os.path.dirname(__file__)
    config = base_user_inputs.copy()  
    config.update({  
        'file_prefix': 'Simple_ODE_Benchmark',  # Replace with your model name  
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',  # Replace with your CSV  
        'model_type': 'cellml_only',  
        'solver': 'CVODE',  
        'param_id_method': 'genetic_algorithm',  
        'pre_time': 0.5,  
        'sim_time': 10.0,  
        'dt': 0.1,  
        'DEBUG': False,  
        'do_mcmc': True,  
        'plot_predictions': False,  
        'do_ia': False,  
        "params_for_id_file": "Simple_ODE_Benchmark_params_for_id.csv",  # Specify which params to identify
        'param_id_obs_path': os.path.join(tests_dir, "test_inputs", 'Simple_ODE_Benchmark_KDE_obs_data.json'),  
        'param_id_output_dir': temp_output_dir,
        'mcmc_options': {
            'mcmc_lib': 'pymc',  # 'emcee' or 'zeus' or 'pymc'
            'num_steps': 200,
            'num_walkers': 8,  # 2*num_params is a common choice
            'burn_in_percentage': 0.0,
            'method': 'mcmc'  # or 'smc' if using SMC
        },
        'optimiser_options': {
            "num_calls_to_function": 10000,  
            "cost_convergence": -100,        # Convergence tolerance  
            "max_patience": 1,              # Max generations without improvement  
            "cost_type": 'gaussian_MLE',
            "relative_tolerance": 0.01,
            "use_relative_tolerance": True
        }
    })  
      
    # Ensure model is generated  
    _ensure_cellml_model_generated(config, mpi_comm)  
      
    # Run parameter identification and MCMC  
    run_param_id(config)  
      
    # Validate results on rank 0  
    if rank == 0:  
        output_dir = os.path.join(  
            temp_output_dir,  
            f"{config['param_id_method']}_{config['file_prefix']}_Simple_ODE_Benchmark_KDE_obs_data"  
        )  
          
        # Check MCMC chain exists  
        chain_file = os.path.join(output_dir, 'mcmc_chain.npy')  
        assert os.path.exists(chain_file), "MCMC chain file should exist"  
          
        # Load and validate samples  
        samples = np.load(chain_file)  
        burn_in_idx = int(samples.shape[0] * config['mcmc_options']['burn_in_percentage'])  
        samples = samples[burn_in_idx:, :, :]  
        flat_samples = samples.reshape(-1, samples.shape[-1])  
          
        # Test plotting  
        mcmc_obj = CVS0DParamID(  
            config['model_path'], config['model_type'], config['param_id_method'], True,  
            config['file_prefix'],  
            params_for_id_path=config.get('params_for_id_path'),  
            param_id_obs_path=config['param_id_obs_path'],  
            sim_time=config['sim_time'], pre_time=config['pre_time'], dt=config['dt'],  
            param_id_output_dir=temp_output_dir, resources_dir=resources_dir,  
            mcmc_options=config['mcmc_options'], DEBUG=config['DEBUG'], one_rank=True  
        )  
        mcmc_obj.plot_mcmc()  

        plot_dir = Path(mcmc_obj.plot_dir)  
        matches = list(plot_dir.glob('mcmc_cornerplot_*.pdf'))
        assert len(matches) > 0

        posterior_stats = mcmc_obj.get_posterior_stats(samples)
        
        assert all(val > 50 for val in posterior_stats['ess_bulk']), f"Some parameters have low bulk ESS: {posterior_stats['ess_bulk']}"
        assert all(val > 50 for val in posterior_stats['ess_tail']), f"Some parameters have low tail ESS: {posterior_stats['ess_tail']}"
      
        for param, r_hat in posterior_stats['r_hat'].to_dict().items():
            assert r_hat < 1.2, f"R-hat value for {param} is too high: {r_hat}"

        # For unimodal case with known solution, check parameter recovery  
        best_params = np.load(os.path.join(output_dir, 'best_param_vals.npy'))  
        analytical_solution = np.array([1.0, 1.0])
        assert np.allclose(best_params, analytical_solution, rtol=0.25)  

        # check std of the solution
        std = posterior_stats['sd'].to_list()
        analytical_std = [0.1, 0.3]
        print(f"Posterior std: {std}, Analytical std: {analytical_std}")
        assert np.allclose(std, analytical_std, atol=0.1)  

    mpi_comm.Barrier()

@pytest.mark.integration  
@pytest.mark.slow  
@pytest.mark.mpi  
def test_mcmc_bimodal_with_validation(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):  
    """  
    Test MCMC pipeline with unimodal distribution and validate results  
    using autocorrelation and effective sample size.  
    """  
    rank = mpi_comm.Get_rank()  
      
    tests_dir = os.path.dirname(__file__)
    config = base_user_inputs.copy()  
    config.update({  
        'file_prefix': 'Simple_ODE_Benchmark',  # Replace with your model name  
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',  # Replace with your CSV  
        'model_type': 'cellml_only',  
        'solver': 'CVODE',  
        'param_id_method': 'genetic_algorithm',  
        'pre_time': 0.5,  
        'sim_time': 10.0,  
        'dt': 0.1,  
        'DEBUG': False,  
        'do_mcmc': True,  
        'plot_predictions': True,  
        'do_ia': False,  
        "params_for_id_file": "Simple_ODE_Benchmark_params_for_id.csv",  # Specify which params to identify
        'param_id_obs_path': os.path.join(tests_dir, "test_inputs", 'Simple_ODE_Benchmark_bimodal_obs_data.json'),  
        'param_id_output_dir': temp_output_dir,
        'mcmc_options': {
            'mcmc_lib': 'pymc',  # 'emcee' or 'zeus' or 'pymc'
            'num_steps': 100,
            'num_walkers': 8,  # 2*num_params is a common choice
            'burn_in_percentage': 0.0,
            'method': 'smc'  # or 'smc' if using SMC
        },
        'optimiser_options': {
            "num_calls_to_function": 10000,  
            "cost_convergence": -100,        # Convergence tolerance  
            "max_patience": 1,              # Max generations without improvement  
            "cost_type": 'gaussian_MLE',
            "relative_tolerance": 0.01,
            "use_relative_tolerance": True
        }
    })  
      
    # Ensure model is generated  
    _ensure_cellml_model_generated(config, mpi_comm)  
      
    # Run parameter identification and MCMC  
    run_param_id(config)  

    # Validate results on rank 0  
    if rank == 0:  
        output_dir = os.path.join(  
            temp_output_dir,  
            f"{config['param_id_method']}_{config['file_prefix']}_Simple_ODE_Benchmark_bimodal_obs_data"  
        )  
          
        # Check MCMC chain exists  
        chain_file = os.path.join(output_dir, 'mcmc_chain.npy')  
        # assert os.path.exists(chain_file), "MCMC chain file should exist"  
          
        # Load and validate samples  
        samples = np.load(chain_file)
        burn_in_idx = int(samples.shape[0] * config['mcmc_options']['burn_in_percentage'])
        samples = samples[burn_in_idx:, :, :]
        flat_samples = samples.reshape(-1, samples.shape[-1])  

        # Test plotting  
        mcmc_obj = CVS0DParamID(  
            config['model_path'], config['model_type'], config['param_id_method'], True,  
            config['file_prefix'],  
            params_for_id_path=config.get('params_for_id_path'),  
            param_id_obs_path=config['param_id_obs_path'],  
            sim_time=config['sim_time'], pre_time=config['pre_time'], dt=config['dt'],  
            param_id_output_dir=temp_output_dir, resources_dir=resources_dir,  
            mcmc_options=config['mcmc_options'], DEBUG=config['DEBUG'], one_rank=True  
        )  
        mcmc_obj.plot_mcmc()  

        plot_dir = Path(mcmc_obj.plot_dir)  
        matches = list(plot_dir.glob('mcmc_cornerplot_*.pdf'))
        # assert len(matches) > 0
        
        recovered_p = flat_samples[:, 0]
        recovered_q = flat_samples[:, 1]

        # --- ANALYTICAL GROUND TRUTH ---
        # parameter p: mode 1 = (1, 0.1^2), mode 2 = (4, 0.1^2) | seperation line = 2.5
        # parameter q: mode 1 = (0.33, 0.1^2), mode 2 = (1.33, 0.1^2) | seperation line = 0.8
        
        # 1. QUANTITATIVE MODE WEIGHT CHECK
        # Check if the MCMC spent roughly equal time in both modes
        p_mode1_mask = recovered_p < 2.5
        p_mode2_mask = recovered_p >= 2.5

        q_mode1_mask = recovered_q < 2.5
        q_mode2_mask = recovered_q >= 2.5
        
        weight_mode1 = np.sum(p_mode1_mask) / len(recovered_p)
        weight_mode2 = np.sum(p_mode2_mask) / len(recovered_p)
        
        print(f"Detected Weights: Mode1={weight_mode1:.3f}, Mode2={weight_mode2:.3f}")
        
        # Assert weights are roughly 50/50 (allowing for sampling noise)
        assert np.isclose(weight_mode1, 0.5, atol=0.15), f"Chain is biased! Mode 1 weight: {weight_mode1}"

        # 3. RECOVERY OF MODE 1 (Means and Stds)
        p_mean_m1 = np.mean(recovered_p[p_mode1_mask])
        q_mean_m1 = np.mean(recovered_q[q_mode1_mask])
        p_std_m1  = np.std(recovered_p[p_mode1_mask])
        q_std_m1  = np.std(recovered_q[q_mode1_mask])

        assert np.isclose(p_mean_m1, 1.0, atol=0.2), f"Mode 1 p-mean failed: {p_mean_m1}"
        assert np.isclose(q_mean_m1, 1.0, atol=0.2), f"Mode 1 q-mean failed: {q_mean_m1}"
        # assert np.isclose(p_std_m1, 0.1, atol=0.2), f"Mode 1 p-std failed: {p_std_m1}"
        # assert np.isclose(q_std_m1, 0.3, atol=0.2), f"Mode 1 q-std failed: {q_std_m1}"

        # 4. RECOVERY OF MODE 2 (Means and Stds)
        p_mean_m2 = np.mean(recovered_p[p_mode2_mask])
        q_mean_m2 = np.mean(recovered_q[q_mode2_mask])
        p_std_m2  = np.std(recovered_p[p_mode2_mask])
        q_std_m2  = np.std(recovered_q[q_mode2_mask])

        assert np.isclose(p_mean_m2, 4.0, atol=0.2), f"Mode 2 p-mean failed: {p_mean_m2}"
        assert np.isclose(q_mean_m2, 4.0, atol=0.2), f"Mode 2 q-mean failed: {q_mean_m2}"
        # assert np.isclose(p_std_m2, 0.1, atol=0.2), f"Mode 2 p-std failed: {p_std_m2}"
        # assert np.isclose(q_std_m2, 0.3, atol=0.2), f"Mode 2 q-std failed: {q_std_m2}"

        # 4. ESS Check
        # For multimodal, ESS is often lower, but should still be healthy
        # assert len(recovered_p[p_mode1_mask]) > 50, "Too few samples in Mode 1"
        # assert len(recovered_p[p_mode2_mask]) > 50, "Too few samples in Mode 2"

        # --- DISTRIBUTION VALIDATION FROM CSV ---  
        mcmc_obj.postprocess_predictions()
        csv_path = os.path.join(output_dir, "posterior_predictions.csv")  
        # assert os.path.exists(csv_path), "Posterior predictions CSV should exist"  
          
        pred_df = pd.read_csv(csv_path)  
          
        # For each feature, compare simulated vs experimental distributions  
        for feature in pred_df["feature"].unique():  
            sim_vals = pred_df[(pred_df["feature"] == feature) &   
                              (pred_df["data_type"] == "simulated")]["value"].values  
            exp_vals = pred_df[(pred_df["feature"] == feature) &   
                              (pred_df["data_type"] == "experimental")]["value"].values  
              
            # Use Kolmogorov-Smirnov test to check if distributions are similar  
            from scipy import stats  
            ks_statistic, p_value = stats.ks_2samp(sim_vals, exp_vals)  
              
            print(f"{feature}: KS statistic={ks_statistic:.4f}, p-value={p_value:.4f}")  
              
            # Assert distributions are not significantly different (p > 0.05)  
            # assert p_value > 0.05, f"Distributions for {feature} are significantly different (p={p_value:.4f})"

    mpi_comm.Barrier()
