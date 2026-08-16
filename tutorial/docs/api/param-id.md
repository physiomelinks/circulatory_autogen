# Parameter identification

`CVS0DParamID` orchestrates calibration of a 0D CVS model against observation
data, across the genetic-algorithm, CMA-ES, Bayesian, and `sp_minimize`
optimisers, plus MCMC sampling.

::: libcuflynx.param_id.paramID.CVS0DParamID
    options:
      members:
        - init_from_dict
        - init_from_all_dicts
        - set_ground_truth_data
        - set_params_for_id
        - set_param_id_method
        - set_optimiser_options
        - set_bayesian_parameters
        - add_user_operation_func
        - add_user_cost_func
        - update_param_range
        - remove_params_by_name
        - remove_params_by_idx
        - run
        - run_mcmc
        - simulate_with_best_param_vals
        - plot_outputs
        - plot_mcmc
        - get_mcmc_samples
        - get_best_param_vals
        - set_best_param_vals
        - get_param_names
        - set_param_names
        - get_param_importance
        - get_collinearity_idx
        - get_collinearity_idx_pairs
        - set_output_dir
        - close_simulation
