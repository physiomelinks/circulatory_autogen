# Sensitivity analysis

`SensitivityAnalysis` runs variance-based (Sobol) global sensitivity analysis
over the parameters selected for identification.

::: libcuflynx.sensitivity_analysis.sensitivityAnalysis.SensitivityAnalysis
    options:
      members:
        - init_from_dict
        - set_ground_truth_data
        - set_params_for_id
        - set_sa_options
        - set_model_out_names
        - add_user_operation_func
        - run_sensitivity_analysis
        - run_sobol_sensitivity
        - choose_most_impactful_params_sobol
