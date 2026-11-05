# Identifiability analysis

`IdentifiabilityAnalysis` quantifies how well parameters can be identified
around a best fit, via the Laplace approximation (profile likelihood is
planned). It wraps an existing `CVS0DParamID` object.

::: libcuflynx.identifiabilty_analysis.identifiabilityAnalysis.IdentifiabilityAnalysis
    options:
      members:
        - init_from_dict
        - set_best_param_vals
        - run
        - run_laplace_approximation
        - run_profile_likelihood
        - plot_laplace_results
