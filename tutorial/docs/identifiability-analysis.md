# Identifiability Analysis

Identifiability Analysis (IA) ensures that identified parameters can be trusted, i.e. have a small uncertainty.

## Prerequisites

- A completed parameter identification run (best-fit parameters computed).
- `param_id_output` directory available for the model and dataset.

## Available Methods

### The Laplace Approximation

The Laplace approximation makes an approximation of your parameter posterior distribution, assuming it is gaussian. This uses the Hessian of the log-likelihood with respect to the parameters.

### Profile Likelihood Analysis

Profile likelihood analysis assesses parameter identifiability by holding each parameter at a sweep of values around its best fit and re-calibrating all the other parameters at each one. For each parameter it:

1. holds the parameter at `num_points` values, within `range_factor` of its `[min, max]` range either side of the best fit (clipped to `[min, max]`);
2. re-calibrates the other parameters at each held value with your `param_id_method`, starting from the optimum at the neighbouring held value (the sweep runs outwards from the best fit; the genetic algorithm and bayesian methods draw their own starting points instead);
3. records the cost at each re-calibrated optimum, and plots cost against the held value.

A grouped params_for_id entry (several `targets` sharing one value) is held as a whole, and a modifier entry is held at its theta and applied through its modifier function, as in calibration.

The resulting profiles reveal:

- **Identifiable parameters**: clear minima with steep cost increases.
- **Non-identifiable parameters**: flat profiles, where other parameter combinations compensate.

**The 95% interval.** Profile likelihood runs on `@is_MLE` costs (such as `gaussian_MLE`), so the cost is a scaled negative log-likelihood: `-ln L = cost * N`, where `N` is the number of weighted observable values (the same relation MCMC uses). The 95% profile-likelihood interval for one parameter is where `-ln L` stays within `chi2(0.95, 1 dof) / 2 = 1.92` of its minimum, so in cost units the threshold is `min cost + 1.92 / N`. It is drawn on the plots and written to each parameter's JSON as `ci_95` (`lower`/`upper` are `null` where the sweep never rises above the threshold, i.e. the interval is open on that side within the sweep).

**Costs.** As for Laplace, an observable whose cost is not `@is_MLE` but has an MLE form is switched to that form for the profile (the best fit it is measured from and every re-fit), with a warning naming the observable: `MSE` becomes `gaussian_MLE`, which has the same minimum. Your calibration's own costs are put back when the profile finishes. A cost with no MLE form (such as `AE`, or a custom cost without `@is_MLE`) is an error naming the observable.

Each profile point is a full calibration, so the run costs about `num_points x num_parameters` calibrations; use `optimiser_options` to give the profile points a smaller budget. With `use_emulator` the re-fits run on the emulator.

## IA in Circulatory_Autogen

IA is run following a parameter identification run.

### Configuration for `user_inputs.yaml`

To run IA, you need to set
```
do_ia: True
```

and add a specific `ia_options` block to your `user_inputs.yaml` configuration file:
#### For the Laplace approximation
```
ia_options:
    method: 'Laplace'
```

#### For profile likelihood
```
ia_options:
  method: 'profile_likelihood'
  num_points: 50              # held values per parameter (>= 2)
  range_factor: 0.2           # sweep half-width, as a fraction of the parameter's [min, max] range
  optimiser_options:          # optional: merged over optimiser_options for each re-calibration
    num_calls_to_function: 200
```

Results go to `<param_id_output_dir>/profile_likelihood/`: one JSON and one PDF per parameter, `profile_likelihood_plots.pdf` with every profile, and `summary.json`. The re-fits' optimisers write no files of their own (no `best_cost.npy` or history CSVs).

## Running identifiability analysis

You can run IA as part of parameter identification by setting `do_ia: True` and running:

```
./run_param_id.sh <NUM_CORES>
```

Or run it separately after parameter identification completes:

```
./run_identifiability_analysis.sh
```

## Expected outcome

Laplace approximation results are saved in your `param_id_output` directory alongside parameter identification outputs; profile likelihood results in its `profile_likelihood/` subdirectory.

## Troubleshooting

- If IA fails with missing files, confirm that parameter identification finished successfully and produced `best_param_vals.npy` and related outputs.

