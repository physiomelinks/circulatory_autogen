# Sensitivity Analysis

Sensitivity Analysis (SA) is a recommended first step in the calibration pipeline because it performs crucial **parameter screening**. Calibration is the process of tuning numerous model parameters (inputs) until the model's output matches experimental data. Without SA, a modeler might waste weeks tuning parameters that have virtually no impact on the final result, or unknowingly tune parameters that are highly correlated, leading to non-unique solutions. SA efficiently identifies:

* **Influential parameters:** Which parameters contribute most significantly to the model's output variance.

* **Non-influential parameters:** Which parameters can be fixed or ignored, drastically simplifying the calibration space.

SA ensures that the calibration effort is focused, efficient, and physically meaningful.

## The Sobol Method

The Sobol method is a powerful, **global, variance-based** sensitivity analysis technique. Unlike local methods that only test parameter changes one at a time, the Sobol method explores the entire input space simultaneously.

* **Key Feature:** It quantifies the contribution of each individual input parameter (First-Order Index, S_i) and the contribution of parameter interactions (Second-Order, Total-Order Index, S_{Ti}) to the overall output variance.

This comprehensive approach allows modelers to understand not only which parameters are important on their own, but also how complex, synergistic interactions between two or more parameters drive the final simulation results.

## Prerequisites

- A generated model and `obs_data.json` file (see [Parameter Identification](parameter-identification.md)).
- A `params_for_id.csv` file defining parameter ranges.
- `pip install "libcuflynx[mpi]"` (and a system MPI toolchain) if running in parallel.

## SA in Circulatory_Autogen
Since Sensitivity Analysis (SA) is intertwined with parameter identification, you will need the same input files as required for parameter identification. This includes both the **`params_for_id.csv`** and the **`obs_data.json`** files. However, the exact values of the data terms in the observation file are not critical for SA itself, as you are simply exploring parameter space and variance, not matching the simulation output to observed data.

Crucially, each data item defined in your `obs_data.json` file is treated as a feature for SA, meaning you will receive a separate set of plots (one for first and total order indices, and one for second order indices) for **each** data item.

### Configuration for `user_inputs.yaml`

To run the Sobol analysis, you need to add a `sa_options` block to your `user_inputs.yaml` configuration file:
```
sa_options: 
    method: 'sobol' 
    num_samples: 1024 
    sample_type: saltelli
    output_dir: <SA_outputs_path>
```
Currently, the available options for `method` are **`'naive'`** and **`'sobol'`**. Available sample types are [**'saltelli'**]. What we call `num_samples` here is actually the `num_samples` in

`actual_num_samples = num_samples (2M+2)`

where M is the number of parameters. This means the `num_samples` that you set doesn't need to be dependent on M.

An indicator that the **sample size may be too low** is the observation of **relatively large negative values for the Sobol indices** in the results; if this occurs, increase the sample size and re-run the analysis.

If `sa_options` is omitted, defaults are applied:

- `method: sobol`
- `num_samples: 32`
- `sample_type: saltelli`
- `output_dir: sensitivity_outputs/<file_prefix>_SA_results`

### Prediction items as extra outputs

Set `include_prediction_items: true` in `sa_options` to also report the sensitivity of the
`prediction_items` in `obs_data.json` that have an `operation` (see
[Prediction items as scalar features](parameter-identification.md#prediction-items-as-scalar-features)).
They are not in the cost, so they are useful for asking which parameters drive a quantity you
predict rather than fit.

```
sa_options:
    method: sobol
    num_samples: 1024
    include_prediction_items: true
```

- **Only scalar prediction items with an operation are included**: `data_type: constant`, or
  no data and a reducing operation such as `max`. Items without an operation, and series, are
  skipped with a warning that names them. An operation that returns more than one number is an
  error.
- They come **after** the data_item outputs. They are labelled by their `data_item_name`, as
  `<data_item_name> (Exp<e>, Sub<s>)`, in the Sobol CSVs
  (`all_outputs_n<N>_Sobol_indices.csv` columns `S1_...`/`ST_...`, and
  `all_outputs_n<N>_Sobol_2nd_order_indices.csv`), the per-output plots and the heatmaps. The
  sub-experiment is the item's `subexperiment_idx` (by default its experiment's last), which is
  the one the operation is applied to. A feature measured in an experiment that has no
  data_items (a validation-only experiment) makes the SA simulate that experiment too. Without
  the option, such an experiment is not simulated.
- The run also writes `sobol_output_features.json`, which lists every output column with its
  `kind` (`data_item`, `prediction_item` or `cost`), `data_item_name`, `experiment_idx` and
  `subexperiment_idx`.
- With `method: local` they are extra rows of `local_sensitivity_absolute.csv` and
  `local_sensitivity_relative.csv`, named by `data_item_name`. `get_local_sensitivities()`
  lists them under `prediction_feature_names`. They are always computed by central finite
  differences with `fd_rel_step`, whatever `gradient_method` gives the data_item rows, which
  costs 2M more simulations.
- `choose_most_impactful_params_sobol` ranks over every column in the CSV, prediction features
  included.
- With `use_emulator: true` the emulator must have been trained with
  `emulator_settings.include_prediction_items: true`. If it was not, the run stops and says to
  retrain.

Without the option, nothing changes: the outputs and files are exactly the data_item ones.

## How to run SA script

First, ensure you have the required sensitivity analysis packages specified in [Getting Started](getting-started.md).

To run the script, use the following command (which utilizes **MPI for parallelized computation** on CPU):

```
./run_sensitivity_analysis.sh <NUM_CORES>
```

After successful execution, you will find the SA plots—including the first, second, and total order indices—in the directory specified by `output_dir`.

## Deriving an observable from other observables

Sensitivity analysis uses the same `data_items` as calibration, so a feature built from other
features -- the difference between two observables, say -- is available here too, and is defined
the same way. See
[Building an observable from other observables](parameter-identification.md#building-an-observable-from-other-observables).

## Expected outcome

You should have Sobol index plots saved to `sa_options.output_dir`.

## Troubleshooting

- If you see errors about `params_for_id_path`, confirm your `params_for_id.csv` filename and `resources_dir`.
- If MPI errors occur, ensure `mpiexec` is available and `mpi4py` is installed in the environment you are running from — it is the `[mpi]` extra, not part of a plain `pip install libcuflynx`.
