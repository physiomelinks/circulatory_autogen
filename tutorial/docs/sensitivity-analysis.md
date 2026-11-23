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
- A `params_for_id.json` file defining parameter ranges.
- `pip install "libcuflynx[mpi]"` (and a system MPI toolchain) if running in parallel.

## SA in Circulatory_Autogen
Since Sensitivity Analysis (SA) is intertwined with parameter identification, you will need the same input files as required for parameter identification. This includes both the **`params_for_id.json`** and the **`obs_data.json`** files. However, the exact values of the data terms in the observation file are not critical for SA itself, as you are simply exploring parameter space and variance, not matching the simulation output to observed data.

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
- `output_dir: <param_id_output_dir>/sensitivity`, in the study's working dir (see
  [The study working dir](parameter-identification.md#the-study-working-dir)). A relative
  `output_dir` is taken under that folder and an absolute one is used as given. With
  `cuflynx_study_copy_inputs: false` (or `write_cuflynx_study: false`) the default is
  `sensitivity_outputs/<file_prefix>_SA_results`, and a relative `output_dir` is taken under
  `sensitivity_outputs/`.

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
  lists them under `prediction_feature_names`. They are computed the same way as the
  data_item rows: by the analytic arm (CasADi, or Myokit CVODES sensitivities), or by central
  finite differences with `fd_rel_step` (2M more simulations) when `gradient_method: FD` or over
  an emulator. Myokit's arm needs every prediction feature in one and the same
  sub-experiment; otherwise (features in two experiments, or in an experiment with several
  sub-experiments) they fall back to finite differences, with a warning.
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

## Parameter reduction

A Sobol analysis can choose the parameters to calibrate, for two reasons:

1. **Redundancy.** Parameters that move the data the same way cannot be told apart. Keep one
   parameter per direction the data can fix. On the 3compartment study
   (`benchmarks/param_reduction_3compartment`) calibrating all seven parameters is not
   identifiable (the Fisher information is singular or has condition number about 1e16), while
   the reduced fits have condition numbers of 12-16 and agree between seeds.
2. **Noise-blindness.** A large Sobol index on an output that moves by less than its noise is
   not worth fitting. The reduction measures every effect in units of the obs_data noise.

With `do_param_reduction: true`,
`run_sensitivity_analysis.sh` follows the Sobol run with a reduction and writes the kept entries
of your params_for_id file as `reduced_params_for_id.json`, ready to calibrate:

```yaml
do_param_reduction: true
param_reduction_options:
  method: pca            # or sobol
  # n_params: 3          # pca: at most this many; sobol: needed
  # index_type: ST       # ST or S1
  # min_mode_std: 1.0    # pca; measured over the whole prior box (global, not local to the fit)
  # variance_fraction: 1.0  # pca; 1 = no relative cut
  # min_index: 0.05      # pca
  # redundancy_check: true  # pca
  # criterion: max       # sobol
  # threshold: 0.01      # sobol
  # sensitivity_dir: ... # default sa_options.output_dir
  # output_dir: ...      # default <sensitivity_dir>/param_reduction
```

**`method: pca`** (the default) asks in how many independent ways the parameters move the data
by more than its noise, and keeps one parameter for each:

1. Each output over the Sobol samples is centred and divided by its obs_data `std` over the
   square root of its `weight`, the scale the `gaussian_MLE` cost measures it on. A change of 1
   is then one noise std.
2. Principal component analysis of these scaled outputs gives modes: independent directions
   in which the parameters move the data, largest variance first. A mode whose std is at least
   `min_mode_std` (default 1, the noise) is **informative**; the others move the data by less
   than the noise and cannot be fitted. Only the leading modes that together explain
   `variance_fraction` of the variance are considered. The default, 1, is no relative cut. A
   smaller value ignores the noise: a mode that moves the data by several noise stds but has a
   small share of the variance can be dropped.
3. Sobol indices are computed for each mode's score, from the same samples. A parameter whose
   largest index over the informative modes is below `min_index` is not a candidate.
4. With `redundancy_check: true` (the default), the reduction asks which candidates move the
   data in different ways, using local derivatives (see "How the redundancy check works" below).
   Of the candidates whose effect is at least `min_mode_std` noise stds, it keeps the one with
   the highest Sobol index over the informative modes, removes that direction from the others'
   effects, and repeats, stopping when no effect left reaches `min_mode_std`: what such a
   parameter would add to the fit is inside the noise. Parameters that move the
   data the same way (only their product or ratio matters, or a scaling symmetry) have the same
   effect, so only one of them is kept, however many modes they drive: the one the Sobol
   indices rank highest. The local derivatives decide which parameters are redundant; the
   Sobol indices, which cover the whole prior box, decide which of them is kept.
5. With `redundancy_check: false` it goes through the informative modes, largest first, and
   keeps the parameter, not kept already, with the largest index for that mode, skipping a mode
   that no free parameter reaches `min_index` for. This keeps at most one parameter per mode, so
   it reduces only when there are fewer informative modes than parameters. Outputs that are a
   time course (series binned into features, an oscillation) have many more modes than
   parameters, and it keeps them all.

Both stop after `n_params`. A parameter that drives an output strongly, but by less than that
output's noise, is not kept.

#### How the redundancy check works

Sobol indices say how much of a mode's variance each parameter drives, but not in which
direction, so they cannot tell that two parameters are interchangeable: when only `a * b`
matters, `a` and `b` get the same large index on the same modes. The check needs, for each
parameter, the direction it moves the data in, and compares those.

**The derivatives.** Every Sobol run also takes local derivatives, set by `sa_options`:

| key | default | what it sets |
|---|---|---|
| `jacobian_points` | 8 | how many points; 0 skips the derivatives |
| `jacobian_location` | `base_points` | `base_points`: the first `jacobian_points` Saltelli base points, the rows starting each block of 2M + 2 samples, which come from a Sobol sequence and so spread over the box; `centre`: one point at the centre of the box |
| `jacobian_scheme` | `central` | `central` (2 runs per parameter and point) or `forward` (1, reusing the base point's own run) |
| `jacobian_step` | 0.01 | the step in ln(parameter), or as a fraction of the range for bounds that straddle 0 |
| `jacobian_check_steps` | false | also take them with half the step (twice the runs), for `redundancy_converged_only` |

With the defaults a run costs `2 x jacobian_points x M` extra simulations, about
`jacobian_points / num_samples` of the Sobol run: 192 next to 6656 (2.9%) for 12 parameters and
`num_samples: 256`; `jacobian_check_steps: true` doubles that. Central differences are the default because a forward step's error is first
order: in an oscillator a step shifts the phase, the error grows along the time course, and in
the Goodwin benchmark it hid an exact symmetry. Each derivative is multiplied by the parameter's
prior range (`ln(max/min)`, or `max - min` for bounds that straddle 0), so it reads as how far the
data would move if that slope held across the whole range. They are saved in
`all_outputs_n<N>_Sobol_local_jacobian.csv`, one row per step, point and parameter.

**The effects.** At each point, each derivative is scaled to noise stds like the outputs in
step 1 and projected onto the informative modes: a parameter's effect at that point, a vector
with one entry per informative mode.

**The choice.** Pivoted Gram-Schmidt, done at each point separately and chosen jointly. For
every parameter, take what is left of its effect at each point once the kept parameters'
effects at that point are removed, and its root mean square length over the points. Keep the
parameter with the largest, remove its direction at every point, and repeat until the largest
is below `min_mode_std`. Because each point is explained by the kept parameters at that same
point, the combination that reproduces a redundant parameter may change across the box, as it
does for a scaling symmetry or for parameters entering as a sum. The report lists each
parameter's `effects` (the statistic before anything is removed) and the
`unexplained_effects` of those not kept (what was left).

Two `param_reduction_options` set how the points are combined:

- `redundancy_statistic`: `median` (the default) keeps a parameter only if it is unexplained at
  most points, which ignores a minority of points where the derivatives are poor, and also a
  parameter that matters in less than half the box; `rms` keeps a parameter that is unexplained
  at any point, however few.
- `redundancy_converged_only` (default false): leave out the points where the derivatives at
  `jacobian_step` and at half of it differ by more than `redundancy_convergence_tol` (default
  0.1) of the effect. It needs `jacobian_check_steps`, and the report lists the
  `unconverged_points`.

The defaults (central differences of 0.01 at 8 base points, the `median` over the points)
come from `benchmarks/param_reduction/compare_redundancy.py`, which runs every setting on the
FitzHugh-Nagumo, Goodwin and Teusink benchmarks and compares the kept count with the Fisher
information at the truth (`redundancy_comparison_results.md`). FitzHugh-Nagumo and Teusink give
the same count under every setting. Goodwin is decided by the points below: `rms` over all
points keeps one parameter too many. `rms` over the converged points and the centre point alone
also give the right count there, but on a stiff model such as the 3-compartment test model
solver noise (which grows as 1/step) leaves no point converged at a step of 0.001, and the centre
point misses a parameter that is flat at the centre. The cost of `median` is a parameter that
matters in less than half the box: set `redundancy_statistic: rms` to keep such parameters.

**Why local derivatives, at several points.**

- *Products and ratios stay exact.* If only `a * b` matters, a 1% step in `a` and a 1% step in
  `b` change the data identically at every point, so once `a` is kept nothing of `b` is left.
  Differences between Saltelli samples cannot do this: each pair of samples jumps by a
  different random amount, and on a curved response even jumps of the same size in ln(a) and
  ln(b) leave a remainder (about 20% of the effect, many noise stds, in a test of `a * b`).
- *A flat spot at one point is not mistaken for no effect.* `p^2` on `[-1, 1]` has slope 0 at
  the centre of its range, and a derivative there alone would drop it. Across 8 points its
  effect is many noise stds, and it is kept.

- *Points where the outputs are not smooth.* In the Goodwin benchmark `ai`, `Ai` and `ki` form
  an exact scaling symmetry, so `ai` is explained by the other two at every point in exact
  arithmetic. At six of eight points the derivatives show it (under 1 noise std left), but at
  the two where `ai` is small next to `Ai` they do not settle as the step shrinks, and leave
  tens of noise stds. The default `median` outvotes those points and drops `ai`;
  `rms` over every point would keep it.

**Limits.** The derivatives are local: a parameter whose effect lies wholly between the points,
such as a sharp switch, can still be missed, and more `jacobian_points` make that less likely.
A point where a run fails is left out, with a warning. `jacobian_points: 0` skips the
derivatives, and then a reduction needs `redundancy_check: false`.

**`method: sobol`** keeps the `n_params` parameters with the highest Sobol index over the
outputs (the `max` or `mean` by `criterion`), dropping any below `threshold`. It ignores the
noise and counts correlated outputs separately.

Both write `param_reduction.json`: the options and the kept parameters with their scores. For
`pca` each kept parameter also has its mode, and the report lists every mode's std and variance
share. `pca_modes.csv` holds each mode's loadings on the outputs, `pca_mode_sobol_indices.csv`
the indices per mode (in the layout of the Sobol indices CSV), and `pca_mode_sobol_ST.png` a
heatmap of them; with the redundancy check the report also has each parameter's `effects` and
the `unexplained_effects` of those not kept. A `pca` reduction reads
`all_outputs_n<N>_Sobol_samples.csv`, `..._outputs.csv`, `..._local_jacobian.csv` and
`sobol_problem.json`, which every Sobol run writes.

A sample whose simulation fails cannot be left out (Saltelli's estimators need every row), so
each output it lacks is filled with that output's mean over the samples that ran. The count is
`failed_samples` in `sobol_problem.json`, with a warning: it shrinks the indices toward zero by
about the failed fraction, so narrow the bounds if it is large. In a calibration
workflow, a `reduction` step does the same (see
[calibration workflows](parameter-identification.md)).

## Deriving an observable from other observables

Sensitivity analysis uses the same `data_items` as calibration, so a feature built from other
features -- the difference between two observables, say -- is available here too, and is defined
the same way. See
[Building an observable from other observables](parameter-identification.md#building-an-observable-from-other-observables).

## Expected outcome

You should have Sobol index plots saved to `sa_options.output_dir`.

## Troubleshooting

- If you see errors about `params_for_id_path`, confirm your `params_for_id.json` filename and `resources_dir`.
- If MPI errors occur, ensure `mpiexec` is available and `mpi4py` is installed in the environment you are running from — it is the `[mpi]` extra, not part of a plain `pip install libcuflynx`.

## Following a run: progress files

A Sobol analysis, an emulator training and a profile likelihood each write `progress.json` into
the directory their results go to (the `sa_options` `output_dir`, the emulator's directory,
`profile_likelihood/`; in a calibration workflow, the step's own folders). MPI rank 0 rewrites
it while the run goes, at most about once a second, through a rename, so a reader never sees
half a file. CUFLynx reads the same file; nothing in it depends on how the run was started.

```json
{"schema_version": 1, "kind": "sobol", "state": "running",
 "started": "2026-11-23T02:10:00+00:00", "updated": "2026-11-23T02:10:41+00:00",
 "phases": [{"name": "sampling", "state": "running", "done": 1200, "total": 4608, "failed": 3}]}
```

- `kind` is `sobol`, `emulation` or `profile_likelihood`.
- `state` is `running`, then `done` once the result files are written, or `failed` if the run
  raised an error. A run that was killed stays `running`: compare `updated` with the clock.
- Each phase counts `done` of `total` items, with those that `failed` among the done, and has
  its own `state` (`queued`, `running`, `done`, `failed`). A phase is added when its size is
  known, so later phases can be missing while an earlier one runs.

| Kind | Phases |
|---|---|
| `sobol` | `sampling` (the model runs of the design), `local_jacobian` (the runs for a PCA reduction's redundancy check, when there are any), `analysis` |
| `emulation` | one `sampling` phase per design stage, with `stage` and `method` (the training simulations), then `training`, whose `models` list each emulator compared: `name`, `state`, `score` (its held-out value of the phase's `metric`) and `duration_s` |
| `profile_likelihood` | one phase per calibrated entry, named by its label, with `param_index`: the points evaluated of `num_points`; a failed point is one whose re-calibration gave no finite cost |

The file's definition is `libcuflynx/schemas/progress.schema.json`
(`libcuflynx.schemas.PROGRESS_SCHEMA`); the writer is `libcuflynx.utilities.progress`. A phase
may gain keys: ignore those you do not know.

A Sobol run also writes, in the same directory:

- `sobol_design.npy`: the design (samples by parameters) in run order, before the first
  simulation, and `sobol_design_params.json` with its columns (`names`, `bounds`,
  `num_samples`, `sample_type`, `rows`).
- `sobol_sample_status.npy`: one int8 per design row, 0 not run, 1 ran, 2 failed (an output
  that is not finite), rewritten as the ranks report and kept after the run. Failed samples are
  filled with each output's mean before the indices are computed, so the outputs CSV does not
  show which they were; this file does.
- `sobol_convergence.json`, at the end: S1 and ST with their confidence half-widths at
  `num_samples` / 8, / 4, / 2 and `num_samples` base samples (from 4 up), each the analysis the
  run would have given had it stopped there:
  `{"n": [32, 64, 128, 256], "outputs": {output: {"S1": {parameter: [[value, conf], ...]}, "ST": {...}}}}`,
  one `[value, conf]` per entry of `n`; a value that is not finite is `null`.
