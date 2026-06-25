# Example `python_user_defined` model — damped oscillator

A minimal, self-contained example of plugging an arbitrary Python ODE model into
circulatory_autogen via `model_type: python_user_defined`. The model is a damped
linear oscillator `x'' + c·x' + k·x = 0`, and the parameters `c` (damping) and
`k` (stiffness) are calibrated / analysed by the standard pipelines.

## Files

| File | Purpose |
|---|---|
| `oscillator_wrapper.py` | The model wrapper (`PARAMETERS`, `STATES`, `OUTPUT_NAMES`, `rhs`). |
| `oscillator_params_for_id.csv` | Parameters to calibrate / sweep (`c`, `k`) with bounds. |
| `oscillator_parameters.csv` | Default parameter values (the calibration start point). |
| `oscillator_obs_data.json` | Target observables (`mean(x)`, `min(x)`, `range(v)`) — computed from the "true" `c=0.7, k=5.0`. |

The wrapper template to copy for your own model is
`funcs_user/model_wrapper_funcs_user.py`.

## How it works

There is **no code generation**. The framework imports `oscillator_wrapper.py`,
builds a model from `rhs` + `STATES` + `PARAMETERS`, and integrates it with the
same scipy `solve_ivp` machinery used by the generated-python backend
(`src/solver_wrappers/python_solver_helper.py`). Calibration, sensitivity
analysis and identifiability analysis then run unchanged.

Only `PARAMETERS` are swept; initial conditions in `STATES` are fixed.
Parameter / output names use the canonical `component/variable` form so they match
the `params_for_id` CSV (`vessel_name`/`param_name`) and the obs_data `operands`.

## `user_inputs.yaml` settings

```yaml
file_prefix: oscillator
input_param_file: oscillator_parameters.csv
model_type: python_user_defined
solver: user_defined
# Default wrapper location is funcs_user/{file_prefix}_wrapper.py. This example
# lives in a subdirectory, so point at it explicitly:
model_wrapper_path: <repo>/funcs_user/example_model/oscillator_wrapper.py
# Point resources_dir at this directory so the CSV/JSON above are found:
resources_dir: <repo>/funcs_user/example_model
pre_time: 0.0
sim_time: 10.0
dt: 0.05
param_id_method: genetic_algorithm
```

## Running

```bash
# Calibration (2 MPI ranks)
./run_param_id.sh 2
# Sensitivity analysis
./run_sensitivity_analysis.sh 2
# Identifiability analysis
./run_identifiability_analysis.sh
```

Calibration should recover `c ≈ 0.7`, `k ≈ 5.0` (the values used to generate
`oscillator_obs_data.json`), starting from the `c=0.5, k=4.0` defaults.

See `tests/test_python_user_defined.py` for an automated end-to-end check of all
three workflows against this example.
