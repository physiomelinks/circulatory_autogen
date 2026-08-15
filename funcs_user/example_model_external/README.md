# Example `external_python` model — 1D heat equation

A self-contained example of plugging a model that **already has its own solver** into
circulatory_autogen via `model_type: external_python`. The model is the 1D heat equation
`dT/dt = k·d²T/dx²` on a unit rod with Dirichlet ends, marched by an explicit finite-difference
scheme that the user's file owns end to end. CA never sees a right-hand side and never chooses a
step; it asks for a sample grid and gets one back.

Calibrated parameters: `k` (diffusivity) and `u_D` (the boundary value). Observables: three fixed
interior probes.

## `external_python` vs `python_user_defined`

| | `python_user_defined` | `external_python` |
|---|---|---|
| You supply | an `rhs(t, y, params)` function | a solver **class** |
| Who integrates | CA, with scipy `solve_ivp` | your code |
| Solver choice | `solver_info.method` (RK45, BDF, …) | yours; CA has none to offer |
| Good for | an ODE you can write down | an FE/FV code, a compiled library, a scheme that is the point |

Use `python_user_defined` (see `funcs_user/example_model/`) when your model really is just an
ODE — you get CA's integrators for free. Use `external_python` when it is not.

## Files

| File | Purpose |
|---|---|
| `heat1d_model.py` | The solver class plus `SIM_HELPER = Heat1D`. |
| `heat1d_params_for_id.csv` | Parameters to calibrate / sweep (`k`, `u_D`) with bounds. |
| `heat1d_parameters.csv` | Default parameter values (the calibration start point). |
| `heat1d_obs_data.json` | Target observables — the three probe means at the "true" `k=0.25, u_D=0.1`. |

## The contract

Your file defines a class and registers it. Nothing is discovered by naming convention:

```python
class MyModel:
    # Self-description. LITERAL values: tools read these two attributes out of the file by
    # AST, without importing it, to build a parameter table before any solver exists.
    parameters = {"heat/k": 0.4, "heat/u_D": 0.0}     # qualified name -> default
    output_names = ["heat/T_p1", "heat/T_p2"]         # qualified names

    # --- required ---
    def init_solver(self, config): ...
        # Called once. Put the expensive setup here (mesh, factorisation, device).
        # config keys: 'dt', 'sim_time', 'pre_time', 'start_time', 'solver_info'
        # (the CA solver_info dict; solver_info['user_config'] is your free-form options).

    def update_times(self, dt, start_time, sim_time, pre_time): ...
        # The sample grid to produce: start_time + i*dt for i in 0..N, where
        # N = int(pre_time/dt) + int(sim_time/dt). Cheap — no re-setup.

    def set_param_vals(self, param_dict): ...
        # A SUBSET of `parameters` with new values. Must not need a re-init.

    def run(self): ...
        # Solve the whole grid, pre_time spin-up included, from the initial condition.
        # Repeatable: calibration calls it thousands of times. Return False if it diverged
        # (CA turns that into an infinite cost, rather than crashing the run).

    def get_results(self): ...
        # {output_name: 1D numpy array of length N+1} for EVERY name in output_names,
        # on the update_times grid, pre_time samples included. CA slices those off itself.

    # --- optional ---
    def get_init_param_vals(self, names): ...  # else CA tracks the values for you
    def reset(self): ...                       # extra clearing between runs; default no-op
    def extra_plots(self): ...                 # -> [matplotlib Figure, ...] views CA cannot draw
    def close(self): ...                       # release resources

SIM_HELPER = MyModel      # required: the class object, at module level
```

Names are `component/variable`, matching the `params_for_id` CSV columns
(`vessel_name`/`param_name`) and the obs_data `operands`.

**Time is CA's, not yours.** `pre_time` is spin-up that `run()` must perform and CA discards; the
logged trace starts at the end of it. `run()` always starts from the initial condition — there is
no state carried between runs, which is why `offline_pre_time` is not available for this model
type.

## `user_inputs.yaml` settings

```yaml
file_prefix: heat1d
input_param_file: heat1d_parameters.csv
model_type: external_python
solver: external
# Default model location is funcs_user/{file_prefix}_model.py. This example lives in a
# subdirectory, so point at it explicitly:
external_model_path: <repo>/funcs_user/example_model_external/heat1d_model.py
# Point resources_dir at this directory so the CSV/JSON above are found:
resources_dir: <repo>/funcs_user/example_model_external
solver_info:
  solver: external
  # Free-form; handed to init_solver as config['solver_info']['user_config']. This model uses
  # it for the explicit scheme's stability margin.
  user_config: {stability_target: 0.4}
pre_time: 0.0
sim_time: 0.5
dt: 0.005
param_id_method: genetic_algorithm
```

There is **no code generation step**: `run_autogeneration.sh` only checks that the file exists.

## Running

```bash
# Calibration (2 MPI ranks)
./run_param_id.sh 2
# Sensitivity analysis
./run_sensitivity_analysis.sh 2
```

Calibration should recover `k ≈ 0.25`, `u_D ≈ 0.1` (the values used to build
`heat1d_obs_data.json`), starting from the `k=0.4, u_D=0.0` defaults.

See `tests/test_external_simulation_helper.py` for an automated end-to-end check.
