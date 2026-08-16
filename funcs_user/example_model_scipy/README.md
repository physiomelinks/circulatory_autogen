# Example `external_python` model — damped oscillator with scipy

The smallest useful example of plugging your own Python model into circulatory_autogen via
`model_type: external_python`. The model is a damped linear oscillator `x'' + c·x' + k·x = 0`,
integrated by `scipy.integrate.solve_ivp` **inside the model class's own `run()`**.

Calibrated parameters: `c` (damping) and `k` (stiffness). Observables: the displacement `x`, the
velocity `v`, and the algebraic total energy `½(v² + k·x²)`.

Start here if your model is an ODE you can write down. Start with
[`../example_model_external/`](../example_model_external/) if you want the same contract on a
model with a hand-written scheme (a 1D heat equation, NumPy only), or with
[`../heat_fenics/`](../heat_fenics/) for the heavyweight case (a FEniCSx finite-element solver).

## Migrating from `python_user_defined`

`model_type: python_user_defined` (solver `user_defined`) has been **removed**. It asked for a
module of `PARAMETERS` / `STATES` / `OUTPUT_NAMES` dicts plus an `rhs(t, y, params)` and called
`solve_ivp` for you. `external_python` covers the same case — you make the `solve_ivp` call
yourself, which is the `run()` below — and having two "bring your own Python" model types whose
names did not distinguish them cost more than the shortcut saved.

`oscillator_model.py` **is** the retired `funcs_user/example_model/oscillator_wrapper.py`,
rewritten under this contract, so the diff is the whole migration:

| `python_user_defined` | `external_python` |
|---|---|
| module-level `PARAMETERS` dict | literal class attribute `parameters` |
| module-level `STATES` dict | the initial condition your `run()` starts from |
| module-level `OUTPUT_NAMES` list | literal class attribute `output_names` |
| `rhs(t, y, params)` | a method, handed to `solve_ivp` by your `run()` |
| `compute_outputs(t, y, params)` | just another entry in the dict `get_results()` returns |
| `solver_info: {method, rtol, atol}` | `solver_info: {user_config: {...}}`, read in `init_solver` |
| `model_wrapper_path` in `user_inputs.yaml` | `external_model_path` |
| CA owned the sample grid | you produce it: `N = int(pre_time/dt) + int(sim_time/dt)` |

That last row is the only real work. CA tells your class the grid through `update_times` and
expects `get_results()` to return arrays of exactly `N + 1` samples at `start_time + i*dt`,
**including the `pre_time` ones** — CA slices those off itself. Compute `N` with that exact
integer arithmetic rather than `np.arange` on floats, so your length and CA's agree exactly
rather than approximately; a short array is a hard error, not a padded one.

A config that still names the removed type does not fail vaguely: the parser refuses
`model_type: python_user_defined`, `solver: user_defined` and a leftover `model_wrapper_path`
key with these instructions.

## Files

| File | Purpose |
|---|---|
| `oscillator_model.py` | The model class plus `SIM_HELPER = Oscillator`. |
| `oscillator_params_for_id.csv` | Parameters to calibrate / sweep (`c`, `k`) with bounds. |
| `oscillator_parameters.csv` | Default parameter values (the calibration start point). |
| `oscillator_obs_data.json` | Target observables — `mean(x)`, `min(x)`, `range(v)` at the "true" `c=0.7, k=5.0`. |

`oscillator/energy` is exposed as an output but not scored by any observable; it is there to show
that an algebraic quantity needs no special hook under this contract — it is one more key in the
dict `get_results()` returns.

## The contract

Your file defines a class and registers it. Nothing is discovered by naming convention:

```python
class MyModel:
    # Self-description. LITERAL values: tools read these two attributes out of the file by
    # AST, without importing it, to build a parameter table before any solver exists.
    parameters = {"oscillator/c": 0.5, "oscillator/k": 4.0}   # qualified name -> default
    output_names = ["oscillator/x", "oscillator/v"]           # qualified names

    # --- required ---
    def init_solver(self, config): ...
        # Called once. Put the expensive setup here (mesh, factorisation, device). For an ODE
        # this small there is none — just read your options out of
        # config['solver_info']['user_config'].

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
        # on the update_times grid, pre_time samples included.

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
file_prefix: oscillator
input_param_file: oscillator_parameters.csv
model_type: external_python
solver: external
# Default model location is funcs_user/{file_prefix}_model.py. This example lives in a
# subdirectory, so point at it explicitly:
external_model_path: <repo>/funcs_user/example_model_scipy/oscillator_model.py
# Point resources_dir at this directory so the CSV/JSON above are found:
resources_dir: <repo>/funcs_user/example_model_scipy
solver_info:
  solver: external
  # Free-form; handed to init_solver as config['solver_info']['user_config']. This model uses it
  # for the solve_ivp settings CA used to own when it did the integrating.
  user_config: {method: RK45, rtol: 1.0e-8, atol: 1.0e-8}
pre_time: 0.0
sim_time: 10.0
dt: 0.05
param_id_method: genetic_algorithm
```

There is **no code generation step**: `run_autogeneration.sh` only checks that the file exists.

## Running

```bash
# The model on its own, no CA involved — prints the sample count and the observable values,
# and is how oscillator_obs_data.json was generated.
venv/bin/python funcs_user/example_model_scipy/oscillator_model.py

# Calibration (2 MPI ranks)
./run_param_id.sh 2
# Sensitivity analysis
./run_sensitivity_analysis.sh 2
```

Calibration should recover `c ≈ 0.7`, `k ≈ 5.0` (the values used to build
`oscillator_obs_data.json`), starting from the `c=0.5, k=4.0` defaults.

See `tests/test_scipy_ode_example.py` for an automated end-to-end check.
