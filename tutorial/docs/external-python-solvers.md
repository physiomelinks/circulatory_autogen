# External Python Solvers

Circulatory Autogen can calibrate, sweep and emulate a model it did not generate and does not
integrate. You write a Python class that owns its own time-stepping — a finite-element solver, a
compiled library behind a `ctypes` wrapper, a hand-rolled scheme, anything — drop it in a file,
and CA drives it through the same calibration, sensitivity-analysis and emulator pipelines it
uses for CellML models.

This is `model_type: external_python` with `solver: external`.

## Which backend do I want?

CA has three ways of getting a model to simulate, and they differ in *who runs the time loop*.

| | You provide | CA does the time-stepping | Use when |
|---|---|---|---|
| `cellml_only` | CellML modules and a vessel array | yes (OpenCOR / Myokit CVODE) | the model is a network of reusable CellML components |
| `python_user_defined` | a right-hand side `rhs(t, y, p)` | yes (`scipy.solve_ivp`) | you have a small ODE system and want CA's integrator |
| **`external_python`** | **a solver class with its own `run()`** | **no — you do** | **your solver has a mesh, an assembled operator, an adaptive scheme, or any state that an RHS callback cannot express** |

!!! note "The distinction that matters"
    `python_user_defined` asks you for a derivative and integrates it. That is the wrong shape
    for a PDE solver: there is a mesh to build, forms to compile, an operator to assemble and a
    factorisation to reuse, and none of it survives being squeezed through a per-step callback.
    `external_python` inverts the relationship — CA hands over the record grid, asks for a run,
    and reads named traces back. Everything in between is yours.

## The contract

Put a class in a Python file and name it `SIM_HELPER` at module level. CA loads the file by
path and instantiates the class once.

```python
import numpy as np


class MyModel:
    # --- self-description (required) ---------------------------------------------
    parameters = {"heat/k": 1.0, "heat/u_D": 0.0}          # name -> default
    output_names = ["heat/T_p1", "heat/T_p2", "heat/T_p3"]

    # --- required methods --------------------------------------------------------
    def init_solver(self, config):
        """Called once. Do the expensive setup here: mesh, assembly, JIT."""

    def update_times(self, dt, start_time, sim_time, pre_time):
        """Set the record grid. Must be cheap — no re-assembly."""

    def set_param_vals(self, param_dict):
        """A subset of `parameters`. Must not require a re-init."""

    def run(self):
        """Solve the whole grid from the initial condition. Returns True/False."""

    def get_results(self):
        """{output_name: 1D np.ndarray}, on the grid, pre_time samples included."""

    # --- optional ----------------------------------------------------------------
    def get_init_param_vals(self, names): ...
    def reset(self): ...
    def extra_plots(self): ...      # -> list[matplotlib.figure.Figure]
    def close(self): ...


SIM_HELPER = MyModel
```

### `parameters` and `output_names` — literals, not expressions

Both are read **by parsing the file, without importing it**, so that a tool (or a machine with
none of your solver's dependencies installed) can list a model's parameters and outputs. Keep
them plain dict and list literals.

```python
parameters = {"heat/k": 1.0}                    # ✅ readable without importing
parameters = {f"heat/{name}": 1.0 for ...}      # ❌ invisible to the parser
parameters = dict(DEFAULTS)                     # ❌ likewise
```

Names use CA's canonical `component/variable` form, so they line up with the `vessel_name` /
`param_name` columns of `params_for_id.csv` and with the `operands` in `obs_data.json` without
any translation.

### `init_solver(config)`

Called once, before anything else. This is where the cost goes: build the mesh, compile the
forms, factorise what can be factorised, locate your probes.

`config` carries:

| key | meaning |
|---|---|
| `dt` | output sampling step |
| `sim_time` | logged simulation duration |
| `pre_time` | unlogged spin-up duration |
| `start_time` | time of the first sample |
| `solver_info` | the `solver_info` block from `user_inputs.yaml` |

`solver_info.get('user_config')` is a free-form dict, untouched by CA, that carries whatever
your solver needs — a mesh resolution, a tolerance, a path to a data file.

```python
def init_solver(self, config):
    user_config = (config.get('solver_info') or {}).get('user_config') or {}
    nx = int(user_config.get('nx', 16))
    ...
```

### `update_times(dt, start_time, sim_time, pre_time)`

Sets the **record grid**, and nothing else. CA may call it many times over a run, so it must be
cheap — no re-meshing, no re-assembly, no recompilation.

After it, `run()` must produce samples at `start_time + i*dt` for `i` in `0..N`, where

```
N = int(pre_time/dt) + int(sim_time/dt)
```

Compute `N` with that exact arithmetic rather than with rounding or `np.arange`, so your length
and CA's agree exactly instead of approximately.

!!! tip
    If a quantity like `dt` appears inside a compiled expression (a UFL form, a generated
    kernel), store it as a mutable constant that the expression already references. Then
    `update_times` is one array write instead of a round trip through the compiler.

### `set_param_vals(param_dict)`

Receives a **subset** of `parameters` — whatever the calibration is currently varying — and must
not require a re-init. A calibration calls this thousands of times on one instance; if each call
had to rebuild the model, the run would be dominated by setup.

Reject unknown names loudly. A typo in `params_for_id.csv` that is silently ignored produces a
calibration that reports success and has fitted nothing.

### `run()`

Solves the whole grid **from the initial condition**, and is repeatable: two calls with the same
parameters must give the same trace. This is the rule most often broken by accident — a solver
that carries its final state into the next call makes sample 500's cost depend on sample 499's
parameters, which shows up as a calibration that "almost works" and never converges.

Return `True` on success and `False` if the solve diverged or produced non-finite values. `False`
is not an error: CA drops that sample and carries on, which is exactly what you want when an
optimiser wanders into a corner of parameter space where your scheme is unstable.

### `get_results()`

Returns `{output_name: 1D np.ndarray}` — one array per name in `output_names`, each of length
`N + 1`, on the record grid. **Include the `pre_time` samples**; CA discards the leading
`int(pre_time/dt)` of them itself.

### The optional four

* **`get_init_param_vals(names)`** — the defaults for the named parameters. CA falls back to the
  `parameters` dict when you do not implement it.
* **`reset()`** — return to the initial state. Useful to call from your own `run()`.
* **`extra_plots()`** — a list of `matplotlib.figure.Figure` objects: fields, meshes, convergence
  histories, anything CA cannot know how to draw. CA collects them alongside its own plots, and
  the CUFLynx GUI surfaces them as extra tabs on the run. Build them with
  `matplotlib.figure.Figure` directly rather than `pyplot` — no global state, no backend to
  configure, safe on a headless node.
* **`close()`** — release whatever needs releasing. Called when CA is done with the instance.

## `user_inputs.yaml`

```yaml
file_prefix: my_model
model_type: external_python
solver: external

# Defaults to funcs_user/{file_prefix}_model.py, so this is only needed when the file
# lives elsewhere. Absolute, or relative to the yaml.
external_model_path: /path/to/my_model.py

pre_time: 0.0
sim_time: 0.05
dt: 0.0005

# Anything under user_config reaches init_solver untouched, as
# config['solver_info']['user_config'].
solver_info:
  user_config:
    nx: 16
    tolerance: 1e-8
```

Everything else — `params_for_id`, `obs_data.json`, `param_id_method`, `resources_dir` — works
exactly as it does for a CellML model. See
[Parameter Identification](parameter-identification.md).

## A minimal example first

`funcs_user/example_model_external/` is the same contract on a one-dimensional NumPy model, with
no external dependencies at all: a few dozen lines, its own explicit time loop, and the
`params_for_id` / `obs_data` pair to calibrate it. Read that one first if you only want to see
the shape of the thing.

## Walkthrough: a FEniCSx heat solver

`funcs_user/heat_fenics/` is the real one — a [FEniCSx](https://fenicsproject.org/) (dolfinx)
finite-element solver for

```
u_t = k Δu    on the unit square,   u = u_D on the boundary
```

with a Gaussian bump as the initial condition, backward Euler in time, P1 Lagrange in space, and
three point probes read out every step as `heat/T_p1`, `heat/T_p2`, `heat/T_p3`. Two parameters
are exposed for calibration: the diffusivity `heat/k` and the boundary value `heat/u_D`.

It is small on purpose — a 16×16 mesh and 100 steps, milliseconds per run once the forms are
compiled — because it is a teaching artefact, not a convergence study.

### 1. Install FEniCSx

dolfinx is a **conda-forge** package. It is not on PyPI, and it is not the legacy `dolfin`.

```bash
conda create -n fenicsx -c conda-forge fenics-dolfinx python=3.11
conda activate fenicsx
```

### 2. Install CA into the same environment

```bash
cd /path/to/circulatory_autogen
pip install -e ".[dev,emulation]"
```

`[dev]` brings pytest, `[emulation]` brings `autoemulate` — needed only if you want to train a
surrogate of the model (see [Emulators](emulators.md)). Drop it otherwise.

### 3. Check it runs

```bash
python funcs_user/heat_fenics/heat_fenics_model.py
```

The example has a `__main__` block that drives the class directly, with no CA involved, and
prints the sample count and the two observable values. A few seconds, including the one-off
form compilation.

!!! note "Tested against dolfinx 0.8.x and 0.9.x"
    The calls most prone to move between dolfinx releases — the function-space constructor, the
    bounding-box tree, the PETSc assembly helpers — are looked up through a small `_resolve`
    helper in the example, which raises a message naming the tested versions instead of an
    `AttributeError` from three frames down. If you are on a newer dolfinx and something raises,
    that message tells you which call moved.

### 4. Point `user_inputs.yaml` at it

```yaml
file_prefix: heat_fenics
model_type: external_python
solver: external
external_model_path: <CA_dir>/funcs_user/heat_fenics/heat_fenics_model.py
resources_dir: <CA_dir>/funcs_user/heat_fenics
param_id_obs_path: <CA_dir>/funcs_user/heat_fenics/heat_fenics_obs_data.json

pre_time: 0.0
sim_time: 0.05
dt: 0.0005

param_id_method: genetic_algorithm

solver_info:
  user_config:
    nx: 16
```

`resources_dir` is where CA looks for `heat_fenics_params_for_id.csv`, which defines the
calibration box:

```csv
vessel_name,  param_name,  param_type,  min,   max,   name_for_plotting
heat,         k,           const,       0.2,   5.0,   k
heat,         u_D,         const,       -0.5,  0.5,   u_{D}
```

and `heat_fenics_obs_data.json` holds the two scalar targets, `mean(heat/T_p2)` and
`min(heat/T_p2)`.

### 5. Calibrate and sweep it

Nothing about the entry points is special-cased for an external model:

```bash
./run_param_id.sh 4               # calibration on 4 MPI ranks
./run_sensitivity_analysis.sh 4   # Sobol indices for k and u_D
```

And a surrogate, if you are heading for MCMC or a large Sobol design:

```yaml
do_emulation: true
emulator_settings:
  num_train_samples: 64
  sample_type: sobol
  models: RadialBasisFunctions
```

## Gotchas worth knowing before you write yours

!!! warning "Build your mesh on `MPI.COMM_SELF`, not `COMM_WORLD`"
    CA parallelises over *independent simulations* — each MPI rank runs its own parameter
    sample. A solver that builds a distributed mesh on `COMM_WORLD` is instead splitting **one**
    problem across the ranks, and it will deadlock the moment two ranks ask for different
    parameters. Each rank must own a complete serial problem.

!!! warning "`run()` must restart from the initial condition"
    See above. Test it: run at parameters A, run at B, run at A again, and assert the first and
    third traces are identical. `tests/test_heat_fenics_example.py` does exactly that.

!!! tip "Pick observables that actually vary"
    The heat example deliberately does *not* use `max(heat/T_p2)`: the centre probe's maximum is
    its initial value, which is 1 for every parameter set. A constant feature contributes nothing
    to the cost and scores `NaN` R² as an emulator target. `min` (the relaxed final value) and
    `mean` both move with the parameters, so that pair is well conditioned.

!!! tip "Give yourself a free correctness check"
    In the heat example, probes p1 and p3 are images of each other under a symmetry that both
    the mesh and the initial bump have, so their traces must agree to round-off. That single
    assertion covers the probe locating and the assembly at once, and it holds on any mesh and
    any step size.

## See also

* `funcs_user/heat_fenics/README.md` — the FEniCSx example in detail, including how to
  regenerate its `obs_data.json` values for your own build.
* `funcs_user/example_model_external/` — the dependency-free NumPy version of the same contract.
* `tests/test_heat_fenics_example.py` — smoke, physics-sanity, plotting and emulator round-trip
  tests you can copy for your own model.
* [Parameter Identification](parameter-identification.md) and
  [Sensitivity Analysis](sensitivity-analysis.md) — the pipelines this backend plugs into.
* [Emulators (Surrogate Models)](emulators.md) — when a surrogate of your solver is worth
  training.
