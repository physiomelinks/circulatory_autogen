# FEniCSx heat equation — the flagship `external_python` example

A [FEniCSx](https://fenicsproject.org/) (dolfinx) finite-element solver driven by
Circulatory Autogen through `model_type: external_python` / `solver: external`.

This is the example to copy when **your solver owns its own time-stepping**. CA's older
`python_user_defined` backend asks you for a right-hand side and integrates it with
`scipy.solve_ivp`; that is the wrong shape for a PDE solver, which has a mesh, an assembled
operator and a time loop of its own. `external_python` inverts the relationship: CA hands
over the record grid, asks for a run, and reads named traces back. Everything in between is
yours.

## What it demonstrates

| Contract feature | Where it shows up here |
|---|---|
| Expensive one-off setup kept out of the hot loop | `init_solver` builds the mesh, the function space, the compiled forms and the probe cells |
| A cheap `update_times` | `dt` lives in the form as a `fem.Constant`, so re-gridding is one array write, not a re-compilation |
| `set_param_vals` without a re-init | `heat/k` and `heat/u_D` are `fem.Constant`s already in the form; setting them writes `.value` in place |
| A repeatable `run` | every call restarts from the same Gaussian initial condition, so a thousand calibration samples reuse one instance |
| Named scalar outputs | three point probes, evaluated every step, named `heat/T_p1..3` |
| `extra_plots` | two field snapshots (mid-time and final time), returned as `matplotlib` Figures for CA / the CUFLynx GUI to place |

## The model

Backward Euler for `u_t = k Δu` on the unit square, P1 Lagrange elements, `u = u_D` on the
whole boundary, and a Gaussian bump as the initial condition:

```
u(x, 0) = exp(-|x - (0.5, 0.5)|² / (2 σ²)),   σ = 0.15
```

Weak form, with `u_n` the previous step:

```
∫ u v dx + dt·k ∫ ∇u·∇v dx  =  ∫ u_n v dx
```

Probes sit at (0.25, 0.25), (0.5, 0.5) and (0.75, 0.75), and are located once in
`init_solver` with `dolfinx.geometry.bb_tree` / `compute_colliding_cells`.

!!! note
    `heat/T_p1` and `heat/T_p3` agree to round-off *by construction*: the bump is radially
    symmetric and a uniform triangulation of the unit square is invariant under a 180°
    rotation about its centre, which maps p1 onto p3. That is a free correctness check on
    the probe locating and the assembly, and `tests/test_heat_fenics_example.py` asserts it.

### Time scales — why `dt = 0.0005`, `sim_time = 0.05`

The slowest mode of the unit square decays at `λ₁ = 2kπ² ≈ 19.7k`, so at the default `k = 1`
the bump has a time constant of ≈ 0.05 s. The suggested grid is therefore **100 steps of
0.0005 s**, covering about one time constant. Across the calibration box `k ∈ [0.2, 5]` that
window spans "barely decayed" to "fully relaxed to `u_D`", which is what makes both
parameters identifiable. A default 16×16 mesh (289 dofs) makes a whole run **milliseconds**
once the forms are compiled; the one-off FFCx compilation in `init_solver` is a few seconds.

## Files

| File | Purpose |
|---|---|
| `heat_fenics_model.py` | The solver class (`parameters`, `output_names`, `init_solver`, `update_times`, `set_param_vals`, `run`, `get_results`, `reset`, `extra_plots`, `close`) and `SIM_HELPER`. |
| `heat_fenics_params_for_id.csv` | The calibration box: `heat/k ∈ [0.2, 5.0]`, `heat/u_D ∈ [-0.5, 0.5]`. |
| `heat_fenics_obs_data.json` | Two scalar observables — `mean(heat/T_p2)` and `min(heat/T_p2)`. |

!!! warning "Why not `max(T_p2)`?"
    The maximum of the centre probe is the *initial* value, which is `exp(0) = 1` for every
    parameter set — a constant feature. It would contribute nothing to the cost and would
    score `NaN` R² as an emulator target. `min(T_p2)` (the final, relaxed value) is
    informative about `u_D`, and `mean(T_p2)` about `k`, so the pair is well conditioned.

## Installing FEniCSx

dolfinx is a **conda-forge** package. It is not on PyPI, and this is *not* legacy `dolfin`.

```bash
# 1. a conda environment with FEniCSx in it
conda create -n fenicsx -c conda-forge fenics-dolfinx python=3.11
conda activate fenicsx

# 2. CA itself, from your checkout, into that same environment
cd /path/to/circulatory_autogen
pip install -e ".[dev,emulation]"
```

`[dev]` brings pytest; `[emulation]` brings `autoemulate`, needed only if you want to train a
surrogate of this model (see below). Drop `emulation` if you do not.

Check the install:

```bash
python funcs_user/heat_fenics/heat_fenics_model.py
```

That drives the class directly, with no CA involved, and prints the sample count, the two
observable values and the p1/p3 symmetry check. It should finish in a few seconds.

!!! note "Tested against dolfinx 0.8.x and 0.9.x"
    The API calls most prone to drift between dolfinx releases — the function-space
    constructor, the bounding-box tree, the PETSc assembly helpers, `Function.x.petsc_vec` —
    are looked up through a small `_resolve` helper that raises a message naming the tested
    versions instead of an `AttributeError` from three frames down. If you are on a newer
    dolfinx and something raises, that message tells you which call moved.

## Running it from CA

Add to `user_run_files/user_inputs.yaml` (paths absolute, or relative to the yaml):

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

# Free-form options handed to init_solver as config['solver_info']['user_config'].
solver_info:
  user_config:
    nx: 16          # mesh resolution; 8 for a fast smoke test, 32 for a finer field
    bump_sigma: 0.15
```

Then the ordinary CA entry points — nothing about them is special-cased for this model:

```bash
./run_param_id.sh 4               # calibration on 4 MPI ranks
./run_sensitivity_analysis.sh 4   # Sobol sensitivity of the two observables to k and u_D
```

Each MPI rank builds its own serial mesh (`MPI.COMM_SELF`), because CA parallelises over
*independent simulations* rather than over one mesh. Do not change that to `COMM_WORLD`: two
ranks solving different parameter samples on a distributed mesh will deadlock.

### Training an emulator of it

The two observables are scalars, so this model is a legitimate emulator target — worthwhile
as soon as you want Sobol indices or MCMC rather than a single calibration:

```yaml
do_emulation: true
emulator_settings:
  num_train_samples: 64
  sample_type: sobol
  models: RadialBasisFunctions
```

See [Emulators](../../tutorial/docs/emulators.md) for the rest.

## About the numbers in `heat_fenics_obs_data.json`

The two values (`mean(T_p2) = 0.385`, `min(T_p2) = 0.172`) are the model's own output at the
default parameters `k = 1.0`, `u_D = 0.0` on the suggested `dt = 0.0005` / `sim_time = 0.05`
grid, estimated from the eigenfunction expansion of the Gaussian bump and quoted with a `std`
that comfortably covers the discretisation error of the 16×16 backward-Euler scheme. They are
targets for a demonstration calibration, not measurements.

To pin them to the exact numbers *your* dolfinx build produces — which is what you want if
you intend the calibration to recover a known answer — run the file directly and paste the
two printed values into the JSON:

```bash
python funcs_user/heat_fenics/heat_fenics_model.py
```

Changing `nx`, `dt` or `sim_time` changes them, so regenerate after any such change.

## See also

* `tutorial/docs/external-python-solvers.md` — the full contract, method by method.
* `funcs_user/example_model_external/` — the same contract on a 1-D NumPy model, with no
  external dependencies at all. Start there if you only want to see the shape of it.
* `funcs_user/example_model/` — the older `python_user_defined` (RHS-only) backend.
* `tests/test_heat_fenics_example.py` — smoke, physics-sanity, plotting and emulator
  round-trip tests for this example.
