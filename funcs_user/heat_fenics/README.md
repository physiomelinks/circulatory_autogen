# FEniCSx heat equation — the flagship `external_python` example

A [FEniCSx](https://fenicsproject.org/) (dolfinx) finite-element solver driven by
Circulatory Autogen through `model_type: external_python` / `solver: external`.

This is the example to copy when **your solver owns its own time-stepping**. A solver you only
have to supply a right-hand side for can call `scipy.solve_ivp` inside `run()` — see
`funcs_user/example_model_scipy/` — but that is the wrong shape for a PDE solver, which has a
mesh, an assembled operator and a time loop of its own. `external_python` inverts the
relationship: libCUFLynx hands
over the record grid, asks for a run, and reads named traces back. Everything in between is
yours.

## What it demonstrates

| Contract feature | Where it shows up here |
|---|---|
| Expensive one-off setup kept out of the hot loop | `init_solver` builds the mesh, the function space, the compiled forms and the probe cells |
| A cheap `update_times` | `dt` lives in the form as a `fem.Constant`, so re-gridding is one array write, not a re-compilation |
| `set_param_vals` without a re-init | `heat/k` and `heat/u_D` are `fem.Constant`s already in the form; setting them writes `.value` in place |
| A repeatable `run` | every call restarts from the same uniform initial condition, so a thousand calibration samples reuse one instance |
| Named scalar outputs | three point probes, evaluated every step, named `heat/T_p1..3` |
| `extra_plots` | two field snapshots (mid-time and final time), returned as `matplotlib` Figures for libCUFLynx / the CUFLynx GUI to place |

## The model

Backward Euler for `u_t = k Δu` on the unit square, P1 Lagrange elements. A **uniformly hot
plate, quenched through its boundary**:

```
u(x, 0) = 1                      everywhere
u = u_D  on the left edge        (x = 0)      -- calibratable
u = 0    on bottom, top, right                -- fixed
```

Weak form, with `u_n` the previous step:

```
∫ u v dx + dt·k ∫ ∇u·∇v dx  =  ∫ u_n v dx
```

Probes sit at (0.25, 0.25), (0.5, 0.5) and (0.75, 0.75), and are located once in
`init_solver` with `dolfinx.geometry.bb_tree` / `compute_colliding_cells`.

!!! note "Why the boundary is split"
    Driving one edge and fixing the other three is what makes the three probes carry
    *independent* information: p1 sits nearest the driven edge and answers mostly to `u_D`,
    p3 sits furthest and answers mostly to `k`. Under a symmetric initial bump with one
    boundary value, p1 and p3 were identical by symmetry, so only one of them was worth
    scoring.

    The corner dofs at (0, 0) and (0, 1) belong to *both* facet sets — the boundary value
    there is genuinely discontinuous. They are assigned to the left edge (`u_D`), which is
    arbitrary but deterministic and documented, rather than left to whichever condition
    dolfinx happens to apply last.

!!! warning "`u_D` defaults to 0.25, not 0"
    At `u_D == 0` the left edge is indistinguishable from the other three, p1 and p3
    coincide, and the boundary split the example exists to demonstrate is invisible at
    exactly the point it is demonstrated from.

### Time scales — why `dt = 0.02`, `sim_time = 2.0`

The slowest mode of the unit square with Dirichlet edges decays at `λ₁ = 2kπ² ≈ 19.7k`, so
across the calibration box `k ∈ [0.001, 0.2]` the plate's time constant runs from ≈ 51 s
down to ≈ 0.25 s. The suggested grid is therefore **100 steps of 0.02 s**: at the default
`k = 0.05` that is about two time constants, while `k = 0.01` leaves the plate only
partially cooled and `k = 0.2` fully relaxes it — both leave a distinct signature, which is
what makes `k` identifiable.

!!! note "The bottom of the box is a bound, not an operating point"
    Below about `k = 0.005` the plate barely cools on this window — at `k = 0.001` it keeps
    ~96% of its heat and every observable saturates at the initial temperature. That is
    exactly what a lower bound should say ("no diffusion"), and a calibration rules it out
    immediately, but it is a *flat* region of the cost surface: a gradient method started
    there has nothing to descend. Lengthen `sim_time` if you want that end informative. A default 16×16 mesh (289 dofs) makes a
whole run **milliseconds** once the forms are compiled; the one-off FFCx compilation in
`init_solver` is a few seconds.

## Files

| File | Purpose |
|---|---|
| `heat_fenics_model.py` | The solver class (`parameters`, `output_names`, `init_solver`, `update_times`, `set_param_vals`, `run`, `get_results`, `reset`, `extra_plots`, `close`) and `SIM_HELPER`. |
| `heat_fenics_params_for_id.csv` | The calibration box: `heat/k ∈ [0.001, 0.2]`, `heat/u_D ∈ [-0.5, 0.5]`. |
| `heat_fenics_obs_data.json` | Six scalar observables — `mean` and `min` of each of the three probes, so every probe is scored against a ground truth rather than only the centre one. |

!!! warning "Why not `max(T_p*)`?"
    The maximum of every probe is its *initial* value, which is the uniform `1.0` for every
    parameter set — a constant feature. It would contribute nothing to the cost and would
    score `NaN` R² as an emulator target. `min(T_p*)` is the temperature the probe reaches
    by the end of the window, and `mean(T_p*)` integrates the path it took to get there;
    together, across three probes at different distances from the driven edge, they pin
    down both parameters.

## Installing FEniCSx

dolfinx is a **conda-forge** package. It is not on PyPI, and this is *not* legacy `dolfin`.

```bash
# 1. a conda environment with FEniCSx in it
conda create -n fenicsx -c conda-forge fenics-dolfinx python=3.11
conda activate fenicsx

# 2. libCUFLynx itself, from your checkout, into that same environment
cd /path/to/circulatory_autogen
pip install -e ".[dev,emulation]"
```

`[dev]` brings pytest; `[emulation]` brings `autoemulate`, needed only if you want to train a
surrogate of this model (see below). Drop `emulation` if you do not.

Check the install:

```bash
python funcs_user/heat_fenics/heat_fenics_model.py
```

That drives the class directly, with no libCUFLynx involved, and prints the sample count, the two
observable values and the p1/p3 symmetry check. It should finish in a few seconds.

!!! note "Tested against dolfinx 0.8.x and 0.9.x"
    The API calls most prone to drift between dolfinx releases — the function-space
    constructor, the bounding-box tree, the PETSc assembly helpers, `Function.x.petsc_vec` —
    are looked up through a small `_resolve` helper that raises a message naming the tested
    versions instead of an `AttributeError` from three frames down. If you are on a newer
    dolfinx and something raises, that message tells you which call moved. The helper lives
    at the foot of `heat_fenics_model.py`, below the class, so the contract is what you read
    first.

## Running it from libCUFLynx

Add to `user_run_files/user_inputs.yaml` (paths absolute, or relative to the yaml):

```yaml
file_prefix: heat_fenics
model_type: external_python
solver: external
external_model_path: <CA_dir>/funcs_user/heat_fenics/heat_fenics_model.py
resources_dir: <CA_dir>/funcs_user/heat_fenics
param_id_obs_path: <CA_dir>/funcs_user/heat_fenics/heat_fenics_obs_data.json

pre_time: 0.0
sim_time: 2.0
dt: 0.02

param_id_method: genetic_algorithm

# Free-form options handed to init_solver as config['solver_info']['user_config'].
solver_info:
  user_config:
    nx: 16          # mesh resolution; 8 for a fast smoke test, 32 for a finer field
```

Then the ordinary libCUFLynx entry points — nothing about them is special-cased for this model:

```bash
./run_param_id.sh 4               # calibration on 4 MPI ranks
./run_sensitivity_analysis.sh 4   # Sobol sensitivity of the two observables to k and u_D
```

Each MPI rank builds its own serial mesh (`MPI.COMM_SELF`), because libCUFLynx parallelises over
*independent simulations* rather than over one mesh. Do not change that to `COMM_WORLD`: two
ranks solving different parameter samples on a distributed mesh will deadlock.

### Training an emulator of it

The six observables are scalars, so this model is a legitimate emulator target — worthwhile
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

The six values are the model's own output at the default parameters `k = 0.05`,
`u_D = 0.25` on the suggested `dt = 0.02` / `sim_time = 2.0` grid — **estimates**, computed
from a matched finite-difference solve of the same problem rather than from dolfinx itself
(the authoring machine had no FEniCSx), and quoted with a `std` that comfortably covers both
the FD-vs-P1-FEM gap and the discretisation error of the 16×16 backward-Euler scheme. They
are targets for a demonstration calibration, not measurements.

To pin them to the exact numbers *your* dolfinx build produces — which is what you want if
you intend the calibration to recover a known answer — run the file directly and paste the
printed values into the JSON (it prints `mean` and `min` for all three probes, in the JSON's
order):

```bash
python funcs_user/heat_fenics/heat_fenics_model.py
```

Changing `nx`, `dt` or `sim_time` changes them, so regenerate after any such change.

## See also

* `tutorial/docs/external-python-solvers.md` — the full contract, method by method.
* `funcs_user/example_model_external/` — the same contract on a 1-D NumPy model, with no
  external dependencies at all. Start there if you only want to see the shape of it.
* `funcs_user/example_model_scipy/` — an ODE solved with `scipy.solve_ivp` inside `run()`,
  the simple end of the same contract.
* `tests/test_heat_fenics_example.py` — smoke, physics-sanity, plotting and emulator
  round-trip tests for this example.
