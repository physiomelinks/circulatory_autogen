# Emulators (Surrogate Models)

An **emulator** (or surrogate) is a fast statistical stand-in for your model. Circulatory Autogen
fits one that maps the parameters in your `params_for_id.csv` to the **scalar features** of your
`obs_data.json` data items — the same numbers the cost function is computed from. Once it is
trained, calibration, sensitivity analysis, UQ/MCMC and identifiability analysis can all evaluate
the emulator instead of running the solver.

!!! warning "The training runs are paid up front"
    Training costs `num_train_samples` simulations. That is only worth it when the downstream use
    is much larger:

    * **Sobol sensitivity analysis** — `num_samples × (2M + 2)` model evaluations. Worth it.
    * **MCMC / UQ** — tens of thousands of evaluations. Worth it.
    * **Identifiability analysis** — a Hessian's worth per parameter pair. Usually worth it.
    * **A single genetic-algorithm calibration** — often *cheaper run directly*. Not worth it.

!!! note "Scope: scalar features, not waveforms"
    The emulator predicts each data item's value **after** its `operation` (`max`, `mean`,
    `max_minus_min`, …). It does not produce simulated traces, so `data_type: series` and
    `frequency` observables, prediction variables and output plots are not available in emulator
    mode — CA refuses them explicitly rather than returning something that looks plausible.
    Emulating full time series is a planned follow-up.

## Installation

The emulator backend is [autoemulate](https://pypi.org/project/autoemulate/), an optional
dependency (it pulls in torch, gpytorch and lightgbm — about 750 MB — and needs Python ≥3.10,
<3.13):

```bash
pip install "libcuflynx[emulation]"
```

Everything else in CA works without it; only `do_emulation` / `use_emulator` need it. CA never
imports autoemulate unless one of those flags is set — it checks whether the package exists
without loading it, so a normal run does not pay torch's import time.

!!! tip "Two installation gotchas"
    **Pin a CPU torch.** Left alone, pip installs the CUDA build and its ~2.5 GB of `nvidia-*`
    wheels. On a machine without a GPU:

    ```bash
    pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
    pip install "autoemulate>=2.1,<3" --extra-index-url https://download.pytorch.org/whl/cpu
    ```

    **autoemulate 2.1.2 cannot sit alongside torch ≥ 2.13.** Its `harmonic` dependency pins
    `setuptools==68.0.0` while torch 2.13 requires `setuptools>=77.0.3`, so pip reports
    `ResolutionImpossible`. Pinning torch to 2.12.x (as above) resolves it. Without a pin, pip
    silently backtracks through several 500 MB torch wheels looking for a combination that works.

## Configuration

Two independent flags, because emulation has two steps:

```yaml
do_emulation: false   # train an emulator against the solver named by `solver`
use_emulator: false   # make the analyses evaluate the trained emulator
```

`solver:` keeps meaning the **truth solver** — the one the emulator is trained against, and the
one you compare it with. That is deliberate: switching to an emulator should not lose track of
what it is approximating.

```yaml
emulator_settings:
  # emulator_dir:            # default <param_id_output_dir>/emulators/<file_prefix>_<obs_prefix>
  models: default            # 'default', 'all', or a comma-separated list of emulator names
  num_train_samples: 128     # simulations run to build the training set
  reuse_samples: false       # refit the samples a previous run saved, instead of simulating
  sample_type: sobol         # sobol | latin_hypercube | random
  log_scale_params: false    # space the design logarithmically (needs every min > 0)
  random_seed: 0
  test_fraction: 0.2         # held out and never trained on -- what R2/RMSE are measured on
  n_splits: 5                # autoemulate cross-validation folds
  n_iter: 10                 # hyper-parameter settings sampled per emulator
  min_r2: 0.9                # refuse to USE an emulator worse than this
  out_of_bounds: error       # error | warn | clip
  fd_rel_step: 1.0e-3        # step for the finite-difference gradient over the emulator
```

The available `models` names come from the installed autoemulate and are discoverable in code:

```python
from libcuflynx.emulators.emulator_trainer import emulator_model_names
print(emulator_model_names())   # GaussianProcessRBF, RadialBasisFunctions, LightGBM, ...
```

## Training

```bash
cd user_run_files
./run_emulator_training.sh 8      # 8 MPI processes
```

The design points are spread across MPI ranks, each runs the real solver, and rank 0 fits and
saves the emulator. The script prints the held-out R² per observable:

```
[emulator] training on 64 samples across 8 rank(s)
[emulator] saved to .../emulators/Simple_ODE_Benchmark_Simple_ODE_Benchmark_obs_data
    held-out R2   1.0000   x_{SS} (steady_state_avg benchmark/x)
    held-out R2   1.0000   y_{SS} (steady_state_avg benchmark/y)
```

**Read those numbers before using it.** They are the only thing standing between you and a set
of Sobol indices for a model you did not simulate.

!!! warning "How good an emulator you can get depends on the parameter box"
    Difficulty grows with the width of the `params_for_id` ranges, the number of parameters and
    how rough the feature is. Two worked examples from this repo, at opposite ends:

    * **`Simple_ODE_Benchmark`** — steady states of `dx/dt = -x + p`, `dy/dt = -3y + q`, i.e. a
      smooth monotone response. 64 Sobol samples give held-out R² of **0.99999** and **0.99997**.
      This is what a well-posed emulation problem looks like.
    * **`Lotka_Volterra`** — `max` of each state over the *full* declared ranges (`alpha` 0.1–7,
      `gamma` 0.1–10, the response spanning 20–3900). A 128-sample Gaussian process manages only
      about **0.2 and 0.5**, and 256 samples does not reliably improve it: `max` of an
      oscillation whose period shifts sharply with the parameters is close to discontinuous.

    If your R² is poor, the useful moves are, in order: narrow the parameter ranges to the
    region you actually care about; add samples; set `log_scale_params: true` when a bound spans
    decades; try `models: all`; and consider whether the feature is a smooth function of the
    parameters at all. What you should *not* do is lower `min_r2` to make the run proceed.

The saved directory contains:

| File | What it is |
|---|---|
| `emulator.joblib` | the fitted emulator |
| `emulator_metadata.json` | per-feature R², RMSE, MAE, bias, max abs error and nRMSE; the training box; the design; provenance |
| `training_data.npz` | the design and its simulated targets, so it can be refitted or extended without re-simulating |
| `emulator_validation.npz` | the held-out points: `theta`, the simulator's `y_true` and the emulator's `y_pred`, in real units |

### Trying another emulator without re-running the simulations

Training is two costs: the `num_train_samples` runs of your model — minutes to hours, and the
whole reason emulators exist — and the fit, which takes seconds. `training_data.npz` above is
what makes the second one repeatable on its own:

```yaml
emulator_settings:
  reuse_samples: true        # refit what emulator_dir already holds; run no new simulations
  models: all                # ... with a different emulator,
  test_fraction: 0.3         # ... or a different split, n_iter, n_splits, min_r2, random_seed
```

```bash
./run_emulator_training.sh 1    # one rank is enough: there is nothing left to parallelise
```

The result is an ordinary emulator bundle, written over the same directory, and its metadata
records `design.reused_samples: true` so the provenance never claims simulations that run did
not perform.

!!! warning "What it does *not* do"
    * It runs **no simulations**, so `num_train_samples`, `sample_type` and `log_scale_params`
      are ignored — the saved design is what gets fitted, however many points it holds. If your
      `num_train_samples` disagrees with what was saved, CA prints the number it is really
      using rather than letting the requested one stand.
    * It needs a previous training run in `emulator_dir`. The first run has to have
      `reuse_samples: false`; that is the run that pays for the simulations.
    * `random_seed` still applies — it seeds the fit and the train/test split — so re-fitting
      with a different seed is a meaningful thing to do.
    * Samples belong to one problem. If the parameter bounds, `obs_data.json`, protocol or the
      model file have changed since they were simulated, CA **refuses** rather than refitting
      them: retrain with `reuse_samples: false` instead. Reusing them would produce an emulator
      that is confidently wrong about a study it was never trained for.

### Analysing the error

The statistics say how wrong the emulator is on average; the held-out points say
**where**, which is what decides whether the region you care about is one of the good
ones. Both are read through the bundle:

```python
from libcuflynx.emulators.emulator_bundle import EmulatorBundle
bundle = EmulatorBundle.load(emulator_dir)

for row in bundle.error_stats():
    print(row['label'], row['r2'], row['bias'], row['nrmse'])

points = bundle.error_points()      # None if the bundle predates this
# points['y_pred'] vs points['y_true']  -> parity plot
# points['residual'] vs points['theta'] -> where in the space it goes wrong
```

`residual` is **prediction minus truth**, fixed here so every consumer agrees on
the sign: positive means the emulator reads high. Why more than R²:

* **`bias`** — a feature can score a good R² and still read systematically high,
  which shifts every downstream cost rather than just adding noise to it.
* **`nrmse`** — RMSE in one feature's units says nothing against another's, so it
  is the only one of these that can rank features against each other.
* **`max_abs_error`** — an emulator that is good almost everywhere still misleads
  a calibration that walks through the one place it is not.

These points are free: training already paid to simulate them and then deliberately
did not fit to them.

## Using it

Set `use_emulator: true` and run any of the usual scripts unchanged:

```bash
./run_param_id.sh 4
./run_sensitivity_analysis.sh 4
./run_identifiability_analysis.sh
```

Nothing else in the configuration changes. To sanity-check a result, run the same analysis with
`use_emulator: false` and compare — that is what keeping `solver:` meaningful is for.

## When CA refuses

An unvalidated emulator does not fail loudly; it returns plausible wrong numbers, and every
downstream index, cost and posterior inherits the error with nothing to show for it. So CA
refuses rather than proceeding quietly:

| Situation | What happens |
|---|---|
| Worst held-out R² below `min_r2` | refused at setup, naming the observable and its R² |
| A parameter outside the training box | refused (or warns/clips, per `out_of_bounds`) |
| Parameter bounds, observables, operations, protocol or the model file changed since training | refused as **stale** — retrain |
| A `series` or `frequency` data item | refused: the emulator predicts scalars only |
| `reuse_samples: true` with no previous emulator, or one saved without its samples | refused, naming the directory it looked in — train once without the setting first |
| `reuse_samples: true` after the bounds, obs_data, protocol or model changed | refused as **stale** — retrain with `reuse_samples: false` |
| `autoemulate` not installed | refused, naming the install command |

An emulator is an interpolant. Outside the box it was trained in it is an extrapolation with no
error estimate at all, which is why `out_of_bounds: error` is the default.

## If saving fails: `model_serialiser`

Training pays for every simulation *before* it writes anything, so a model that cannot be
pickled costs the whole run rather than just the save. Some fitted emulators hold an
uninitialised C-extension descriptor that `pickle` cannot take apart, and the run ends with:

```
TypeError: cannot pickle '_abc._abc_data' object
```

`emulator_settings.model_serialiser` decides which container is used:

| Value | Behaviour |
|---|---|
| `auto` (default) | joblib, then cloudpickle, then dill, until one works — with a warning saying which |
| `joblib` | joblib only — fail rather than switch container |
| `cloudpickle` | cloudpickle only |
| `dill` | dill only |

**None of the three is a superset of the others**, which is why `auto` falls back in order
rather than simply preferring the most capable one. Measured against autoemulate 2.1.2:

| | an object pickle cannot name | a torch-backed emulator |
|---|---|---|
| joblib | fails | works |
| cloudpickle | works | works |
| dill | works | **fails** (a `PyCapsule` it recurses on) |

So joblib stays first — it is what `autoemulate` itself writes and reads — and switching to
dill outright would break the common case to fix the rare one.

Which container wrote a bundle is recorded in `emulator_metadata.json` as `model_serialiser`,
so it reads back without the setting having to be repeated; a bundle written before the
setting existed still loads, because all three are tried. Note that a bundle saved with a
fallback needs that library present wherever it is loaded.

If a training run dies while saving, leave this at `auto` and make sure the fallbacks are
installed (`pip install "libcuflynx[emulation]"` brings them), or name one outright.

### One failure no container can fix

```
PicklingError: Can't pickle sentinel: it's not the same object as typing_extensions.sentinel
```

This one is not about the container, and changing `model_serialiser` will not help — joblib,
cloudpickle and dill all fail identically. A [PEP 661](https://peps.python.org/pep-0661/)
sentinel pickles by *name*: its `__reduce__` returns a string, and pickle stores it as a global,
checking on the way back in that the name still refers to the same object.
`typing_extensions` 4.16.0 ships one where that check cannot pass:

```python
_marker = sentinel("sentinel")     # named "sentinel", bound to _marker
```

`typing_extensions.sentinel` is the *class*, so the identity check fails for any object holding
`_marker`. CA handles it by reducing sentinels to where they actually live rather than to what
they call themselves, so nothing needs configuring — but if you meet this outside CA, the
workaround is `pip install "typing_extensions!=4.16.0"` (4.15.0 is unaffected).

## Gradients

Over an emulator the only gradient source is **finite differences on the emulator itself**. The
analytic arms (CasADi AD, Myokit CVODES FSA, AADC) all differentiate the *real* model, which is
not the function an emulator run is evaluating — using one would mean the optimiser descends a
different function than the cost it reports.

This costs `2M` emulator evaluations per gradient, which is a matrix multiply apiece rather than
`2M` simulations, so gradient-based calibration (`sp_minimize`, `multi_start_sp_minimize`) and
the Laplace identifiability analysis work normally. `do_ad` is turned off automatically, with a
message, when `use_emulator` is set.

## From Python

```python
from libcuflynx.emulators.emulator_trainer import EmulatorTrainer
from libcuflynx.param_id.paramID import CVS0DParamID

inp["do_emulation"] = True
inp["emulator_settings"] = {"num_train_samples": 200, "models": "GaussianProcessRBF"}
bundle = EmulatorTrainer.init_from_dict(inp).train()
print(dict(zip(bundle.feature_labels, bundle.meta["feature_r2"])))

inp["use_emulator"] = True
pid = CVS0DParamID.init_from_dict(inp)   # calibrates against the emulator
```

`EmulatorTrainer.init_from_dict` always builds its engine with `use_emulator` forced off, so
training runs the real solver even when the config asks for an emulator elsewhere.

## Notes

* The training targets are computed through the **same code path as the cost**
  (`libcuflynx.param_id.fd_backend.observable_features`), so the emulator approximates exactly what your
  calibration is fitting rather than a second implementation of it.
* Parameters are mapped to the unit box and features are standardised before fitting. CA
  parameters routinely span a compliance near `1e-9` and a resistance near `1e8`, and autoemulate
  works in float32; the transforms are stored in the bundle and inverted on prediction.
* Modifier and grouped parameters are supported: the emulator is trained on θ — one value per
  `params_for_id` entry — not on the expanded per-parameter values.
* `params_to_change` from your protocol are held at the values they had during training. Change
  the protocol and the emulator is refused as stale.
