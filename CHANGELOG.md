# Changelog

Notable changes to libcuflynx (circulatory_autogen). Entries under *Unreleased* ship with the
next release; add to that section as you land a change.

## Unreleased

## 0.7.0 — 2026-09-01

### Changed — one spelling per concept in `obs_info` and `prediction_info`

The obs_data *file* vocabulary was split by #466; the parsed dicts were not, and kept the old
spellings as live aliases. `obs_info` held `obs_names` with `data_item_names` aliased to the
same list, and `names_for_plotting` aliased to `item_names_for_plotting`. `prediction_info`
was worse: its `names_for_plotting` meant the **trace** label where obs_info's meant the
**item** label, so one key named two different things depending on which dict you held.

Now: `obs_info` has `data_item_names`, `trace_names_for_plotting` and
`item_names_for_plotting` and nothing else; `prediction_info` mirrors it exactly, with its
`names` becoming `operands` (a list per item, as on obs_info) and its `names_for_plotting`
becoming `trace_names_for_plotting`. `obs_item_names` / `obs_item_labels` /
`obs_trace_labels` / `obs_operand_lists` now work on either dict.

**If you build one of these dicts by hand**, `ParamID` normalises what it is handed and warns
once per superseded key; a dict setting both an old key and its replacement is an error.
Reading the old keys off a parsed dict no longer works, because they are no longer written.

Two bugs fell out. `posterior_predictive` captioned a series panel with the item's *identity*
where it meant the trace label, and `print_observable_errors` named one of its six branches by
identity where the other five used the label.


## 0.6.0 — 2026-08-31

### Removed — the flat-import shims (#428)

`import parsers`, `from param_id.paramID import CVS0DParamID` and the other nine top-level
names were moved under `libcuflynx` in 0.4.0 and kept working since as deprecation shims that
warned once. They are gone: the eleven `src/<name>/` packages, the `_AliasFinder` meta-path
hook and `libcuflynx._deprecated_aliases` are all deleted, and the old spellings now raise
`ModuleNotFoundError`.

**Migrate by prefixing the import with `libcuflynx.`** — `from libcuflynx.param_id.paramID
import CVS0DParamID`. Nothing else changed: the modules, classes and functions are the same
objects at the same paths they have had since 0.4.0, so this is a rename of the import line
and nothing more. An installed package is importable from any directory, so a
`sys.path.insert(0, 'src')` that preceded the old spelling can go too.

This was announced for 0.5.0 and deferred once, so that one release did not ask for two
unrelated migrations alongside the #466 obs_data break.

### Note — `names_for_plotting` is still here

0.4.1 said this `obs_info` alias would go in 0.6.0. It has not: the engine still reads it
internally (`paramID.py`), so removing it is a change to the parser and its call sites rather
than a deletion, and it does not belong in the same release as the namespace removal for the
same reason that removal was deferred out of 0.5.0. Prefer `data_item_names`,
`trace_names_for_plotting` or `item_names_for_plotting`; the alias will go in a later release.
*(Done in Unreleased — see "one spelling per concept" above.)*

## 0.5.1 — 2026-08-25

### Changed — the MCMC ensemble is evaluated in one emulator call (#490)

An ensemble sampler asks for its whole walker population at every step, and a fitted surrogate
costs almost the same at sixty-four points as at one: per-call overhead dominates. Measured on a
two-phase RBF emulator with 84 outputs, one parameter vector took 84.8 ms and sixty-four took
355 ms — 5.6 ms each. The emulator-backed sampler now asks once per step instead of once per
walker.

### Changed — `OpencorParamID` is now `ParamID`, and `OpencorMCMC` is `MCMC` (#491)

They are the parameter-identification and MCMC engines, and they run against myokit/CVODE,
casadi and trained emulators as readily as against OpenCOR — the name came from the only backend
that existed when they were written. **The old names still work**: `OpencorParamID` and
`OpencorMCMC` remain as aliases, so nothing importing them has to change.


## 0.5.0 — 2026-08-24

### Added — check a posterior against the data it was fitted to (#478, #473)

A chain says what the parameters could be. It does not say whether the model, at those
parameters, reproduces what was measured — a calibration that fits badly can still produce a
tidy posterior. `libcuflynx.param_id.posterior_predictive` draws from the chain, simulates each
draw and scores the result: predictions in units of each measurement's own std, so observables on
different scales share one axis, plus coverage against its nominal level. The draws are spread
across MPI ranks, and both engines honour `use_emulator`, without which a check against the
solver is only affordable on a chain that was.

### Added — the plotting scripts ship with the engine (#479)

`plot_utilities.py` and `plot_outputs.py` are written into every generated bundle, so a folder
that reproduces a study can also draw it. Three panels nothing drew before: the pairwise
posterior (`plot_corner`), the posterior predictive against the measurements, and coverage
against its nominal level. All three read what `posterior_predictive` writes and return quietly
when it is absent, so they are safe on a calibration-only run.

### Added — emulator designs can be drawn in stages (#488)

A Sobol design spreads points evenly over the parameter box, which is right when nothing is known
about the response and wasteful once something is. `emulator_settings` now accepts staged
designs: a first stage explores, and a later one aims at where the model is hard — by output
gradient (`gradient_weighted`) or by the emulator's own held-out error (`error_weighted`), with a
weight dial between "follow the signal" and "keep exploring". Validation stays on a Sobol-only
subset, so the score is not measured on the points the design deliberately clustered.

### Added — two-phase emulators for observables that sit on a floor (#486)

autoemulate's emulators are regressors, and a regressor is the wrong shape for an observable
pinned at one value over most of its range: a smooth fit splits the difference across the flat
region and undershoots beyond it. A two-phase emulator classifies which side of the cliff a point
is on, then regresses only on the side that varies. One rule now decides which non-scalar
observables an emulator must refuse, rather than several places each deciding differently.

### Added — draw the recorded trace behind the posterior draws (#485)

A recorded trace carried in an obs_data purely to be plotted is now drawn behind the model's own
draws, the trace panels say which line is which (simulation, data trace, data max, sim max), and
the model's own value of each statistic travels in the plot metadata rather than only the
scalars.

### Fixed — sensitivity analysis on an obs_data holding a recorded trace (#487)

`generate_outputs_mpi` looked up an operation func for every data_item, and a trace carried for
plotting has none, so a Sobol run died on `KeyError: None` before the first sample finished. A
series observable's weight is now checked before it is interpolated, not after, so a
zero-weighted trace costs nothing and asks the model for nothing.

### Changed — tested on 3.12, and 3.9 is dropped (#459, #480)

The declared dependencies have required >= 3.10 since 0.4.0; the test matrix now says so.

### Changed — the flat-import shims now go away in 0.6.0, not 0.5.0

They were promised for removal in this release. They survive it instead. This release already
asks users to migrate their obs_data files (below), and removing the `libcuflynx.` namespace
shims at the same time would make one upgrade demand two unrelated migrations — one touching
data, one touching imports. `REMOVAL_VERSION` now reads `0.6.0`, and every shim's warning, the
README, CONTRIBUTING, CLAUDE.md and the tutorial say so.

The tests that check those docs agree now read `REMOVAL_VERSION` rather than restating it: the
literal `"0.5.0"` in three test files would have passed happily while every document said
something else.


### Added — a whole study in one call, shipped for readers outside this repo (#478)

`libcuflynx.external_testing.full_pipeline_run.build_full_pipeline_run()` runs sensitivity, an
emulator, a calibration, a chain and a posterior predictive check into one directory, small
enough for a test. Every stage already had its own test; nothing tested the combination, which
is what the readers downstream depend on — the generated `plot_outputs.py` and CUFLynx's
outputs-directory loader both find files by CA's naming rule. Those readers were checked against
fixtures written by hand, which agree with the reader by construction and cannot notice CA
renaming a file.

It ships in the package rather than living in `tests/` because the wheel carries no `tests/`:
CUFLynx resolves CA through whatever `libcuflynx` is installed, so a builder it cannot import is
a builder it cannot use. It sits in `external_testing` — shipped code whose only callers are
tests outside this repository — and deliberately not in `checks`, which validates a *user's*
model during generation. Nothing an ordinary run reaches imports it.

### Changed — the scaling benchmark now reports how much work it did (#344)

The 3compartment core-scaling sweep reported CMA-ES speeding up **22.8x on 8 cores**, past the
physical ceiling, in three independent sweeps. The cause is that the population methods stop when
they reach `cost_convergence`, so the work they do is whatever the search happened to need — and
unseeded, that is a fresh draw every run. The wall-clock ratio was measuring how soon each core
count happened to converge, not throughput.

Two things change, and it is worth being precise about which does what:

- **Every run records the cost evaluations it performed** (`num_cost_evals.npy`, carried through
  `BenchmarkRow.evals`). The scaling sweep prints them per core count and states in the table's
  env note whether they matched. This is the load-bearing part: wall-clock on this hardware
  varies 1.3–2.1x on identical code, so a ratio only reads as throughput once the counts are
  known to agree.
- **`seed` is honoured by `genetic_algorithm` and `CMA-ES`** (it already was by
  `multi_start_sp_minimize`), and is published in `PARAM_ID_METHODS`. This makes a run repeatable
  at a given rank count. Measured, it does **not** make CMA-ES rank-independent: it asks one
  candidate per MPI rank, so the ask/tell interleaving still changes with the rank count. With a
  fixed batch, 1/2/4 ranks agree and 8 does not, because nevergrad also derives internal settings
  from `num_workers`. Equal work across core counts needs the population decoupled from the rank
  count, which has **not** been done — so #344 stays open, with the numbers now visible rather
  than implied.

Also: `--leg-timeout` (default 1 hour) kills a hung core-count leg instead of stalling the sweep —
the AADC path has gone silent for 87 minutes against an 8.6 s nominal — and the benchmarks CI job
gets `timeout-minutes: 180` instead of inheriting GitHub's 6-hour default. The README's claim that
"every core count runs the same work" was true only of multi-start and has been corrected.

### Added — an observable can be built from one in another experiment (#466, #127)

An `operation_kwargs` value naming another `data_item_name` may now name an item in a different
`experiment_idx` / `subexperiment_idx`. The table those names resolve against used to be cleared
once per sub-experiment, so a reference could only ever see the segment being evaluated; it now
spans a whole cost evaluation, with segments visited in order, so a reference reaches backwards
across experiments. This makes the difference between a baseline run and a treated one — often
the quantity actually measured — expressible as an observable.

Two related fixes fall out of it. A reference to an item that has not been computed yet now
raises, naming the item to move, instead of passing the name through as a plain string (which
surfaced as `str - str`, or as a plausible wrong number). And on the sensitivity path the table
is cleared per sample rather than once per run, so a forward reference can no longer read the
previous sample's value.

The Myokit CVODES FSA and CasADi AD gradients refuse a cross-segment reference: each builds its
observables from one sub-experiment's operands, so it would differentiate a different feature
than the cost is built from. Finite differences and the gradient-free methods are unaffected.

### Changed (breaking) — an obs_data item's name is now separate from its labels (#466)

`variable` and `name_for_plotting` each named two different things, and one of the collisions
was silently producing wrong numbers. Four fields replace them:

| was | now | what it is |
|---|---|---|
| `variable` | **`data_item_name`** | the item's identity; **must be unique** across `data_items` and `prediction_items` |
| `variable` (as a fallback operand) | **`operands`** | the model variable(s) the item reduces — now always required |
| `name_for_plotting` | **`trace_name_for_plotting`** | the axis label of the trace; may repeat |
| `name_for_plotting` | **`item_name_for_plotting`** | the label of the scalar feature; defaults to `"<trace> (<operation>)"` |

`prediction_items` take the same four.

**Why uniqueness matters.** A string in `operation_kwargs` that names another item is how an
observable is built from other observables, and `data_item_name` is what it resolves against.
When two items answered to one name the reference took whichever was computed last — so the
shipped `resources/3compartment_extra_ops_obs_data.json`, which asks for the max minus the mean
of one trace, was computing `max - max`, a constant `0.0` against a ground truth of `4e-4`. Every
test passed, because none asserted a value. A repeated `data_item_name` is now an error that
names the offenders, and the example is fixed and pinned by a test.

**Migrating.** Run:

```
cuflynx-migrate-obs-data path/to/resources        # or a single obs_data.json; --dry-run to look
```

It renames the keys, gives `prediction_items` the `operands` they never had, and — the part that
is not mechanical — makes `data_item_name` unique, deriving a name from whatever actually
distinguishes the colliding items (the operation first, then the experiment and sub-experiment).
Edits are textual, so each file keeps its own formatting and the diff shows only what moved. An
`operation_kwargs` value that referenced a renamed item is followed through; where one name split
into several the command says so rather than guessing.

Measured against the 27 obs_data files shipped before this change: 18 load as-is with a
deprecation warning and 9 are refused; after running the command, 25 load with no warning at all.
(The remaining 2 fail on an unrelated `state_or_alg` key that has never been in the schema.)

Without the command, the old keys still load with a `DeprecationWarning` naming their
replacements — `variable` is read as `data_item_name`, `name_for_plotting` as
`trace_name_for_plotting`, in files, in a hand-built dataframe, and in `ObsDataCreator`. Two
things are not automatic:

- **`operands` is required.** An item that relied on `obs_type: min|max|mean` taking its operand
  from `variable` now raises; state the model variable in `operands`.
- **Names must be made unique.** Where one variable carried several features (the mean and the
  max of a trace, or one variable measured across experiments), give each item its own
  `data_item_name` — `"mean flow aortic root"` / `"max flow aortic root"` — and let them keep a
  shared `trace_name_for_plotting`. `ObsDataCreator` now rejects a repeat at the
  `add_data_item` call rather than leaving it to surface at parse time.

Also fixed: the mean of `heart/u_la` in `resources/3compartment_obs_data.json` was labelled
`u_{AR}`, so it plotted as an aortic-root pressure. It is now `u_{LA}`.

Reading `obs_info` from python: `names_for_plotting` remains as a deprecated alias of
`item_names_for_plotting` and was to be removed in 0.6.0 (deferred -- see 0.6.0's note). Prefer `data_item_names`,
`trace_names_for_plotting` or `item_names_for_plotting`.

## 0.4.1 — 2026-08-19

Three bug fixes, all found by running 0.4.0. Nothing about the API or the packaging changed,
so upgrading is a `pip install -U libcuflynx`.

### Fixed — an mpi4py that is imported but not initialised no longer kills the process

`mpi_utils` treated "`mpi4py.MPI` is in `sys.modules`" as "MPI is open", and read the rank
through it. There is a third state: `MPI4PY_RC_INITIALIZE=0` (or `mpi4py.rc.initialize = False`)
loads the library and skips `MPI_Init`. In that state every routine except `MPI_Initialized` and
`MPI_Finalized` is erroneous, and MPICH and Microsoft MPI both answer by printing

```
Attempting to use an MPI routine before initializing MPI
```

and killing the process — not raising, so the `except Exception` around the call could not help.

`PrimitiveParsers` reads `rank = mpi_utils.rank()` at module scope, so *importing* libcuflynx
was enough to hit it. It killed the CUFLynx v0.4.0 Windows release build twice: PyInstaller
imports every bundled package into one isolated child, mpi4py.futures loaded MPI uninitialised,
and the child then died importing `libcuflynx.solver_wrappers`.

`rank`/`size`/`is_root` and `get_MPI` now ask `mpi_is_live()` first and fall back to the
one-rank answers, which are the right ones for a process whose MPI was never opened. Nothing
changes when MPI is genuinely open, and a rank started by a launcher is left alone — N ranks
each believing they are rank 0 would be worse than the abort.

### Fixed — reading a constant back from a Myokit run raised KeyError

`get_results('component/some_constant')` on the `CVODE_myokit` backend raised
`KeyError: 'component.some_constant'` (issue #453). `_make_log` deliberately leaves constants
out of the log — Myokit cannot log something that never varies — while `_resolve_name`
classifies a constant as a `"var"`, like every other non-state. So the read indexed a log that
was never going to contain it.

The arm that answers correctly was already two lines below and simply unreachable while a log
existed: for a `"var"`, evaluate the variable. It now falls through to it.

### Fixed — an unsupported `model_type` says so, instead of failing several frames later

Calibrating a `model_type` that parameter identification cannot run — `cpp` is the case that
gets there by being *valid*, a real model type for generation that neither `param_id` nor
`solver_wrappers` can simulate — surfaced as
`AttributeError: 'CVS0DParamID' object has no attribute 'param_id'`, from inside
`EmulatorTrainer.init_from_dict` (CUFLynx #270). `__init__` built `self.param_id` only for the
supported types and `set_output_dir` then dereferenced it unconditionally, so the failure named
an internal attribute rather than the setting the user chose.

`CVS0DParamID` now checks `model_type` against `PARAM_ID_MODEL_TYPES` up front and says which
part is missing, quoting the list it was checked against.

## 0.4.0 — 2026-08-18

The release that makes the project installable: `pip install libcuflynx` now gives you
generation, simulation and calibration from any directory, with no checkout and nothing to put
on `sys.path`. The distribution is named `libcuflynx` and every module moved under the
`libcuflynx` namespace.

### Breaking — the flat import names are deprecated, and go away in 0.5.0

`import parsers`, `from param_id.paramID import CVS0DParamID` and the rest of the flat names
still work in 0.4.0: each is a shim that emits one `DeprecationWarning` and then hands back the
real `libcuflynx.*` package — the *same* module object, so `isinstance` checks and
monkeypatching against classes reached through the old name are unaffected. **They are removed
in 0.5.0**, which is one full release of overlap. Migrate by prefixing the import:

```python
from param_id.paramID import CVS0DParamID            # 0.4.0: warns
from libcuflynx.param_id.paramID import CVS0DParamID  # do this instead
```

### Added — a console command per pipeline stage

Every stage now has a command that setuptools puts on `PATH`, so nothing has to know where the
package was installed:

| Command | Stage |
|---|---|
| `cuflynx-generate` | generate a model from the CSV arrays |
| `cuflynx-param-id` | generate + calibrate (`mpiexec -n N` for parallel) |
| `cuflynx-sequential-param-id` | staged calibration — declared, not yet implemented |
| `cuflynx-sensitivity` | Sobol sensitivity analysis |
| `cuflynx-identifiability` | Laplace / profile-likelihood identifiability |
| `cuflynx-train-emulator` | train a surrogate of the obs features |
| `cuflynx-plot` | plot calibration results |

Each takes `--help` and no options beyond the optional `True|False` the two generation
launchers pass; the configuration is still `user_run_files/user_inputs.yaml`.
`user_run_files/*.sh` invoke these commands rather than `${python_path}
../src/libcuflynx/scripts/<file>.py`, which stopped being a real path the moment libcuflynx
was installed anywhere but a checkout — **so `pip install -e .` (or `pip install libcuflynx`)
is now a prerequisite for every launcher**, and `cuflynx_entry_point.sh` says so, before
`mpiexec` is reached, rather than letting rank 3 raise an ImportError an hour into a queue.
Set `CUFLYNX_PYTHON` to run a stage under a specific interpreter without putting its `bin/`
on `PATH`. Three utilities without a console command —
`read_and_insert_parameters.sh`, `run_multiple_param_id.sh`, `run_module_generator.sh` — go
through `python -m libcuflynx.scripts.<module>` instead.

### Added — `CUFLYNX_USER_DIR`, and nothing is written inside the package

A dozen modules used to compute "the repo root" from their own location, which resolves
inside `site-packages` once installed — a meaningless place to look for a user's model
inputs, and a writable one, so per-run artefacts were scattered through the installed package
(#431). Inputs and outputs now default under a *user directory*: `$CUFLYNX_USER_DIR` if set,
otherwise the checkout being run from if this is one, otherwise the current working
directory. `resources_dir`, `generated_models_dir`, `param_id_output_dir` and
`external_modules_dir` in `user_inputs.yaml` still override individually. The CellML module
library and the other non-Python files ship as *package data* and are located with
`importlib.resources` (#432).

### Changed — the heavy dependencies are extras

A default `pip install libcuflynx` brings what a generate + simulate + calibrate run needs
and nothing else (~540 MB of site-packages). The rest are extras, each named by the error you
get when it is missing (#435):

| Extra | Brings | Why it is optional |
|---|---|---|
| `[mpi]` | mpi4py, schwimmbad | only 5 MB, but it compiles against a system MPI at install time — the commonest `pip install` failure on macOS and Windows. A serial run needs none of it. |
| `[casadi]` | casadi | 221 MB; only `model_type: casadi_python`, `solver: casadi_integrator` and symbolic-adjoint AD use it |
| `[uq]` | pymc, arviz | 65 MB and pytensor builds C extensions; the built-in emcee sampler needs neither |
| `[emulation]` | autoemulate | ~750 MB of torch/gpytorch/lightgbm, and requires Python >=3.10,<3.13 |
| `[cpp]` | *(nothing)* | `model_type: cpp` needs a toolchain, not a Python package |
| `[all]` | every runtime extra | inherits `[emulation]`'s narrower Python range |

`requires-python` is `>=3.9` — the floor `importlib.resources.files` actually needs. It
previously claimed `>=3.7`, on which `pip install` succeeded and the first generator import
then failed.

### Docs — every documented import is now `libcuflynx.`

The tutorial (`tutorial/docs/`, published at
<https://physiomelinks.github.io/circulatory_autogen/>), the interactive notebooks, the README
and the shipped `example_format_obs_data_json_file.py` all import from the `libcuflynx`
namespace, install with `pip install libcuflynx`, and no longer put anything on the import path.
The API reference's mkdocstrings identifiers moved with them
(`::: libcuflynx.param_id.paramID.CVS0DParamID`). The README carries the measured install sizes
per extra, and says in as many words that the repository is `circulatory_autogen` while the
package is `libcuflynx` — cite the former, `pip install` the latter.

### Renamed — `model_type: cellml_only` is now `model_type: cellml`

The `_only` distinguished it from nothing. `cellml` is the default `model_type`, so this is the
most-read name in `user_inputs.yaml`. **The old spelling still works**: it is what every config
written before this release says — including the dated copies `save_dated_user_inputs` archives
beside every run — so `cellml_only` is translated with a warning naming the replacement, and
will be removed in a later release. `SOLVER_SCHEMA` advertises `cellml` only, so a tool building
its menus from the schema writes the current name into new configs.

### Changed — `cellml` defaults to `CVODE_myokit`

A config that omits `solver` used to be routed to `CVODE_opencor`, the one backend a pip
install cannot provide (OpenCOR's `opencor` module is not on PyPI). The default is now
`CVODE_myokit`, which is a drop-in replacement: same CellML model, same CVODE integrator, no
OpenCOR. `solver: CVODE_opencor` is unchanged and still works inside an OpenCOR install; asking
for it anywhere else now fails with a message naming `CVODE_myokit` rather than a traceback.

### Breaking — action required if you edited `funcs_user/*_funcs_user.py`

**The built-in cost, operation and modifier functions moved into the package** (issue #433).

| Was | Is now |
|---|---|
| `funcs_user/cost_funcs_user.py` | `libcuflynx.funcs.cost_funcs_user` |
| `funcs_user/operation_funcs_user.py` | `libcuflynx.funcs.operation_funcs_user` |
| `funcs_user/modifier_funcs_user.py` | `libcuflynx.funcs.modifier_funcs_user` |

They were library code living outside the package and imported by bare module name
(`import cost_funcs_user`), which only resolved because CA appended the repo's `funcs_user/`
directory to `sys.path`. That cannot work from an installed package, and a library must not
ask its users to edit files inside it.

**If you added your own cost / operation / modifier function by editing one of those files in
place, it will silently stop being registered** — the registry now reads the package copy, and
your edited file is no longer loaded at all. Move your functions into a file of your own and
name it in `user_run_files/user_inputs.yaml`:

```yaml
cost_funcs_external_path:      funcs_user/my_costs.py
operation_funcs_external_path: funcs_user/my_ops.py
modifier_funcs_external_path:  funcs_user/my_modifiers.py
```

Relative paths like those resolve from the repository root (from the directory of the config
named by `user_inputs_path_override` if you use one, and from `$CUFLYNX_USER_DIR` or the
current directory when libcuflynx is pip-installed with no checkout). Absolute paths are
taken as-is.

Nothing else about your functions has to change: the external file's top-level functions are
merged into the *same* registry as the built-ins, with `@is_MLE` / `@cost_combiner` /
`@differentiable` / `@series_to_constant` / `@modifier_func` intact and reported by
`cost_func_metadata()` and `param_modifiers()`; a function reusing a built-in name overrides
it. This mechanism is not new (issues #303, #383) — it is now the only supported way to add
your own.

Templates to copy: `funcs_user/cost_funcs_example.py`, `funcs_user/operation_funcs_example.py`,
`funcs_user/modifier_funcs_example.py`. See `funcs_user/README.md`.

Imports of the *shipped* functions change accordingly — `from funcs_user.cost_funcs_user import
cost_func_metadata` becomes `from libcuflynx.funcs.cost_funcs_user import cost_func_metadata`.
For external funcs files, the old bare-name imports (`from cost_funcs_user import is_MLE`)
still work: CA aliases the three legacy module names onto the package modules when it loads an
external file. New files should import `from libcuflynx.funcs.cost_funcs_user import is_MLE`.

`funcs_user/` itself stays, and is now purely yours: the three templates above, plus the
unrelated `model_type: external_python` examples (`example_model_scipy/`,
`example_model_external/`, `heat_fenics/`).
