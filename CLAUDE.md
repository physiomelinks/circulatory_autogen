# circulatory_autogen — Agent Guide

Python framework for **automated generation and parameter identification of computational physiology models** (CellML-based circulatory / electrophysiology models). It turns module/vessel CSV arrays into flat CellML (or Python/C++/CasADi) models, then calibrates and analyses them.

This file documents the conventions, gotchas, and entry points an agent needs that are **not** obvious from the code. Don't re-explain what the code already makes clear; do read the referenced files before changing behavior around them.

## Build / run / test

**Do not use OpenCOR's `pythonshell` — run everything in a normal venv.** The bundled interpreter is deprecated and going away: it will be replaced by a plain `pip install libopencor` into a standard Python env once libOpenCOR reaches PyPI, and `python_path.sh` / `opencor_pythonshell_path.sh` will be removed with it. Don't add code paths, docs or test setups that assume it.

- **Set up**: `python -m venv venv && venv/bin/pip install -e ".[dev]"`. Nothing in the default path needs OpenCOR — the default solver is `CVODE_myokit`, and `CVODE_opencor` is the only backend that does.
- **`user_run_files/python_path.sh` sets `python_path`**, the interpreter every `user_run_files/*.sh` (including `run_pytest.sh`) invokes. **Point it at your venv** — `python_path=/abs/path/to/circulatory_autogen/venv/bin/python` — not at a `pythonshell`. The runner is otherwise interpreter-agnostic; there is nothing OpenCOR-specific left in it. Note `python_path.sh` does **not** source `opencor_pythonshell_path.sh`: that file is legacy and unread, and is on the same removal list.
- `./run_pytest.sh` — full suite, 1 MPI rank.
- `./run_pytest.sh -n 4 -v -s` — `-n N` sets **MPI rank count** (it is *not* pytest-xdist; xdist is force-disabled with `-p no:xdist` because ranks and xdist workers conflict).
- `./run_pytest.sh -m "not slow"` / `-m "not compare_optimisers"` — deselect expensive tests. `-k <expr>` and other args pass straight through to pytest.
- Equivalent without the script (useful when `python_path.sh` holds someone else's path): `mpiexec -n 1 venv/bin/python -m pytest -p no:xdist <args>`.
- **Without OpenCOR installed, deselect the tests that truly need it**: `-m "not need_opencor"`. `need_opencor` is a plain marker with **no auto-skip**, so those tests *fail* rather than skip if OpenCOR is absent — that is expected, not a regression.
- Editable install: `pip install -e ".[dev]"` (the test runner auto-installs dev deps into whatever `python_path` points at, if pytest is missing there).
- Legacy MPI gotcha, only if you are still on a pythonshell: OpenCOR bundles a **dual-ABI `mpi4py`** (`MPI.mpich.*.so` **and** `MPI.openmpi.*.so`, dispatched by `mpi4py._mpiabi`). If the variant picked at import doesn't match the MPI owning the system `mpiexec`, every rank aborts at `MPI_Init` with `unsupported PMI version PMIx`. Fix by pinning the ABI to the installed launcher — `export MPI4PY_MPIABI=openmpi` (or `mpich`) — **not** by installing a second MPI. A venv's pip-installed `mpi4py` links the single system MPI and avoids the ambiguity entirely; this is one more reason to move off the pythonshell.
- pyproject.toml holds deps, pytest config, markers, black (line-length 100), coverage. Python `>=3.7`.

## How users actually drive it

Runs are launched via shell scripts in `user_run_files/`. Each one invokes a **console command** declared in `[project.scripts]` — `mpiexec -n 4 cuflynx-param-id`, not a path into `src/` — so a launcher never has to know where the package was installed. That means **`pip install -e .` is a prerequisite for every one of them**; without it the command is not on `PATH` and `user_run_files/cuflynx_entry_point.sh` says so before `mpiexec` is reached. All read config from **`user_run_files/user_inputs.yaml`** (overridable via `user_inputs_path_override` in that file).

| Shell script (`user_run_files/`) | Command → `libcuflynx.scripts.` | Purpose |
|---|---|---|
| `run_autogeneration.sh` | `cuflynx-generate False` → `script_generate_with_new_architecture` | Generate model from CSV arrays |
| `run_autogeneration_with_id_params.sh` | `cuflynx-generate True` → (same) | Regenerate using previously fitted params |
| `run_param_id.sh` (arg: `num_processors`, uses `mpiexec`) | `cuflynx-param-id` → `param_id_run_script` | Generate + calibrate |
| `run_sequential_param_id.sh` | `cuflynx-sequential-param-id` → `sequential_param_id_run_script` | Staged/sequential calibration — **not currently implemented**, the `SequentialParamID` class it drives is not in the tree; the command says so and exits 2 |
| `run_multiple_param_id.sh` | *(no command yet)* → `run_multiple_param_id.py` by path | Batch calibration over models |
| `run_sensitivity_analysis.sh` | `cuflynx-sensitivity` → `sensitivity_analysis_run_script` | Sobol SA (`mpiexec`) |
| `run_identifiability_analysis.sh` | `cuflynx-identifiability` → `identifiability_run_script` | Laplace / profile-likelihood |
| `run_emulator_training.sh` (arg: `num_processors`, uses `mpiexec`) | `cuflynx-train-emulator` → `train_emulator_run_script` | Train a surrogate of the obs features |
| `plot_param_id.sh` | `cuflynx-plot` → `plot_param_id_script` | Plot calibration results |

Every command takes `--help` and no options beyond the optional `True|False` the two generation launchers pass; the configuration is the yaml. Each is a `main()` in the named module — `tests/test_console_entry_points.py` is table-driven off `[project.scripts]`, so a new entry point cannot be added without being tested.

`python_path.sh` / `opencor_pythonshell_path.sh` are **not** sourced by these launchers any more; they remain for the OpenCOR route and for `run_pytest.sh`. To run a stage under a specific interpreter without putting its `bin/` on `PATH`, set `CUFLYNX_PYTHON` and the launcher uses `$CUFLYNX_PYTHON -m libcuflynx.scripts.<module>` instead.

Other useful scripts in `src/libcuflynx/scripts/`: `generate_obs_json.py`, `example_format_obs_data_json_file.py`, `generate_modules_files.py`, `convert_0d_to_1d.py`, `read_and_insert_parameters.py`, `generate_omex_analysis_script.py`.

## Calling from Python (programmatic API)

The whole pipeline can be driven directly from Python instead of the shell scripts — this is how the interactive tutorials work (`tutorial/interactive/generation_and_calibration.ipynb`). **Import from the `libcuflynx` namespace and put nothing on `sys.path`**: an installed package (`pip install libcuflynx`, or `pip install -e .` on a checkout) is importable from any directory. The flat names (`import solver_wrappers`, `from param_id.paramID import ...`) still resolve in 0.4.0 as deprecation shims that warn, and are **removed in 0.5.0** — don't write new code or docs against them. Nothing here needs OpenCOR: the default solver is `CVODE_myokit`.

Instead of editing `user_inputs.yaml`, build the config dict in code and mutate it:
```python
from libcuflynx.utilities.utility_funcs import get_default_inp_data_dict
inp = get_default_inp_data_dict(file_prefix, input_param_file, resources_dir)  # == user_inputs.yaml defaults
inp["sim_time"], inp["pre_time"] = 2, 20.0
inp["DEBUG"] = True
```
Then call the same stages the scripts call, all taking that dict:
- **Generate**: `from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture` → `generate_with_new_architecture(inp_data_dict=inp)` (returns success bool).
- **Simulate**: `from libcuflynx.solver_wrappers import get_simulation_helper_from_inp_data_dict` → `sim = get_simulation_helper_from_inp_data_dict(inp)`; `sim.run()`, `sim.get_results(names, flatten=True)`, `sim.get_time()`.
- **Calibrate**: `from libcuflynx.param_id.paramID import CVS0DParamID` → `pid = CVS0DParamID.init_from_dict(inp)`; then `set_ground_truth_data(obs)`, `set_params_for_id(params_for_id_dict)`, `set_param_id_method(...)`, `set_optimiser_options(...)`, `run()`, `simulate_with_best_param_vals()`, `plot_outputs()`; results under `pid.output_dir`.
- **Sensitivity**: `from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis` → `SensitivityAnalysis.init_from_dict(inp)`; `set_ground_truth_data`, `set_params_for_id`, `set_sa_options`, `run_sensitivity_analysis(sa_options)`, `choose_most_impactful_params_sobol(top_n=..., index_type='ST', ...)`.
- **Build obs data in code**: `from libcuflynx.utilities.obs_data_helpers import ObsDataCreator` → `add_protocol_info(pre_times, sim_times, params_to_change, offline_pre_time=...)`, `add_data_item(entry)`, `get_obs_data_dict()` — produces the same structure as an `obs_data.json` file.
- **Custom features**: register a Python function with `add_user_operation_func(fn)` on the param-id / SA object, then reference it by name in a data item's `"operation"` (operands map to the function args). Set `fn.series_to_constant = True` for series→scalar features so auto-plotting works.

`params_for_id_dict` is a list of `{vessel_name, param_name, min, max, name_for_plotting}` (the in-memory equivalent of `{prefix}_params_for_id.csv`); `vessel_name` may be a single name or a list to share one calibrated param across many vessels.

## `user_inputs.yaml` — key fields

- `file_prefix` — model name; ties together `{prefix}_vessel_array.csv`, `{prefix}_parameters.csv`, `{prefix}_obs_data.json` in `resources/`.
- `model_type` — `cellml_only` (default) | `python` | `casadi_python` | `cpp` | `aadc_python` | `external_python`.
- `solver` — `CVODE_myokit` (default) | `CVODE_opencor` | `solve_ivp` (python models) | `casadi_integrator` (casadi_python models) | `RK4_cpp` | `external` (external_python).
- `solver_info` — `MaximumStep`, `MaximumNumberOfSteps`, and `method` (e.g. `RK45` for solve_ivp; `cvodes`/`idas`/`collocation`/`rk` for CasADi). Validated — see `tests/test_solver_info_validation.py`.
- **The user-supplied model type.** `external_python` is the **one** way to bring your own Python model: a **solver class** that does its own time stepping — an FE/FV code, a compiled library, or a plain `scipy.solve_ivp` call — which CA only wraps (`external_model_path`, default `funcs_user/{prefix}_model.py`). The file must define `SIM_HELPER = YourClass`, and the class must declare literal `parameters` / `output_names` attributes (read by AST, without importing) plus `init_solver` / `update_times` / `set_param_vals` / `run` / `get_results`; `get_init_param_vals` / `reset` / `extra_plots` / `close` are optional. Its only solver_info setting is `user_config`, a free-form dict handed to `init_solver`. There is no generation step. Two shipped examples: `funcs_user/example_model_scipy/` (a damped oscillator integrated by scipy inside `run()` — the simple case) and `funcs_user/example_model_external/` (a 1D heat equation with a hand-written scheme); `funcs_user/heat_fenics/` is the heavyweight FEniCSx one. See `src/libcuflynx/solver_wrappers/external_simulation_helper.py` for the full contract and `tutorial/docs/external-python-solvers.md` for the how-to.
- `pre_time` / `sim_time` / `dt` — steady-state spin-up, logged simulation duration, output sampling step. `dt` must be ≤ every dt in the obs_data.json. (Timeline can also be set per-experiment in obs_data.json; see below.)
- `param_id_method` — `genetic_algorithm` | `CMA-ES` | `bayesian` | `sp_minimize`. x0 auto-loaded from `{prefix}_parameters.csv`.
- `optimiser_options` / `debug_optimiser_options` — when `DEBUG: true`, the debug block overrides. `cost_type` (e.g. `gaussian_MLE`) selects the cost function from the registry (`libcuflynx.funcs.cost_funcs_user` plus any `cost_funcs_external_path`).
- Feature flags: `do_ad`, `do_sensitivity` (`sa_options`), `do_mcmc` (`mcmc_options`), `do_ia` (`ia_options`).
- Path overrides (recommended for real work, to keep inputs/outputs outside the repo): `resources_dir`, `generated_models_dir`, `param_id_output_dir`, `external_modules_dir`.
- `operation_funcs_external_path` / `cost_funcs_external_path` — optional paths to external Python files with extra user operation / cost funcs, loaded and registered alongside the built-ins the package ships in `src/libcuflynx/funcs/` (same decorators). Since #433 this is the **only** supported way to add your own: the built-in modules are library files that an upgrade replaces, so editing them in place silently loses the edits. Absent/empty is a no-op; templates in `funcs_user/*_funcs_example.py`, migration note in `CHANGELOG.md`. Threaded from the config into the param-id, MCMC and sensitivity runs via `scriptFunctionParser(operation_funcs_external_path=…, cost_funcs_external_path=…)`.
- `modifier_funcs_external_path` — same pattern for **modifier functions** (issue #383): a params_for_id `modifies` entry's `modifier` key names a registered function `p_i = fn(theta, baseline_i, **inputs)` (built-ins `scale`, `remainder`; user funcs go in an external file via this key, decorated `@modifier_func(inputs={name: 'float'|'list'})`). Entry `inputs` supply model qnames whose *defaults* the function receives (resolved once, like baselines). Functions must be affine in theta (probed at setup) so the FSA chain-rule weight `a = dp/dtheta` is constant and theta's x0 comes from inverting at the baseline. Works on `cellml_only` + `CVODE_myokit` (FSA) **and** `casadi_python` (symbolic AD): the CasADi arm takes one symbol per *member* and folds its per-member jacobian with the same affine weights, so both backends report the same d/dtheta. Discoverable via `param_modifiers(external_path)` — each record carries `description`/`inputs`/`user_defined`. Vocabulary: an **operation** acts on an *output* (obs_data), a **modifier** acts on a *parameter* (params_for_id); the #378 `operation` entry key is a deprecated alias that warns.

**Never commit `resources/user_inputs_<yymmdd>.yaml`.** Every run archives its resolved config there via `save_dated_user_inputs` (`src/libcuflynx/parsers/PrimitiveParsers.py`), so these files appear constantly as modified or untracked — they are per-machine run artifacts, not inputs. They bake in absolute local paths (`resources_dir`, `generated_models_dir`, and `/tmp/pytest-*` dirs when a test run writes them), so committing them adds churn and leaks one machine's layout into the repo. They are gitignored; leave them untracked, and do not `git add -A` them back in. The same goes for anything else that shows up modified purely because you ran the suite.

- `do_emulation` / `use_emulator` + `emulator_settings` — **emulators (surrogate models, issue #333)**. The emulator maps theta (the params_for_id vector, one slot per entry) to the *scalar* features of the obs_data data_items, i.e. the value after the `operation` reduction — the same numbers the cost uses. `do_emulation` trains one against the solver named by `solver`; `use_emulator` makes calibration/SA/UQ/IA evaluate it. `solver` deliberately keeps meaning the truth solver, so an emulator run can be compared with it. Backend is the optional `autoemulate` (`pip install ".[emulation]"`, needs Python >=3.10,<3.13). Series/frequency observables, prediction traces and output plots are **refused** in emulator mode, not approximated.

## Discoverable schemas (keep settings machine-readable for CUFLynx)

The GUI front-end **CUFLynx** auto-populates its menus and settings forms by reading discoverable schemas in `src/libcuflynx/parsers/PrimitiveParsers.py` — it does **not** hardcode the options. **Whenever you add or change a user-configurable setting, update the matching schema in the same PR**, or CUFLynx silently won't expose it:

- `SOLVER_SCHEMA` — model types (including `external_python`, whose solver `external` has the single placeholder method `external`: the user's class owns its scheme, so there is nothing for CA to offer), solvers per model type, integrator `method`s per solver, and `solver_info_fields_by_solver` (`SOLVER_INFO_FIELDS`: the `solver_info` settings per solver). `_SOLVER_INTEGRATOR_KEYS` (validation) is **derived** from `SOLVER_INFO_FIELDS`, so add a solver_info field there. Also carries per-integrator analytic-gradient suitability for a tool's Gradient menu: `ad_suitable_methods` (casadi_integrator methods that support CasADi AD — **derived** from `_CASADI_ADJOINT_METHODS`, i.e. every method except the adjoint `cvodes`/`idas`), `fsa_suitable_methods` (Myokit CVODES FSA methods per solver), and `default_method_by_solver` (advisory default integrator, e.g. `casadi_integrator` → `bdf`).
- `PARAM_ID_METHODS` — calibration methods, each with an `options` list = the `optimiser_options` it reads. Add a new optimiser knob here.
- `gradient_sources(model_type, solver, method=None)` — the gradient sources (FD / AD / FSA) the gradient-based methods (`sp_minimize`, `multi_start_sp_minimize`) can use for a given model, each with the `do_ad` flag it implies. There is **no** per-method "gradient" option: AD vs FD is the top-level `do_ad` flag, and which analytic backend runs follows from `model_type`/`solver` (mirrors `OpencorParamID.get_gradient`). When the integrator `method` is passed, the analytic source is additionally gated on `ad_suitable_methods`/`fsa_suitable_methods` (e.g. no CasADi AD for `cvodes`/`idas`). Keep it in step with that dispatch.
- `ANALYSIS_OPTIONS` — the `sa_options` / `mcmc_options` / `ia_options` / `emulator_settings` settings for the sensitivity / MCMC / identifiability / emulation modes. `emulation` is the only entry with a **`use_flag`** as well as an `enable_flag`, because it has a train step and a use step. `gradient_sources(..., use_emulator=True)` collapses to FD alone (the analytic arms differentiate the real model, which an emulator run is not evaluating). Emulator model names are a runtime registry like the cost funcs — `libcuflynx.emulators.emulator_trainer.emulator_model_names()`.
- Cost functions are a runtime registry (user-extensible), so they're discovered via `libcuflynx.funcs.cost_funcs_user.cost_func_metadata()` (names + `is_MLE`/`is_combiner`/`differentiable` flags), not a static schema. Both this and the operation-funcs dict take an optional `external_path` (and `scriptFunctionParser` an `operation_funcs_external_path` / `cost_funcs_external_path`) so the merged set — including funcs from an external file — is introspectable.

Each setting is a descriptor `{name, type, default, required, description, choices?}`. Accessors: `solver_info_fields(solver)`, `param_id_method_options(method)`, `analysis_options(mode)`, `gradient_sources(model_type, solver, method=None)`. Tests in `tests/test_solver_info_validation.py` lock every schema's shape **and** its correspondence to what the code actually reads (e.g. an option the optimiser doesn't read, or a read the schema omits, fails the suite) — extend those when you add settings.

## Source layout (`src/libcuflynx/`)

| Dir | Contents / purpose |
|---|---|
| `solver_wrappers/` | `SimulationHelper` backends + `get_simulation_helper()` factory (`__init__.py`). Backends: `myokit_helper.py`, `opencor_helper.py`, `python_solver_helper.py`, `casadi_python_solver_helper.py`, `emulator_solver_helper.py` (answers from a trained emulator; `emulates_features = True` tells the two reduction sites to skip the obs `operation`), `external_simulation_helper.py` (wraps a user-supplied solver class for `model_type: external_python`; the wrapper owns the timeline, the user owns the stepping). `name_resolver.py` maps variable names. |
| `generators/` | `CVSCellMLGenerator.py`, `PythonGenerator.py` (libCellML Analyser, strict ODE), `CVSCppGenerator.py`, `Python1DModelFilesGenerator.py`. |
| `param_id/` | `paramID.py` (calibration), `optimisers.py`, `differentiable.py` + `math_backend.py` + `operation_funcs.py` (AD), `plot_outputs.py`. |
| `protocol_runners/` | `protocol_runner.py`, `protocol_executor.py` — the multi-experiment/sub-experiment simulation loop. |
| `sensitivity_analysis/` | `sensitivityAnalysis.py`, `sobolSA.py`. |
| `emulators/` | `emulator_trainer.py` (design -> simulate -> fit -> validate -> persist; `emulator_settings.reuse_samples` skips the first two and refits the samples a previous run saved, refusing when their fingerprint no longer matches the run at hand), `emulator_bundle.py` (the artefact + every refusal rule, plus `error_stats()` / `error_points()` -- the per-feature held-out statistics and the held-out points themselves, which is what an error analysis is drawn from). Backend: optional `autoemulate`. |
| `identifiabilty_analysis/` | `identifiabilityAnalysis.py` (note the dir is spelled `identifiabilty`). |
| `parsers/` | `ModelParsers.py`, `PrimitiveParsers.py`, `OMEXParsers.py` — CSV/YAML/JSON/OMEX loading. |
| `models/` | `LumpedModels.py` (`CVS0DModel`). `checks/LumpedModelChecks.py` validates structure/connectivity. |
| `funcs/` | The built-in cost / operation / modifier funcs the library ships (`cost_funcs_user.py`, `operation_funcs_user.py`, `modifier_funcs_user.py`). Library files — users add their own via the `*_funcs_external_path` config keys, never by editing these (#433). |
| `utilities/` | `utility_funcs.py`, `protocol_funcs.py`, `libcellml_utilities.py`, `obs_data_helpers.py`, `diagnostics.py`, plotting helpers. |
| `scripts/` | Entry points (see table above). |
| `coupler/`, `solver1d/` | 0D–1D coupling / 1D solver (in development). |
| `obsolete/` | Dead code — don't extend. |

User-extensible (kept outside the package): `module_config_user/` (custom CellML modules), `funcs_user/` (your own cost / operation / modifier funcs, plus the `external_python` model examples — see `funcs_user/README.md`). `resources/` holds input CSVs and obs_data.json; `generated_models/` is build output.

## `SimulationHelper` API (common across backends)

`get_simulation_helper(...)` in `src/libcuflynx/solver_wrappers/__init__.py` returns the backend for the configured `solver`. Common methods: `run()`, `update_times(...)`, `get_results(var_names)`, `get_all_results()`, `get_all_variable_names()`, `get_init_param_vals()`, `set_param_vals(names, values)`. When adding a backend, implement this full surface and register it in the factory.

## `obs_data.json` — experiment descriptor

```json
{
  "protocol_info": {
    "pre_times":  [0.0, 0.0],
    "sim_times":  [[5], [5]],
    "params_to_change": { "component/param": [[exp0_sub0, …], [exp1_sub0, …]] },
    "protocol_traces": { "trace_key": {"t": [...], "values": [...]} },
    "protocol_shapes": { "trace_key": {"events": [{"level": 1.0, "start": 100,
                                                   "length": 2, "period": 1000,
                                                   "multiplier": 0}]} }
  },
  "data_items": [...],
  "prediction_items": [...]
}
```
A `params_to_change` value is a **float** (constant) or a **string** (trace key into `protocol_traces`). Series entries currently must have a `std` set (single-likelihood assumption — see commit history).

`protocol_shapes` is the **same waveform written as Myokit `[[protocol]]` events** rather than as a table of points, in Myokit's own field names (`level` / `start` / `length` / `period` / `multiplier`, plus an optional `baseline`) — so a protocol imported from a `.mmt` needs no translation. `libcuflynx.utilities.protocol_shapes.materialise_shapes` expands them **into `protocol_traces`** at parse time (and again, idempotently, in `set_protocol_info`), so **nothing downstream needs to know shapes exist** — keep it that way and add new shape types there rather than teaching the solvers about them. A shape is sized by the sub-experiment that references it unless it names its own `duration`; a name belongs to `protocol_traces` **or** `protocol_shapes`, not both.

**Timeline conventions** (subtle — get these right):
- `pre_times[j]` is the **unlogged pre-pass** before the first sub-experiment of experiment `j`; `sim_times[j][k]` is the duration of sub-experiment `(j, k)`.
- `protocol_traces[...].t` are **seconds from the start of the Myokit segment** where the trace applies; match or exceed the segment length in `sim_times[j][k]`.
- Myokit (`myokit_helper`): each `update_times` calls `simulation.reset()`; logging instants are shifted so the first requested output time aligns with `simulation.time()` after `pre(pre_time)`.
- **`offline_pre_time`**: a generic steady state can be reached **offline** once and reused, so per-run `pre_time` only needs to cover parameter-specific settling — speeds up calibration. See obs_data / tutorial docs for `offline_pre_time`, `val_path`, `t_path`.

## Backend caveats

- **SN_simple / SN_full**: `cellml_only` generation + **Myokit** accept the emitted CellML (including state initial values referencing `*_init` params). **`PythonGenerator`** uses libCellML Analyser and requires a strict **ODE** model; the same SN CellML fails `ANALYSER_VARIABLE_NON_CONSTANT_INITIALISATION`, so `model_type: python` codegen is **not expected to work for SN** until the generator/model satisfies the analyser. The two tests that hit this (`test_generate_python_model_succeeds[SN_simple…]`, `test_python_BDF_solver[SN_simple…]`) are `xfail(strict=True)`, so a libCellML upgrade that fixes it turns them into an `XPASS(strict)` **failure** telling you to drop the marker — rather than leaving two permanently-red tests nobody reads.
- **CasADi**: piecewise/conditional models emit `ca.if_else` so they stay symbolically differentiable (needed for AD-based param ID). See `tests/test_casadi_conditionals.py`.

## Testing (required for every change)

Add/extend tests in `tests/` for **every feature and bugfix**; a bugfix should include a test that fails before the fix.

| Test file | Covers |
|---|---|
| `test_solvers.py` | All four solver backends (markers: `solver`, `need_opencor`) |
| `test_autogeneration.py` | CSV → model generation (markers: `integration`, `slow`, `autogen_rank(idx)`) |
| `test_param_id.py` | Calibration across optimisers (`integration`, `slow`, `mpi`, `compare_optimisers`) |
| `test_sensitivity_analysis.py` | Sobol SA (`integration`, `mpi`) |
| `test_emulator_settings.py`, `test_emulator_solver_helper.py` | Emulator artefact, refusals and the cost-path seam, with a stub emulator (`unit`, no autoemulate) |
| `test_emulator_training.py` | Emulator training end-to-end (`integration`, `slow`, `mpi`; the real fit is `importorskip('autoemulate')`) |
| `test_external_simulation_helper.py`, `test_scipy_ode_example.py` | The user-supplied model type, end to end (`unit` for the contract checks; `integration`/`slow`/`mpi` for calibration, SA and local sensitivities). |
| `test_protocol_funcs.py`, `test_unit_conversion.py`, `test_solver_info_validation.py`, `test_casadi_conditionals.py` | Unit-level (`unit`) |
| `test_omex_analysis_pipeline.py` | OMEX/SED-ML pipeline (`integration`, `misc_task`) |

- Markers are declared in pyproject.toml (`--strict-markers` is on, so an unregistered marker fails the run). Notable: `slow`, `integration`, `unit`, `mpi`, `solver`, `need_opencor`, `compare_optimisers`, and the rank/task-coordination markers (`one_rank_task`, `autogen_rank`, `solver_task`, `misc_task`) used for MPI ordering.
- **`manual`** is stronger than `slow`: those tests are **not collected at all** without `--run-manual`, so they run neither in CI nor in a default local run. It is for a test that is too expensive to justify on every run *and* has a faster stand-in — the full-model UQ posterior-recovery tests in `test_UQ.py` (~80 min; the bimodal one alone took 56 on CI) versus `test_uq_on_emulator.py`, which makes the same assertions through an emulator in ~5. Reach for `manual` only with that pairing; a slow test with no cheaper equivalent should stay `slow` and keep running.
- Fixtures in `tests/conftest.py`: `base_user_inputs`, `resources_dir`, `temp_output_dir`, `temp_generated_models_dir`, `mpi_comm`.
- Reuse existing CellML models / obs_data.json patterns from `resources/`; put new fixtures under `tests/test_inputs/` when needed.

## Docs

- Tutorial (authoritative how-to): https://physiomelinks.github.io/circulatory_autogen/ — source under `tutorial/docs/` (`getting-started`, `design-model`, `parameter-identification`, `sensitivity-analysis`, `identifiability-analysis`, `other-features`, `running-on-hpc`).
- AI wiki (orientation only): https://deepwiki.com/FinbarArgus/circulatory_autogen/1-overview
- Main dev branch is `devel`; PRs target `master`.
- This repo is a fork (`origin` = your fork, e.g. `FinbarArgus/circulatory_autogen`). **PRs must be opened against the upstream `physiomelinks/circulatory_autogen` repo** (the `upstream` remote), targeting its `master` branch — not against the fork. With `gh`, pass `--repo physiomelinks/circulatory_autogen` and a cross-fork head ref like `FinbarArgus:<branch>`. Note `gh pr create` can fail to resolve refs on a fork ("Head sha can't be blank"); the REST API works: `gh api repos/physiomelinks/circulatory_autogen/pulls -f head=FinbarArgus:<branch> -f base=master -f title=... -f body=...`.
