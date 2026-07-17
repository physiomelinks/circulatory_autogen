# circulatory_autogen — Agent Guide

Python framework for **automated generation and parameter identification of computational physiology models** (CellML-based circulatory / electrophysiology models). It turns module/vessel CSV arrays into flat CellML (or Python/C++/CasADi) models, then calibrates and analyses them.

This file documents the conventions, gotchas, and entry points an agent needs that are **not** obvious from the code. Don't re-explain what the code already makes clear; do read the referenced files before changing behavior around them.

## Build / run / test

- **Tests MUST run under the OpenCOR Python shell** — not the system Python. Always use `./run_pytest.sh`, which sources `user_run_files/python_path.sh` and runs pytest under `mpiexec`. The path to the OpenCOR python is in `user_run_files/opencor_pythonshell_path.sh` (sourced by `python_path.sh`).
  - `./run_pytest.sh` — full suite, 1 MPI rank.
  - `./run_pytest.sh -n 4 -v -s` — `-n N` sets **MPI rank count** (it is *not* pytest-xdist; xdist is force-disabled with `-p no:xdist` because ranks and xdist workers conflict).
  - `./run_pytest.sh -m "not slow"` / `-m "not compare_optimisers"` — deselect expensive tests.
  - `-k <expr>` and other args pass straight through to pytest.
- Editable install: `pip install -e ".[dev]"` (the test runner auto-installs dev deps into the OpenCOR env if pytest is missing).
- pyproject.toml holds deps, pytest config, markers, black (line-length 100), coverage. Python `>=3.7`.

## How users actually drive it

Runs are launched via shell scripts in `user_run_files/`, which call entry-point scripts in `src/scripts/`. All read config from **`user_run_files/user_inputs.yaml`** (overridable via `user_inputs_path_override` in that file).

| Shell script (`user_run_files/`) | Entry script (`src/scripts/`) | Purpose |
|---|---|---|
| `run_autogeneration.sh` | `script_generate_with_new_architecture.py` | Generate model from CSV arrays |
| `run_autogeneration_with_id_params.sh` | (same) | Regenerate using previously fitted params |
| `run_param_id.sh` (arg: `num_processors`, uses `mpiexec`) | `param_id_run_script.py` | Generate + calibrate |
| `run_sequential_param_id.sh` | `sequential_param_id_run_script.py` | Staged/sequential calibration |
| `run_multiple_param_id.sh` | `run_multiple_param_id.py` | Batch calibration over models |
| `run_sensitivity_analysis.sh` | `sensitivity_analysis_run_script.py` | Sobol SA (`mpiexec`) |
| `run_identifiability_analysis.sh` | `identifiability_run_script.py` | Laplace / profile-likelihood |
| `plot_param_id.sh` | `plot_param_id_script.py` | Plot calibration results |

Other useful scripts in `src/scripts/`: `generate_obs_json.py`, `example_format_obs_data_json_file.py`, `generate_modules_files.py`, `convert_0d_to_1d.py`, `read_and_insert_parameters.py`, `generate_omex_analysis_script.py`.

## Calling from Python (programmatic API)

The whole pipeline can be driven directly from Python instead of the shell scripts — this is how the interactive tutorials work (`tutorial/interactive/generation_and_calibration.py` / `.ipynb`, run under OpenCOR's Jupyter: `./path/to/OpenCOR/jupyternotebook ...`). Same OpenCOR-Python requirement applies. Add `src/` to `sys.path` first, since the package imports are relative to `src/` (e.g. `import solver_wrappers`, `from param_id.paramID import ...`).

Instead of editing `user_inputs.yaml`, build the config dict in code and mutate it:
```python
from utilities.utility_funcs import get_default_inp_data_dict
inp = get_default_inp_data_dict(file_prefix, input_param_file, resources_dir)  # == user_inputs.yaml defaults
inp["sim_time"], inp["pre_time"] = 2, 20.0
inp["DEBUG"] = True
```
Then call the same stages the scripts call, all taking that dict:
- **Generate**: `from scripts.script_generate_with_new_architecture import generate_with_new_architecture` → `generate_with_new_architecture(inp_data_dict=inp)` (returns success bool).
- **Simulate**: `from solver_wrappers import get_simulation_helper_from_inp_data_dict` → `sim = get_simulation_helper_from_inp_data_dict(inp)`; `sim.run()`, `sim.get_results(names, flatten=True)`, `sim.get_time()`.
- **Calibrate**: `from param_id.paramID import CVS0DParamID` → `pid = CVS0DParamID.init_from_dict(inp)`; then `set_ground_truth_data(obs)`, `set_params_for_id(params_for_id_dict)`, `set_param_id_method(...)`, `set_optimiser_options(...)`, `run()`, `simulate_with_best_param_vals()`, `plot_outputs()`; results under `pid.output_dir`.
- **Sensitivity**: `from sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis` → `SensitivityAnalysis.init_from_dict(inp)`; `set_ground_truth_data`, `set_params_for_id`, `set_sa_options`, `run_sensitivity_analysis(sa_options)`, `choose_most_impactful_params_sobol(top_n=..., index_type='ST', ...)`.
- **Build obs data in code**: `from utilities.obs_data_helpers import ObsDataCreator` → `add_protocol_info(pre_times, sim_times, params_to_change, offline_pre_time=...)`, `add_data_item(entry)`, `get_obs_data_dict()` — produces the same structure as an `obs_data.json` file.
- **Custom features**: register a Python function with `add_user_operation_func(fn)` on the param-id / SA object, then reference it by name in a data item's `"operation"` (operands map to the function args). Set `fn.series_to_constant = True` for series→scalar features so auto-plotting works.

`params_for_id_dict` is a list of `{vessel_name, param_name, min, max, name_for_plotting}` (the in-memory equivalent of `{prefix}_params_for_id.csv`); `vessel_name` may be a single name or a list to share one calibrated param across many vessels.

## `user_inputs.yaml` — key fields

- `file_prefix` — model name; ties together `{prefix}_vessel_array.csv`, `{prefix}_parameters.csv`, `{prefix}_obs_data.json` in `resources/`.
- `model_type` — `cellml_only` (default) | `python` | `casadi_python` | `cpp`.
- `solver` — `CVODE_myokit` (default) | `CVODE_opencor` | `solve_ivp` (python models) | `casadi_integrator` (casadi_python models) | `RK4_cpp`.
- `solver_info` — `MaximumStep`, `MaximumNumberOfSteps`, and `method` (e.g. `RK45` for solve_ivp; `cvodes`/`idas`/`collocation`/`rk` for CasADi). Validated — see `tests/test_solver_info_validation.py`.
- `pre_time` / `sim_time` / `dt` — steady-state spin-up, logged simulation duration, output sampling step. `dt` must be ≤ every dt in the obs_data.json. (Timeline can also be set per-experiment in obs_data.json; see below.)
- `param_id_method` — `genetic_algorithm` | `CMA-ES` | `bayesian` | `sp_minimize`. x0 auto-loaded from `{prefix}_parameters.csv`.
- `optimiser_options` / `debug_optimiser_options` — when `DEBUG: true`, the debug block overrides. `cost_type` (e.g. `gaussian_MLE`) selects the cost function in `funcs_user`.
- Feature flags: `do_ad`, `do_sensitivity` (`sa_options`), `do_mcmc` (`mcmc_options`), `do_ia` (`ia_options`).
- Path overrides (recommended for real work, to keep inputs/outputs outside the repo): `resources_dir`, `generated_models_dir`, `param_id_output_dir`, `external_modules_dir`.

## Discoverable schemas (keep settings machine-readable for CUFLynx)

The GUI front-end **CUFLynx** auto-populates its menus and settings forms by reading discoverable schemas in `src/parsers/PrimitiveParsers.py` — it does **not** hardcode the options. **Whenever you add or change a user-configurable setting, update the matching schema in the same PR**, or CUFLynx silently won't expose it:

- `SOLVER_SCHEMA` — model types, solvers per model type, integrator `method`s per solver, and `solver_info_fields_by_solver` (`SOLVER_INFO_FIELDS`: the `solver_info` settings per solver). `_SOLVER_INTEGRATOR_KEYS` (validation) is **derived** from `SOLVER_INFO_FIELDS`, so add a solver_info field there.
- `PARAM_ID_METHODS` — calibration methods, each with an `options` list = the `optimiser_options` it reads. Add a new optimiser knob here.
- `ANALYSIS_OPTIONS` — the `sa_options` / `mcmc_options` / `ia_options` settings for the sensitivity / MCMC / identifiability modes.
- Cost functions are a runtime registry (user-extensible), so they're discovered via `funcs_user.cost_funcs_user.cost_func_metadata()` (names + `is_MLE`/`is_combiner`/`differentiable` flags), not a static schema.

Each setting is a descriptor `{name, type, default, required, description, choices?}`. Accessors: `solver_info_fields(solver)`, `param_id_method_options(method)`, `analysis_options(mode)`. Tests in `tests/test_solver_info_validation.py` lock every schema's shape **and** its correspondence to what the code actually reads (e.g. an option the optimiser doesn't read, or a read the schema omits, fails the suite) — extend those when you add settings.

## Source layout (`src/`)

| Dir | Contents / purpose |
|---|---|
| `solver_wrappers/` | `SimulationHelper` backends + `get_simulation_helper()` factory (`__init__.py`). Backends: `myokit_helper.py`, `opencor_helper.py`, `python_solver_helper.py`, `casadi_python_solver_helper.py`. `name_resolver.py` maps variable names. |
| `generators/` | `CVSCellMLGenerator.py`, `PythonGenerator.py` (libCellML Analyser, strict ODE), `CVSCppGenerator.py`, `Python1DModelFilesGenerator.py`. |
| `param_id/` | `paramID.py` (calibration), `optimisers.py`, `differentiable.py` + `math_backend.py` + `operation_funcs.py` (AD), `plot_outputs.py`. |
| `protocol_runners/` | `protocol_runner.py`, `protocol_executor.py` — the multi-experiment/sub-experiment simulation loop. |
| `sensitivity_analysis/` | `sensitivityAnalysis.py`, `sobolSA.py`. |
| `identifiabilty_analysis/` | `identifiabilityAnalysis.py` (note the dir is spelled `identifiabilty`). |
| `parsers/` | `ModelParsers.py`, `PrimitiveParsers.py`, `OMEXParsers.py` — CSV/YAML/JSON/OMEX loading. |
| `models/` | `LumpedModels.py` (`CVS0DModel`). `checks/LumpedModelChecks.py` validates structure/connectivity. |
| `utilities/` | `utility_funcs.py`, `protocol_funcs.py`, `libcellml_utilities.py`, `obs_data_helpers.py`, `diagnostics.py`, plotting helpers. |
| `scripts/` | Entry points (see table above). |
| `coupler/`, `solver1d/` | 0D–1D coupling / 1D solver (in development). |
| `obsolete/` | Dead code — don't extend. |

User-extensible (kept outside `src/`): `module_config_user/` (custom CellML modules), `funcs_user/` (cost functions, protocols). `resources/` holds input CSVs and obs_data.json; `generated_models/` is build output.

## `SimulationHelper` API (common across backends)

`get_simulation_helper(...)` in `src/solver_wrappers/__init__.py` returns the backend for the configured `solver`. Common methods: `run()`, `update_times(...)`, `get_results(var_names)`, `get_all_results()`, `get_all_variable_names()`, `get_init_param_vals()`, `set_param_vals(names, values)`. When adding a backend, implement this full surface and register it in the factory.

## `obs_data.json` — experiment descriptor

```json
{
  "protocol_info": {
    "pre_times":  [0.0, 0.0],
    "sim_times":  [[5], [5]],
    "params_to_change": { "component/param": [[exp0_sub0, …], [exp1_sub0, …]] },
    "protocol_traces": { "trace_key": {"t": [...], "values": [...]} }
  },
  "data_items": [...],
  "prediction_items": [...]
}
```
A `params_to_change` value is a **float** (constant) or a **string** (trace key into `protocol_traces`). Series entries currently must have a `std` set (single-likelihood assumption — see commit history).

**Timeline conventions** (subtle — get these right):
- `pre_times[j]` is the **unlogged pre-pass** before the first sub-experiment of experiment `j`; `sim_times[j][k]` is the duration of sub-experiment `(j, k)`.
- `protocol_traces[...].t` are **seconds from the start of the Myokit segment** where the trace applies; match or exceed the segment length in `sim_times[j][k]`.
- Myokit (`myokit_helper`): each `update_times` calls `simulation.reset()`; logging instants are shifted so the first requested output time aligns with `simulation.time()` after `pre(pre_time)`.
- **`offline_pre_time`**: a generic steady state can be reached **offline** once and reused, so per-run `pre_time` only needs to cover parameter-specific settling — speeds up calibration. See obs_data / tutorial docs for `offline_pre_time`, `val_path`, `t_path`.

## Backend caveats

- **SN_simple / SN_full**: `cellml_only` generation + **Myokit** accept the emitted CellML (including state initial values referencing `*_init` params). **`PythonGenerator`** uses libCellML Analyser and requires a strict **ODE** model; the same SN CellML fails `ANALYSER_VARIABLE_NON_CONSTANT_INITIALISATION`, so `model_type: python` codegen is **not expected to work for SN** until the generator/model satisfies the analyser.
- **CasADi**: piecewise/conditional models emit `ca.if_else` so they stay symbolically differentiable (needed for AD-based param ID). See `tests/test_casadi_conditionals.py`.

## Testing (required for every change)

Add/extend tests in `tests/` for **every feature and bugfix**; a bugfix should include a test that fails before the fix.

| Test file | Covers |
|---|---|
| `test_solvers.py` | All four solver backends (markers: `solver`, `need_opencor`) |
| `test_autogeneration.py` | CSV → model generation (markers: `integration`, `slow`, `autogen_rank(idx)`) |
| `test_param_id.py` | Calibration across optimisers (`integration`, `slow`, `mpi`, `compare_optimisers`) |
| `test_sensitivity_analysis.py` | Sobol SA (`integration`, `mpi`) |
| `test_protocol_funcs.py`, `test_unit_conversion.py`, `test_solver_info_validation.py`, `test_casadi_conditionals.py` | Unit-level (`unit`) |
| `test_omex_analysis_pipeline.py` | OMEX/SED-ML pipeline (`integration`, `misc_task`) |

- Markers are declared in pyproject.toml (`--strict-markers` is on, so an unregistered marker fails the run). Notable: `slow`, `integration`, `unit`, `mpi`, `solver`, `need_opencor`, `compare_optimisers`, and the rank/task-coordination markers (`one_rank_task`, `autogen_rank`, `solver_task`, `misc_task`) used for MPI ordering.
- Fixtures in `tests/conftest.py`: `base_user_inputs`, `resources_dir`, `temp_output_dir`, `temp_generated_models_dir`, `mpi_comm`.
- Reuse existing CellML models / obs_data.json patterns from `resources/`; put new fixtures under `tests/test_inputs/` when needed.

## Docs

- Tutorial (authoritative how-to): https://physiomelinks.github.io/circulatory_autogen/ — source under `tutorial/docs/` (`getting-started`, `design-model`, `parameter-identification`, `sensitivity-analysis`, `identifiability-analysis`, `other-features`, `running-on-hpc`).
- AI wiki (orientation only): https://deepwiki.com/FinbarArgus/circulatory_autogen/1-overview
- Main dev branch is `devel`; PRs target `master`.
- This repo is a fork (`origin` = your fork, e.g. `FinbarArgus/circulatory_autogen`). **PRs must be opened against the upstream `physiomelinks/circulatory_autogen` repo** (the `upstream` remote), targeting its `master` branch — not against the fork. With `gh`, pass `--repo physiomelinks/circulatory_autogen` and a cross-fork head ref like `FinbarArgus:<branch>`. Note `gh pr create` can fail to resolve refs on a fork ("Head sha can't be blank"); the REST API works: `gh api repos/physiomelinks/circulatory_autogen/pulls -f head=FinbarArgus:<branch> -f base=master -f title=... -f body=...`.
