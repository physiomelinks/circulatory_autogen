# C++ generation templates (`model_type: cpp`)

These Jinja2 templates render the C++ that wraps a generated model. `template_docs.html` in this folder is an illustrated version of this README: open it in a browser to see each template beside the C++ it produced, with diagrams of the generation pipeline and of what runs when. The model equations are
never in a template. libCellML writes them as C (`model0d_core.c/.h`), and they are left
unmodified. The templates add everything around the equations:

- the `Model0d` class, including the solvers and output files;
- the code that supplies the values of external variables;
- a driver program, `main0d`;
- a CMake build;
- optionally, a C++ class that another program calls (an api "provider").

The rendering code lives next to this folder:

| File | Role |
|---|---|
| `../generator.py` | `CVS0DCppGenerator`: flattens and analyses the model, writes the libCellML C code, builds the template context, renders the templates. |
| `../externals.py` | Collects the external variables (FV 1D pipe inputs, delays, api-set values), resolves them to array indices, and generates the hook code for named-pipe apis. |
| `../api.py` | Validates `api` blocks and holds the unit-conversion table. |

## How a model is rendered

1. **Generate the CellML.** `generate_cellml()` writes it, and `generate_cpp()` flattens it (`<prefix>_flat.cellml`).
2. **Collect the external variables** from the module configs:
   - **Pipe inputs:** from `api` blocks with `role: consumer` and `transport: named_pipe`, for FV 1D coupling, using `conn_1d_0d_info` from the module-array split. Their pipes come from the `role: process` entry they name (`FV1D_solver`).
   - **Delays:** from `delay_info` entries. Each delayed variable depends on the variable to delay and on the delay amount.
   - **API-set values:** from module-array rows whose `api` block has `role: provider`. Every `set` function's variable, resolved through the row's ports, becomes external.
3. **Analyse.** The libCellML analyser runs with those external variables. Every state and variable gets an index (`states[i]`, `variables[i]`); `externals.ModelRef` records it.
4. **Write the equations.** libCellML's C output goes to `model0d_core.c/.h`, unchanged.
5. **Render.** Each template is rendered with the context described below. Jinja2 runs with `trim_blocks`, `lstrip_blocks` and `StrictUndefined`, so a misspelt variable is an error rather than empty text.

Files written, to `cpp_generated_models_dir` (or the model folder):

| File | From |
|---|---|
| `model0d_core.c`, `model0d_core.h` | libCellML (unmodified) |
| `model0d.h` | `model0d.h.j2` |
| `model0d.cpp` | `model0d.cpp.j2`, which includes `solver_init.j2`, `pipes.j2` and `io.j2` |
| `main0d.cpp` | `main0d.cpp.j2` |
| `CMakeLists.txt` | `CMakeLists.txt.j2` |
| `circulation_api.h`, `circulation_api.cpp` | `api_provider.h.j2`, `api_provider.cpp.j2` (only when a provider api is connected) |
| `api_test_driver.cpp` | `api_test_driver.cpp.j2` (only with a provider) |
| `<name>_coupler1d0d.json` | written directly by `generator.py` (only when coupled to 1D) |
| `coupler_config.json` | `coupler_config.json.j2`, from the `process` api block the connections name (only when coupled to 1D) |

Build:

```bash
cmake -S <dir> -B <dir>/build [-DSUNDIALS_DIR=<sundials install prefix>]
cmake --build <dir>/build
```

## Readable names

The generated code names every state and variable index instead of using bare numbers (`human_readable=True`, the default, as in the Python generator):

- **Header:** `model0d_core.h` declares `StateIndex` and `VariableIndex` enums. There is one entry per index, `S_<component>_<variable>` for states and rates and `V_<component>_<variable>` for variables, each with a `/* component.variable [units] type */` comment.
- **libCellML functions:** in `initialiseVariables`, `computeComputedConstants`, `computeRates` and `computeVariables`, every `states[3]`, `rates[3]`, `variables[12]` and the index argument of `externalVariable(...)` is rewritten to the name. Generation fails if a numeric index is left, so a change in libCellML's output format can't leave a mix.
- **Other libCellML output:** signatures, `STATE_INFO`/`VARIABLE_INFO` and the array helpers are untouched.
- **Templates and hooks:** they write the same names, via `ModelRef.name` (e.g. `setExternal(V_heart_u_lv_ext, ...)`, `s[S_terminal_1_module_v]`), and keep the `// component/variable` comments.

The enum values are the same integers, so the compiled model and its results are identical. The coupler JSON keeps plain integer indices, because the coupler and the 1D solver read them at run time.

The names come from `generators/naming.py`, shared with the Python generator: `var.pvn_module_u` in generated Python is `variables[V_pvn_module_u]` in C. A repeated name gets `_2`, `_3`, … in index order.

## Runtime structure

### The libCellML functions the templates call

libCellML 0.6 generates these functions:
- `initialiseVariables`
- `computeComputedConstants`
- `computeRates`
- `computeVariables`

When the model has external variables, each of them also takes an `ExternalVariable` callback. The generated code then calls `externalVariable(voi, states, rates, variables, index)` for every external variable it needs. The templates pass a callback that returns `Model0d::ext_cache[index]`, so **every external value is read from one cache, and each hook's job is to fill that cache**. `Model0d::setExternal(index, value)` writes both the cache and `variables[index]`.

### One right-hand-side evaluation: `Model0d::computeRates(voi, dt_stage, s, r, v)`

Every solver calls this, including CVODE through `cvodeRhs`, and each stage of the explicit schemes.

```
hookRhsStart   delays: cache <- history(voi - delay); named-pipe "rhs_start" calls (FV 1D: send [voi, dt])
computeVariables (libCellML)   algebraic variables from the states and the externals received so far
hookRhs        named-pipe "rhs" calls (FV 1D: send each connection's port value, read its BC into the cache)
computeRates   (libCellML)    rates, reading the externals just received
```

**The dt passed in:**

| Solver | dt passed on each call |
|---|---|
| explEul | `0` |
| Heun | `0`, then `dt` |
| midpoint | `0`, then `dt/2` |
| RK4 | `0`, `dt/2`, `dt/2`, `dt` |
| CVODE | `hcur` on the first call of a new internal step, `0` otherwise (Newton and Jacobian calls) |

The FV 1D solver weights its own update by these values, so don't change them.

### One step: `Model0d::solveOneStep(dt)`

```
hookStepStart  named-pipe "step_start" calls (FV 1D: send [voi, dt], adopt the global dt sent back)
integrate      CVODE (to voi + dt, max internal step = dt_solver) or explEul/Heun/midpoint/RK4
hookStepEnd    named-pipe "step_end" calls (FV 1D: send [voi, -999], read the 1D volume sum)
computeVariables, then store delay histories
```

`initialiseVariablesAndComputeConstants()` runs these in order:
1. `initialiseVariables`
2. optional loading of initial values from a previous run's output
3. `computeComputedConstants`
4. `hookInit` (FV 1D: read the volume sum)
5. `computeVariables`
6. the first delay-history entry
7. `CVodeReInit`

## The templates

### `model0d.h.j2`
Declares `Model0d`. It includes the libCellML header inside `extern "C"`, because libCellML's profile can't close an `extern "C"` block itself. For CVODE it also includes the SUNDIALS headers, with `#if SUNDIALS_VERSION_MAJOR` guards.

- **Public members:**
  - `states`, `rates`, `variables`, `voi`, `dt`
  - `set_ode_solver`, `initialiseVariablesAndComputeConstants`, `computeRates`, `computeVariables`, `solveOneStep`
  - output, input and pipe functions
  - `snapshot()`/`restore()`, which save and restore the whole state for trial steps
  - `reinitialiseSolver()`, to call after changing states from outside
  - `N1d0d`/`N1d0dTot`, kept for drivers written against the old generator
- **Private members:**
  - the hooks and `ext_cache`
  - the RK work arrays
  - the pipe streams (one `std::ofstream`/`std::ifstream` member per pipe)
  - a `DelayHistory` per delay
  - the CVODE objects

### `model0d.cpp.j2`
Implements `Model0d`:
- the callback;
- `cvodeRhs`, which works out the CVODE `dtLoc` from the step counter;
- `DelayHistory`, a time-stamped history with linear interpolation. It returns the first value before `t0 + delay`, and grows as needed, so it works with variable CVODE steps;
- the constructor, which zero-fills the arrays (libCellML fills them with NaN) and seeds `ext_cache`;
- `initialise`/`snapshot`/`restore`, `computeRates`, and `solveOneStep` with every solver.

The hook bodies are `{% for line in hooks.<hook> %}` loops over lines generated by `externals.build_named_pipe_code`. The delay lines are rendered directly from `delays`.

### `solver_init.j2`
The body of `set_ode_solver()`. `CVS0DCppGenerator._build_solver_init_function()` also renders it on its own, for tests.

- **CVODE:**
  - creates the context: `SUN_COMM_NULL` on SUNDIALS 7, `NULL` on 6, none on 5;
  - creates the BDF solver, a dense matrix and a dense linear solver;
  - applies `MaximumNumberOfSteps`, `dt_solver` as the maximum step, and `rtol`/`atol`.
- **Explicit schemes** only set the RK4 weights. A CVODE build still runs the explicit schemes when `ODEsolver` asks for one.
- **PETSC:** records the settings only; `generate_cpp()` refuses PETSC.

### `pipes.j2`
`openPipes()`, `closePipes()`, `pipeWrite()` and `pipeRead()` for named-pipe apis.

- **Opening:** every send pipe, then every receive pipe, in the order the api block declares its `channels`. A FIFO open blocks until the other end opens it, so this order must match the coupler's.
- **Which channels:** only those that some call uses. The coupler creates the volume pipe only when the model has a volume sum.
- **Without pipes:** empty stubs.

### `io.j2`
- **Writing:** `openOutputFiles()` creates the full folder path, and `writeOutput()` writes `sol0D_states.txt` and `sol0D_variables.txt`. The header format, `# 0: t[s]; 1: module/var[units]; ...`, is what `loadFromFile()`, the 1D solver and the plotting scripts read. States are written from `STATE_INFO`; algebraic and external variables from `VARIABLE_INFO`, when they belong to a `*_module` or to `parameters`.
- **Reading:** `loadFromFile()` reads the last row, or the first, of a previous run's output into the arrays, matching columns by name.

### `main0d.cpp.j2`
The driver.

- **Standalone:** `./main0d [-ODEsolver S] [-tEnd t] [-tSave t] [-outDir d] [-initStatePath p]`. The defaults come from the user inputs: end time `pre_time + sim_time`, and saving from `pre_time`.
- **Coupled:** `./main0d -coupled 1 -ODEsolver S -T0 t -nCC n -networkName s -pipePath p -initStatePath p`. This is the command line the coupler uses, so don't change it.
- **Stepping and saving:** it steps at `dt_solver` and writes output every `dt`. When coupled it saves from cycle `nCC - 2`.

### `coupler_config.json.j2`
What the coupler launches and with what, for a model coupled to 1D. `generator._write_coupler_config` fills it:
- `T0`: the global parameter `T`; `nCC`: whole periods covering `pre_time + sim_time` (`generator._coupled_run_length`). `main0d`'s coupled defaults use the same two values.
- `solver1d_path`: from the process's `program`; `python_path`: `sys.executable`; `solver0d_path`: `build/main0d`.
- `tmp_pipe_path`: user input `coupler_pipe_dir`, default `cuflynx_pipes/<model>/` in the system temp folder (`tempfile.gettempdir()`, so `TMPDIR` moves it).

The rendered text is parsed with `json.loads` before it is written, so a template mistake fails generation.

### `CMakeLists.txt.j2`
- **Libraries:** `model0d_core` (C, the libCellML code) and `model0d` (the C++ wrapper).
- **Executable:** `main0d`.
- **SUNDIALS:** tried first as a CMake config package (`SUNDIALS::cvode`, …; point `SUNDIALS_DIR` at the install prefix), then with `find_path`/`find_library`.
- **Providers:** `extra_targets` adds the `circulation_api` library and `api_test_driver`.

### `api_provider.h.j2` / `api_provider.cpp.j2`
The class generated for a provider api, e.g. `lifex::Circulation`. It owns a `Model0d` and has one method per entry in the api's `functions`:

| kind | Generated method | Body |
|---|---|---|
| `set` | `void name(const double &arg)` | `setExternal(index, value * factor)`, then recompute variables |
| `get` | `double name() const` | `states[i]` or `variables[i]`, divided by `factor` |
| `set_state` | `void name(const double &arg)` | overwrite a state, restart the solver |
| `set_indexed` | `void name(const Chamber &, const double &)` | `switch` over the chamber enum, using `pressure_set` |
| `get_indexed` | `double name(const Chamber &) const` | `switch` over the chamber enum, using the named `field` |
| `time` | `void name(const double &time, const unsigned int &n)` | store the target time |
| `time_discretization` | lifex's 5-argument signature | store the time step, move the model time |
| `step` | `void name()` | advance to the time from `set_time` (or one step) |
| `noop` | variadic template | accepts any arguments, does nothing |
| `extrapolate` | lifex's `get_volume_chamber_extrapolated` signature | snapshot, trial step(s), finite-difference dV/dp, restore |

**Unit factors:** `factor` converts API units to model units (`model = api * factor`). It comes from `api_units`, using the table in `api.py`, or from an explicit `factor` in the function entry.

**Solver restarts:** a value set from outside is a discontinuity for CVODE's multistep history, so setters raise `inputsChanged_`. `advance()` then calls `reinitialiseSolver()` before the next step.

**Chamber enum:** with `-DCVS_WITH_LIFEX`, `chamber_enum.type` (e.g. `iheart::Chamber`) comes from `chamber_enum.header`. Otherwise an enum with the value names from the api block (`LA`, `LV`, …) is declared in the generated header.

**Getting from a Compartment to a Chamber:** `extrapolate` converts lifex's `Compartment` keys to `Chamber` with `static_cast`. This assumes both enums list the chambers in the same order.

### `api_test_driver.cpp.j2`
A stand-in for the program that calls the api:

```
api_test_driver [-replay pressures.csv] [-tEnd t] [-dt dt] [-solver S] [-out results.csv]
```

- **`-replay`:** sets chamber pressures (api units, columns `t, p_<chamber>...`) from a file every step and writes the resulting volumes. For example, replaying pressures from the 0D-heart model should reproduce its volumes, up to the error of holding each pressure constant over a step.
- **Without `-replay`:** couples implicitly to a simple elastance law, `p = E(t) (V - V0)`, with Newton iterations that use the `extrapolate` function. That is the volume-constrained coupling lifex does.
- **Also printed:** a unit round trip and one extrapolation.

## Template context

Available to every template:

| Name | Meaning |
|---|---|
| `file_prefix`, `model_name` | the model's prefix (`model_name` drops a trailing `_0d`) |
| `core_header`, `core_source` | `model0d_core.h`, `model0d_core.c` |
| `libcellml_version` | for the generated-by comment |
| `solver` | `CVODE`, `RK4`, `Heun`, `midpoint` or `explEul` |
| `n_max_steps`, `dt_solver`, `dt_sample`, `reltol`, `abstol` | from `solver_info` / `dt` |
| `has_externals` | whether the libCellML functions take the callback (their signatures differ) |
| `externals` | `ExternalSpec` list: `.ref` (`.kind` = `state`/`variable`, `.index`, `.name` = the named index, e.g. `V_heart_u_lv_ext`, `.label` = `component/variable`, `.cpp('s','v')`), `.source` (`pipe`/`delay`/`api`), `.initial` |
| `delays` | the delay `ExternalSpec`s; `.meta.source` / `.meta.amount` are `ModelRef`s |
| `pipes` | `None`, or `{api_name, message_length, send_pipes, recv_pipes, hooks}` for a named-pipe api |
| `hooks` | `{init, step_start, rhs_start, rhs, step_end}` → lists of generated C++ lines |
| `n_connections`, `n_connections_total` | 1D-0D connections, without / with volume-sum entries |
| `output_dir` | default output folder for `main0d` |
| `default_T0`, `default_nCC`, `default_end_time`, `default_save_time` | `main0d` defaults |
| `cmake_project`, `extra_targets` | CMake project name; extra CMake text (provider targets) |

Added for the provider templates:

| Name | Meaning |
|---|---|
| `api` | the api block (dict) |
| `api_functions` | its `functions`, each with `factor` and, for `set`/`get`/`set_state`, `ref` (a `ModelRef`; write indices as `ref.name`) |
| `chambers` | from `chamber_enum.values`: `{name, <field>: ModelRef, ...}` |
| `chamber_units` | factors for `chamber_enum.units` (`pressure`, `volume`) |
| `namespace`, `class_name` | where the class goes and what it's called |
| `chamber_type`, `chamber_header` | the enum type and lifex header |
| `provider_vessel` | the provider's module-array row name |

## `api` blocks in brief

The full description is in `tutorial/docs/design-model.md`; validation is in `../api.py`.

- **Python external model** (`role: provider`, `transport: python`): a Python class (e.g. FEniCS) stepped with the model.
  - `externals.collect_python_exchange` maps the row's port variables to every connected CellML module, and decides each direction from an analysis of the flat model: a constant (boundary condition) is set by the class and becomes an external variable; anything computed is read by it.
  - `model0d_capi.cpp.j2` renders a C interface over `Model0d` (exchange table, get/set/step/snapshot/restore/output, status codes), built as the shared library `model0d_capi` (only `cf_*` exported); `external_models.json` says what to run. `libcuflynx.coupling` loads both.
  - The tutorial's "Coupling to External Models" describes it from the user's side.
- **Process** (`role: process`, e.g. `FV1D_solver` in `resources/coupling_modules_config.json`): a program run alongside the model.
  - **`program`:** what to launch (`package` + `script`, or `path` + `script`); **`coordinator`:** `coupler`.
  - **`channels`, `message_length`:** the pipes, shared by every consumer that names the process. No `calls`.
  - Gives `coupler_config.json`. It can be a module-array row with no connections; it doesn't have to be.
- **Named-pipe consumer** (e.g. the FV1D entries in `resources/coupling_modules_config.json`):
  - **`process`:** the process entry whose `channels` and `message_length` it uses (`api.resolve_process_apis` copies them in when the configs load). It may add `channels` of its own.
  - **`channels`:** pipe names; `{i}` is replaced by the connection number.
  - **`calls`:** each has a `when` (`init`, `step_start`, `rhs_start`, `rhs`, `step_end`) and a `kind` (`send`, `recv`, `send_recv`).
  - **What a call sends:** `$voi`, `$dt`/`$dt_stage`, `port.flow`/`port.pressure`/`port.input`/`port.output`, `control.<port_type>|default`, or numbers.
  - **What a call receives:** `port.input`, a model variable, `$dt`, or `$ignore`.
  - **Per connection:** calls marked `"per": "connection"` are repeated for each connection. All the sends are emitted before all the receives.
  - **Where the code goes:** `externals.build_named_pipe_code` turns the calls into the `hooks` lines.
- **Provider** (`transport: cpp_class`):
  - The provider is a module-array row with `module_format: external_api`, connected to CellML modules through ports matched by `port_type`, with variables paired by position.
  - Its `functions` name its own port variables, or `component/variable`.
  - `externals.provider_port_refs` resolves them.

## Changing or adding a template

- **Rendering and output:** add the template here and render it in `CVS0DCppGenerator._render_all`. New context values go in the `ctx` dict there, or in `_provider_context` for provider-only values. Package data already includes `templates/*.j2`.
- **Generated code that differs per model** (hooks, index lists) is best built in Python (`externals.py`) and passed as data, so templates stay mostly static C++.
- **Changes to the pipe protocol** (message order, the dt values, `-999`, which pipes are opened, the `main0d` command line) must be matched in `coupler/coupler.cpp` and `solver1d/main1D.py`. `tests/test_cpp_template_generator.py::test_fv1d_api_matches_the_coupler_and_1d_solver` checks the pipe names and message length.
- **Tests** are in `tests/test_cpp_template_generator.py`. They build with CMake, set `SUNDIALS_DIR` if needed, and include a full coupled FV 1D run. They also check the delay model and a provider module coupled through a port. `tests/test_issue_fixes.py` checks the rendered solver settings.
