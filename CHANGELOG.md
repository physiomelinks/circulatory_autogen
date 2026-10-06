# Changelog

Notable changes to libcuflynx (circulatory_autogen). Entries under *Unreleased* ship with the
next release; add to that section as you land a change.

## Unreleased

### Changed! — `model_type: cpp` is generated from libCellML's C output and Jinja2 templates

The C++ generator is rewritten (`libcuflynx/generators/cpp/`). The model equations are
libCellML's C code, written unmodified (`model0d_core.c/.h`); the `Model0d` class, the solvers,
`main0d` and a `CMakeLists.txt` are rendered from templates. Build with CMake (add
`-DSUNDIALS_DIR=<prefix>` if SUNDIALS isn't found; versions 5-7 work). Solvers are CVODE and the
fixed-step RK4/Heun/midpoint/explEul; **PETSC is no longer supported** and is refused with an
error. Generated models no longer get the `solver1d/Make_files` copied next to them (they built
the old C++), and the coupler builds with CMake. FV_1d coupled output is identical to before.

### Added — `api` blocks: couplings to other models described in module configs

A module config entry can carry an `api` block. `role: consumer` describes calls the generated
C++ makes (the FV 1D named-pipe protocol is now described this way in
`coupling_modules_config.json`); `role: provider` generates a C++ class another program calls,
e.g. a drop-in `lifex::Circulation`. A provider is its own vessel-array row
(`module_format: external_api`) coupled to CellML modules through ports; values it sets become
libCellML external variables. `external_modules_dir` also accepts a list of directories.

### Added — the FV 1D solver as a `process` module; `coupler_config.json` is generated

A new `api` role, `process`, describes a program run alongside the generated model: what to
launch (`program`), who launches it (`coordinator: coupler`) and the pipes it talks over
(`channels`, `message_length`). `FV1D_solver` in `coupling_modules_config.json` is the FV 1D
solver. The `FV1D_vessel` and `FV1D_volume_sum` entries name it (`"process": "FV1D_solver"`)
instead of each repeating the pipe names. From it, C++ generation of a model coupled to 1D now
writes `coupler_config.json`, which until now had to be written by hand:
- `T0` is the global parameter `T`;
- `nCC` is the number of whole periods covering `pre_time + sim_time`;
- the pipe folder is the user input `coupler_pipe_dir` (default `cuflynx_pipes/<model>/` in the
  system temp folder, which `TMPDIR` moves);
- `python_path` is the generating Python, and the 1D solver is the installed one.

`convert_0d_to_1d` adds an `FV1D_solver` row to the hybrid vessel array, and reads
`<model>_vessel_array.csv` when there is no `<model>_0d_vessel_array.csv`. The coupler creates
the pipe folder, and its default pipe folder and Python are no longer paths on one machine.
`main0d`'s coupled defaults (`T0`, `nCC`) match the configuration. New example model
`aortic_bif_0d` (all 0D); a test runs it against the same model with its vessels in 1D.

### Added — coupling generated C++ to external Python models (e.g. FEniCS)

A module config entry with `"module_format": "external_api"` and
`"api": {"role": "provider", "transport": "python", "python": {"file": ..., "class": ...}}` is an
external Python model: a row of the vessel array, connected to CellML modules through its ports.
- **Generation.** Generating the model as C++ also writes a C interface (`model0d_capi.cpp`,
  built as the shared library `model0d_capi`) and `external_models.json`.
- **Running.** `cuflynx-couple <model folder>` (or `libcuflynx.coupling.run_coupled`) builds the
  library, then steps the C++ model and the class together.
- **Directions.** No function list is needed: each port variable connected to a boundary
  condition of the 0D model is set by the class, and each connected to a computed variable is
  read by it. A port connected to several modules exchanges arrays.
- **Coupling scheme.** Explicit staggered coupling (first order), or `subiterations` for a
  trapezoidal fixed point (second order).
- **Also:** MPI (the 0D model runs on every rank), output on the model's `dt`, and timings.

New tutorial section "Coupling to External Models": an overview, coupling your own Python
model, FEniCS tissue O2 with capillaries, FEniCS NE around a sympathetic varicosity, 1D
finite-volume coupling, and troubleshooting. The FEniCS modules and system models live in
circulatory-autogen-modules.

### Fixed — C++ generation

- `Model0d::solveOneStep` returns a status (and `lastError()`) instead of calling `exit(1)`, and
  `main0d` now exits non-zero when the solver fails (it returned 0).
- CVODE restarts from the last good state after an error-test or convergence failure, so
  models with time switches (a stimulus pulse train) run.
- State initial values given by computed variables (e.g. a gate starting at its steady state)
  are evaluated with Myokit; libCellML 0.6 accepts only constants there, so these models
  could not be generated as C++ before.
- CellML generation connects variables of equivalent units under different names (e.g.
  `mol_per_m3` and `millimolar`) directly. It used to create an unconnected "converter", and the
  converter name was not unique per connection.
- `GE_capillary` listed `d_1` ... `s_2` twice in its config, and `capillary_GE` and
  `pulmonary_GE_5_lobe_type` used `saturation_cap` without declaring it.

### Added — readable generated C/C++

Generated C code names every state and variable index: `rates[S_heart_module_q_lv] =
(variables[V_parameters_r_pvn] ...)` instead of `rates[3] = (variables[12] ...)`.
`model0d_core.h` declares the `StateIndex`/`VariableIndex` enums, with each name's component,
variable, units and type, and the wrapper, pipe hooks and api classes use the same names. The
names are the generated Python's attribute names (shared `generators/naming.py`). Results are
unchanged.

### Fixed — cpp generation and 1D coupling

- `model_type: cpp` with CVODE always failed solver-settings validation.
- Models with more than one delay variable did not compile; delays now use a time-stamped
  history that works with variable CVODE steps.
- 1D volume sum: the 1D solver opened its volume pipe in an order that deadlocked with the
  coupler, and sent the volume in cm³ instead of m³.
- 1D input generation wrote an unknown artery/vein type for vessels not named `A_*`/`V_*`;
  an `art_ven_type_<vessel>` parameter now sets it.

### Added — JSON vessel arrays and supermodules

A vessel array can be a JSON list of records, `[file_prefix]_vessel_array.json`, with PhLynx's
keys (`name, module_type, module_subtype, inp_instances, out_instances`) or libcuflynx's (`name,
vessel_type, BC_type, inp_vessels, out_vessels`). A CSV array is now read by converting each row
to the same record, so both are processed identically; the `.json` file is preferred when both
exist, then `.csv`, then PhLynx's `_module_array.json`/`.csv`. Convert CSV arrays with
`python -m libcuflynx.utilities.config_schemas to-json <csv>... [--style phlynx|libcuflynx]`.
Every model in `resources/` that generated before generates byte-identical CellML from its CSV
and from its JSON conversion.

A module config entry with `"module_format": "supermodule"` and a list of `submodules` defines a
supermodule. An instance of it in a vessel array expands into `[instance]_[submodule]` modules
before anything else reads the array; `per_submodule_inputs`/`per_submodule_outputs` on the
instance link its hosts to individual submodules. Supermodules may nest. An optional
`default_parameters` CSV supplies parameters (renamed to the expanded names) wherever the
model's parameters file does not set them. See `tutorial/docs/design-model.md`.

JSON Schemas for both files ship in `libcuflynx/schemas/` (`vessel_array.schema.json`,
`module_config.schema.json`). The loaders check the same rules without a schema library;
`jsonschema` is a `[dev]` dependency only, for the tests.

The 0D/1D split for `couple_to_1d` now always writes `[file_prefix]_0d_vessel_array.csv` and
`[file_prefix]_1d_vessel_array.csv`, and the 1D generator reads the file the split wrote. Before,
an input named `_module_array.csv` gave split files the 1D generator could not find.

### Added — PhLynx module-config and vessel-array schemas; `"Sum"` and `"Multiply"` multi_ports

The module library is moving its configs to PhLynx's key names, and libcuflynx now reads both
schemas. In PhLynx's schema, `module_type` is the vessel_type, `module_subtype` is the BC_type,
`component_file` is the module_file and `component_type` is the CellML component name. Each
entry is detected on its own, by `module_subtype`, `component_file` or `component_type`, and is
converted to the libcuflynx names when the config is loaded
(`libcuflynx.utilities.config_schemas.normalise_module_config_entry`). An entry that mixes the
two schemas is an error. Vessel arrays can use PhLynx's export layout too
(`name, module_type, module_subtype, inp_instances, out_instances`). The layout is detected from
the header. `[file_prefix]_module_array.csv` is read when there is no
`[file_prefix]_vessel_array.csv`.

`multi_port` values are now case-insensitive. Before, only lowercase `"sum"` summed, and PhLynx's
`"Sum"` behaved like `"True"`. On a `volume_port`, `"Sum"` now gives the same `sum_blood_volume`
sum as `"sum"`. On any other port, `"sum"`/`"Sum"` needs exactly one variable, and that variable
is the sum of the connected modules' corresponding variables. This is PhLynx's semantics, and it
is implemented as the list form `["sum"]`. The module library's microvasculature_network
`v_out_sum` ports need it. The new `"Multiply"` value, with an optional numeric
`multiply_factor` (default 1), sets each downstream neighbour's variable to the factor times
this module's variable, as PhLynx does. All the models in `resources/` that generated before
still generate byte-identical CellML.

### Added — per-variable `multi_port`, for flow merge and split nodes

A port's `multi_port` can now be a list with one entry per port variable. Use it when a port
connected to several modules must sum some variables and share others:

```json
{"port_type": "vessel_port", "variables": ["v_in", "u"], "multi_port": ["sum", "True"]}
```

A `"sum"` variable (an input) equals the sum, over every module connected through the port, of
the neighbour's corresponding port variable. A generated algebraic component,
`multiport_sum_<vessel>_<variable>`, computes it. The component scales a neighbour in different
but compatible units, and generation stops on incompatible units. With no neighbour, the variable
is 0 and a warning is printed. A `"True"` variable (normally an output) is mapped to the
corresponding variable of every neighbour. This works on entrance, exit and general ports.

The module library's algebraic `flow_merge` (many inflows, one outflow) and `flow_split` (one
inflow, many outflows) nodes need this: they sum the flows and share the node pressure. The string
forms (`"True"`, `"sum"`) are unchanged, and models without a list-form port generate
byte-identical CellML. The model checks reject a malformed list (wrong length, or an entry other
than `"sum"`/`"True"`). The C++ generator's 0D-1D coupling (`couple_to_1d`) does not support
list-form ports and raises `NotImplementedError`. Plain C++ generation works from the generated
CellML and is not affected. See *Designing a model* in the tutorial.

### Added — `module_library_dirs` and `use_builtin_modules`

Models can now be generated from an external module library laid out one module per
directory, such as circulatory-autogen-modules (`modules/<name>/<name>_modules.cellml`,
`<name>_modules_config.json`, `<name>_units.cellml`). `module_library_dirs` takes one path
or a list, each searched recursively; only `*_modules_config.json` / `*_module_config.json`
files are read as configs there, so parameter or obs_data JSON can sit next to a module.
`use_builtin_modules: false` stops the built-in and `module_config_user` modules and units
from loading, so a library can be the only source of modules and can redefine a built-in
`(vessel_type, BC_type)` without the duplicate-entry exit. Both default to today's behaviour.

### Fixed — units files in `external_modules_dir` are loaded

The docs said a `user_units.cellml` in `external_modules_dir` was picked up; it was not.
Every `*units.cellml` there (and in `module_library_dirs`) is now merged into the generated
units file. A unit defined identically in several files is written once; one defined
differently in two files raises a `ValueError` naming both files.

### Fixed

- `model_type: python` (and the CasADi / AADC variants) now defines every helper libCellML's
  Python profile can emit: `eq_func`, `neq_func`, `or_func`, `xor_func`, `not_func`, `min` and
  the reciprocal trig functions `sec` ... `acoth`. A model using `<eq/>`, `<or/>`, `<min/>` etc.
  used to fail at run time with `NameError: name 'eq_func' is not defined` (#526).
- A `sum` multi-port (e.g. a `volume_sum` vessel) with no inputs connected is now 0, with a
  warning naming the vessel, and is still mapped to the vessel's port variable. Generation used
  to fail with `IndexError: list index out of range` (#525).
- The built-in constant boundary conditions (`inlet_pressure`, `outlet_pressure`, `inlet_flow`,
  `outlet_flow` with BC_type `nn_constant`) list their port's flow / pressure variable in
  `variables_and_units`, so they can be connected. Generation used to stop with "the port variable
  v is not a variable for vessel type: inlet_pressure" (#529).
- Generic junctions (`Min_junction`, `Nout_junction`, `MinNout_junction`) now include every
  neighbour with a `vessel_port` facing the junction node, taking its flow and pressure from that
  port, instead of skipping any neighbour whose BC_type starts with `nn`. A junction fed directly by
  boundary conditions such as `inlet_flow nn_constant` used to fail with "Min_junction junc has NO
  other vessels connected to its inlet node". Existing models generate byte-identical CellML (#524).

## 0.7.3 — 2026-09-05

### Changed! — `calculate_two_observable_difference` takes `subtract_from` / `subtract_this`

Two changes to one operation: its inputs are declared, and they are renamed.

**Declared.** It was written `(x=None, series_output=False, **kwargs)` and read its two inputs
out of `kwargs` in the body, so nothing that introspects an operation could see them:
`get_operation_kwarg_spec` could only answer *accepts anything*. A form built from it had no
fields to offer, and a misspelled key was accepted and ignored — leaving the real one unset,
to surface later as a missing value. They are now ordinary keyword arguments. Both keep a
default, which is what makes them keyword arguments rather than operands: a parameter with no
default is filled positionally from the data_item's `operands`, and these come from
`operation_kwargs`. The vestigial `x` is gone — an operation that takes no operands was never
handed it.

**Renamed.** `pred1` / `pred2` said nothing about which way round the subtraction went. The
function returns `pred2 - pred1`, which you could only learn by reading it, and getting it
backwards is a sign error that looks like a plausible number. The names now say it:

```
    result = subtract_from - subtract_this
```

**This changes obs_data.** An item using this operation must be updated, and the mapping is
*not* positional — `pred2` is the value subtracted **from**:

| before | after |
|---|---|
| `"pred2": "max flow aortic root"` | `"subtract_from": "max flow aortic root"` |
| `"pred1": "mean flow aortic root"` | `"subtract_this": "mean flow aortic root"` |

Swapping the two flips the sign of the result. Nothing silently accepts the old spelling: the
inputs are declared now, so `pred1` and `pred2` are unknown keys and the ordinary
`operation_kwargs` check refuses them, naming the operation. A file that is not updated fails
at parse time rather than calibrating against a wrong number — which is exactly why the
declaration had to come first. Both shipped files using this operation
(`resources/3compartment_extra_ops_obs_data.json` and the cross-experiment test fixture) are
updated here.

**The paired GUI change is CUFLynx #351.** CUFLynx builds its obs_data editor from these
accessors, and this operation was the one it could not render a form for: with the inputs
undeclared, `get_operation_kwarg_spec` answered *accepts anything* and there were no fields to
offer. CUFLynx #351 deletes the workaround it had grown for that (an `ast` parse of the
function body) and reads the signature like it does for every other operation. That release is
the one to take this with, and it picks the new names up automatically — it reads them from
the signature rather than restating them. CUFLynx's floor (`libcuflynx>=0.7.1`) is unchanged,
but a study whose obs_data still says `pred1`/`pred2` needs editing either way.

No change to the dependency extras, the minimum Python, or the flat-import shims (removed in
0.6.0, and unaffected here).

## 0.7.2 — 2026-09-04

### Fixed — one reconstruction page per trace, not per feature (#515)

Before #466 a data_item's `variable` was both its identity and the model variable it reduced,
so `max` and `min` of one pressure shared a `variable` and were drawn on one figure. #466
split those two jobs and made `data_item_name` unique by construction — and
`plot_reconstruction_pages` still grouped on it. Every feature therefore got a page of its
own: the 3compartment example went from four figures to six, and a study with several
features per trace lost the merged figures entirely.

Pages are now keyed on `trace_name_for_plotting` — the series a feature is measured on, which
is allowed to repeat and is already what the y-axis is labelled with. On the shipped fixtures
that is 3compartment back to four pages from six, with `mean`/`max` of `v_{AR}` and
`mean`/`max_minus_min` of `u_{AR}` reunited, while the four-experiment multi-trace fixture
stays at eight, one per trace per experiment.

Subexperiment is deliberately *not* part of the key, which is the second half of the issue: a
trace measured on several subexperiments is one page carrying a segment for each, drawn along
the experiment's timeline and sharing a single legend entry. `Lotka_Volterra_multisub` goes
from four pages to two, each spanning both subexperiments. A subexperiment nothing was
measured on is still left out, so #474's narrowed axis is unchanged — the window is the union
of the segments actually measured, not the whole experiment.

The y-axis label now comes from the same resolved name the page is grouped by, so a page
cannot be grouped under one name and labelled with another.

Not affected: the error vectors stay one entry per data_item. Grouping changes how many
figures are drawn, not what is scored — `percent_error_vec` / `std_error_vec` and the
`error_vec_names` beside them (#341) are untouched.

### Fixed — a partial `apt-get update` no longer fails a job several steps later

`apt-get update` exits 0 when it could not fetch an index: it says *"Some index files failed
to download. They have been ignored, or old ones used instead"* and carries on with the stale
one. The install then asks for the package version that index names, the mirror has moved to
a newer point release, and the fetch 404s — so a mirror wobble arrives as a hard failure in an
unrelated job rather than as a slow update. On this release's own PR it took out `test-uq` and
`test-param-id-serial-full-scale` over `libcurl4-openssl-dev`.

`install-system-deps` now notices the partial update and re-runs it (three attempts, backing
off), and gives the install one refresh-and-retry of its own. The per-fetch timeouts and
retries already there bound a single request; neither can see that the update as a whole came
back incomplete. A genuinely broken apt still fails the job — the retries do not swallow it.

## 0.7.1 — 2026-09-01

### Added — a contract for what `process_obs_info` publishes

`obs_info` is a public API with no schema and no version, indexed by string literal around
three hundred times here and thirty-one times in CUFLynx. Renaming two of its keys in 0.7.0
broke the then-current CUFLynx release, and the first thing to notice was an hour-long CI job
that downloads a built binary.

`tests/test_obs_info_contract.py` now describes all thirty-two keys: which index space each
belongs to, and whether anything outside this repository may read it. Renaming one fails a
test whose id is the key's own name, in seconds. Eighteen of them are public with no
accessor; those are the ones a rename has to plan for.

### Fixed — `process_obs_info` on frames it said it accepted

Two crashes, both on inputs the docstrings invite:

* A caller handing it a `gt_df` it built itself hit `KeyError: 'prob_dist_params'` -- a column
  the schema adds, read with a bare `[...]`.
* An obs_data carrying only `protocol_info` parsed to a frame with no rows *and no columns*,
  so every read raised and such a file could not be opened at all.

### Fixed — the obs_info accessors on numpy columns

`obs_item_names` and its siblings returned `... or []`, which raises "the truth value of an
array with more than one element is ambiguous" when handed a numpy array. The keys they read
happen to be lists, so this never fired here; several sibling `obs_info` values *are* arrays,
and a downstream adapter copying the idiom onto them broke forty-seven tests. They now test
for `None`.

### Added — `obs_experiment_indices`, `obs_subexperiment_indices`, `obs_scalar_rows`

So the keys CUFLynx reads have the same rename-immunity the label keys already had.

### Changed — the release waits for PyPI to serve what it published

PyPI accepts an upload before its index offers it, and two downstream jobs went red in that
window on the 0.7.0 release with a message that reads like a dependency mistake.

### Changed — the solver wrapper says what it skipped

Four `except Exception: pass` sites in `myokit_helper` now log at debug. A dropped
`MaximumStep`, or a variable with no default value, was previously silent.

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
