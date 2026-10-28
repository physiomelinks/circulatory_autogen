# Changelog

Notable changes to libcuflynx (circulatory_autogen). Entries under *Unreleased* ship with the
next release; add to that section as you land a change.

## Unreleased

### Added — `BE_batched_c`: generated C, one block of states at a time, on several threads

A CellML solver for models made of many weakly connected parts, such as a module array of cells
(#573). `get_simulation_helper(..., solver='BE_batched_c')` takes the same CellML file and protocols
as `CVODE_myokit`.

- **When to choose it.** `CVODE_myokit` stays the general-purpose default. `BE_batched_c` and
  `CVODES_batched_c` are good when you have many perturbations through time, because CVODE would keep
  stopping and restarting its adaptive time steps. They are opt-in: they run only when named in the config.
  The solver section of `tutorial/docs/parameter-identification.md` says so.
- **Blocks.** `generators/kernel_ir.py` reads libCellML's analysis into a language-neutral form and
  splits the states into blocks that do not depend on one another. Blocks with the same equations
  share one kernel, so 64 identical cells compile one kernel.
- **Exact Jacobians.** Each kernel's Jacobian is made by forward-mode differentiation of the
  equations. Only its structurally non-zero entries are written.
- **C.** `generators/cpp/batched.py` writes the kernels as C next to libCellML's own C (as
  `model_type: cpp` writes it, which gives initial values and outputs), renders the runtime from
  `templates/batched_*.j2` and builds it with CMake. A built model is cached under a digest of its
  CellML files, so later helpers load it in milliseconds.
- **Integrators.** Each block gets its own adaptive, L-stable one-step method, chosen by
  `solver_info` `method`:
  - `backward_euler`, the default;
  - `trbdf2`, second order, which is far more accurate at the same tolerance and usually faster.

  Both use modified Newton, with the Jacobian and LU factors kept across steps. The step size changes only
  when the error control allows more than 20% growth (or asks for any shrink), so the factors serve several
  steps: 1.10-1.14x faster on AdaptOrDeity's colonies at the same error.
- **Inputs that change at every call.** `bk_set_stop_at_outputs` (`BatchedSimulation.set_stop_at_outputs`, off
  by default) ends steps on each run's last output time; a block whose inputs changed there restarts softly,
  keeping its step size, Jacobian and LU factors (never from an interpolated state). With an input changed every
  0.1 s, as a game's macro step does, that is 4.3-4.6x faster; with inputs that rarely change, 0.93-1.02x, hence
  off. `CVODES_batched_c` accepts the call and ignores it.
- **Threads and restarts.** OpenMP runs over the blocks, and results are the same for any number
  of threads. A block whose inputs did not change between sub-experiments keeps its steps.
- **Protocol traces.** A `params_to_change` entry naming a trace drives that constant along it, as on
  `CVODE_myokit`: piecewise linear on the model's clock, held at its ends. Steps end on the trace's
  breakpoints, and the kernels compute their computed constants, so everything that depends on the
  traced constant follows it. SN_simple's protocol agrees with `CVODE_myokit` within 5e-4 of each
  variable's range, 20x faster.
- **Instances and outputs.** `bk_instance_*` steps copies of the model from the caller's own threads
  (AdaptOrDeity's cells); `outputs` computes a few algebraic variables alone after each step.
- **Limits.**
  - It is meant for moderate tolerances (default `rtol` 1e-4).
  - Its LU is dense, so a single block of hundreds of states is slow.
  - It does not take models with algebraic loops or external variables.
  - Excitable models want a `MaximumStep` or a tighter `rtol`: at `rtol` 1e-6 and no step cap, an action
    potential in SN_simple lands 0.25 ms late (1.3 mV on its upstroke); with `MaximumStep` 1e-3, 0.09 mV.
  - It has no sensitivities.
- **user_inputs.** `solver: BE_batched_c` takes `method` `backward_euler` (the default) or `trbdf2`; each CellML
  solver's methods are checked against its own list, not `CVODE_myokit`'s.
- **First step after a restart.** Hairer, Norsett and Wanner's starting step, from weighted RMS norms over the
  block's states and a second-derivative estimate, at least 64 ulps of t, so a state restarting at zero (weight
  1/atol) no longer forces a step too small for t + h to differ from t (late restarts failed as "step too small").

### Added — `CVODES_batched_c`: one CVODES per block, on the same generated C

`get_simulation_helper(..., solver='CVODES_batched_c')` builds `BE_batched_c`'s kernels, blocks, traces and
outputs with a different runtime (`templates/batched_cvodes_runtime.c.j2`, which extends `batched_runtime.c.j2`).
Each block gets its own CVODES: BDF with Newton, a dense LU and the exact Jacobian (CVODES's difference
quotients with `jacobian: finite_difference`).

- **Threads.** Blocks share nothing, so OpenMP runs over them with results that are the same for any number
  of threads.
- **Restarts.** A block is reinitialised (`CVodeReInit`) only when its inputs, its states, a trace or its time
  origin change. A trace's breakpoints are CVODES stop times.
- **Swapping solvers.** The library exports every `bk_*` function of `BE_batched_c`, plus
  `bk_block_cvodes_flag`, so a program that steps `bk_instance_*` copies can swap one library for the other.
  The one difference is `bk_instance_copy`: a copy starts CVODES afresh from the copied states, so it
  follows its original to the tolerance rather than bit for bit.
- **Settings.** The defaults are `CVODE_myokit`'s tolerances (`rtol` 1e-6, `atol` 1e-8).
  `MaximumNumberOfSteps` counts the steps of one block in one run, as on `BE_batched_c`.
- **Needs SUNDIALS 6 or later.** It is found from `SUNDIALS_DIR` or CMake's search path. Without SUNDIALS the
  build fails with a message that says so, and its tests skip.
- Both batched libraries also get `bk_solver_name`, and `bk_instances_advance`, which advances many
  instances on OpenMP threads.

### Fixed — libCellML 0.7's C for models with a state set by a computed constant, or with no algebraic variables

The one-`variables`-array rewrite of libCellML 0.7's C now keeps `voi`, `states` and `rates` in
`computeComputedConstants`. libCellML sets the states initialised by a computed constant there, so
`SN_simple` did not compile. The rewrite also no longer leaves a stray `{` in `VARIABLE_INFO` when
a kind of variable is empty.

### Changed — libCellML 0.7.2 or later

libCellML 0.7 generates Python and C with constants, computed constants, algebraic and external variables in separate
arrays. `PythonGenerator` and the C++ generator join them into the one `variables` array the solvers and the C++ model
use, so generated models behave as before: on the test models, the values (Python and C) and the generated C++
statements are the same as with 0.6.3. 0.6.3 flattened some models with wrong units (a conversion factor's units
replaced by their inverse), which 0.7.1 fixes (#271). The requirement is `libcellml>=0.7.2`, the PyPI
release: 0.7.1 drops an imported unit's exponent (SN_simple and the coupling `k_reduce` fail with a
units mismatch), which 0.7.2 fixes (cellml/libcellml#1498).

### Changed — a `volume_port` "sum" is an ordinary multi_port sum

No `port_type` has a meaning of its own. A whole-port `"sum"` (any case) on any port, a `volume_port`
included, is the list form with `"sum"` for every port variable: each variable is the sum, over every
module connected through the port, of the neighbour's corresponding variable, computed in
`multiport_sum_<module>_<variable>`. A port with several variables sums each of them, and on a
general port a neighbour counts whichever side of the connection it is drawn on, as in PhLynx.

- The `sum_blood_volume` component is gone. A blood-volume total such as the 3compartment
  `volume_sum` is now read as `multiport_sum_volume_sum_q/q` (it was
  `sum_blood_volume/q_volume_sum_sum`); obs_data, plots and checks that name the old output need
  the new name. The values are the same for every model in `resources/` and the module library.
- `FV1D_volume_sum`'s exit `volume_port` maps `q_sum`, so it is an ordinary term of the volume sum it
  feeds.
- With `couple_to_1d`, only a module next to a 1D vessel is refused a `"sum"`, `"Multiply"` or
  list-form port; sums elsewhere in the 0D model are allowed.

### Changed — a variable may be summed through several ports

A variable marked `"sum"` on two or more of a module's ports is the sum over the neighbours of all
of those ports, in one `multiport_sum_<module>_<variable>` component.

### Added — `"multiply"` in a list-form `multi_port`

A list-form port's entry may be `"multiply"` (any case), as PhLynx exports a port whose
variables have different types: that variable reaches its neighbour times its factor, like a
whole-port `"Multiply"`. `multiply_factor` may be a list aligned with the port's variables (a
number for each Multiply variable, null for the others). A variable pair cannot be Multiply on
both ends.

### Changed — a generated model has one parameters file, nested JSON; the generated CSV is gone

Generation writes one parameters file beside the generated model, `<prefix>_parameters.json` (in
the generated-models directory; the model's own input file of that name stays in its resources
directory), and no `<prefix>_parameters.csv`: every value the model uses, in the form of a
model's parameters file (`aortic_root/C`, `heart/lv/E_A`, globals under `global`), with its units
and data_reference. A value the model's parameters file does not set (a supermodule instance
written `"heart": {}`, a module's default parameterisation, a supermodule parameterisation's
global) names its source at the start of its data_reference:
`"from <module_type>/<module_subtype> parameterisation <name>: <its own data_reference>"`; a
data_reference already starting `from ` is not prefixed again, and none is cut at 80 characters.
Generating again from it alone, with the same module array, gives the same model. A
`<prefix>_parameters.csv` an older libcuflynx left in the generated model's directory is removed
when the model is generated again.

Everything that read the generated CSV reads the JSON, naming each parameter as the model does
with `parameters_file.model_parameter_rows` (`C_aortic_root`, `E_A_heart__lv`, `R`): the
calibration workflow (fixed values, identifiability's start point, the target's check of the
calibrated names) and the methods LaTeX (`cuflynx-methods-latex --parameters` takes the JSON; a
parameters file that does not exist, named or the default beside the model, is an error naming its
path rather than a methods section citing no source).

A calibration workflow's calibrated parameters (`write_calibrated`,
`write_calibrated_parameters_to`) are the target's parameters file completed with every other
value its model uses, with the calibrated values set, so a result reproduces the run without the
module library; `write_calibrated_parameters_to` builds on the run's own generated target
parameters file (regenerating the target only when the run has none) and still writes only its
`path`.

Renamed, so the names match the file: `generate_model(...)['used_parameters_path']` and
`workflow_result['target_model']['used_parameters_path']` are `parameters_path` (which named the
CSV); `runner.run_target_used_parameters` is `run_target_parameters`,
`runner.target_used_parameters` is `target_generated_parameters`, and the `used_path` argument
of `write_calibrated_parameters` is `generated_path`; `parameters_file.used_parameters_file_name`
is gone (`parameters_file_name`). A workflow run whose `workflow_result.json` records a parameters
CSV for its target model is refused when its calibrated parameters are written: run the workflow
again.

### Added — calibration workflow steps that select instances of the target

A step's model may be `{"select": ["C"]}` (or `["heart/lv"]`, or several records) against a
module-array target. One instance runs as its library module placed at that instance; several run
as the module array they form, cut from the target. The step starts from the target's values for
them, and its obs_data and params_for_id name vessels as the target does. A boundary condition
the selection leaves open takes its value from the step's protocol (`params_to_change`), and one
with no value anywhere raises `UnsetParametersError`, which lists them. Any workflow step's
protocol now gives the parameters its model would otherwise leave unset. A nested selection starts
from the model's values under its path (`heart/lv`). The calibrated values are written
(`write_calibrated`, and `write_calibrated_parameters_to(source, output_dir, path)`, which writes
only `path`) as a copy of the target's parameters file with them set, a value inside a super
instance under its path; the target's own files and the supermodule version's template are not
changed.

### Fixed — `CVODE_myokit` on modules in other time units (#604)

- A module whose time variable is in other units than the model's (e.g. `t` in ms in CA's
  seconds-based model) has its derivatives taken w.r.t. its own time, as CellML defines them.
  Myokit converted the module's `t` but not `dx/dt`, so such a module ran 1000x too slowly
  under `CVODE_myokit` (libCellML's analyser, behind the python and C++ models, already
  applies the factor). Results of
  models with such modules change: `module_config_user/tutorial_modules.cellml`'s
  `i_piezo_Na_tutorial` and `smc_hernandez_with_piezo` are the shipped ones.

### Changed — a supermodule names its submodule versions; its instances are its own module array

A supermodule version's config lists the module versions it is built from, `submodule_versions:
[{"module_type", "module_subtype"}]`. Its instances and their connections are a module array of their
own, `<module_type>_<module_subtype>_module_array.json` beside the config, whose records name no
`parameterisation`. Its default parameterisation is the template of a new instance's values. An inline
`submodules` list is refused; `cuflynx-upgrade module-config <file> --module-library-dir <library>`
converts it: the records to the module array, and each instance's values (from the parameterisation it
named, else its version's default, with the supermodule's default parameterisation's
`<instance>/<variable>` rows over them) into the default parameterisation, nested by instance.

### Changed — parameter values are nested JSON

A model's parameter values are `<prefix>_parameters.json` beside its module array (schema
`parameters.schema.json`), nested instance, sub-instance, variable:

```json
{"schema_version": 1,
 "global": {"R": {"value": 8.314, "units": "J_per_mol_K", "data_reference": "..."}},
 "aortic_root": {"C": {"value": 1.2e-8, "units": "m3_per_Pa", "data_reference": "Liang2009"}},
 "heart": {"lv": {"E_A": {"value": 366575000, "units": "J_per_m6", "data_reference": "..."}}}}
```

- A parameter is an object with `value` (a number, or text such as `EMPTY_MUST_BE_FILLED`; an empty
  one is read as `EMPTY_MUST_BE_FILLED`), `units`, `data_reference` and any other string key (a
  library's `sourced`). A number keeps the text it is written with. Every other object is an instance
  or sub-instance. `global` holds the global constants. `global`, `value` and `schema_version` are
  reserved: no module-array record, submodule or variable may have those names.
- A path is the name users write: `aortic_root/C`, `heart/lv/E_A`, `global/R` or `R`.
- A parameterisation, `parameterisations/<name>/<name>_parameters.json`, has the same form rooted at
  its module: a component's own variables at the top (`{"C": {...}}`) and its global constants under
  `global`; a supermodule's sub-instances (`{"lv": {"E_A": {...}}}`) and its globals.
- A supermodule instance's values are in the model's file under its name. `cuflynx-upgrade
  supermodule-instances <module array> --module-library-dir <library> [--instance N] [--replace]
  [--parameters FILE]` copies them once from the version's template (its default parameterisation,
  or the one the record names in its place, which must set every instance value the default sets,
  and nested supermodules' templates under it); an instance that
  has an entry keeps it. Generation refuses a supermodule instance the file has no entry for, naming that command.
  Of a supermodule's parameterisation it reads only the globals; a value the model's file lacks comes
  from the submodule version's `default_parameterisation`.
- Calibration workflows write `<name>_calibrated_parameters.json` in the same form: a module array
  target's full file with the calibrated values set, or a parameterisation's.
- A parameters CSV is refused. `cuflynx-upgrade parameters <file | *_modules_config.json | directory>`
  converts it to the `.json` beside it: a model's file (and the per-instance
  `<prefix>_supermodule_parameters/` files beside its module array, moved in under each instance), a
  parameterisation (globals by its config), a config's parameterisations (and a supermodule version's
  `<module_type>_<module_subtype>_parameters.json`, folded into its default parameterisation), or every
  one below a directory. Name the `.json` file in `input_param_file`.
- Generation writes the generated model's parameters as `<prefix>_parameters.json` beside the
  generated model (see "a generated model has one parameters file" above). A model with missing
  values gets `<prefix>_parameters_unfinished.json` in its resources directory.

### Changed — params_for_id is JSON

A calibration's parameters are `<prefix>_params_for_id.json` (schema `params_for_id.schema.json`), a
flat list of entries, not nested by instance: an entry groups parameters or modifies them, so it is
not one parameter.

```json
{"version": 1,
 "params": [
  {"targets": ["aortic_root/C"], "min": 1e-9, "max": 5e-8, "name_for_plotting": "C_{ao}"},
  {"targets": ["heart/lv/E_A", "heart/rv/E_A"], "min": 1e8, "max": 5e8},
  {"name": "R_scale", "modifies": ["venous_svc/R", "venous_ivc/R"], "modifier": "scale", "min": 0.5, "max": 2}]}
```

- Parameters are named as users write them: `<vessel>/<variable>`, a nested vessel by its path
  (`heart/lv/E_A`, read as the model's `heart__lv`) and a global as `global/<variable>`.
- A params_for_id CSV is refused (`params_for_id_file`, the default `<prefix>_params_for_id.*`, a
  workflow step's `params_for_id`). `cuflynx-upgrade params-for-id <CSV | directory>` writes the
  `.json` beside each CSV and removes it: one entry per row, `vessel_name "a b"` with `param_name "C"`
  as `"targets": ["a/C", "b/C"]`, the `prior_*` columns under `prior_params`, numbers as written.
  `ObsAndParamDataParser.params_for_id_csv_to_json` returns the same document.
- Calibration workflows write a step's params_for_id with its priors' parameters, and a reduction's,
  as `<step>_params_for_id.json`; a prior over parameters of different names is no longer refused.

### Added — trace inputs and time courses in `prediction_plots`

- An input reference (`x`, `series`) takes an optional `reduce` that makes a trace one number:
  `"final"`, `"mean"`, `"max"`, `"min"` or `{"at": <seconds from the start of the
  sub-experiment>}`. A trace without `reduce` is no longer refused when the obs_data is read:
  that experiment's point is left out with a `PredictionPlotWarning`.
- Two kinds drawn from a simulation: `input_and_output` (per experiment, a `params_to_change`
  input and a model output over time on two y-axes, with sub-experiment boundaries and any
  series data for the output) and `output_vs_input` (the output against the input at the same
  times, one line per experiment). Both take `{"name", "kind", "input": {"params_to_change"},
  "output", "experiments"?}`; the schema enforces each kind's keys.

### Changed — the heart is connected like any other module

Nothing in generation depends on a module being a heart or on an instance's name:
- A module's inputs take its entrance ports of one `port_type` in the order its
  `inp_instances` lists them (instances with no port of that type skipped), as its outputs
  already took its exit ports in `out_instances` order. The rows' order no longer decides it:
  a junction with two entrance ports of a type took its inputs in the order of their rows.
  A model whose junction inputs were listed against their rows' order lists them in the order
  it was generated with (CA's `physiological`, `control_phys*`, `cerebral*`, `elic`,
  `FinalModel` arrays are reordered; their models are unchanged).
- No zero-flow vena cava is added for a heart with one vena cava. List a `zero_flow`
  instance (`module_type` `zero_flow`, `module_subtype` `nn`, a new built-in module with a
  `vessel_port` exit) first among the heart's vessel inputs, as `3compartment` now does.
  Otherwise its two vessel inputs take the `v_ivc` and `v_svc` ports, and `heart/v_pvn` is a
  boundary condition to fill, named in the missing-parameters report. The `zero_flow` component
  is in a model's `_modules.cellml` only when an instance uses it.
- A model may have more than one heart.
- A 1D vessel is coupled to the port of its 0D neighbour that joins them, found as above,
  in the cpp generator and in `convert_0d_to_1d` (which takes the module sources as
  `module_sources_inputs`); a heart's 1D outflow vessel no longer needs `aorta` in its name,
  nor its venous inputs `ivc`, `svc` or `pvn`.
- The CasADi solver diagnostics report, for each stiff mode, the inertance `I_<module>` of the
  module of its dominant state and the inertance that would make it steppable, in place of the
  `par` and `systemic_T` estimates.

### Changed — `solver: CVODE` for a cellml model is `CVODE_myokit`

`solver: CVODE` (or `solver_info.solver: CVODE`) with `model_type: cellml` is read as
`CVODE_myokit` when user_inputs are read, with one printed line saying so; a `solver_info` error
then names the shorthand. The dated `user_inputs_<yymmdd>.yaml` keeps `CVODE`, as written; a
pipeline bundle's config records `CVODE_myokit`, since its `run_pipeline.py` does not go through
the reader. `CVODE` for a cpp model is unchanged. The shorthand is listed in
`SOLVER_SCHEMA['solver_aliases_by_model_type']`.

### Changed — a nested vessel is written by its path

An output or obs_data operand names a nested vessel by its path, as a host parameter already
does: `SN/soma/membrane/V` (everything before the last `/` is the vessel's path) for the
model's `SN__soma__membrane/V`. The flat `__` form is still accepted. A host parameter without
a vessel (`R`, `global/R`) that no module of the model declares as a global constant is
refused, as `<vessel>/<variable>` names the model doesn't have already are; models with 1D
vessels are not checked.

### Removed — older module config, module array, parameter name and obs_data forms

libcuflynx reads only the current forms of its input files. A file in an older form is refused
with an error naming the `cuflynx-upgrade` command that converts it once, in place. Convert a
model's module configs and module array before its parameter names.
- Module configs (`cuflynx-upgrade module-config FILE`): the keys `vessel_type`, `BC_type`,
  `module_file` and `module_type` (the component) -> `module_type`, `module_subtype`,
  `component_file`, `component_type`; `default_instance` -> `default_parameterisation`; a
  supermodule's `default_parameters` CSV -> its parameterisation `parameterisations/default/`;
  `multi_port: "False"` (leave it out, or `null`/`false`).
- Module arrays (`cuflynx-upgrade module-array FILE`): the keys and columns `vessel_type`,
  `BC_type`, `inp_vessels`, `out_vessels`; `<prefix>_vessel_array.*` file names (renamed
  `<prefix>_module_array.*`). A heart with one output no longer gets `par` and `pvn` added, by
  the generator or the converter: a heart is a module like any other, and the array lists every
  module. A model that relied on them adds the two rows the generator used to add,
  `par,arterial_simple,vp,<heart>,pvn` and `pvn,arterial_simple,vp,par,<heart>`, and their
  parameters. A heart's vessel inputs fill its ports in order, `v_ivc`, `v_svc`, then `v_pvn`,
  and its vessel outputs `v_aov` (aorta) then `v_puv` (`par`): list the inferior vena cava, the
  superior vena cava, then `pvn` in its inputs, and `par` after the aorta in its outputs. A
  heart with only one vena cava lists a `zero_flow` instance first in its inputs, as the
  `3compartment` arrays do. Without the two rows the heart's pulmonary ports are unconnected,
  so `heart/u_par` and `heart/v_pvn` are boundary conditions that need values.
- Generation lists the missing parameters, and names each boundary condition that no connected
  module supplies with its module, so an unconnected port is easy to find.
- Parameter names (`cuflynx-upgrade parameter-names --module-array ARRAY` or
  `--supermodule-config CONFIG`, with `--module-library-dir` for library modules): a host
  parameters file names a module's parameter `<vessel>/<variable>` (`aortic_root/C`), and
  `C_aortic_root` is refused; a name without `/` is a global. A supermodule parameterisation row
  is `<submodule>/<variable>`; `{var}_{sub}` is refused. Nested levels joined with a single `_`
  are refused (`__` is the join).
- Parameterisations: a version with an `instances/` directory and no `parameterisations/` is
  refused; rename the directory.
- obs_data (`cuflynx-upgrade obs-data FILE --user-inputs user_inputs.yaml`, or `--pre-time T
  --sim-time T`, which win over the yaml's values): an obs_data must
  have a `protocol_info` with `pre_times` and `sim_times`. A bare list of data items, or an
  object without `protocol_info` timed by user_inputs.yaml's `pre_time`/`sim_time`, is refused.
  `parse_obs_data_json` no longer takes `pre_time` / `sim_time`.
- `python -m libcuflynx.utilities.config_schemas to-json` no longer takes `--style`, and
  `module_array_to_json` no longer takes `style`: they write the current keys.

### Added — `cuflynx-upgrade`

`cuflynx-upgrade module-config | module-array | obs-data | parameter-names` converts files written
for an earlier libcuflynx, in place, and prints what it changed; `--dry-run` only prints. A file
already in the current form is left as it is. It replaces `cuflynx-migrate-parameter-names`, which
is removed (`cuflynx-upgrade parameter-names` takes the same options).

### Removed — old obs_data and params_for_id spellings, and `cuflynx-migrate-obs-data`

No shipped or library file used them, so they are no longer read; the schema refuses them by name
("'variable' was unexpected"):
- obs_data items: `variable` (use `data_item_name` and `operands`), `name_for_plotting` (use
  `trace_name_for_plotting` / `item_name_for_plotting`), `obs_type` (use `operation`),
  `vm_path` / `im_path` and paths inside a `source` object (use `t_path` and `value_path`), and
  `data_type: "timeseries"` (use `"series"`); the top-level `data_item` key (use `data_items`).
- params_for_id: `operation` on a modifier entry (use `modifier`), and on a saved modifiers
  record.

- obs_data's `data_type: "prob_dist"` no longer gets its own advice: it is refused like any
  other unknown data_type. An observable scored against a distribution is a `constant` with a
  distribution `cost_type` and `prob_dist_params`.

`cuflynx-migrate-obs-data`, which rewrote those spellings, is removed with them. A bare list of
data items is refused too (see above).

### Removed — old configuration keys and Python names

`user_inputs.yaml` keys and values that are no longer read are refused with a `ValueError`
naming what to write instead:
- `ga_options` / `debug_ga_options` -> `optimiser_options` / `debug_optimiser_options`.
- `do_mcmc` / `mcmc_options` / `debug_mcmc_options` -> `do_uq` / `UQ_options` /
  `debug_UQ_options`.
- `model_type: cellml_only` -> `cellml`.
- AADC `method: bdf_tape` / `bdf_kernel` -> `method: semi_implicit_signed` with
  `gradient_strategy: tape` / `kernel`.
- `sa_options.gradient_method: analytic` -> `auto`.
- solver_info keys of another solver are no longer moved across: `MaximumStep` on solve_ivp
  (`max_step`), casadi_integrator (`max_step_size`) or the cpp solvers (`dt_solver`), and
  `MaximumNumberOfSteps` on solve_ivp or CVODE_myokit (no such setting) or casadi_integrator
  (`max_num_steps`). The error lists the solver's own keys.

External funcs files import the built-in funcs as `libcuflynx.funcs.cost_funcs_user` (and
`operation_funcs_user`, `modifier_funcs_user`); the bare `from cost_funcs_user import ...` is
refused with that import.

Python names removed (use the second):
- `OpencorParamID`, `OpencorMCMC` -> `ParamID`, `MCMC`.
- `CVS0DParamID.run_mcmc()` -> `run_UQ()`; the `mcmc_options=` argument of `CVS0DParamID`,
  `run_UQ`, `MCMC`, `MCMC.from_param_id` and `MCMC_plotter` -> `UQ_options=`.
- `PARAM_MODIFIER_OPERATIONS`, `DEFAULT_PARAM_MODIFIER_OPERATION`, `param_modifier_operations`
  -> `PARAM_MODIFIERS`, `DEFAULT_PARAM_MODIFIER`, `param_modifiers`.
- `MODEL_TYPE_ALIASES`, `normalise_model_type`, `AADC_LEGACY_METHOD_ALIASES`,
  `migrate_legacy_solver_info_keys` (none needed), `paramID.TAPE_CONSISTENT_AADC_METHODS` ->
  `aadc_backend.TAPE_CONSISTENT_METHODS`, `CSV0DModelParser.split_0d_1d_vessel_array` ->
  `split_0d_1d_module_array`, `sobol_SA.create_SA_info` -> `_create_SA_info`.
- `libcuflynx.utilities.protocol_funcs` -> `libcuflynx.protocol_runners.ProtocolRunner`, and
  the dead `libcuflynx.obsolete` package.

The exported `run_pipeline.py` imports `libcuflynx.*` only and reads `do_uq` / `UQ_options`.

### Changed — input files are checked against their JSON Schemas on read

obs_data, params_for_id, module config, module array and calibration workflow files are
validated against the schemas shipped in `libcuflynx/schemas/` whenever libcuflynx reads them,
so what an editor, PhLynx or CUFLynx validates is what libcuflynx accepts. `jsonschema` is a
core dependency.

- An error lists every problem, each with where it is (an item by its name) and what the field
  is for: `at data_items[2] ('systolic').value: [1.0, 2.0] is not of type 'number' (A
  constant's value and std are single numbers.)`.
- obs_data and params_for_id: the schema replaces the readers' own key and type checks, which
  are gone. Errors about unknown keys, types and missing settings read differently. Unknown
  `prior_params` keys, which were ignored, are refused, and prior names are still read in any
  case.
- Calibration workflows: schema first too. `spec.py` keeps only what spans keys and steps
  (which kinds use which sources, an emulator step's own values, where a model sits).
- Module configs and module arrays: their readers' checks run first, then the schema.
  The schema refuses what the readers did not check, for example `variables_and_units` access
  other than `access`/`no_access`, a variable kind other than `constant`, `global_constant`,
  `boundary_condition` or `variable`, and a port with no `variables`.
- Enums are generated from libcuflynx's registries: `python -m libcuflynx.schemas.registries
  --write`. The variable kind is now an enum, and `module_config_user/BVC_Kidney_module_config.json`
  had two `"variable "` (trailing space) fixed.
- PhLynx main's CA-format files and value lists are tested against the readers and schemas
  (`tests/test_phlynx_consistency.py`, `PHLYNX_DIR`; CI checks PhLynx out).

### Changed — `time` means the same on every backend (#557)

- `get_results(['time'])` and `get_time()` are the cumulative time after the experiment's
  warm-up, the axis of the executor's `t_by_exp`, in every sub-experiment. **With
  `pre_time > 0` this changes what the `python`, `casadi_python`, `aadc_python`,
  `external_python` and OpenCOR backends return**: they gave the raw solver clock
  (warm-up included); OpenCOR's `get_time` restarted at 0 in each sub-experiment; Myokit was
  `pre_time` too large after the first sub-experiment. A time-based operation (e.g.
  `first_peak_time`) on those backends now reads the same times Myokit gives.

### Fixed — traced inputs on Myokit (#556, #559)

- A trace or shape of an experiment's first sub-experiment starts at that sub-experiment, not
  at the start of the warm-up, and holds its first value through the warm-up.
- Several inputs can follow traces in one sub-experiment: each is bound to its own Myokit label
  (`pace`, `pace_1`, ...; a variable that already has a binding keeps it).
- The offline warm-up (`offline_pre_time`) holds each traced input at its first value of
  experiment 0, instead of 0.

### Fixed — PhLynx's `instance_parameters` / `global_parameters` (#558)

- Parameters PhLynx (v0.3.0 on) hoists into `instance_parameters` and `global_parameters` resolve,
  in the solvers, in `model_qname_candidates`, the variable mapping, methods and the
  calibrated-value writer. One list, `parameter_names.HOISTED_PARAMETER_CONVENTIONS`, holds the
  conventions they all read.

### Added — predictions without a calibration (#569)

- `libcuflynx.protocol_runners.run_prediction_items(model, obs, dt, solver_info, ...)` runs an
  obs_data's protocol and returns its prediction items as traces, scalar features, per-segment
  results and validation; `predict_on` does the same on an executor you have.
- `cuflynx-run-predictions MODEL OBS_DATA --dt ...` writes `traces.npz`, `features.csv`,
  `validation_results.json` and the `prediction_plots` figures.
- `ProtocolRunner.run_protocols` takes `result_variables` and `return_segments`; `model_path` is
  optional. **Its join of sub-experiments pads a scalar result to its own segment**
  (`int(sim_time/dt)+1` samples, as the solvers step); it used to pad sub-experiment 0 to the
  whole experiment and use `round`, so the result could be longer than `t`.
- `save_prediction_data` is built on `predict_on`; its files are unchanged.

### Added — `prediction_plots` in obs_data (#568)

- An optional top-level list of figures: a scalar feature against another feature or against
  the number an input is set to, across experiments, optionally one line per value of a series
  input. A group is the data_items and prediction_items sharing an `item_name_for_plotting`.
  Drawn by `save_prediction_data` and `cuflynx-run-predictions`; see the tutorial.

### Changed — PhLynx's key names (#163)

- Module configs and module arrays use PhLynx's keys: `module_type`, `module_subtype`,
  `component_file`, `component_type`, `inp_instances`, `out_instances`. **Files CA writes use
  them**, and so do `resources/`, `module_config_user/` and the built-in module configs. The
  older `vessel_type`, `BC_type`, `module_file`, `module_type` (= the component, beside the other
  older keys), `inp_vessels` and `out_vessels` are refused; `cuflynx-upgrade module-config` and
  `cuflynx-upgrade module-array` convert them. Code reading a module dataframe's columns, or a
  normalised record's keys, sees the new names.

### Changed — nested names, path references and interfaces (#582)

- **Nested levels are joined with `__`**: the submodule `lv` of the supermodule instance `heart`
  is `heart__lv`, its parameter `E_heart__lv`. A module-array or submodule name may not contain
  `__` nor begin or end with `_`. Names joined with a single `_` (`heart_lv/q`, `E_heart_lv`)
  are refused in parameters files, params_for_id and obs operands; `cuflynx-upgrade
  parameter-names` converts a parameters file.
- A host may name a submodule by its path (`heart/ra`, `neuron/soma/membrane`) in its own
  inp/out list, keeping its own order; a `per_submodule_*` key may be a path into a nested
  instance.
- A supermodule entry may declare an `interface` that hosts bind to by port type.
- `phlynx_*` keys on records and config entries are read and ignored.

### Fixed — docs of `offline_pre_time` (#560)

- The tutorial lists every backend that runs it (now also OpenCOR and `aadc_python`) and the
  calibration-parameter restriction; the benchmarks note and the Myokit docstring describe the
  run-once behaviour.

### Added — packing a calibration workflow

`pack_workflow(workflow, dest_dir)` and `cuflynx-calibration-workflow <workflow> --pack DIR`
copy every file a workflow needs to run into one directory: the workflow rewritten to the
packed files, each step's obs_data, params_for_id and module array with its parameters file,
each module version it uses (with every units file of its library), and the files its settings
name. Results are not packed. CUFLynx's OMEX export packs a workflow with it.

### Changed — workflow settings paths are relative to the workflow file

`external_modules_dir` and the `*_funcs_external_path` settings of a calibration workflow are
read relative to the workflow file, like its other paths, not to the directory it is run from.

### Added — analysis steps in calibration workflows

A workflow step has a `kind`: `calibration` (the default, so existing files are unchanged),
`sensitivity`, `reduction`, `emulation` or `identifiability`, so one `calibration_workflow.json`
records the whole order of a study.

- `sensitivity` runs Sobol or local sensitivity analysis (`settings.sa_options`).
- `reduction` ranks a Sobol step's parameters (`sensitivity_from`) and keeps the `top_n`
  (`reduction`: `index_type`, `criterion`, `threshold`), writing those rows of its
  params_for_id; a calibration fits them with `params_for_id_from`.
- `emulation` trains an emulator (`settings.emulator_settings`); a calibration or sensitivity
  step with `emulator_from` runs on it, with a copy of the emulation step's model.
- `identifiability` runs a Laplace analysis (`settings.ia_options`) at the values of the
  calibration in its `result_from`.

An emulation step may be trained on a reduction's parameters (`params_for_id_from`), and a
calibration with the same `params_for_id_from` and `emulator_from` pointing at it can sample
its posterior on that emulator (`store_distribution`). `SobolReduction`
(`libcuflynx.sensitivity_analysis.parameter_reduction`) is the reduction, usable on its own.

Only calibrations are merged into the target. The study manifest, `--dry-run` plan, status and
methods LaTeX (text and flow chart) show each step's kind, what it takes from where and its
outputs. `run_identifiability_analysis` takes `best_param_vals`, and Sobol indices are ranked by
`libcuflynx.sensitivity_analysis.sobol_ranking`.

### Removed — `cuflynx-sequential-param-id`

The command and `user_run_files/run_sequential_param_id.sh` are gone, with the script and the
two obsolete copies behind them. The stage was never implemented: the command only said so.
Ordered calibrations and analyses are run by `cuflynx-calibration-workflow`.

### Changed — multi-port coupling follows PhLynx (#575)

- **Multiply on either end.** A `"Multiply"` port scales its variable into the neighbour
  whichever end of the connection it is on; a downstream Multiply port was mapped like `"True"`
  and its `multiply_factor` ignored. Multiply on both ends of a connection is a `ValueError`.
- **One sum port per variable.** A variable summed through two different ports is a
  `ValueError`; terms used to be merged into one sum.
- **Boundary conditions.** Only a BC that a plain mapping connects to a non-blank partner, or
  that is a sum result or Multiply output, is removed. A BC that is only a sum term, the source
  of a Multiply, or mapped to a blank (`""`) slot keeps its own value (it used to lose it and
  leave the model underdetermined). BCs connected only to each other are a `ValueError`.
  The built-in `pulmonary_GE` computes `C_CO2_a` and `C_O2_a`, so they are now `variable`, not
  `boundary_condition`.
- **Units.** A coupling between different but compatible units is mapped directly, and the
  solver converts it in the direction the value flows; the generated `unit_converter_*`
  components are gone. Couplings between affine units (kelvin and celsius) are a `ValueError`.
- **Port sizes.** A plain port pair with different numbers of variables is a `ValueError`; it
  used to raise an `IndexError` or silently shift later mappings.

### Added — JSON Schemas for all four input files

- **`params_for_id.schema.json`**, new: the JSON form of params_for_id (version 1) that a CSV is converted to on read. Its entry keys, priors and prior settings are tested against `PARAMS_FOR_ID_ENTRY_KEYS` and `PARAM_PRIOR_TYPES`.
- **`obs_data.schema.json`** now describes the data items, prediction items and `protocol_info` (with `protocol_traces` and `protocol_shapes`), not just the top level: every key with its default, and the rules the parser enforces (unknown keys, required keys, constant vs series shapes, a ground truth per item).
- **`module_config.schema.json`, `module_array.schema.json`**: every key described; `variables_and_units` rows checked as `[name, units, access, kind]`; `multiply_factor` only on a Multiply port (and as a numeric string); list-form `multi_port` entries `"sum"`/`"True"`; `only_one_port` and `delay_info` added; names may carry surrounding whitespace, as the reader strips it.
- Tests check each schema against its reader: key sets, and inputs both refuse.

### Added — parameter names written `<vessel>/<variable>`

- **The `/` form.** A parameters file may name a module's parameter `aortic_root/C` (or `heart/lv/E`, for a supermodule's expanded submodule; `global/R` or `R` for a global). A supermodule's parameterisation and `default_parameters` rows name a submodule's parameter `lv/E`, or `soma/i_M/rho_M` when nested.
  - Generated models keep their names (`C_aortic_root`), because CellML names cannot contain `/`.
- **Why.** The `_` form cannot be split reliably when names contain `_`, which supermodule rows had to do: `g_leak_Na` is `g` of `leak_Na` or `g_leak` of `Na`.
- **The old form is refused**, in host parameters files and supermodule rows; `cuflynx-upgrade parameter-names` converts it.
- **Checks on the new form.**
  - A host `/` row naming a vessel or variable the model does not have, or a global constant through a vessel, is an error.
  - A supermodule row naming a path that is not a submodule is an error.
  - A parameterisation's rows must stay bare variable names.
- **`cuflynx-upgrade parameter-names`** rewrites old names, in a host file (`--module-array`) or in a supermodule's files (`--supermodule-config`). It rewrites a name only when exactly one reading of it names a real variable of that module, and reports the others.

### Changed — local SA differentiates prediction features analytically

With `sa_options.include_prediction_items`, local SA's prediction-feature rows now use the same
arm as the data_item rows (CasADi jacobian, or Myokit CVODES sensitivities) instead of always
central finite differences: one extra solve rather than 2M. The prediction features are written
as the data_items of a second engine and differentiated by the existing code. Finite
differences are still used with `gradient_method: FD`, over an emulator, and, with a warning,
when Myokit's arm cannot take them: it needs every prediction feature in one and the same
sub-experiment.

### Added — every analysis records itself in a study manifest CUFLynx opens

Calibration, UQ, sensitivity analysis and emulator training now record themselves in
`<param_id_output_dir>/<study_name>_cuflynx_study.json` (`libcuflynx.utilities.cuflynx_study`):
the model (the calibrated one once written), obs_data, params_for_id, the
`*_funcs_external_path` files, the solver settings and `dt`, and each output directory —
including sensitivity results written outside `param_id_output_dir`. A calibration or MCMC run
is entered when it starts (`status: running`), so a run cut short is there with what it wrote
each iteration, and is marked `finished`, `stopped` or `failed` when it ends. Dropping that
file into CUFLynx reopens the whole study. Two new user_inputs keys: `study_name` (default
`file_prefix`) and `write_cuflynx_study` (default `true`). Writing is atomic, merges rather
than overwrites, and only ever warns on failure. `cuflynx-param-id` now also forwards
`modifier_funcs_external_path`, which it used to drop.

A calibration workflow writes one manifest for the whole run,
`<output_dir>/<workflow_name>_cuflynx_study.json`, whose `workflow` key records the workflow as
run, its result, the target (as named and as generated, and its calibrated parameters file) and
each step (model, obs_data, params_for_id, sources, status, run directory). Each step's own
manifest links back with `workflow: {manifest, step}`. `write_cuflynx_study` is also a workflow
setting.

### Changed — a module type several sources define comes from the most specific one

A `(vessel_type, BC_type)` defined in more than one module source used to keep the first entry
in the type registry, then stop generation with a print and `exit()`. That is why a module
library that copies built-in modules needed `use_builtin_modules: false`.

- **Precedence.** The type is now taken from the most specific source: `external_modules_dir`,
  then `module_library_dirs` in order, then `module_config_user`, then the built-in modules.
- **One rule for everything.** The type registries (so instances resolve next to the winning
  config), the module-config join and the CellML modules file all apply it, through
  `ModuleSources` (`excluded_entries`, `excluded_components`, `component_registry()`,
  `supermodule_registry()`).
- **What is left out.** The shadowed definitions' config entries and CellML components. A
  component that a type still in use needs is kept.
- **Competition is reported.** A `ModuleShadowWarning` lists every type defined more than once
  and the source that won.
- **Within one source, a duplicate is a `ValueError`** naming the files. So is a repeated type
  at the module-config join, which used to `exit()`.

### Changed — vessel arrays are now called module arrays

The file is `[file_prefix]_module_array.json` or `.csv`, and every model in `resources/` has been
renamed. A `[file_prefix]_vessel_array.json`/`.csv` is refused; `cuflynx-upgrade module-array`
renames and converts it. The
functions and constants that carried the old name are renamed in the same way
(`module_array_path`, `read_module_array_records`, `load_module_array`, `module_array_to_json`,
`PHLYNX_MODULE_ARRAY_COLUMNS`, ...), without aliases, as none of them was released;
`CSV0DModelParser.split_0d_1d_vessel_array`, which was, is now `split_0d_1d_module_array`
(the old name is removed). The 0D/1D split writes `[file_prefix]_{0d,1d}_module_array.csv`,
`convert_0d_to_1d` writes `[model]_hybrid_module_array.*`, and the JSON Schema is
`libcuflynx/schemas/module_array.schema.json`. The `vessels_csv_abs_path` config key is unchanged (the record keys are renamed by #163, above).

### Added — writing a model and its calibration up in LaTeX

- **`cuflynx-variable-mapping`** writes a model's `<prefix>_variable_mapping.csv`.
  - It lists every variable and parameter by its canonical `component/variable` name, with a LaTeX symbol from a default rule (Greek names, `_` subscripts, the module as a last subscript).
  - Regenerating the file keeps edited symbols.
- **`cuflynx-methods-latex`** writes a methods section in LaTeX, for a calibration workflow or for one calibration:
  - the equations of each module in the mapped symbols, with their parameter tables;
  - a table of every parameter with its value and source: BibTeX keys in a parameter's
    `data_reference` are cited (from the modules' `*_references.bib`, or `--bib`), the cited
    entries are written to a `.bib` beside the `.tex`, and a calibrated parameter names the
    step that calibrated it;
  - a TikZ flow chart of the calibration;
  - the protocol, observables, parameters, method, cost function, fixed values, priors, MCMC settings and results of each step.
- **`libcuflynx.reporting`** is the Python API for both.

### Added — calibration workflows

`cuflynx-calibration-workflow <workflow.json>` (and
`libcuflynx.calibration_workflow.run_calibration_workflow`) runs calibrations in order and merges
their results into one target model. Each step names the three inputs of an ordinary param_id
run -- `model`, `obs_data` and `params_for_id`, paths relative to the workflow file -- so a
workflow works in any repository. A model is a CA module array, or a module of a module library
(`{module_type, version, parameterisation}`).

- **Names between models.** A module array keeps its vessel names (`vessel_map` renames them
  into the target's); a library module sits at a `submodule` of a supermodule target or a
  `vessel` of a module-array target, inferred when it occurs once. `--dry-run` checks that every
  calibrated parameter lands on a vessel of the target.
- **Fixed values.** A step can take earlier steps' calibrated values as fixed parameter values
  (`fixed_from`); it runs after the steps it takes values from.
- **Distributions and priors.** A step can store its MCMC posterior (`store_distribution`:
  chain, samples, stats; `StoredDistribution` samples and evaluates it). A later step can use it
  as a joint prior (`priors_from`), represented as a multivariate normal, independent normals or
  a KDE, each on a bounds-respecting scale.
- **Reruns.** `--from-step` / `--only` rerun part of a workflow, refusing when an input it would
  reuse has changed. `--dry-run` checks everything against the library and prints the plan.
  `--write-calibrated` writes the merged set beside the target's parameters file, as
  `<name>_calibrated_parameters.csv`, naming parameters `<vessel>/<variable>` and replacing rows
  written in either form.
- **A parameter calibrated by two steps** takes the later step's value, with a warning that
  suggests `priors_from` for a Bayesian update instead.
- **A step's `submodule`** is written `soma/i_M` when nested (`soma_i_M` is also read).
- **Schema.** `libcuflynx/schemas/calibration_workflow.schema.json`.
- **param_id.** `ParamID.set_joint_priors` / `CVS0DParamID.set_joint_priors` add log densities
  over several parameters to the log prior. `run_param_id` passes `joint_priors` from its input
  dict to both the calibration and the MCMC, and now returns
  `{'output_dir', 'best_param_vals', 'param_id', 'mcmc'}` instead of None.
- **Testing.** `libcuflynx.external_testing.workflow_library.build_workflow_library` writes a
  small module library with two workflows, for tests here and in CUFLynx.
- **Docs.** The params_for_id prior columns are now documented; the docs had said only uniform
  priors existed.

### Changed! — `model_type: cpp` is generated from libCellML's C output and Jinja2 templates

The C++ generator is rewritten (`libcuflynx/generators/cpp/`). The model equations are
libCellML's C code (`model0d_core.c/.h`), with its non-state variables in one `variables` array
and named indices; the `Model0d` class, the solvers, `main0d` and a `CMakeLists.txt` are rendered
from templates. Build with CMake (add `-DSUNDIALS_DIR=<prefix>` if SUNDIALS isn't found; versions
5-7 work). Solvers are CVODE and the fixed-step RK4/Heun/midpoint/explEul; **PETSC is no longer
supported** and is refused when `user_inputs.yaml` is validated. Generated models no longer get
the `solver1d/Make_files` copied next to them (they built the old C++), and the coupler builds
with CMake. FV_1d coupled output is identical to before.

A model coupled to 1D can have a volume sum over its 1D vessels: the 1D solver sends their volume,
which is a term of the volume sum, and it needs no value in the parameters file (the old generator
stopped with `KeyError: 'bc_inlet0_or_outlet1'`). A value the 1D solver sends never needs one.

### Added — `api` blocks: couplings to other models described in module configs

A module config entry can carry an `api` block. `role: consumer` describes calls the generated
C++ makes (the FV 1D named-pipe protocol is now described this way in
`coupling_modules_config.json`); `role: provider` generates a C++ class another program calls,
e.g. a drop-in `lifex::Circulation`. A provider is its own module-array row
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

`convert_0d_to_1d` adds an `FV1D_solver` row to the hybrid module array, and reads
`<model>_module_array.csv` when there is no `<model>_0d_module_array.csv`. The coupler creates
the pipe folder, and its default pipe folder and Python are no longer paths on one machine.
`main0d`'s coupled defaults (`T0`, `nCC`) match the configuration. New example model
`aortic_bif_0d` (all 0D); a test runs it against the same model with its vessels in 1D.

### Added — coupling generated C++ to external Python models (e.g. FEniCS)

A module config entry with `"module_format": "external_api"` and
`"api": {"role": "provider", "transport": "python", "python": {"file": ..., "class": ...}}` is an
external Python model: a row of the module array, connected to CellML modules through its ports.
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
  `mol_per_m3` and `millimolar`) directly. It used to create an unconnected "converter".
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

### Changed — module instances are now called parameterisations

A module version's named parameter sets live in `parameterisations/<name>/<name>_parameters.csv`
(with `<name>_obs_data.json` and `<name>_params_for_id.csv`); a record or submodule picks one with
`"parameterisation"`, and a config entry names its default with `"default_parameterisation"`.
"Instance" now means one use of a version in a module array, as a PhLynx node does. The older
names are refused: rename an `instances/` directory to `parameterisations/`, and convert
`"default_instance"` with `cuflynx-upgrade module-config`. A module-array record or submodule must use
`"parameterisation"`: `"instance"` there is a ValueError, and invalid in the JSON Schemas. The
module is `libcuflynx.utilities.parameterisations` (was `module_instances`), with
`record_parameterisation`, `parameterisations_dir`, `parameterisation_parameters_path`,
`available_parameterisations`, `parameterisation_rows` and `component_parameterisation_rows`.

### Added — prediction items as scalar features

A `prediction_item` may carry an `operation` and `operation_kwargs`, with the same vocabulary and
checks as a data_item. Such an item is a scalar feature, e.g. the max of a trace over its
experiment, reduced over the last sub-experiment of that experiment. `prediction_info` gains
parallel `operations` and `operation_kwargs` columns (`None` / `{}` when absent).

- **Validation.** Held-out data on such an item (`data_type: constant`) is compared with
  `operation(operands)`. `save_prediction_data` records every operand of the item, so an
  operation with two operands works. Each item in `validation_results.json` now names its
  `operation`.
- **SA.** New `sa_options.include_prediction_items` (bool, default false). Sobol and local SA
  report these features as extra outputs, labelled `<data_item_name> (Exp<e>, Sub<s>)`. The
  run also writes `sobol_output_features.json`, which says what each output column is. Local SA
  computes these rows by finite differences.
- **Emulators.** New `emulator_settings.include_prediction_items` (bool, default false). The
  emulator is also trained on these features. The bundle records them in
  `prediction_feature_labels`, and they get a separate `prediction_sha256` fingerprint.
  Calibration on such an emulator still fits and checks only the data_item features. An SA that
  asks for prediction features on an emulator trained without them stops and says to retrain.

**Sub-experiments and validation-only experiments.** A prediction item may set
`subexperiment_idx` (default: its experiment's last; `prediction_info` gains
`subexperiment_idxs`). Validation, the features and `save_prediction_data` use that segment, and
series times run from its start, as for a data_item series. An item on a non-default segment is
saved in `prediction_variable_data_exp_<e>_sub_<s>.npy`; the existing files are unchanged. For a
multi-sub-experiment experiment, held-out series times now start at the start of the
sub-experiment rather than the experiment. An experiment that no data_item belongs to is not
simulated during calibration: not in any cost path, the best-fit check, or the
all-outputs npz. It is simulated for the prediction data and validation, and for SA/emulator
training only when a prediction feature needs it. A one-line message names such experiments.
`libcuflynx.parsers.PrimitiveParsers.cost_experiment_idxs(protocol_info)` lists the ones the
cost uses. An obs_data with no data_items at all has nothing to fit: calibration and UQ refuse it
with a ValueError before simulating anything (it used to crash with an IndexError).

**Scalar or series.** Every data_item and prediction_item is a scalar (`constant`: value and
std are numbers) or a series (`series`: value is a list, std is a number or a list of the same
length, obs_dt is required). The parser checks this and names the item; before, a `constant` with
a list value was accepted, and a `series` with a number crashed. A series prediction item may
have an operation, and is validated with the operation's series. Only scalar prediction items
are features.

Only scalar prediction items with an operation become features. The others are skipped, with a
`PredictionFeatureWarning` that names them, and an operation that does not return a scalar
raises `NonScalarPredictionFeatureError`. With both options off, SA outputs and emulator
fingerprints are unchanged. Callers can feature-detect with
`libcuflynx.sensitivity_analysis.SUPPORTS_PREDICTION_FEATURES`.

### Added — held-out data in `prediction_items`, scored after calibration

A `prediction_item` may carry data that calibration never fits: `value` (with `data_type`,
`std`, and `obs_dt` for a series). `prediction_info` gains parallel `data_types`, `values`,
`stds` and `obs_dts` columns. After a calibration, `libcuflynx.param_id.validation` compares the
calibrated model's prediction with the held-out data (a series at `k*obs_dt` within the run, a
constant at the end of its experiment), and `save_prediction_data` writes
`validation_results.json` with rmse, nrmse, mean |z|, the fraction within 2 std, and the model
and data at the observation times. Prediction items without a `value` behave as before. See
`tutorial/docs/parameter-identification.md`.

The `std` is optional; without it there are no z-scores. When given, it is checked like a data
item's: one finite positive number for a constant, and for a series one such number or one per
point. A zero, negative or wrong-length `std` is a parse error naming the item.

### Added — module versions and instances

A module library can lay a module version out as `<module_type>/versions/<version>/` with named
parameter sets in `instances/<instance>/<instance>_parameters.csv` (names without a vessel
suffix). A module-array record, or a supermodule's submodule, picks one with
`"instance": "<name>"`; without it, the config entry's `"default_instance"` is used if that file
exists. The instance directory is found next to the config file the record's type came from.
Rows become `{var}_{vessel}`, except the module's `global_constant`s, and are merged as default
parameters: the host parameters file wins, then a supermodule's instance, then its
`default_parameters`, then submodule/component instances. A supermodule entry can have instances
too, named like `default_parameters` (`{var}_{submodule}` or global); `default_parameters` still
works. An unknown instance is an error that lists the version's instances. Models without
instances generate byte-identical CellML.

A global constant is one value for the whole model, so when instances set one to different values
the first is used and a `ConflictingGlobalWarning` names each value, its units and the record that
set it. The warning is skipped when the host parameters file sets the global, and when a
supermodule's instance overrides the instances inside it. It also says when the units differ: two
modules then most likely mean different quantities by one name (in the module library, `T` is a
temperature in the ion channels and a period in the cardiac clock).

obs_data files accept a top-level `"obs_data_name"` (returned as `obs_data_name` by
`parse_obs_data_json`); a file in `instances/<name>/` whose `obs_data_name` is not `<name>` is
warned about. The JSON Schemas gain `instance` and `default_instance`, and a new
`obs_data.schema.json` describes the top level of an obs_data file.

### Added — JSON module arrays and supermodules

A module array can be a JSON list of records, `[file_prefix]_module_array.json`, with PhLynx's
keys (`name, module_type, module_subtype, inp_instances, out_instances`) or libcuflynx's (`name,
vessel_type, BC_type, inp_vessels, out_vessels`). A CSV array is now read by converting each row
to the same record, so both are processed identically; the `.json` file is preferred when both
exist, then `.csv`. Convert CSV arrays with
`python -m libcuflynx.utilities.config_schemas to-json <csv>... [--style phlynx|libcuflynx]`.
Every model in `resources/` that generated before generates byte-identical CellML from its CSV
and from its JSON conversion.

A module config entry with `"module_format": "supermodule"` and a list of `submodules` defines a
supermodule. An instance of it in a module array expands into `[instance]_[submodule]` modules
before anything else reads the array; `per_submodule_inputs`/`per_submodule_outputs` on the
instance link its hosts to individual submodules. Supermodules may nest. An optional
`default_parameters` CSV supplies parameters (renamed to the expanded names) wherever the
model's parameters file does not set them. See `tutorial/docs/design-model.md`.

JSON Schemas for both files ship in `libcuflynx/schemas/` (`module_array.schema.json`,
`module_config.schema.json`). The loaders check the same rules without a schema library;
`jsonschema` is a `[dev]` dependency only, for the tests.

The 0D/1D split for `couple_to_1d` now always writes `[file_prefix]_0d_module_array.csv` and
`[file_prefix]_1d_module_array.csv`, and the 1D generator reads the file the split wrote. Before,
an input named `_module_array.csv` gave split files the 1D generator could not find.

### Fixed — a BC_type's first two letters are a BC pair only for vessels

Non-vessel modules (cells, ion channels, controllers, heart parts, BC modules) can now have any
`BC_type` / `module_subtype`, with no `nn` prefix. Before, every connection whose two sides did
not start with `nn` had their first two letters checked as vessel BCs, so a version named
`lv_...`, `rv_...` or `Up...` stopped generation, and a non-vessel neighbour of an Nout junction
lost its connection. A module now counts as a vessel when its `BC_type` starts with `vv`, `vp`,
`pv` or `pp` and it has vessel ports (a `vessel_port`, or a `flow_port` and a `pressure_port` at
its entrance/exit); see `libcuflynx.utilities.vessel_bc`. Vessel BC pairs are checked as before,
and all the built-in example models generate byte-identical CellML.

### Added — PhLynx module-config and module-array schemas; `"Sum"` and `"Multiply"` multi_ports

The module library is moving its configs to PhLynx's key names, and libcuflynx now reads both
schemas. In PhLynx's schema, `module_type` is the vessel_type, `module_subtype` is the BC_type,
`component_file` is the module_file and `component_type` is the CellML component name. Each
entry is detected on its own, by `module_subtype`, `component_file` or `component_type`, and is
converted to the libcuflynx names when the config is loaded
(`libcuflynx.utilities.config_schemas.normalise_module_config_entry`). An entry that mixes the
two schemas is an error. Module arrays can use PhLynx's export layout too
(`name, module_type, module_subtype, inp_instances, out_instances`). The layout is detected from
the header.

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
- Unit converter components are named after the connection they sit on,
  `unit_converter_[from module]_[from variable]_to_[to module]_[to variable]`, with one per
  variable pair. They were named `unit_converter_[from units]_to_[to units]`, so a module output
  shared (multi_port `"True"`) with two modules that both needed the same conversion gave two
  components with the same name, and libCellML / Myokit rejected the model ("Component name must
  be unique within model").
- A venous module fed by a terminal takes the terminal flow into `v_in` through the
  `terminal_venous_connection` when one of its *other* entrance ports has a list-form
  multi_port, e.g. a separate `blood_uptake_port` with `multi_port: "sum"`. Only a list-form
  multi_port on the module's `vessel_port` entrance (which sums the terminal flow itself) skips
  that mapping now; before, any list-form entrance port did, and `v_in` was left unconnected.

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
