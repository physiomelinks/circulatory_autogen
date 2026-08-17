# Changelog

Notable changes to libcuflynx (circulatory_autogen). Entries under *Unreleased* ship with the
next release; add to that section as you land a change.

## Unreleased — 0.4.0

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
