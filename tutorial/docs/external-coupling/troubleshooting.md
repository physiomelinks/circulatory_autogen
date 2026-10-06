# Troubleshooting external coupling

**`cuflynx-couple --check <model folder>` first.** It lists each external model, its file and
class, its parameters, and every exchanged variable with its direction, units and connected
modules, without running anything.

## Generation

| Message | Cause and fix |
|---|---|
| `python api … has unknown key(s)` | a typo in the api block; the message lists the allowed keys |
| `python api …: model file … not found` | `api.python.file` is relative to the folder of the module config that declares it |
| `Port '…' of external module '…' is not connected to any CellML module with a matching port` | name the 0D module(s) in the row's `inp_vessels` (for an entrance port) or `out_vessels` (exit port), and check they have a port of that `port_type` |
| `'row/var' connects to …, some computed by the 0D model and some not` | the connected modules disagree on who sets the variable; give `api.variables[var].direction` |
| `Parameter X_row (constant X of external module 'row') is not in the parameters file` | external modules' constants are parameters like any module's |
| `A model coupled to Python external models … cannot also be coupled to the 1D solver` | not supported yet: one kind of coupling per model |

## Building the shared library

The first `cuflynx-couple` run builds `model0d_capi` with CMake (`<model>/build`).

- **`cmake not found`**: install CMake. In a conda environment: `mamba install cmake cxx-compiler sundials`.
- **SUNDIALS not found** (CVODE models): set `SUNDIALS_DIR` to its install prefix. Inside conda,
  the environment's prefix is searched first.
- **`Cannot load …libmodel0d_capi.so: … GLIBCXX_3.4.xx not found`**: the library was built
  with a newer C++ compiler than the C++ runtime your Python loads, typically the system
  compiler with a conda Python. Install `cxx-compiler` in the environment, delete
  `<model>/build`, and run again.
- **A stale build**: the library is rebuilt whenever the generated sources are newer than it. A
  regenerated model gets a fresh build.

## Running

| Symptom | Cause and fix |
|---|---|
| `… needs N value(s), one per connected module` | `step()` returned the wrong length; return an array per variable, ordered as `neighbours[var]` (a scalar is broadcast) |
| `… did not return [...]` | `step()` and `initial_outputs()` must return every variable the model sets (`info['directions']`) |
| `CVODE failed at t = … with flag …` | the 0D model failed: often a jump in an external value is too large for the coupling step. Reduce `coupling_dt`, or use `subiterations` |
| results depend on `coupling_dt` | explicit coupling is first order. Halve the step and compare, or use `subiterations` (second order) |
| MPI runs disagree between ranks, or hang | `step()` must return the same values on every rank: reduce with `comm.allreduce`. The 0D model runs on every rank. |

## Units

Values cross in the CellML units of the 0D variables (`info['units']`). Concentrations are
`millimolar` (= mol/m³), fluxes `mol_per_s`, pressures `J_per_m3` (Pa). Convert in the class,
or give `api.variables[var].api_units` / `factor`.
