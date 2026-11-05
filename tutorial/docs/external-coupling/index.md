# Coupling to external models

A model generated as C++ (`model_type: cpp`) can run alongside another model, such as a PDE
solver, a 1D blood-flow solver, or any code of your own. The two exchange values every
**coupling step**. The 0D side stays generated C++, so it runs fast, and the other model keeps
its own solver, mesh and time stepping.

To CA, the other model is a **module**: a row of the module array, connected to CellML modules
through ports like any other. What makes it external is its module config entry:

- `"module_format": "external_api"`: it has no CellML;
- an `api` block saying how the generated code talks to it.

## Choosing a path

| Your other model is… | `api` | How it runs | Example |
|---|---|---|---|
| **a Python class** (FEniCS, numpy, a wrapped library…) | `role: provider`, `transport: python` | `cuflynx-couple` steps the generated C++ (loaded as a shared library) and your class in one process | [Couple your own Python model](python-models.md), [tissue O2 with FEniCS](fenics-tissue-o2.md), [NE around a varicosity with FEniCS](fenics-ne-varicosity.md) |
| **the FV 1D blood-flow solver** shipped with CA | `role: consumer`, `transport: named_pipe`, `process: FV1D_solver` | the coupler launches `main0d` and the 1D solver, which talk over named pipes | [1D finite-volume coupling](fv-1d.md) |
| **a C++ program** that drives the 0D model step by step | `role: provider`, `transport: cpp_class` | your program links the generated library and calls a generated class (`set_…`, `get_…`, `solve_time_step`) | the provider section of `src/libcuflynx/generators/cpp/templates/README.md` |

!!! note "Not the same as `model_type: external_python`"
    [External Python Solvers](../external-python-solvers.md) is for calibrating, sweeping or
    emulating a model CA does not generate at all: CA runs it whole and reads its outputs. On
    this page, CA generates the 0D model and runs it **together with** your model.

## What crosses the coupling

The variables exchanged are the external module's own **port variables**. Ports connect by
`port_type`, and variables pair up by position, exactly as between CellML modules. For a Python
model, the direction of each variable is worked out from the CellML side, so you never list
functions:

- connected to a variable the 0D model **computes** (a state, or the result of an equation):
  the external model **receives** it each step;
- connected to a **boundary condition** (or constant) of the 0D model: the external model
  **sets** it.

  For example, `GE_capillary`'s `ub_O2_t` (tissue O2) is a boundary condition, so a tissue
  model sets it, and `flux_O2_c` is computed, so the tissue model receives it. A variable that
  is set becomes a libCellML *external variable* of the generated C. Its parameter value is the
  starting value until the external model sets it.

`cuflynx-couple --check <model folder>` prints this table for a generated model.

A port connected to **several** 0D modules (several capillaries feeding one tissue model, say)
exchanges one value per module, as an array ordered as the module array lists them.

Values are in the CellML units of the 0D variables (mol/s, mM, Pa, …). Add `api_units` or a
`factor` per variable in `api.variables` if your model works in other units.

## Coupling step and accuracy

Each coupling step from `t` to `t + dt` is **explicit and staggered**:

1. the external model receives the 0D values at `t`, advances to `t + dt`, and returns what it sets;
2. those values are held in the 0D model while it steps to `t + dt`.

This is first order in `dt`: halving the step halves the coupling error. For stiff exchanges, or
when the step can't be small, set `"subiterations": n` in the api block. Each step is then
iterated from snapshots of both models until the exchanged values change by less than `tol`:
each side holds the average of the other's start and end values, which is second order.
`tests/test_external_coupling.py` checks both orders against the all-CellML model.

## The api block

```json
"api": {
  "name": "my_tissue",
  "role": "provider",
  "transport": "python",
  "python": {"file": "my_tissue.py", "class": "MyTissue"},
  "coupling_dt": 0.01,
  "subiterations": 0,
  "variables": {"C_t": {"api_units": "mM"}}
}
```

| Key | Meaning |
|---|---|
| `python.file`, `python.class` | the model; the file path is relative to the module config's folder |
| `coupling_dt` | the coupling step [s] (default: the model's `dt`). A module constant `coupling_dt` (set per instance in the parameters file) overrides it |
| `subiterations`, `tol`, `relaxation` | fixed-point iteration of each step (default 0: explicit) |
| `variables` | optional per-variable `direction` (`to_external` / `from_external`), `api_units` or `factor` |

The block is checked when the module configs load. An unknown key, a missing model file or a
bad value is reported with the module's name.

The other transports are described on their pages: `named_pipe` and `process` in
[1D coupling](fv-1d.md), and `cpp_class` in `src/libcuflynx/generators/cpp/templates/README.md`.

## Pages

- [Couple your own Python model](python-models.md): step by step, with a numpy example
- [Tissue O2 and a microvasculature model, in FEniCS](fenics-tissue-o2.md)
- [Norepinephrine around a sympathetic varicosity, in FEniCS](fenics-ne-varicosity.md)
- [1D finite-volume coupling](fv-1d.md)
- [Troubleshooting](troubleshooting.md)
