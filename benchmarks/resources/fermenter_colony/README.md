# Fermenter colony: the batched solvers' scaling model

`module_library/` is a module library (as `module_library_dirs` reads it) holding one module,
`generic_adaptive_cell` version `fermenter`: a whole cell (membrane pumps, exchangers and leaks for H, Na, K and
Ca; glucose uptake and energy metabolism; protein pools; growth; a flagellar motor) assembled into one CellML
component of 23 states and about 280 parameters. `benchmarks/batched_solver_scaling.py` builds colonies of N unconnected copies
of it (`write_colony_inputs` writes the module array and parameters file for each N) and times the batched
solvers on them.

## Where it comes from

- **Source module:** `modules/cell/generic_adaptive_cell/versions/fermenter` in
  [circulatory-autogen-modules](https://github.com/physiomelinks/circulatory-autogen-modules) (branch
  `homogenisation-checks`, not yet merged), a supermodule of the library's membrane, metabolism and protein
  modules.
- **Assembly:** AdaptOrDeity's (github.com/FinbarArgus/AdaptOrDeity) `aod-kernelgen-flatten` (its ADR 0007) takes
  libcuflynx's flat model of the supermodule and merges it into one component. These files are AdaptOrDeity's
  `content/assembled/generic_adaptive_cell/versions/fermenter` at commit `66a6024` (2026-10-11), with the
  `default` parameterisation only.
- **Licence:** CC0-1.0, as its modules config records (`licence`, `assembled_from`). Authors: the
  circulatory-autogen-modules contributors.

## Regenerating

In AdaptOrDeity, with the libcuflynx venv:

```bash
tools/kernelgen/.venv/bin/aod-kernelgen-flatten <flat.cellml> content/assembled generic_adaptive_cell fermenter \
    --modules-root <circulatory-autogen-modules>/modules --supermodule-config <config.json>
```

then copy `content/assembled/generic_adaptive_cell/versions/fermenter` here (the two `.cellml` files, the
modules config and its default parameters). If they are in an older form (an `instances/` directory, a parameters
CSV), rename `instances/` to `parameterisations/` and convert them with
`cuflynx-upgrade module-config <modules_config.json>`; the result is `parameterisations/default/default_parameters.json`.
