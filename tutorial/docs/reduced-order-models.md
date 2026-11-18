# Reduced-order models (ROMs)

A **reduced-order model** is a [surrogate](surrogates.md) that is itself a dynamic model: a
smaller system of equations, built from runs of the full model, that is integrated in time and
returns the outputs as traces. An [emulator](emulators.md) answers only for the scalar features it
was trained on; a ROM answers for any output and protocol it was built to cover.

This release of libcuflynx does not build ROMs: build one with your own tools. CA then runs it as
a model:

- Wrap the ROM in a solver class and name it as an
  [external Python model](external-python-solvers.md): `model_type: external_python`, with
  `external_model_path` naming the file. The class declares the ROM's parameters and outputs and
  steps it in time.
- Calibration, sensitivity analysis and the other analyses then run on it as they do on any
  external model, with the same obs_data and params_for_id. Series data, prediction items and
  plots work, because the ROM returns traces.
- In a [calibration workflow](parameter-identification.md#calibration-workflows) the ROM is a
  step's model, `{"external_model_path": "my_rom.py"}` (the builder's `external_model(path)`).
  A step on the full model and a step on the ROM can then sit in one workflow.

A ROM needs no key of its own: `use_emulator`, `emulator_from` and `emulator_dir` name emulators
only. An emulator can also be trained on a ROM, as on any model.

Check a ROM against the full model before relying on it: run both on the same obs_data and
compare the features and traces over the parameter ranges the analysis will visit.
