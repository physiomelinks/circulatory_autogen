# Surrogates

A **surrogate** is a cheaper stand-in for a model, built from runs of that model. Use one when an
analysis needs many more model evaluations than you can afford: a Sobol design, MCMC, or a
calibration of a slow solver.

Circulatory Autogen has two kinds. They differ in what they stand in for.

| | [Emulator](emulators.md) | [Reduced-order model (ROM)](reduced-order-models.md) |
|---|---|---|
| Stands in for | the map from the calibrated parameters to the **scalar features** of the obs_data | the **model**: it is integrated in time and returns traces |
| Gives | one number per data item, after its `operation` | the outputs over time, for any protocol the ROM covers |
| Built by | `do_emulation`, `cuflynx-train-emulator`, or a workflow `emulation` step | you, outside CA |
| Used by | `use_emulator`, or a workflow step's `emulator_from` / `emulator_dir` | running it as the model: `model_type: external_python` |
| Series data, prediction traces, plots | refused | available, as for any model |

- **[Emulators](emulators.md)**: training, the settings, when CA refuses one, and gradients.
- **[Reduced-order models (ROMs)](reduced-order-models.md)**: how a ROM runs as a model.

The settings keep the name of the kind they configure: `do_emulation`, `use_emulator`,
`emulator_settings`, `emulator_from` and `emulator_dir` are an emulator's. A ROM has no settings
of its own, because CA runs it as a model.
