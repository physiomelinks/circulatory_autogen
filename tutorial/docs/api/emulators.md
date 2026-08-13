# Emulators (surrogate models)

`EmulatorTrainer` fits a surrogate of the model's scalar observable features, and
`EmulatorBundle` is the saved artefact plus the checks that decide whether it may be
used. See [Emulators](../emulators.md) for the workflow.

::: emulators.emulator_trainer.EmulatorTrainer
    options:
      members:
        - init_from_dict
        - design
        - evaluate
        - fit
        - train
        - feature_labels

::: emulators.emulator_trainer.emulator_model_names

::: emulators.emulator_trainer.resolve_emulator_dir

::: emulators.emulator_bundle.EmulatorBundle
    options:
      members:
        - predict
        - check_quality
        - check_bounds
        - check_matches
        - save
        - load
        - make_scale

::: emulators.emulator_bundle.fingerprint

The emulator is used through the ordinary simulation-helper interface, so every analysis
reaches it the same way it reaches a solver.

::: solver_wrappers.emulator_solver_helper.SimulationHelper
    options:
      members:
        - set_theta
        - set_param_vals
        - run
        - get_results
        - get_predicted_features
        - set_obs_map
        - update_times
        - get_init_param_vals
