"""Surrogate (emulator) models of the forward model -- issue #333.

An emulator maps theta (the ``params_for_id`` vector) to the scalar features of the
``obs_data.json`` data_items, i.e. the same numbers the cost is computed from. Fit it once, and
Sobol SA, calibration, MCMC and identifiability analysis can all evaluate it instead of the
simulator, because they all reach the model through that one mapping.

Public surface:

* :class:`emulators.emulator_trainer.EmulatorTrainer` -- design, simulate, fit, validate, save.
* :class:`emulators.emulator_bundle.EmulatorBundle` -- the saved artefact and its refusal rules.
* :func:`emulators.emulator_trainer.emulator_model_names` -- the emulator names the settings
  accept, discovered from autoemulate rather than hardcoded.

``autoemulate`` is an optional dependency; importing this package without it works, and only
fitting (or loading a bundle fitted with it) raises.
"""
from libcuflynx.emulators.emulator_bundle import (EmulatorBoundsError, EmulatorBundle,
                                       EmulatorQualityError, fingerprint)
from libcuflynx.emulators.emulator_trainer import (EmulatorTrainer, autoemulate_available,
                                        emulator_model_names, resolve_emulator_dir)

__all__ = [
    'autoemulate_available',
    'EmulatorBoundsError',
    'EmulatorBundle',
    'EmulatorQualityError',
    'EmulatorTrainer',
    'emulator_model_names',
    'fingerprint',
    'resolve_emulator_dir',
]
