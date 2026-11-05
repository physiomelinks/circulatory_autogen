"""Sobol and local sensitivity analysis."""

#: Feature flag for callers (CUFLynx) that must work with several libcuflynx versions: True
#: when ``sa_options.include_prediction_items`` (and ``emulator_settings.include_prediction_items``)
#: are supported, i.e. prediction items with an ``operation`` can be SA / emulator outputs.
#: Absent in older versions, so test with ``getattr(module, 'SUPPORTS_PREDICTION_FEATURES', False)``.
SUPPORTS_PREDICTION_FEATURES = True
