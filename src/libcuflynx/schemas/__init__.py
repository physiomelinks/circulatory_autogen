"""Machine-readable JSON Schemas of libcuflynx's input files, shipped as package data.

* ``module_array.schema.json`` -- ``<prefix>_module_array.json``: a list of instance records,
  in the current (PhLynx) or the older libcuflynx keys, with the optional ``per_submodule_inputs`` /
  ``per_submodule_outputs`` of a supermodule instance and the optional ``parameterisation``
  (an ``instance`` key is refused).
* ``module_config.schema.json`` -- ``*_modules_config.json``: component entries in either key
  style, with their ports and ``variables_and_units`` (and an optional
  ``default_parameterisation``, older name ``default_instance``), and supermodule entries.
* ``parameters.schema.json`` -- ``<prefix>_parameters.json`` (a model's parameter values) and
  ``parameterisations/<name>/<name>_parameters.json`` (a module's): nested instance,
  sub-instance, variable (``utilities/parameters_file.py``).
* ``obs_data.schema.json`` -- ``<name>_obs_data.json``: the optional ``obs_data_name``, the
  ``data_items``, ``prediction_items`` and ``protocol_info`` (with ``protocol_traces`` and
  ``protocol_shapes``).
* ``params_for_id.schema.json`` -- ``<prefix>_params_for_id.json``: the parameters a
  calibration identifies, a flat list of entries (version 1; ``utilities/params_for_id_file.py``).
  A params_for_id CSV is refused; ``cuflynx-upgrade params-for-id`` converts it.
* ``calibration_workflow.schema.json`` -- ``calibration_workflow.json``: ordered calibrations
  of module parameterisations merged into a supermodule parameterisation
  (``libcuflynx.calibration_workflow``, whose ``spec.py`` and ``resolve.py`` check the same
  rules and those that need the library).

Every reader checks its file against its schema when it loads it
(:mod:`libcuflynx.schemas.validation`), so a schema is the definition of its file's structure:
what an editor, PhLynx or CUFLynx accepts is what libcuflynx accepts. The readers check only
what a schema cannot (that a file exists, that a vessel is in the model). For obs_data,
params_for_id and parameters the schema runs first; module configs, module arrays and
calibration workflows
are checked by their readers first, for those readers' more specific messages, and then by
the schema. Enums that libcuflynx keeps as registries (prior types, step kinds, data types,
variable kinds ...) are generated into the schemas by :mod:`libcuflynx.schemas.registries`.
Use :func:`schema_file` or :func:`load_schema` to get at them.
"""

import json

from libcuflynx.utilities.package_resources import package_data_file

MODULE_ARRAY_SCHEMA = 'module_array.schema.json'
MODULE_CONFIG_SCHEMA = 'module_config.schema.json'
OBS_DATA_SCHEMA = 'obs_data.schema.json'
PARAMS_FOR_ID_SCHEMA = 'params_for_id.schema.json'
CALIBRATION_WORKFLOW_SCHEMA = 'calibration_workflow.schema.json'
PARAMETERS_SCHEMA = 'parameters.schema.json'


def schema_file(name):
    """The shipped schema ``name`` as an :class:`importlib.abc.Traversable`."""
    return package_data_file('libcuflynx.schemas', name)


def load_schema(name):
    """The shipped schema ``name`` (e.g. ``MODULE_ARRAY_SCHEMA``) as a dict."""
    return json.loads(schema_file(name).read_text(encoding='utf-8'))
