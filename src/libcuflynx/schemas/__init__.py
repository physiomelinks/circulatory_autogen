"""Machine-readable JSON Schemas of libcuflynx's input files, shipped as package data.

* ``module_array.schema.json`` -- ``<prefix>_module_array.json``: a list of instance records,
  in PhLynx or libcuflynx keys, with the optional ``per_submodule_inputs`` /
  ``per_submodule_outputs`` of a supermodule instance and the optional module ``instance``.
* ``module_config.schema.json`` -- ``*_modules_config.json``: component entries in either key
  style (with an optional ``default_instance``), and supermodule entries.
* ``obs_data.schema.json`` -- ``<name>_obs_data.json``: the top level of an obs_data file,
  including the optional ``obs_data_name``.

libcuflynx does not need a JSON Schema library: ``utilities/config_schemas.py`` checks the same
rules when it loads the files. The schemas are for editors, other tools (PhLynx) and tests.
Use :func:`schema_file` or :func:`load_schema` to get at them.
"""

import json

from libcuflynx.utilities.package_resources import package_data_file

MODULE_ARRAY_SCHEMA = 'module_array.schema.json'
MODULE_CONFIG_SCHEMA = 'module_config.schema.json'
OBS_DATA_SCHEMA = 'obs_data.schema.json'


def schema_file(name):
    """The shipped schema ``name`` as an :class:`importlib.abc.Traversable`."""
    return package_data_file('libcuflynx.schemas', name)


def load_schema(name):
    """The shipped schema ``name`` (e.g. ``MODULE_ARRAY_SCHEMA``) as a dict."""
    return json.loads(schema_file(name).read_text(encoding='utf-8'))
