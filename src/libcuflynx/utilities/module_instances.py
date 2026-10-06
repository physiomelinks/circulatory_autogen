'''
Module *instances*: named parameter sets of one module version.

A module library (e.g. circulatory-autogen-modules) lays a module out as::

    modules/<category>/<module_type>/versions/<version>/
        <module_type>_<version>_modules.cellml
        <module_type>_<version>_modules_config.json   one entry, module_subtype == <version>
        <module_type>_<version>_units.cellml
        instances/<instance>/
            <instance>_parameters.csv     variable_name,units,value,data_reference[,sourced]
            <instance>_obs_data.json      optional; "obs_data_name": "<instance>"
            <instance>_params_for_id.csv  optional

An instance only changes parameters, never the math. Its parameter names carry no instance
suffix: a row ``C`` becomes ``C_<vessel>`` for the module-array record that uses it, except a
row naming one of the module's ``global_constant`` variables, which keeps its plain name.

A module-array record (or a supermodule's submodule) picks an instance with
``"instance": "<name>"``; without one, the config entry's ``"default_instance"`` is used if
that instance's parameters file exists, and otherwise nothing is loaded. The instance
directory is looked for next to the config file the (vessel_type, BC_type) entry came from:
``<config dir>/instances/<instance>/<instance>_parameters.csv``.

Instance parameters are *default* parameters: the host model's parameters file wins over
them (``ModelParsers.merge_default_parameters``). The order of precedence is

    host parameters file > supermodule instance > supermodule default_parameters
                         > submodule / component instance

and within one tier the first record in the (expanded) module array wins, so a global set by
several instances is added once.

A supermodule entry may also live in a version directory with instances. Its instance's rows
are named ``{var}_{submodule}`` (local) or are globals, like ``default_parameters``, and are
renamed to ``{var}_{instance}_{submodule}`` (``supermodules.rename_default_parameter``).
'''

import os

import pandas as pd

INSTANCES_DIR = 'instances'
GLOBAL_CONSTANT = 'global_constant'

# the columns a parameters CSV (an instance's, or a supermodule's default_parameters) must
# have; config_schemas.PARAMETER_COLUMNS re-exports this
PARAMETER_COLUMNS = ('variable_name', 'units', 'value', 'data_reference')


def check_instance_name(value, where, key='instance'):
    '''``value`` if it is a usable instance name; ValueError naming ``where`` otherwise.'''
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where}: "{key}" must be a non-empty string (an instance name), '
                         f'not {value!r}.')
    value = value.strip()
    if value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError(f'{where}: "{key}" is "{value}", but an instance name is a directory '
                         f'name under instances/, not a path.')
    return value


def instances_dir(config_path):
    '''``<config dir>/instances``: where the instances of the config at ``config_path`` live.'''
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), INSTANCES_DIR)


def instance_parameters_path(config_path, instance):
    '''``<config dir>/instances/<instance>/<instance>_parameters.csv``'''
    return os.path.join(instances_dir(config_path), instance, f'{instance}_parameters.csv')


def available_instances(config_path):
    '''The instances next to the config at ``config_path``: the subdirectories of
    ``instances/`` that hold a ``<name>_parameters.csv``, sorted.'''
    root = instances_dir(config_path)
    if not os.path.isdir(root):
        return []
    return sorted(name for name in os.listdir(root)
                  if not name.startswith('.')
                  and os.path.isfile(os.path.join(root, name, f'{name}_parameters.csv')))


def read_parameter_rows(path, where, what='parameters file'):
    '''
    The rows of the parameters CSV at ``path`` as ``{variable_name, units, value,
    data_reference}`` dicts of stripped strings, in file order, skipping rows without a
    variable_name. Other columns (``sourced``...) are ignored. ValueError for missing columns.
    '''
    df = pd.read_csv(path, dtype=str, na_filter=False)
    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in PARAMETER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{where}: {what} {path} is missing the columns {missing}; it needs '
                         f'{list(PARAMETER_COLUMNS)}.')
    rows = []
    for row in df[list(PARAMETER_COLUMNS)].itertuples(index=False, name=None):
        values = {c: str(v).strip() for c, v in zip(PARAMETER_COLUMNS, row)}
        if values['variable_name']:
            rows.append(values)
    return rows


def _entry_label(entry):
    return f'({entry.get("vessel_type")}, {entry.get("BC_type")})'


def instance_parameter_rows(entry, instance, where):
    '''
    ``(instance_name, rows)`` for the config ``entry`` (a normalised module config entry,
    component or supermodule, carrying the ``config_path`` it was loaded from) and the
    record's ``instance`` (None when the record names none). The rows are the raw ones of
    the instance's parameters file, not yet renamed; ``([], None)`` when nothing applies.

    * an explicit ``instance`` must exist: ValueError naming the version directory and the
      instances it has, or saying that the config has no instances directory;
    * with none, ``entry["default_instance"]`` is used if its parameters file exists.
    '''
    config_path = entry.get('config_path')
    if instance is None:
        default = entry.get('default_instance')
        if not default or not config_path:
            return None, []
        path = instance_parameters_path(config_path, default)
        if not os.path.isfile(path):
            return None, []
        return default, read_parameter_rows(path, where, f'default instance "{default}" file')

    label = _entry_label(entry)
    if not config_path:
        raise ValueError(
            f'{where}: names instance "{instance}" of {label}, but that module config was not '
            f'loaded from a module library directory, so there is nowhere to look for '
            f'instances. Instances live in <module>/versions/<version>/instances/<instance>/ '
            f'next to the version\'s *_modules_config.json.')
    version_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isdir(instances_dir(config_path)):
        raise ValueError(
            f'{where}: names instance "{instance}" of {label}, but its module config '
            f'{config_path} has no {INSTANCES_DIR}/ directory next to it, so {label} has no '
            f'instances. Instances live in {version_dir}/{INSTANCES_DIR}/<instance>/'
            f'<instance>_parameters.csv.')
    path = instance_parameters_path(config_path, instance)
    if not os.path.isfile(path):
        existing = available_instances(config_path)
        raise ValueError(
            f'{where}: unknown instance "{instance}" of {label}: {path} does not exist. The '
            f'instances in {version_dir} are {existing or "none"}.')
    return instance, read_parameter_rows(path, where, f'instance "{instance}" file')


def global_constants(entry):
    '''The names of ``entry``'s ``global_constant`` variables (variables_and_units kind).'''
    names = set()
    for row in entry.get('variables_and_units') or []:
        if isinstance(row, (list, tuple)) and len(row) > 3 and row[3] == GLOBAL_CONSTANT:
            names.add(row[0])
    return names


def component_instance_rows(records, component_registry, source=None):
    '''
    The instance parameter rows of every record in ``records`` (normalised, expanded vessel
    records) whose (vessel_type, BC_type) is in ``component_registry`` (see
    ``config_schemas.load_component_registry``), in record order: each row ``{var}`` renamed
    to ``{var}_{record name}``, or kept if ``var`` is a global constant of the module.

    A record naming an instance of a type with no config entry is an error; one naming no
    instance of such a type is left to the module-config join to report.
    '''
    source = source or 'module array'
    rows = []
    for record in records:
        key = (record['vessel_type'], record['BC_type'])
        instance = record.get('instance')
        where = f'{source}: "{record["name"]}" {key}'
        entry = component_registry.get(key)
        if entry is None:
            if instance is not None:
                raise ValueError(f'{where}: names instance "{instance}", but no module config '
                                 f'entry of that type was found, so there is no module library '
                                 f'directory to look for its instances in.')
            continue
        _, raw = instance_parameter_rows(entry, instance, where)
        globals_ = global_constants(entry)
        for row in raw:
            row = dict(row)
            if row['variable_name'] not in globals_:
                row['variable_name'] = f'{row["variable_name"]}_{record["name"]}'
            rows.append(row)
    return rows


def first_rows_win(rows):
    '''``rows`` with each variable_name kept once, the first occurrence winning.'''
    seen = set()
    out = []
    for row in rows:
        if row['variable_name'] not in seen:
            seen.add(row['variable_name'])
            out.append(row)
    return out
