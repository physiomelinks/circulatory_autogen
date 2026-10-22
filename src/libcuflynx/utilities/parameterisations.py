'''
Module *parameterisations*: named parameter sets of one module version.

A module library (e.g. circulatory-autogen-modules) lays a module out as::

    modules/<category>/<module_type>/versions/<version>/
        <module_type>_<version>_modules.cellml
        <module_type>_<version>_modules_config.json   one entry, module_subtype == <version>
        <module_type>_<version>_units.cellml
        parameterisations/<name>/
            <name>_parameters.json    nested parameter values (utilities/parameters_file.py)
            <name>_obs_data.json      optional; "obs_data_name": "<name>"
            <name>_params_for_id.json optional

A module-array record names its parameterisation with ``"parameterisation"``; ``"instance"``
there is an error, since an instance is the record itself. A version directory with the
older ``instances/`` and no ``parameterisations/`` is an error naming the rename, and a
parameterisation with a ``<name>_parameters.csv`` and no ``.json`` one is an error naming
``cuflynx-upgrade parameters``.

A parameterisation only changes parameters, never the math. Its parameters file is rooted at
the module: a component module's own variables at the top (``{"C": {...}}``), which become
``C_<vessel>`` for the module-array record that uses it, and the module's ``global_constant``
variables it sets under ``global``, which keep their plain names. A supermodule's
parameterisation holds its sub-instances' values (``{"lv": {"E_A": {...}}}``) and its globals.

A module-array record picks a parameterisation with ``"parameterisation": "<name>"``; without
one, the config entry's ``"default_parameterisation"`` is used if that parameterisation's
parameters file exists, and otherwise nothing is loaded. The directory is looked for next to
the config file the (module_type, module_subtype) entry came from:
``<config dir>/parameterisations/<name>/<name>_parameters.json``.

Parameterisation parameters are *default* parameters: the model's parameters file wins over
them (``ModelParsers.merge_default_parameters``). The order of precedence is

    model's parameters file > supermodule parameterisation's globals
        > submodule / component parameterisation

and within one tier the first record in the (expanded) module array wins, so a global set by
several parameterisations is added once. A global is one value for the whole model, so a
parameterisation that set it to another value than the one used is warned about
(``ConflictingGlobalWarning``): its module runs at a value it did not choose. The exceptions
are a supermodule overriding the parameterisations inside it, which is what a supermodule
parameterisation is for, and a global the model's parameters file sets.

A supermodule instance's sub-instance values are in the model's parameters file, under the
instance (``heart/lv/E_A``). Its version's default parameterisation is the template they are
copied from when the instance is made (``supermodules.write_instance_parameters``); generation
reads only a supermodule parameterisation's globals (``supermodules.read_supermodule_parameters``).
'''

import os
import warnings

from libcuflynx.utilities.parameters_file import (CONVERT_COMMAND, GLOBAL, MODULE,
                                                    SEPARATOR, read_parameter_rows as _read_rows)

PARAMETERISATIONS_DIR = 'parameterisations'
# the name PARAMETERISATIONS_DIR had before; a version directory with only it is refused
OLD_PARAMETERISATIONS_DIR = 'instances'
# the record key naming a parameterisation, and the older one that is refused
PARAMETERISATION_KEY, INSTANCE_KEY = 'parameterisation', 'instance'
DEFAULT_PARAMETERISATION_KEY = 'default_parameterisation'
GLOBAL_CONSTANT = 'global_constant'

# the keys every parameter row has; config_schemas.PARAMETER_COLUMNS re-exports this
PARAMETER_COLUMNS = ('variable_name', 'units', 'value', 'data_reference')
#: a parameterisation's global constants are rows named global/<variable>
GLOBAL_PREFIX = GLOBAL + SEPARATOR


def check_parameterisation_name(value, where, key=PARAMETERISATION_KEY):
    '''``value`` if it is a usable parameterisation name; ValueError naming ``where``
    otherwise.'''
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where}: "{key}" must be a non-empty string (a parameterisation '
                         f'name), not {value!r}.')
    value = value.strip()
    if value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError(f'{where}: "{key}" is "{value}", but a parameterisation name is a '
                         f'directory name under {PARAMETERISATIONS_DIR}/, not a path.')
    return value


def reject_instance_key(record, where, what='A module-array record'):
    '''``what`` (a module-array record, a workflow target) names its parameter set with
    ``"parameterisation"``; an ``"instance"`` key is a ValueError naming ``where``.'''
    if INSTANCE_KEY in record:
        raise ValueError(f'{where} has "{INSTANCE_KEY}": {record[INSTANCE_KEY]!r}. {what} '
                         f'names its parameter set with "{PARAMETERISATION_KEY}"; rename '
                         f'the key.')


def record_parameterisation(record, where):
    '''The parameterisation a module-array record names (``"parameterisation"``), None when it
    names none.'''
    reject_instance_key(record, where)
    return record.get(PARAMETERISATION_KEY)


def parameterisations_dir(config_path):
    '''Where the parameterisations of the config at ``config_path`` live:
    ``<config dir>/parameterisations``. A directory with only the older ``instances/`` is a
    ValueError asking for the rename.'''
    version_dir = os.path.dirname(os.path.abspath(config_path))
    path = os.path.join(version_dir, PARAMETERISATIONS_DIR)
    old = os.path.join(version_dir, OLD_PARAMETERISATIONS_DIR)
    if not os.path.isdir(path) and os.path.isdir(old):
        raise ValueError(f'{old} has the older name of {PARAMETERISATIONS_DIR}/, which libcuflynx '
                         f'no longer reads: rename it {path}.')
    return path


def parameterisation_parameters_path(config_path, name):
    '''``<config dir>/parameterisations/<name>/<name>_parameters.json``'''
    return os.path.join(parameterisations_dir(config_path), name, f'{name}_parameters.json')


def available_parameterisations(config_path):
    '''The parameterisations next to the config at ``config_path``: the subdirectories of
    ``parameterisations/`` that hold a ``<name>_parameters.json``, sorted.'''
    root = parameterisations_dir(config_path)
    if not os.path.isdir(root):
        return []
    return sorted(name for name in os.listdir(root)
                  if not name.startswith('.')
                  and os.path.isfile(os.path.join(root, name, f'{name}_parameters.json')))


def _refuse_csv_parameterisation(path, where):
    '''A parameterisation with only the older ``<name>_parameters.csv`` is a ValueError naming
    the converter.'''
    csv_path = os.path.splitext(path)[0] + '.csv'
    if not os.path.isfile(path) and os.path.isfile(csv_path):
        raise ValueError(f'{where}: {csv_path} is a parameters CSV; libcuflynx reads a '
                         f'parameterisation from nested JSON ({os.path.basename(path)}). Convert '
                         f'it once with: {CONVERT_COMMAND} {csv_path}')


def read_parameter_rows(path, where, what='parameters file'):
    '''
    The rows of the parameterisation's parameters file at ``path`` as ``{variable_name, units,
    value, data_reference}`` dicts of strings, in file order: its own variables named as they
    are (``C``, ``lv/E_A``), its globals ``global/<variable>``. Other keys (``sourced``...) are
    dropped.
    '''
    _refuse_csv_parameterisation(path, f'{where}: {what}')
    return [{c: row[c] for c in PARAMETER_COLUMNS}
            for row in _read_rows(path, f'{where}: {what} ', MODULE)]


def is_global_row(row):
    '''Whether a parameterisation row sets a global (``global/<variable>``).'''
    return row['variable_name'].startswith(GLOBAL_PREFIX)


def _entry_label(entry):
    return f'({entry.get("module_type")}, {entry.get("module_subtype")})'


def parameterisation_rows(entry, parameterisation, where):
    '''
    ``(name, rows)`` for the config ``entry`` (a normalised module config entry, component or
    supermodule, carrying the ``config_path`` it was loaded from) and the record's
    ``parameterisation`` (None when the record names none). The rows are the raw
    ones of the parameterisation's parameters file, not yet renamed; ``(None, [])`` when
    nothing applies.

    * an explicit parameterisation must exist: ValueError naming the version directory and
      the parameterisations it has, or saying that the config has no parameterisations
      directory;
    * with none, ``entry["default_parameterisation"]``
      is used if its parameters file exists.
    '''
    config_path = entry.get('config_path')
    if parameterisation is None:
        default = entry.get(DEFAULT_PARAMETERISATION_KEY)
        if not default or not config_path:
            return None, []
        path = parameterisation_parameters_path(config_path, default)
        _refuse_csv_parameterisation(path, where)
        if not os.path.isfile(path):
            return None, []
        return default, read_parameter_rows(path, where,
                                            f'default parameterisation "{default}" file')

    label = _entry_label(entry)
    if not config_path:
        raise ValueError(
            f'{where}: names parameterisation "{parameterisation}" of {label}, but that module config '
            f'was not loaded from a module library directory, so there is nowhere to look for '
            f'parameterisations. Parameterisations live in <module>/versions/<version>/'
            f'{PARAMETERISATIONS_DIR}/<name>/ next to the version\'s *_modules_config.json.')
    version_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isdir(parameterisations_dir(config_path)):
        raise ValueError(
            f'{where}: names parameterisation "{parameterisation}" of {label}, but its module config '
            f'{config_path} has no {PARAMETERISATIONS_DIR}/ '
            f'directory next to it, so {label} has no parameterisations. Parameterisations '
            f'live in {version_dir}/{PARAMETERISATIONS_DIR}/<name>/<name>_parameters.json.')
    path = parameterisation_parameters_path(config_path, parameterisation)
    _refuse_csv_parameterisation(path, where)
    if not os.path.isfile(path):
        existing = available_parameterisations(config_path)
        raise ValueError(
            f'{where}: unknown parameterisation "{parameterisation}" of {label}: {path} does not '
            f'exist. The parameterisations in {parameterisations_dir(config_path)} are '
            f'{existing or "none"}.')
    return parameterisation, read_parameter_rows(path, where, f'parameterisation "{parameterisation}" file')


class ConflictingGlobalWarning(UserWarning):
    '''Module parameterisations set the global constant ``name`` to different values;
    ``values`` is ``[(value, units, who)]`` in precedence order, the first used.
    ``ModelParsers.load_model`` drops it when the host parameters file sets ``name``.'''

    def __init__(self, message, name=None, values=()):
        super().__init__(message)
        self.name = name
        self.values = list(values)


def _same_value(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def inside(record, supermodule):
    '''Whether the expanded record ``record`` comes from the supermodule instance
    ``supermodule`` (expansion names it ``<supermodule>_<submodule>``, nesting included).'''
    return record.startswith(supermodule + '_')


def warn_conflicting_globals(settings, source):
    '''One ConflictingGlobalWarning per global in ``settings`` -- ``[(name, value, units, who,
    record)]`` in precedence order, the first one used -- that a record's parameterisation set
    to a different value than the one used, unless that record is inside the supermodule whose
    value is used. Different units are called out: two modules then most likely mean
    different quantities by one name (``T``, a temperature in one and a period in another).'''
    by_name = {}
    for name, value, units, who, record in settings:
        by_name.setdefault(name, []).append((value, units, who, record))
    for name, entries in by_name.items():
        first_value, first_units, first_who, first_record = entries[0]
        others = [(v, u, w) for v, u, w, record in entries[1:]
                  if not _same_value(v, first_value) and not inside(record, first_record)]
        if not others:
            continue
        values = [(first_value, first_units, first_who)] + others
        listed = '; '.join(f'{v} {u} by {w}' for v, u, w in values)
        clash = ('' if len({u for _, u, _ in values}) == 1 else
                 f' The units differ too, so these modules may mean different quantities by '
                 f'"{name}"; one of them needs renaming.')
        warnings.warn(ConflictingGlobalWarning(
            f'{source}: the global constant "{name}" is set to different values by module '
            f'parameterisations ({listed}). The model uses {first_value} {first_units} (the '
            f'first), so the other modules run at a value their parameterisation did not '
            f'choose. Set "{name}" in '
            f'the host parameters file to choose the value for the whole model.{clash}',
            name, values), stacklevel=3)


def reissue_warnings(held, settled=()):
    '''Re-issues the warnings in ``held`` (a ``catch_warnings(record=True)`` list), except a
    ``ConflictingGlobalWarning`` for a global in ``settled``: one a higher level of precedence
    (a supermodule parameterisation, or the host parameters file) sets, which ends the
    conflict.'''
    for w in held:
        if isinstance(w.message, ConflictingGlobalWarning) and w.message.name in settled:
            continue
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def global_constants(entry):
    '''The names of ``entry``'s ``global_constant`` variables (variables_and_units kind).'''
    names = set()
    for row in entry.get('variables_and_units') or []:
        if isinstance(row, (list, tuple)) and len(row) > 3 and row[3] == GLOBAL_CONSTANT:
            names.add(row[0])
    return names


def component_parameterisation_rows(records, component_registry, source=None, settings=None):
    '''
    The parameterisation rows of every record in ``records`` (the normalised records of an
    expanded module array, whose parameterisation is in ``record["parameterisation"]``) whose
    (module_type, module_subtype) is in ``component_registry`` (see
    ``config_schemas.load_component_registry``), in record order: each of the module's own
    variables ``{var}`` renamed to ``{var}_{record name}``, and each global (``global/{var}``)
    named ``{var}``. A global of the module at the top, or a ``global`` entry that is not one of
    its global constants, is a ValueError.

    A record naming a parameterisation of a type with no config entry is an error; one naming
    none of such a type is left to the module-config join to report.

    The globals the parameterisations set are added to ``settings`` (see
    ``warn_conflicting_globals``) when it is given, for the caller to check with the
    supermodules'; otherwise they are checked here.
    '''
    source = source or 'module array'
    rows = []
    check_here = settings is None
    settings = [] if check_here else settings
    for record in records:
        key = (record['module_type'], record['module_subtype'])
        where = f'{source}: "{record["name"]}" {key}'
        parameterisation = record_parameterisation(record, where)
        entry = component_registry.get(key)
        if entry is None:
            if parameterisation is not None:
                raise ValueError(f'{where}: names parameterisation "{parameterisation}", but no module '
                                 f'config entry of that type was found, so there is no module '
                                 f'library directory to look for its parameterisations in.')
            continue
        used, raw = parameterisation_rows(entry, parameterisation, where)
        globals_ = global_constants(entry)
        for row in raw:
            row = dict(row)
            is_global = is_global_row(row)
            if is_global:
                row['variable_name'] = row['variable_name'][len(GLOBAL_PREFIX):]
            if SEPARATOR in row['variable_name']:
                raise ValueError(
                    f'{where}: parameterisation "{used}" names "{row["variable_name"]}", but a '
                    f'parameterisation\'s parameters are the module\'s own variables, at the top '
                    f'of its file ("C", not "lv": {{"C"}}); the vessel is the module-array record '
                    f'that uses it.')
            if is_global != (row['variable_name'] in globals_):
                raise ValueError(
                    f'{where}: parameterisation "{used}" has "{row["variable_name"]}" '
                    + (f'under "{GLOBAL}", but it is not one of the module\'s global constants '
                       f'({sorted(globals_) or "none"}); put it at the top of the file.'
                       if is_global else
                       f'at the top of its file, but it is a global constant of the module; put '
                       f'it under "{GLOBAL}".'))
            if not is_global:
                row['variable_name'] = f'{row["variable_name"]}_{record["name"]}'
            else:
                who = f'"{record["name"]}" (parameterisation "{used}")'
                settings.append((row['variable_name'], row['value'], row['units'], who,
                                 record['name']))
            rows.append(row)
    if check_here:
        warn_conflicting_globals(settings, source)
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
