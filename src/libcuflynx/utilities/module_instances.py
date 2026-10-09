'''
Module *parameterisations*: named parameter sets of one module version.

A module library (e.g. circulatory-autogen-modules) lays a module out as::

    modules/<category>/<module_type>/versions/<version>/
        <module_type>_<version>_modules.cellml
        <module_type>_<version>_modules_config.json   one entry, module_subtype == <version>
        <module_type>_<version>_units.cellml
        parameterisations/<name>/
            <name>_parameters.csv     variable_name,units,value,data_reference[,sourced]
            <name>_obs_data.json      optional; "obs_data_name": "<name>"
            <name>_params_for_id.csv  optional

Parameterisations used to be called *instances* (the word now means one use of a version in
a module array), and the old names are still read: an ``instances/`` directory is used when
there is no ``parameterisations/`` one, ``"instance"`` is read as ``"parameterisation"`` and
``"default_instance"`` as ``"default_parameterisation"``. The code below keeps the old names
internally: the config loaders (``config_schemas.py``) normalise both spellings to
``instance`` / ``default_instance``.

A parameterisation only changes parameters, never the math. Its parameter names carry no
suffix: a row ``C`` becomes ``C_<vessel>`` for the module-array record that uses it, except a
row naming one of the module's ``global_constant`` variables, which keeps its plain name.

A module-array record (or a supermodule's submodule) picks a parameterisation with
``"parameterisation": "<name>"``; without one, the config entry's
``"default_parameterisation"`` is used if that parameterisation's parameters file exists, and
otherwise nothing is loaded. The directory is looked for next to the config file the
(vessel_type, BC_type) entry came from:
``<config dir>/parameterisations/<name>/<name>_parameters.csv``.

Parameterisation parameters are *default* parameters: the host model's parameters file wins
over them (``ModelParsers.merge_default_parameters``). The order of precedence is

    host parameters file > supermodule parameterisation > supermodule default_parameters
                         > submodule / component parameterisation

and within one tier the first record in the (expanded) module array wins, so a global set by
several parameterisations is added once. A global is one value for the whole model, so a
parameterisation that set it to another value than the one used is warned about
(``ConflictingGlobalWarning``): its module runs at a value it did not choose. The exceptions
are a supermodule overriding the parameterisations inside it, which is what a supermodule
parameterisation is for, and a global the host parameters file sets.

A supermodule entry may also live in a version directory with parameterisations. Its
parameterisation's rows are named ``{var}_{submodule}`` (local) or are globals, like
``default_parameters``, and are renamed to ``{var}_{instance}_{submodule}``
(``supermodules.rename_default_parameter``), ``instance`` being the supermodule record's name.
'''

import os
import warnings

import pandas as pd

PARAMETERISATIONS_DIR = 'parameterisations'
# the older name of PARAMETERISATIONS_DIR, used when a version has no parameterisations/
INSTANCES_DIR = 'instances'
# the record / config entry keys naming a parameterisation: (new, older) -> internal (older)
PARAMETERISATION_KEY, INSTANCE_KEY = 'parameterisation', 'instance'
DEFAULT_PARAMETERISATION_KEY = 'default_parameterisation'
DEFAULT_INSTANCE_KEY = 'default_instance'
GLOBAL_CONSTANT = 'global_constant'

# the columns a parameters CSV (an instance's, or a supermodule's default_parameters) must
# have; config_schemas.PARAMETER_COLUMNS re-exports this
PARAMETER_COLUMNS = ('variable_name', 'units', 'value', 'data_reference')


def check_instance_name(value, where, key='instance'):
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


def merge_renamed_key(mapping, new_key, old_key, where):
    '''
    A copy of ``mapping`` with its ``new_key`` (e.g. ``parameterisation``) stored under the
    older ``old_key`` (``instance``), which the code reads. Either spelling may be given;
    both with different values is a ValueError naming ``where``. ``mapping`` is returned
    unchanged when it has no ``new_key``.
    '''
    if new_key not in mapping:
        return mapping
    if old_key in mapping:
        new, old = mapping[new_key], mapping[old_key]
        same = (isinstance(new, str) and isinstance(old, str) and new.strip() == old.strip()) \
            or new == old
        if not same:
            raise ValueError(f'{where} has both "{new_key}" ({new!r}) and its older name '
                             f'"{old_key}" ({old!r}). Give only "{new_key}".')
    return {(old_key if k == new_key else k): v for k, v in mapping.items() if k != old_key}


def record_instance(record, where):
    '''The parameterisation a module-array record names (``"parameterisation"``, or the older
    ``"instance"``), None when it names none. Normalised records only carry ``"instance"``;
    this also takes records that were not normalised.'''
    return merge_renamed_key(record, PARAMETERISATION_KEY, INSTANCE_KEY, where).get(INSTANCE_KEY)


def instances_dir(config_path):
    '''Where the parameterisations of the config at ``config_path`` live:
    ``<config dir>/parameterisations``, or the older ``<config dir>/instances`` when only that
    exists.'''
    version_dir = os.path.dirname(os.path.abspath(config_path))
    path = os.path.join(version_dir, PARAMETERISATIONS_DIR)
    legacy = os.path.join(version_dir, INSTANCES_DIR)
    if not os.path.isdir(path) and os.path.isdir(legacy):
        return legacy
    return path


def instance_parameters_path(config_path, instance):
    '''``<config dir>/parameterisations/<name>/<name>_parameters.csv`` (or under
    ``instances/``, see ``instances_dir``)'''
    return os.path.join(instances_dir(config_path), instance, f'{instance}_parameters.csv')


def available_instances(config_path):
    '''The parameterisations next to the config at ``config_path``: the subdirectories of
    ``parameterisations/`` (or ``instances/``) that hold a ``<name>_parameters.csv``, sorted.'''
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
    ``(name, rows)`` for the config ``entry`` (a normalised module config entry, component or
    supermodule, carrying the ``config_path`` it was loaded from) and the record's
    parameterisation ``instance`` (None when the record names none). The rows are the raw
    ones of the parameterisation's parameters file, not yet renamed; ``(None, [])`` when
    nothing applies.

    * an explicit parameterisation must exist: ValueError naming the version directory and
      the parameterisations it has, or saying that the config has no parameterisations
      directory;
    * with none, ``entry["default_instance"]`` (the normalised ``default_parameterisation``)
      is used if its parameters file exists.
    '''
    config_path = entry.get('config_path')
    if instance is None:
        default = entry.get('default_instance')
        if not default or not config_path:
            return None, []
        path = instance_parameters_path(config_path, default)
        if not os.path.isfile(path):
            return None, []
        return default, read_parameter_rows(path, where,
                                            f'default parameterisation "{default}" file')

    label = _entry_label(entry)
    if not config_path:
        raise ValueError(
            f'{where}: names parameterisation "{instance}" of {label}, but that module config '
            f'was not loaded from a module library directory, so there is nowhere to look for '
            f'parameterisations. Parameterisations live in <module>/versions/<version>/'
            f'{PARAMETERISATIONS_DIR}/<name>/ next to the version\'s *_modules_config.json.')
    version_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isdir(instances_dir(config_path)):
        raise ValueError(
            f'{where}: names parameterisation "{instance}" of {label}, but its module config '
            f'{config_path} has no {PARAMETERISATIONS_DIR}/ (or older {INSTANCES_DIR}/) '
            f'directory next to it, so {label} has no parameterisations. Parameterisations '
            f'live in {version_dir}/{PARAMETERISATIONS_DIR}/<name>/<name>_parameters.csv.')
    path = instance_parameters_path(config_path, instance)
    if not os.path.isfile(path):
        existing = available_instances(config_path)
        raise ValueError(
            f'{where}: unknown parameterisation "{instance}" of {label}: {path} does not '
            f'exist. The parameterisations in {instances_dir(config_path)} are '
            f'{existing or "none"}.')
    return instance, read_parameter_rows(path, where, f'parameterisation "{instance}" file')


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


def component_instance_rows(records, component_registry, source=None, settings=None):
    '''
    The parameterisation rows of every record in ``records`` (the normalised records of an
    expanded module array, whose parameterisation is in ``record["instance"]``) whose
    (vessel_type, BC_type) is in ``component_registry`` (see
    ``config_schemas.load_component_registry``), in record order: each row ``{var}`` renamed
    to ``{var}_{record name}``, or kept if ``var`` is a global constant of the module.

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
        key = (record['vessel_type'], record['BC_type'])
        where = f'{source}: "{record["name"]}" {key}'
        instance = record_instance(record, where)
        entry = component_registry.get(key)
        if entry is None:
            if instance is not None:
                raise ValueError(f'{where}: names parameterisation "{instance}", but no module '
                                 f'config entry of that type was found, so there is no module '
                                 f'library directory to look for its parameterisations in.')
            continue
        used, raw = instance_parameter_rows(entry, instance, where)
        globals_ = global_constants(entry)
        for row in raw:
            row = dict(row)
            if row['variable_name'] not in globals_:
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
