'''
Reads module configs and module arrays written in either of the two schemas in use, and
converts them to libcuflynx's internal names.

Module config entries (``*_modules_config.json``)
=================================================

A module config file is a JSON list. Each entry is either a *component* entry (a module
that lives in a CellML/C++ file) or a *supermodule* entry (a named group of modules, below).

Component entries:

==================  ==================  ============================================
libcuflynx          PhLynx              meaning
==================  ==================  ============================================
``vessel_type``     ``module_type``     the kind of module named in a module array
``BC_type``         ``module_subtype``  its boundary-condition variant
``module_file``     ``component_file``  the CellML file the module lives in
``module_type``     ``component_type``  the CellML component name
==================  ==================  ============================================

``module_type`` means different things in the two schemas, so the PhLynx schema is recognised
by ``module_subtype``, ``component_file`` or ``component_type``, never by ``module_type``. An
entry that mixes keys of both schemas is an error. One library may use both schemas across
(or even within) its files: each entry is read on its own.

Supermodule entries are recognised by ``"module_format": "supermodule"`` (or by having a
``submodules`` list):

======================  ==================  ===============================================
libcuflynx              PhLynx              meaning
======================  ==================  ===============================================
``vessel_type``         ``module_type``     the type an instance names in a module array
``BC_type``             ``module_subtype``  its variant (conventionally ``supermodule``)
``module_format``       ``module_format``   ``"supermodule"``
``submodules``          ``submodules``      a module array (list of records, below) whose
                                            names are local to the supermodule
``default_parameters``  same                optional parameters CSV, relative to the
                                            config file's directory
``description``         same                optional free text
======================  ==================  ===============================================

A submodule's inp/out lists name sibling submodules only. A submodule may itself be a
supermodule instance, with its own ``per_submodule_inputs``/``per_submodule_outputs`` naming
siblings. ``default_parameters`` rows are ``variable_name,units,value,data_reference``; a row
named ``{var}_{submodule}`` is a local parameter of that submodule, any other row a global.
Supermodule entries never reach the module dataframe or the (vessel_type, BC_type) join:
they are collected by ``load_supermodule_registry`` and expanded out of the module array
(``utilities/supermodules.py``) before anything else reads it.

Module arrays
=============

``<prefix>_module_array.json`` (preferred) or ``<prefix>_module_array.csv``, looked for in
that order (``module_array_path``). Module arrays used to be called vessel arrays, and
``<prefix>_vessel_array.json``/``.csv`` are still read, with a warning to rename them.

The JSON form is a list of instance records:

==================  ==================  ==================================================
libcuflynx          PhLynx              meaning
==================  ==================  ==================================================
``name``            ``name``            the instance name (required)
``vessel_type``     ``module_type``     the module type (required)
``BC_type``         ``module_subtype``  its boundary-condition variant (required)
``inp_vessels``     ``inp_instances``   list of upstream instance names (optional)
``out_vessels``     ``out_instances``   list of downstream instance names (optional)
==================  ==================  ==================================================

A record uses one key style; mixing the two in one record is an error. A list may also be
given as one space-separated string. Other keys are allowed. A supermodule instance adds

* ``per_submodule_inputs`` -- ``{submodule: [host, ...]}``: the hosts feeding that
  submodule. Each host must list the instance in its out list.
* ``per_submodule_outputs`` -- ``{submodule: [host, ...]}``: the hosts that submodule
  feeds. Each host must list the instance in its inp list.

Either may also be written as a list of one-key objects, ``[{"ra": ["svc"]}, ...]``.

The CSV forms have the columns

* libcuflynx: ``name, BC_type, vessel_type, inp_vessels, out_vessels``
* PhLynx's Circulatory Autogen export: ``name, module_type, module_subtype, inp_instances,
  out_instances``

with space-separated lists; every other cell keeps its first token. A CSV is read by
converting each row to the JSON record above (``read_module_array_records``), so both forms
are processed identically. ``python -m libcuflynx.utilities.config_schemas to-json <csv>...``
converts CSV arrays to JSON.

Machine-readable JSON Schemas for both files ship as package data in ``libcuflynx/schemas/``
(``module_array.schema.json``, ``module_config.schema.json``). The loaders here check the
same rules themselves, so no JSON Schema library is needed at run time.

Normalisation happens where the files are loaded, so everything downstream only ever sees
the libcuflynx names.
'''

import argparse
import json
import warnings
import os
import sys

import pandas as pd

from libcuflynx.generators.multi_port import normalise_port_multi_port

# PhLynx key -> libcuflynx key, for module config entries
PHLYNX_MODULE_KEYS = {
    'module_type': 'vessel_type',
    'module_subtype': 'BC_type',
    'component_file': 'module_file',
    'component_type': 'module_type',
}
# keys that only a PhLynx-schema entry has
_PHLYNX_ONLY_KEYS = ('module_subtype', 'component_file', 'component_type')
# keys that only a libcuflynx-schema entry has
_LIBCUFLYNX_ONLY_KEYS = ('vessel_type', 'BC_type', 'module_file')

_PORT_KEYS = ('entrance_ports', 'exit_ports', 'general_ports')

SUPERMODULE_FORMAT = 'supermodule'

# module arrays used to be called vessel arrays: <prefix>_vessel_array.json/.csv are still read
LEGACY_ARRAY_SUFFIXES = ('_vessel_array.json', '_vessel_array.csv')

# PhLynx column -> libcuflynx column, for module arrays
PHLYNX_MODULE_ARRAY_COLUMNS = {
    'module_type': 'vessel_type',
    'module_subtype': 'BC_type',
    'inp_instances': 'inp_vessels',
    'out_instances': 'out_vessels',
}
_PHLYNX_ONLY_COLUMNS = ('module_subtype', 'inp_instances', 'out_instances')
_LIBCUFLYNX_ONLY_COLUMNS = ('BC_type', 'vessel_type', 'inp_vessels', 'out_vessels')
_LIBCUFLYNX_COLUMN_ORDER = ('name', 'BC_type', 'vessel_type', 'inp_vessels', 'out_vessels')
# a vessel record's keys whose values are lists of instance names
_LIST_COLUMNS = ('inp_vessels', 'out_vessels')
PER_SUBMODULE_KEYS = ('per_submodule_inputs', 'per_submodule_outputs')

# the columns a parameters CSV (and a supermodule's default_parameters) must have
PARAMETER_COLUMNS = ('variable_name', 'units', 'value', 'data_reference')


# --------------------------------------------------------------------------------------------
# module config entries
# --------------------------------------------------------------------------------------------

def _describe(entry, source):
    where = f' in {source}' if source else ''
    name = entry.get('vessel_type', entry.get('module_type', '?'))
    subtype = entry.get('BC_type', entry.get('module_subtype', '?'))
    return f'module config entry ({name}, {subtype}){where}'


def is_supermodule_entry(entry):
    '''Whether a module config entry (either schema) is a supermodule entry.'''
    return isinstance(entry, dict) and (entry.get('module_format') == SUPERMODULE_FORMAT
                                        or 'submodules' in entry)


def normalise_supermodule_entry(entry, source=None):
    '''
    A supermodule config entry, in either key style, with ``vessel_type``/``BC_type`` keys
    first and its ``submodules`` normalised to libcuflynx vessel records (see
    ``normalise_vessel_record``). ``entry`` itself is not modified.

    Raises ValueError for mixed key styles, a component key (a supermodule has none), a
    missing type/subtype, an empty or malformed ``submodules`` list, repeated submodule
    names, or a submodule connection to something that is not a sibling submodule.
    '''
    description = 'supermodule ' + _describe(entry, source)[len('module config '):]
    has_phlynx = [k for k in ('module_type', 'module_subtype') if k in entry]
    has_libcuflynx = [k for k in ('vessel_type', 'BC_type') if k in entry]
    if has_phlynx and has_libcuflynx:
        raise ValueError(f'{description} mixes the libcuflynx keys {has_libcuflynx} and the '
                         f'PhLynx keys {has_phlynx}. Use vessel_type/BC_type or '
                         f'module_type/module_subtype.')
    for component_key in ('component_file', 'component_type', 'module_file'):
        if component_key in entry:
            raise ValueError(f'{description} has "{component_key}": a supermodule has no '
                             f'component of its own, only "submodules".')
    type_key, subtype_key = (('module_type', 'module_subtype') if has_phlynx
                             else ('vessel_type', 'BC_type'))
    for key in (type_key, subtype_key):
        if not isinstance(entry.get(key), str) or not entry.get(key):
            raise ValueError(f'{description} needs a non-empty string "{key}".')
    if entry.get('module_format', SUPERMODULE_FORMAT) != SUPERMODULE_FORMAT:
        raise ValueError(f'{description} has submodules but module_format '
                         f'"{entry.get("module_format")}"; a supermodule entry has '
                         f'module_format "{SUPERMODULE_FORMAT}".')
    submodules = entry.get('submodules')
    if not isinstance(submodules, list) or not submodules:
        raise ValueError(f'{description} needs "submodules", a non-empty list of records.')
    default_parameters = entry.get('default_parameters')
    if default_parameters is not None and not isinstance(default_parameters, str):
        raise ValueError(f'{description}: "default_parameters" must be a file name (string), '
                         f'not {default_parameters!r}.')

    records = [normalise_vessel_record(sub, source=f'{description}, submodules', index=i, what=None)
               for i, sub in enumerate(submodules)]
    names = [r['name'] for r in records]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f'{description}: submodule names {duplicated} are repeated.')
    for i, record in enumerate(records):
        for key in _LIST_COLUMNS:
            unknown = [n for n in record[key] if n not in names]
            if unknown:
                raise ValueError(
                    f'{description}, submodules[{i}] ("{record["name"]}"): "{key}" names '
                    f'{unknown}, which are not submodules of this supermodule. Submodules '
                    f'connect only to each other; connections to the host model are made by '
                    f'the instance\'s per_submodule_inputs/per_submodule_outputs.')
        for key in PER_SUBMODULE_KEYS:
            for sub, hosts in record.get(key, {}).items():
                unknown = [h for h in hosts if h not in names]
                if unknown:
                    raise ValueError(
                        f'{description}, submodules[{i}] ("{record["name"]}"): "{key}" names '
                        f'{unknown} for "{sub}", which are not submodules of this supermodule.')

    normalised = {'vessel_type': entry[type_key], 'BC_type': entry[subtype_key]}
    for key, value in entry.items():
        if key not in (type_key, subtype_key):
            normalised[key] = value
    normalised['module_format'] = SUPERMODULE_FORMAT
    normalised['submodules'] = records
    return normalised


def normalise_module_config_entry(entry, source=None):
    '''
    One module config entry, in either schema, as a libcuflynx-schema entry.

    A libcuflynx entry comes back with its keys unchanged; a PhLynx entry has its keys
    renamed, with ``vessel_type`` and ``BC_type`` first and the rest in their original order.
    In both, ``multi_port`` values are put in canonical form (see
    ``generators/multi_port.normalise_port_multi_port``). ``entry`` itself is not modified.
    A supermodule entry is normalised by ``normalise_supermodule_entry``.

    Raises ValueError for an entry that mixes the two schemas, a PhLynx entry missing one of
    its four keys, or an entry in neither schema.
    '''
    if not isinstance(entry, dict):
        raise ValueError(f'module config entry{" in " + source if source else ""} is a '
                         f'{type(entry).__name__}, not a JSON object: {entry!r}')
    if is_supermodule_entry(entry):
        return normalise_supermodule_entry(entry, source)
    phlynx_keys = [k for k in _PHLYNX_ONLY_KEYS if k in entry]
    libcuflynx_keys = [k for k in _LIBCUFLYNX_ONLY_KEYS if k in entry]
    if phlynx_keys and libcuflynx_keys:
        raise ValueError(
            f'{_describe(entry, source)} mixes the libcuflynx schema (has {libcuflynx_keys}) '
            f'and the PhLynx schema (has {phlynx_keys}). Use one schema per entry: either '
            f'vessel_type/BC_type/module_file/module_type, or '
            f'module_type/module_subtype/component_file/component_type.')

    if phlynx_keys:
        missing = [k for k in PHLYNX_MODULE_KEYS if k not in entry]
        if missing:
            raise ValueError(
                f'{_describe(entry, source)} is in the PhLynx schema (has {phlynx_keys}) but '
                f'is missing {missing}. A PhLynx-schema entry needs module_type, '
                f'module_subtype, component_file and component_type.')
        renamed = {PHLYNX_MODULE_KEYS.get(k, k): v for k, v in entry.items()}
        normalised = {'vessel_type': renamed.pop('vessel_type'), 'BC_type': renamed.pop('BC_type')}
        normalised.update(renamed)
    elif libcuflynx_keys:
        missing = [k for k in ('vessel_type', 'BC_type', 'module_file', 'module_type') if k not in entry]
        if missing:
            raise ValueError(
                f'{_describe(entry, source)} is in the libcuflynx schema (has {libcuflynx_keys}) but '
                f'is missing {missing}. A libcuflynx-schema entry needs vessel_type, BC_type, '
                f'module_file and module_type (the CellML component).')
        normalised = dict(entry)
    else:
        raise ValueError(
            f'Cannot tell the schema of {_describe(entry, source)}: it has none of '
            f'vessel_type/BC_type/module_file (libcuflynx) or '
            f'module_subtype/component_file/component_type (PhLynx). Keys: {list(entry)}')

    description = _describe(normalised, source)
    for port_key in _PORT_KEYS:
        ports = normalised.get(port_key)
        if isinstance(ports, list):
            normalised[port_key] = [normalise_port_multi_port(port, description) for port in ports]
    return normalised


def normalise_module_config(entries, source=None):
    '''A list of module config entries (one JSON file), each normalised on its own.'''
    if not isinstance(entries, list):
        raise ValueError(f'module config {source or ""} must be a JSON list of entries, '
                         f'not a {type(entries).__name__}')
    return [normalise_module_config_entry(entry, source) for entry in entries]


def load_module_config(path, include_supermodules=False):
    '''
    The normalised entries of the module config JSON file at ``path``. Supermodule entries
    are left out unless ``include_supermodules`` (they are read by
    ``load_supermodule_registry``, never joined to a module array as components).
    '''
    with open(path, encoding='utf-8-sig') as f:
        entries = normalise_module_config(json.load(f), source=str(path))
    if include_supermodules:
        return entries
    return [e for e in entries if not is_supermodule_entry(e)]


def load_supermodule_registry(config_files):
    '''
    ``{(vessel_type, BC_type): supermodule entry}`` for every supermodule entry in
    ``config_files``. Each entry also gets ``config_path`` (the file it came from), against
    whose directory its ``default_parameters`` is resolved. Raises ValueError if the same
    (vessel_type, BC_type) supermodule is defined twice.
    '''
    registry = {}
    for path in config_files:
        with open(path, encoding='utf-8-sig') as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not any(is_supermodule_entry(e) for e in raw):
            continue
        for entry in normalise_module_config(raw, source=str(path)):
            if not is_supermodule_entry(entry):
                continue
            key = (entry['vessel_type'], entry['BC_type'])
            if key in registry:
                raise ValueError(f'supermodule {key} is defined twice: in '
                                 f'{registry[key]["config_path"]} and in {path}.')
            entry['config_path'] = str(path)
            registry[key] = entry
    return registry


# --------------------------------------------------------------------------------------------
# module arrays
# --------------------------------------------------------------------------------------------

def normalise_module_array_columns(df, source=None):
    '''
    A module-array dataframe with libcuflynx column names.

    A PhLynx-layout array (``module_type``, ``module_subtype``, ``inp_instances``,
    ``out_instances``) has its columns renamed and put in the libcuflynx order (``name,
    BC_type, vessel_type, inp_vessels, out_vessels``, then any others). A libcuflynx-layout
    array is returned unchanged. Column names are expected to be stripped already.
    '''
    columns = [str(c) for c in df.columns]
    phlynx_columns = [c for c in _PHLYNX_ONLY_COLUMNS if c in columns]
    libcuflynx_columns = [c for c in _LIBCUFLYNX_ONLY_COLUMNS if c in columns]
    where = f' {source}' if source else ''
    if phlynx_columns and libcuflynx_columns:
        raise ValueError(
            f'module array{where} mixes libcuflynx columns {libcuflynx_columns} and PhLynx '
            f'columns {phlynx_columns}. Use either name,BC_type,vessel_type,inp_vessels,'
            f'out_vessels or name,module_type,module_subtype,inp_instances,out_instances.')
    if not phlynx_columns:
        return df
    missing = [c for c in ('name',) + tuple(PHLYNX_MODULE_ARRAY_COLUMNS) if c not in columns]
    if missing:
        raise ValueError(f'module array{where} is in the PhLynx layout (has {phlynx_columns}) '
                         f'but is missing the columns {missing}.')
    df = df.rename(columns=PHLYNX_MODULE_ARRAY_COLUMNS)
    ordered = list(_LIBCUFLYNX_COLUMN_ORDER) + [c for c in df.columns if c not in _LIBCUFLYNX_COLUMN_ORDER]
    return df[ordered]


def read_module_array_csv(path, **read_csv_kwargs):
    '''``pd.read_csv`` of a module array, with its columns normalised to libcuflynx names.'''
    df = pd.read_csv(path, **read_csv_kwargs)
    df = df.rename(columns=lambda c: str(c).strip())
    return normalise_module_array_columns(df, source=str(path))


def module_array_path(resources_dir, file_prefix):
    '''
    The module array of ``file_prefix`` in ``resources_dir``: the first that exists of
    ``<prefix>_module_array.json`` and ``<prefix>_module_array.csv``, then the older names
    ``<prefix>_vessel_array.json`` and ``<prefix>_vessel_array.csv``, which are still read with
    a FutureWarning asking for the file to be renamed. When none exists the
    ``_module_array.csv`` path is returned, so the error names the usual file.
    '''
    candidates = [os.path.join(resources_dir, file_prefix + suffix)
                  for suffix in ('_module_array.json', '_module_array.csv') + LEGACY_ARRAY_SUFFIXES]
    existing = [c for c in candidates if os.path.exists(c)]
    if len(existing) > 1:
        # e.g. after `config_schemas to-json`, which writes the JSON next to the CSV: edits to
        # the CSV would otherwise be ignored without a word
        warnings.warn(f'{len(existing)} module arrays for "{file_prefix}" in {resources_dir}: '
                      f'{[os.path.basename(c) for c in existing]}. Using '
                      f'{os.path.basename(existing[0])}; the others are ignored. Remove or rename '
                      f'the ones you do not mean.', UserWarning, stacklevel=2)
    if not existing:
        return candidates[1]
    chosen = existing[0]
    if chosen.endswith(LEGACY_ARRAY_SUFFIXES):
        # FutureWarning, not DeprecationWarning: this is for the person running the model, and
        # Python hides DeprecationWarning outside __main__
        renamed = os.path.basename(chosen).replace('_vessel_array.', '_module_array.')
        warnings.warn(f'{os.path.basename(chosen)} uses the old name "vessel array"; rename it to '
                      f'{renamed}. The old name is still read for now.', FutureWarning, stacklevel=2)
    return chosen


def _record_where(source, index, record=None, what='module array'):
    where = f'{what} {source}' if what and source else (source or what)
    if index is not None:
        where += f', record {index}'
    if isinstance(record, dict) and isinstance(record.get('name'), str):
        where += f' ("{record["name"]}")'
    return where


def _name_list(value, where, key):
    '''A list of names from a JSON list of strings or one space-separated string.'''
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list):
        names = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f'{where}: "{key}" must be a list of names (strings), but '
                                 f'contains {item!r}.')
            names += item.split()
        return names
    raise ValueError(f'{where}: "{key}" must be a list of names or a space-separated string, '
                     f'not {value!r}.')


def normalise_per_submodule(value, where, key):
    '''
    ``per_submodule_inputs``/``per_submodule_outputs`` as an ordered ``{submodule: [hosts]}``
    dict, from a dict or a list of one-key dicts.
    '''
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = []
        for i, item in enumerate(value):
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f'{where}: "{key}"[{i}] must be an object with exactly one '
                                 f'key (a submodule name), not {item!r}.')
            items += list(item.items())
    else:
        raise ValueError(f'{where}: "{key}" must be an object {{submodule: [hosts]}} or a list '
                         f'of one-key objects, not {value!r}.')
    out = {}
    for sub, hosts in items:
        if sub in out:
            raise ValueError(f'{where}: "{key}" lists submodule "{sub}" twice.')
        out[sub] = _name_list(hosts, where, f'{key}.{sub}')
    return out


def is_heart_vessel_type(vessel_type):
    """Whether ``vessel_type`` is the monolithic heart (heart, heart_ASD, heart_nonstiff, ...),
    which the generators special-case: its venous (ivc/svc) and pulmonary inputs, and the
    pulmonary circuit added when it has one output. Not the heart_effector* controllers. The
    generators used to find the heart by its *name*, "heart", which a heart inside a
    supermodule (named <instance>_<submodule>) cannot have."""
    vessel_type = str(vessel_type or '')
    return vessel_type.startswith('heart') and not vessel_type.startswith('heart_effector')


def normalise_vessel_record(record, source=None, index=None, what='module array'):
    '''
    One module-array record, in either key style, as a libcuflynx record: ``name, BC_type,
    vessel_type, inp_vessels, out_vessels`` (the last two lists of names), then any other
    keys in their original order, with ``per_submodule_*`` as ordered dicts. ``record``
    itself is not modified.

    Raises ValueError naming ``source``, ``index`` and the key for a record that is not an
    object, mixes the two key styles, lacks name/type/subtype, or has a malformed list.
    '''
    where = _record_where(source, index, record, what=what)
    if not isinstance(record, dict):
        raise ValueError(f'{where} is a {type(record).__name__}, not a JSON object: {record!r}')
    phlynx_keys = [k for k in PHLYNX_MODULE_ARRAY_COLUMNS if k in record]
    libcuflynx_keys = [k for k in _LIBCUFLYNX_ONLY_COLUMNS if k in record]
    if phlynx_keys and libcuflynx_keys:
        raise ValueError(
            f'{where} mixes libcuflynx keys {libcuflynx_keys} and PhLynx keys {phlynx_keys}. '
            f'Use either name/vessel_type/BC_type/inp_vessels/out_vessels or '
            f'name/module_type/module_subtype/inp_instances/out_instances.')
    if phlynx_keys:
        rename = PHLYNX_MODULE_ARRAY_COLUMNS
        type_key, subtype_key = 'module_type', 'module_subtype'
    else:
        rename = {}
        type_key, subtype_key = 'vessel_type', 'BC_type'
    for key in ('name', type_key, subtype_key):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{where}: "{key}" is required and must be a non-empty string '
                             f'(got {value!r}).')
        if len(value.split()) > 1:
            # names become CellML component and variable names, and the generator's frame
            # keeps a cell's first word only: "a b" was silently truncated to "a"
            raise ValueError(f'{where}: "{key}" {value!r} contains whitespace. Names, module '
                             f'types and versions cannot contain spaces; use underscores.')
    out = {'name': record['name'].strip(),
           'BC_type': record[subtype_key].strip(),
           'vessel_type': record[type_key].strip()}
    for key, value in record.items():
        new_key = rename.get(key, key)
        if new_key in ('name', 'BC_type', 'vessel_type'):
            continue
        if new_key in _LIST_COLUMNS:
            out[new_key] = _name_list(value, where, key)
        elif new_key in PER_SUBMODULE_KEYS:
            out[new_key] = normalise_per_submodule(value, where, key)
        else:
            out[new_key] = value
    for key in _LIST_COLUMNS:
        out.setdefault(key, [])
    ordered = {k: out[k] for k in _LIBCUFLYNX_COLUMN_ORDER}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    return ordered


def normalise_vessel_records(records, source=None):
    '''A list of vessel records (one JSON file), each normalised on its own.'''
    if not isinstance(records, list):
        raise ValueError(f'module array {source or ""} must be a JSON list of records, not a '
                         f'{type(records).__name__}.')
    return [normalise_vessel_record(r, source=source, index=i) for i, r in enumerate(records)]


def _first_token(cell):
    tokens = cell.split() if isinstance(cell, str) else []
    return tokens[0].strip() if tokens else ''


def module_array_csv_to_records(path):
    '''
    The rows of a CSV module array (either layout) as raw libcuflynx-keyed records: list
    columns split on whitespace, every other cell reduced to its first token ('' if empty),
    exactly as the CSV reader has always treated them.
    '''
    df = pd.read_csv(path, dtype=str, na_filter=False)
    df = df.rename(columns=lambda c: str(c).strip())
    df = normalise_module_array_columns(df, source=str(path))
    records = []
    for row in df.itertuples(index=False, name=None):
        record = {}
        for column, cell in zip(df.columns, row):
            if column in _LIST_COLUMNS:
                record[column] = cell.split() if isinstance(cell, str) else []
            else:
                record[column] = _first_token(cell)
        records.append(record)
    return records


def read_module_array_records(path):
    '''
    The records of the module array at ``path`` -- a ``.json`` list of records, or a CSV
    converted row by row -- normalised to libcuflynx keys (``normalise_vessel_record``).
    Supermodule instances are not expanded here (see ``load_module_array``).
    '''
    path = str(path)
    if path.lower().endswith('.json'):
        with open(path, encoding='utf-8-sig') as f:
            try:
                raw = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f'module array {path} is not valid JSON: {e}') from e
        return normalise_vessel_records(raw, source=path)
    return normalise_vessel_records(module_array_csv_to_records(path), source=path)


def _frame_cell(value, is_list):
    if is_list:
        return list(value)
    if value is None:
        return []
    if not isinstance(value, str):
        value = str(value)
    tokens = value.split()
    return tokens[0].strip() if tokens else []


def vessel_records_to_frame(records):
    '''
    Normalised vessel records as the list-form dataframe the generator uses: the libcuflynx
    columns first, then every other scalar key (in the order first seen); inp/out cells are
    lists, every other cell its first token, or ``[]`` if empty -- the frame
    ``CSVFileParser.get_data_as_dataframe_multistrings(..., vessel_array=True)`` returns for
    a CSV. Keys holding objects or lists (``per_submodule_*``, positions...) are not columns.
    '''
    columns = list(_LIBCUFLYNX_COLUMN_ORDER)
    for record in records:
        for key, value in record.items():
            if key not in columns and not isinstance(value, (dict, list)):
                columns.append(key)
    df = pd.DataFrame(index=range(len(records)), columns=columns, dtype=object)
    for i, record in enumerate(records):
        for j, column in enumerate(columns):
            df.iat[i, j] = _frame_cell(record.get(column), column in _LIST_COLUMNS)
    return df


def vessel_records_to_string_frame(records):
    '''
    Normalised vessel records as a dataframe of strings (lists joined by spaces, empty cells
    ''), the form a CSV module array is written from.
    '''
    frame = vessel_records_to_frame(records)
    for column in frame.columns:
        frame[column] = [' '.join(v) if column in _LIST_COLUMNS else (v if isinstance(v, str) else '')
                         for v in frame[column]]
    return frame


def load_expanded_vessel_records(path, supermodule_registry=None):
    '''
    The records of the module array at ``path`` with every supermodule instance expanded,
    and the supermodules' default parameter rows: ``(records, extra_param_rows)``.
    '''
    from libcuflynx.utilities.supermodules import expand_supermodules
    return expand_supermodules(read_module_array_records(path), supermodule_registry or {},
                               source=str(path))


def load_module_array(path, supermodule_registry=None):
    '''
    The module array at ``path`` (JSON or CSV), with every supermodule instance expanded
    (``utilities/supermodules.py``), as ``(frame, extra_param_rows)``:

    * ``frame`` -- the list-form dataframe of ``vessel_records_to_frame``;
    * ``extra_param_rows`` -- the supermodules' default parameters under the expanded names,
      a list of ``{variable_name, units, value, data_reference}`` dicts, to be used wherever
      the model's parameters file does not set that name.
    '''
    records, extra_param_rows = load_expanded_vessel_records(path, supermodule_registry)
    return vessel_records_to_frame(records), extra_param_rows


def record_to_style(record, style='phlynx'):
    '''A normalised (libcuflynx-keyed) record with PhLynx or libcuflynx key names.'''
    if style == 'libcuflynx':
        return dict(record)
    if style != 'phlynx':
        raise ValueError(f'style must be "phlynx" or "libcuflynx", not {style!r}')
    out = {'name': record['name'],
           'module_type': record['vessel_type'],
           'module_subtype': record['BC_type'],
           'inp_instances': list(record['inp_vessels']),
           'out_instances': list(record['out_vessels'])}
    for key, value in record.items():
        if key not in _LIBCUFLYNX_COLUMN_ORDER:
            out[key] = value
    return out


def dump_vessel_records(records, style='phlynx'):
    '''JSON text of a module array: a list with one record per line, indent 1.'''
    lines = [' ' + json.dumps(record_to_style(r, style)) for r in records]
    return '[\n' + ',\n'.join(lines) + '\n]\n' if lines else '[]\n'


def module_array_to_json(csv_path, json_path=None, style='phlynx'):
    '''
    Convert the CSV module array at ``csv_path`` to JSON records at ``json_path`` (default:
    the same name with ``.json``), with PhLynx keys (``style='phlynx'``, the default) or
    libcuflynx keys. Returns the JSON path.
    '''
    csv_path = str(csv_path)
    if json_path is None:
        json_path = os.path.splitext(csv_path)[0] + '.json'
    records = read_module_array_records(csv_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(dump_vessel_records(records, style))
    return str(json_path)


def main(argv=None):
    '''``python -m libcuflynx.utilities.config_schemas to-json <csv>... [--style ...]``'''
    parser = argparse.ArgumentParser(
        prog='python -m libcuflynx.utilities.config_schemas',
        description='Tools for libcuflynx module arrays and module configs.')
    commands = parser.add_subparsers(dest='command', required=True)
    to_json = commands.add_parser(
        'to-json', help='convert CSV module arrays to JSON records, written next to each CSV')
    to_json.add_argument('csv', nargs='+', help='CSV module array(s) to convert')
    to_json.add_argument('--style', choices=('phlynx', 'libcuflynx'), default='phlynx',
                         help='key names to write (default: phlynx)')
    args = parser.parse_args(argv)
    if args.command == 'to-json':
        for csv_path in args.csv:
            print(module_array_to_json(csv_path, style=args.style))
    return 0


if __name__ == '__main__':
    sys.exit(main())
