'''
``cuflynx-upgrade``: convert, once, files written for an earlier libcuflynx.

libcuflynx reads only the current forms of its input files. A reader that meets an older form
refuses the file and names the command here that converts it::

    cuflynx-upgrade module-config <*_modules_config.json>... [--module-library-dir <dir>]
    cuflynx-upgrade module-array <*_module_array|_vessel_array .json|.csv>...
    cuflynx-upgrade obs-data <*_obs_data.json>... --user-inputs <user_inputs.yaml>
    cuflynx-upgrade parameter-names --module-array <array> --module-library-dir <dir> ...
    cuflynx-upgrade parameters <parameters CSV | *_modules_config.json | directory>...
    cuflynx-upgrade supermodule-instances <*_module_array .json|.csv>... --module-library-dir <dir>
    cuflynx-upgrade params-for-id <params_for_id CSV | directory>...

Each converts its files in place (a ``_vessel_array`` file is renamed; a parameters CSV becomes
the ``.json`` beside it) and prints what it changed. Convert a model's module configs and module
array, then its parameter names (they are checked against the converted model), then its
parameters files. A file already in the current form is left as it is. ``--dry-run`` prints the
changes without writing them.

What each converts:

* ``module-config``: the older keys ``vessel_type``/``BC_type``/``module_file``/``module_type``
  (component) to ``module_type``/``module_subtype``/``component_file``/``component_type``;
  ``default_instance`` to ``default_parameterisation``; a ``multi_port`` of ``"False"`` is
  dropped; a supermodule's ``default_parameters`` CSV becomes its ``default`` parameterisation;
  a supermodule's inline ``submodules`` become its ``submodule_versions``, its own module array
  file and its default parameterisation's instance values: each instance's from the
  parameterisation it named (else its version's default), with the supermodule's
  ``<instance>/<variable>`` rows over them (``--module-library-dir`` says where the submodule
  versions are). The config's parameterisations are then converted as ``parameters`` does.
* ``module-array``: the keys or columns ``vessel_type``/``BC_type``/``inp_vessels``/
  ``out_vessels`` to ``module_type``/``module_subtype``/``inp_instances``/``out_instances``;
  ``<prefix>_vessel_array.*`` is renamed ``<prefix>_module_array.*``. No module is added: an
  array lists every module of its model (a heart with one output gets no pulmonary artery and
  vein; add ``par`` and ``pvn`` to the array if the model has them).
* ``obs-data``: a bare list of data items, or an obs_data with no ``protocol_info``, gets the
  one-experiment ``protocol_info`` of ``pre_time`` and ``sim_time`` from user_inputs.yaml.
* ``parameters``: a parameters CSV to nested JSON (``utilities/parameters_file.py``), the
  ``.json`` beside it, which replaces it:

  - a model's (``<prefix>_parameters.csv``): ``<vessel>/<variable>`` rows nest by instance
    (``heart/lv/E_A`` under ``heart``, ``lv``), globals (``R``, ``global/R``) go under ``global``;
    the older ``<prefix>_supermodule_parameters/<instance>.json`` files beside the module array
    move in under each instance, the CSV's rows over them, and the directory is removed (give the
    ``.json`` when the parameters are JSON already);
  - a parameterisation (``parameterisations/<name>/<name>_parameters.csv``, or its
    ``_calibrated_parameters.csv``), rooted at its module: a component's own variables at the
    top and its global constants (by the config beside) under ``global``; a supermodule's
    ``<sub>/<variable>`` rows nest by instance and its other rows are globals;
  - a ``*_modules_config.json``: every parameterisation beside it, and a supermodule version's
    older ``<module_type>_<module_subtype>_parameters.json`` folded into its default
    parameterisation (each instance's values under its name), which it replaces;
  - a directory: all of these below it.

  Of two rows for one parameter the first is kept (generation used it) and the other reported.
* ``supermodule-instances``: writes the values of each supermodule instance of a model into the
  model's parameters file (``<prefix>_parameters.json`` beside its module array, made if
  needed), under the instance's name: a copy of its version's template, its default
  parameterisation, or the parameterisation the record names in its place, which must set every
  instance value the default sets (and the templates of the
  supermodules nested in it under them). It is how a new supermodule instance gets its values. An instance the file has an
  entry for keeps it; ``--instance`` picks instances and ``--replace`` writes theirs again from the
  version.
* ``params-for-id``: a params_for_id CSV (``vessel_name,param_name,min,max,...``) to the JSON
  beside it, which replaces it (``utilities/params_for_id_file.py``): one entry per row, its
  ``targets`` the row's vessels' ``<vessel>/<param_name>`` (a nested ``heart__lv`` as its path,
  ``heart/lv/E``), its ``prior_*`` columns under ``prior_params``; a directory: every CSV
  below it with vessel_name and param_name columns.
* ``parameter-names``: ``{variable}_{vessel}`` parameter names to ``<vessel>/<variable>``, and
  names joined with a single ``_`` across nested levels to ``__`` (see
  ``scripts/migrate_parameter_names.py``).
'''

import argparse
import json
import os
import re
import shutil
import sys

COMMAND = 'cuflynx-upgrade'

#: older module config key -> key; an older component entry's ``module_type`` is its component
MODULE_CONFIG_RENAMES = (('module_type', 'component_type'), ('module_file', 'component_file'),
                         ('vessel_type', 'module_type'), ('BC_type', 'module_subtype'),
                         ('inp_vessels', 'inp_instances'), ('out_vessels', 'out_instances'),
                         ('default_instance', 'default_parameterisation'))
#: older module array key or column -> key
MODULE_ARRAY_RENAMES = (('vessel_type', 'module_type'), ('BC_type', 'module_subtype'),
                        ('inp_vessels', 'inp_instances'), ('out_vessels', 'out_instances'))
OLD_ARRAY_SUFFIX, ARRAY_SUFFIX = '_vessel_array', '_module_array'


def hint(kind, path, extra=''):
    '''The sentence a reader's error ends with: how to convert ``path``.'''
    return f'Convert it once with: {COMMAND} {kind} {path}{extra}'


def _objects(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


def _rename_keys(text, renames):
    '''``text`` with each JSON key of ``renames`` renamed, formatting and order kept.'''
    for old, new in renames:
        text = re.sub(r'"%s"(\s*):' % re.escape(old), r'"%s"\1:' % new, text)
    return text


def _write(path, text, dry_run, changes):
    print(f'{path}: ' + '; '.join(changes))
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)


# ------------------------------------------------------------------------------ module configs

def upgrade_module_config(path, dry_run=False, module_library_dirs=(), builtin_modules=True):
    '''Convert the module config at ``path``; returns the changes made (empty if none).
    ``module_library_dirs`` (and the built-in modules, unless ``builtin_modules`` is False) are
    where a supermodule's submodule versions, whose parameterisations give its instances' values,
    are looked for.'''
    text = open(path, encoding='utf-8-sig').read()
    doc = json.loads(text)
    changes = []
    old = [o for o in _objects(doc) if 'vessel_type' in o or 'BC_type' in o]
    if old:
        current = [o for o in _objects(doc)
                   if ('module_subtype' in o or 'component_type' in o) and o not in old]
        if current:
            raise SystemExit(f'{path} mixes entries in the older keys with entries in the current '
                             f'ones; convert the older entries by hand.')
        text = _rename_keys(text, MODULE_CONFIG_RENAMES)
        changes.append('older keys renamed')
    elif any('default_instance' in o for o in _objects(doc)):
        text = _rename_keys(text, MODULE_CONFIG_RENAMES[-1:])
        changes.append('default_instance renamed default_parameterisation')
    doc = json.loads(text)
    rewritten = len(changes)
    for entry in doc if isinstance(doc, list) else []:
        for key in ('entrance_ports', 'exit_ports', 'general_ports'):
            for port in entry.get(key) or []:
                if isinstance(port.get('multi_port'), str) and port['multi_port'].lower() == 'false':
                    del port['multi_port']
                    changes.append(f'multi_port "False" dropped ({port.get("port_type")})')
        if entry.get('default_parameters'):
            changes.append(_default_parameters_to_parameterisation(path, entry, dry_run))
        if isinstance(entry.get('submodules'), list):
            changes += _submodules_to_versions(path, entry, dry_run, module_library_dirs,
                                               builtin_modules, doc)
    if len(changes) > rewritten:
        text = json.dumps(doc, indent=2, ensure_ascii=False) + '\n'
    if changes:
        _write(path, text, dry_run, changes)
        if not dry_run:
            upgrade_parameters(path)
    return changes


def _version_entries(module_library_dirs, builtin_modules):
    '''``{(module_type, module_subtype): (entry, config path)}`` of every config entry in the
    module sources, read as written (so older supermodules are found too).'''
    # listed, not read through ModuleSources, which refuses the older forms being converted
    from libcuflynx.utilities.module_library import (_is_any_json, _is_library_config_json,
                                                     _list_flat, _list_recursive)
    from libcuflynx.utilities.package_resources import builtin_modules_dir
    from libcuflynx.utilities.paths import default_module_config_user_dir
    files = []
    for directory in module_library_dirs:
        files += _list_recursive(directory, _is_library_config_json)
    if builtin_modules:
        files += (_list_flat(default_module_config_user_dir(), _is_any_json)
                  + _list_flat(builtin_modules_dir(), _is_any_json))
    found = {}
    for config in files:
        try:
            entries = json.load(open(config, encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        for e in entries if isinstance(entries, list) else []:
            if isinstance(e, dict) and 'module_type' in e and 'module_subtype' in e:
                found.setdefault((e['module_type'], e['module_subtype']), (e, config))
    return found


def _read_rows(path):
    import csv
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


def _parameterisation_file(config_path, name):
    '''The parameters file of parameterisation ``name`` beside ``config_path``: the JSON one,
    else the older CSV one, else the JSON path.'''
    from libcuflynx.utilities.parameterisations import parameterisation_parameters_path
    path = parameterisation_parameters_path(config_path, name)
    csv_path = os.path.splitext(path)[0] + '.csv'
    return csv_path if not os.path.isfile(path) and os.path.isfile(csv_path) else path


def _load_parameterisation(path):
    '''``(columns, rows)`` of a parameterisation file in either format; a JSON one's globals are
    ``global/<variable>``.'''
    if path.endswith('.csv'):
        return _read_rows(path)
    from libcuflynx.utilities.parameters_file import MODULE, read_parameter_rows
    rows = read_parameter_rows(path, kind=MODULE)
    return ['variable_name', 'units', 'value', 'data_reference'], rows


def _save_supermodule_parameterisation(path, columns, rows):
    '''Write a supermodule's parameterisation rows in its file's format (a JSON file's rows
    without a ``/`` are globals).'''
    if path.endswith('.csv'):
        _write_rows(path, columns, rows)
        return
    from libcuflynx.utilities.parameters_file import MODULE, write_parameter_rows
    write_parameter_rows(path, [dict(r, variable_name=r['variable_name'] if '/' in
                                     r['variable_name'] else 'global/' + r['variable_name'])
                                for r in rows], MODULE)


def _write_rows(path, columns, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows([{c: r.get(c, '') for c in columns} for r in rows])


def parameter_from_row(row):
    '''A parameters CSV row as a parameters file's parameter: value (its text kept),
    units, data_reference and any other column that is set.'''
    from libcuflynx.utilities.parameters_file import leaf_from_row
    return leaf_from_row(row)


def _submodules_to_versions(config_path, entry, dry_run, module_library_dirs, builtin_modules,
                            siblings=()):
    '''A supermodule entry's inline ``submodules``: its instances to its own module array file,
    ``submodule_versions`` in the entry, and its parameters file. Each instance's values there
    come from the parameterisation it named (its version's default if none), with the rows
    ``<instance>/<variable>`` of the supermodule's default parameterisation over them; those rows
    leave that parameterisation. A global a named parameterisation sets is added to the
    supermodule's parameterisations that lack it. The versions are looked for in the module
    sources and in ``siblings``, the (converted) entries of the same config.'''
    from libcuflynx.utilities.config_schemas import supermodule_array_path
    from libcuflynx.utilities.parameterisations import (PARAMETERISATIONS_DIR,
                                                        parameterisations_dir)
    records = entry.pop('submodules')
    versions = list(dict.fromkeys((r['module_type'], r['module_subtype']) for r in records))
    entry['submodule_versions'] = [{'module_type': t, 'module_subtype': s} for t, s in versions]
    named = {r['name']: r.pop('parameterisation') for r in records if 'parameterisation' in r}
    array = supermodule_array_path(config_path, entry)
    if os.path.exists(array):
        raise SystemExit(f'{config_path}: {array} exists already; remove it or convert by hand.')
    own = entry.setdefault('default_parameterisation', 'default')
    changes = [f'submodules moved to {os.path.basename(array)}, with submodule_versions',
               f'their parameter values written to its default parameterisation "{own}"']
    sources = _version_entries(module_library_dirs, builtin_modules)
    sources.update({(e['module_type'], e['module_subtype']): (e, config_path) for e in siblings
                    if isinstance(e, dict) and 'module_type' in e and 'module_subtype' in e})
    instances, folded = {}, []
    for record in records:
        sub, key = record['name'], (record['module_type'], record['module_subtype'])
        if key not in sources:
            raise SystemExit(f'{config_path}: submodule "{sub}" is of {key}, which no module '
                             f'source given has; pass --module-library-dir for its library.')
        version, config = sources[key]
        default = version.get('default_parameterisation', version.get('default_instance'))
        used = named.get(sub) or default
        nested = version.get('module_format') == 'supermodule' or 'submodules' in version \
            or 'submodule_versions' in version
        parameters = {}
        # a supermodule instance at its default takes its values from its own parameters file
        if used and not (nested and used == default):
            path = _parameterisation_file(config, used)
            if not os.path.isfile(path) and used == default:
                path = None             # a default with no file gives nothing, as when generating
            elif not os.path.isfile(path):
                raise SystemExit(f'{config_path}: submodule "{sub}" uses parameterisation '
                                 f'"{used}" of {key}, but {path} does not exist.')
            globals_ = {v[0] for v in version.get('variables_and_units') or []
                        if len(v) > 3 and v[3] == 'global_constant'}
            for row in _load_parameterisation(path)[1] if path else []:
                name = row['variable_name'].strip()
                if name.startswith('global/'):
                    is_global = True
                elif nested:
                    is_global = '/' not in name     # a supermodule's other rows are paths
                else:
                    is_global = name in globals_
                if is_global:
                    if used != default:
                        # a global of the named parameterisation, as the supermodule's
                        folded.append(dict(row, variable_name=name.split('/')[-1]))
                    continue
                parameters[name] = parameter_from_row(row)
        instances[sub] = parameters
        if used != default:
            changes.append(f'"{sub}" starts from its parameterisation "{used}"')

    # the supermodule's default parameterisation's <instance>/<variable> rows go over them
    own_path = _parameterisation_file(config_path, own)
    columns, own_rows = (_load_parameterisation(own_path) if os.path.isfile(own_path) else
                         (['variable_name', 'units', 'value', 'data_reference'], []))
    kept, moved = [], 0
    for row in own_rows:
        head, _, rest = row['variable_name'].strip().partition('/')
        if rest and head in instances:
            instances[head][rest] = parameter_from_row(row)
            moved += 1
        else:
            kept.append(row)
    if moved:
        changes.append(f'{moved} <instance>/<variable> rows of parameterisation "{own}" '
                       f'moved under their instances')
    root = parameterisations_dir(config_path)
    targets = sorted(n for n in (os.listdir(root) if os.path.isdir(root) else ())
                     if os.path.isfile(_parameterisation_file(config_path, n)))
    if own not in targets:
        targets.append(own)
    writes = {}
    for name in targets:
        path = _parameterisation_file(config_path, name)
        cols, rows = (columns, kept) if name == own else _load_parameterisation(path)
        have = {r['variable_name'].strip().split('/')[-1] for r in rows
                if r['variable_name'].strip().startswith('global/') or '/' not in r['variable_name']}
        added = []
        for r in folded:          # the first of two submodules' values for one global is kept
            if r['variable_name'] not in have:
                have.add(r['variable_name'])
                added.append(r)
        if added:
            changes.append(f'globals {[r["variable_name"] for r in added]} added to '
                           f'parameterisation "{name}"')
        if added or name == own:
            writes[path] = (cols, rows + added)
    # written last, once everything is found, so a refusal leaves nothing behind
    if not dry_run:
        for path, (cols, rows) in writes.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _save_supermodule_parameterisation(path, cols, rows)
        with open(array, 'w', encoding='utf-8') as f:
            f.write('[\n' + ',\n'.join(' ' + json.dumps(r) for r in records) + '\n]\n')
        upgrade_parameters(config_path)
        from libcuflynx.utilities.parameters_file import (MODULE, read_parameters_file,
                                                          write_parameters_file)
        path = _parameterisation_file(config_path, own)
        doc = read_parameters_file(path, kind=MODULE)
        for sub, values in instances.items():
            node = doc.setdefault(sub, {})
            for var, leaf in values.items():
                _set_path(node, var.split('/'), leaf)
        write_parameters_file(path, doc)
    # a named parameterisation replaces the default as an instance's values, where the older
    # form layered it over them: it takes the default's instance values it does not set
    for name in targets:
        if name == own:
            continue
        completed = _complete_parameterisation(config_path, own, name, dry_run)
        if completed:
            changes.append(f'parameterisation "{name}" completed with the default\'s values of '
                           f'{completed}')
    return changes


def _complete_parameterisation(config_path, default, name, dry_run):
    '''Add to the supermodule parameterisation ``name`` every instance value its default
    parameterisation ``default`` sets and it does not (nothing in a dry run). Returns the paths
    added (``sub/var``).'''
    from libcuflynx.utilities.parameters_file import (MODULE, get_leaf, parameter_rows,
                                                      read_parameters_file, set_leaf,
                                                      write_parameters_file, GLOBAL)
    json_path = os.path.splitext(_parameterisation_file(config_path, name))[0] + '.json'
    default_json = os.path.splitext(_parameterisation_file(config_path, default))[0] + '.json'
    if dry_run or not os.path.isfile(json_path) or not os.path.isfile(default_json):
        return []
    base = read_parameters_file(default_json, kind=MODULE)
    doc = read_parameters_file(json_path, kind=MODULE)
    added = []
    for row in parameter_rows(base, MODULE):
        path = row['variable_name'].split('/')
        if path[0] == GLOBAL or get_leaf(doc, path) is not None:
            continue
        set_leaf(doc, path, get_leaf(base, path))
        added.append(row['variable_name'])
    if added:
        write_parameters_file(json_path, doc)
    return added


def _set_path(node, parts, leaf):
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = leaf


def _default_parameters_to_parameterisation(config_path, entry, dry_run):
    '''Move a supermodule's default_parameters CSV to ``parameterisations/default/``.'''
    from libcuflynx.utilities.parameterisations import PARAMETERISATIONS_DIR
    if entry.get('default_parameterisation'):
        raise SystemExit(f'{config_path}: supermodule ({entry.get("module_type")}) has both '
                         f'default_parameters and a default_parameterisation; move the '
                         f'default_parameters rows into that parameterisation by hand.')
    config_dir = os.path.dirname(os.path.abspath(config_path))
    source = os.path.join(config_dir, entry.pop('default_parameters'))
    target_dir = os.path.join(config_dir, PARAMETERISATIONS_DIR, 'default')
    target = os.path.join(target_dir, 'default_parameters.csv')
    if os.path.exists(target):
        raise SystemExit(f'{config_path}: {target} exists already; move {source} by hand.')
    entry['default_parameterisation'] = 'default'
    if not dry_run:
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(source, target)
    return f'default_parameters moved to {os.path.relpath(target, config_dir)}'


# ------------------------------------------------------------------------------- module arrays

def upgrade_module_array(path, dry_run=False):
    '''Convert the module array at ``path`` (and rename a ``_vessel_array`` file); returns the
    changes made.'''
    import pandas as pd
    changes = []
    if path.lower().endswith('.json'):
        text = open(path, encoding='utf-8-sig').read()
        if any(k in o for o in _objects(json.loads(text)) for k, _ in MODULE_ARRAY_RENAMES):
            text = _rename_keys(text, MODULE_ARRAY_RENAMES)
            changes.append('older keys renamed')
    else:
        df = pd.read_csv(path, dtype=str, na_filter=False)
        df = df.rename(columns=lambda c: str(c).strip()).apply(lambda col: col.str.strip())
        renames = dict(MODULE_ARRAY_RENAMES)
        if any(c in renames for c in df.columns):
            df = df.rename(columns=renames)
            order = ['name', 'module_type', 'module_subtype', 'inp_instances', 'out_instances']
            df = df[order + [c for c in df.columns if c not in order]]
            changes.append('older columns renamed')
        text = df.to_csv(index=False)
    stem, ext = os.path.splitext(path)
    target = path
    if stem.endswith(OLD_ARRAY_SUFFIX):
        target = stem[:-len(OLD_ARRAY_SUFFIX)] + ARRAY_SUFFIX + ext
        if os.path.exists(target):
            raise SystemExit(f'{path}: {target} exists already; remove one of them.')
        changes.append(f'renamed {os.path.basename(target)}')
    if changes:
        _write(target, text, dry_run, changes)
        if target != path and not dry_run:
            os.remove(path)
    return changes


# ------------------------------------------------------------------------------------ obs_data

def upgrade_obs_data(path, pre_time, sim_time, dry_run=False):
    '''Give the obs_data at ``path`` a protocol_info (one experiment of ``pre_time`` then
    ``sim_time``) if it has none; returns the changes made.'''
    text = open(path, encoding='utf-8-sig').read()
    doc = json.loads(text)
    if isinstance(doc, dict) and 'protocol_info' in doc:
        return []
    protocol_info = {'pre_times': [float(pre_time)], 'sim_times': [[float(sim_time)]],
                     'params_to_change': {}}
    if isinstance(doc, list):
        doc = {'protocol_info': protocol_info, 'data_items': doc}
        change = 'bare list wrapped in {protocol_info, data_items}'
    else:
        doc = {'protocol_info': protocol_info, **doc}
        change = 'protocol_info added'
    _write(path, json.dumps(doc, indent=2, ensure_ascii=False) + '\n', dry_run,
           [f'{change} (pre_time {pre_time}, sim_time {sim_time})'])
    return [change]


def _times(args):
    '''pre_time and sim_time for obs-data: --pre-time/--sim-time win over --user-inputs.'''
    pre, sim = args.pre_time, args.sim_time
    if args.user_inputs:
        import yaml
        with open(args.user_inputs) as f:
            inputs = yaml.safe_load(f) or {}
        pre = inputs.get('pre_time') if pre is None else pre
        sim = inputs.get('sim_time') if sim is None else sim
    if pre is None or sim is None:
        raise SystemExit('obs-data needs pre_time and sim_time: give --user-inputs '
                         '<user_inputs.yaml> (its pre_time and sim_time), or --pre-time and '
                         '--sim-time.')
    return pre, sim


# ---------------------------------------------------------------------------------- parameters

_PARAMETERS_CSV_COLUMNS = ('variable_name', 'value')


def _is_parameters_csv(path):
    '''Whether ``path`` is a parameters CSV: a ``.csv`` with a ``_parameters`` name and the
    variable_name and value columns.'''
    name = os.path.basename(path)
    if not name.lower().endswith('.csv') or '_parameters' not in name:
        return False
    try:
        columns, _ = _read_rows(path)
    except (OSError, UnicodeDecodeError):
        return False
    return all(c in [str(x).strip() for x in columns] for c in _PARAMETERS_CSV_COLUMNS)


def _parameterisation_config(path):
    '''``(config path, entries)`` of the module version a parameterisation file at
    ``<version>/parameterisations/<name>/<file>`` belongs to; ``(None, [])`` for a file that is
    not in a parameterisation.'''
    from libcuflynx.utilities.parameterisations import PARAMETERISATIONS_DIR
    directory = os.path.dirname(os.path.abspath(path))
    if os.path.basename(os.path.dirname(directory)) != PARAMETERISATIONS_DIR:
        return None, []
    version_dir = os.path.dirname(os.path.dirname(directory))
    configs = sorted(n for n in os.listdir(version_dir) if n.endswith('_modules_config.json'))
    if not configs:
        raise SystemExit(f'{path}: a parameterisation, but {version_dir} has no '
                         f'*_modules_config.json, which says which of its rows are globals.')
    config = os.path.join(version_dir, configs[0])
    entries = json.load(open(config, encoding='utf-8-sig'))
    return config, [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def _row_name(row):
    return str(row.get('variable_name') or '').strip()


def _csv_document(path):
    '''``(doc, duplicates)``: the parameters CSV at ``path`` as a parameters document, a
    parameterisation's rooted at its module.'''
    from libcuflynx.utilities.parameters_file import MODEL, MODULE, rows_to_document
    columns, rows = _read_rows(path)
    rows = [{str(k).strip(): v for k, v in r.items() if k is not None} for r in rows]
    config, entries = _parameterisation_config(path)
    duplicates = []
    if config is None:
        return rows_to_document(rows, MODEL, f'{path}: ', duplicates), duplicates
    supermodule = any(e.get('module_format') == 'supermodule' or 'submodule_versions' in e
                      or 'submodules' in e for e in entries)
    globals_ = {v[0] for e in entries for v in e.get('variables_and_units') or []
                if isinstance(v, list) and len(v) > 3 and v[3] == 'global_constant'}
    named = []
    for row in rows:
        name = _row_name(row)
        if name and not name.startswith('global/') and \
                (('/' not in name) if supermodule else (name in globals_)):
            name = 'global/' + name
        named.append(dict(row, variable_name=name))
    return rows_to_document(named, MODULE, f'{path}: ', duplicates), duplicates


def _instance_entries(doc_path, where):
    '''``{instance: {path: parameter}}`` of an older supermodule parameters file
    (``{"schema_version", "instances": {name: {"parameterisation", "parameters"}}}``), each
    number's text kept.'''
    with open(doc_path, encoding='utf-8-sig') as f:
        doc = json.loads(f.read(), parse_float=str, parse_int=str)
    if not isinstance(doc, dict) or not isinstance(doc.get('instances'), dict):
        raise SystemExit(f'{where}{doc_path}: not a supermodule parameters file '
                         f'({{"schema_version", "instances"}}).')
    return {name: dict((entry or {}).get('parameters') or {})
            for name, entry in doc['instances'].items()}


def _nest(target, values, over=True):
    '''Put ``{path: parameter}`` into the node ``target``; with ``over`` false a value already
    there is kept.'''
    from libcuflynx.utilities.parameters_file import get_leaf, set_leaf
    for path, leaf in values.items():
        parts = path.split('/')
        if over or get_leaf(target, parts) is None:
            set_leaf(target, parts, leaf)


def _fold_instance_dir(json_path, doc, dry_run, changes, from_csv):
    '''Move the older ``<prefix>_supermodule_parameters/<instance>.json`` files beside the model
    parameters file ``json_path`` into ``doc``, under each instance: values ``doc`` already has
    (the model's own rows) are kept.'''
    stem = os.path.basename(json_path)[:-len('_parameters.json')]
    directory = os.path.join(os.path.dirname(os.path.abspath(json_path)),
                             f'{stem}_supermodule_parameters')
    if not os.path.isdir(directory):
        return None
    for name in sorted(os.listdir(directory)):
        instance, ext = os.path.splitext(name)
        if ext != '.json':
            continue
        values = _instance_entries(os.path.join(directory, name), '')
        node = doc.setdefault(instance, {})
        for sub, parameters in values.items():
            _nest(node.setdefault(sub, {}), parameters, over=False)
        changes.append(f'{os.path.basename(directory)}/{name} moved under "{instance}"')
    return directory


def _convert_parameters_csv(path, dry_run):
    '''A parameters CSV to the ``.json`` beside it (and, for a model's, its older instance
    files folded in); returns the changes.'''
    from libcuflynx.utilities.parameters_file import write_parameters_file
    target = os.path.splitext(path)[0] + '.json'
    if os.path.exists(target):
        raise SystemExit(f'{path}: {target} exists already; remove one of them.')
    doc, duplicates = _csv_document(path)
    changes = [f'written as {os.path.basename(target)}']
    if duplicates:
        changes.append(f'rows set twice, the first kept: {sorted(set(duplicates))}')
    directory = (_fold_instance_dir(target, doc, dry_run, changes, True)
                 if os.path.basename(target).endswith('_parameters.json') else None)
    print(f'{path}: ' + '; '.join(changes))
    if not dry_run:
        write_parameters_file(target, doc)
        os.remove(path)
        if directory:
            shutil.rmtree(directory)
    return changes


def _fold_model_json(path, dry_run):
    '''A model's parameters JSON with the older instance directory beside it: the directory
    folded in.'''
    from libcuflynx.utilities.parameters_file import read_parameters_file, write_parameters_file
    doc = read_parameters_file(path)
    changes = []
    directory = _fold_instance_dir(path, doc, dry_run, changes, False)
    if directory:
        print(f'{path}: ' + '; '.join(changes))
        if not dry_run:
            write_parameters_file(path, doc)
            shutil.rmtree(directory)
    return changes


def _convert_config(path, dry_run):
    '''Every parameterisation CSV beside the module config ``path``, then a supermodule version's
    older parameters file folded into its default parameterisation.'''
    from libcuflynx.utilities.parameterisations import parameterisations_dir
    from libcuflynx.utilities.parameters_file import (MODULE, new_document, read_parameters_file,
                                                      write_parameters_file)
    changes = []
    root = parameterisations_dir(path)
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        for file in sorted(os.listdir(os.path.join(root, name))):
            candidate = os.path.join(root, name, file)
            if _is_parameters_csv(candidate):
                changes += _convert_parameters_csv(candidate, dry_run)
    entries = json.load(open(path, encoding='utf-8-sig'))
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or 'submodule_versions' not in entry:
            continue
        from libcuflynx.utilities.config_schemas import supermodule_parameters_path
        old = supermodule_parameters_path(path, entry)
        if not os.path.isfile(old):
            continue
        default = entry.get('default_parameterisation')
        if not default:
            default = entry['default_parameterisation'] = 'default'
            if not dry_run:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(entries, indent=2, ensure_ascii=False) + '\n')
        target = os.path.join(root, default, f'{default}_parameters.json')
        csv_target = os.path.splitext(target)[0] + '.csv'
        if dry_run and os.path.isfile(csv_target):
            doc = _csv_document(csv_target)[0]
        else:
            doc = (read_parameters_file(target, kind=MODULE) if os.path.isfile(target)
                   else new_document())
        for instance, values in _instance_entries(old, f'{path}: ').items():
            _nest(doc.setdefault(instance, {}), values)
        line = (f'{os.path.basename(old)} folded into parameterisation "{default}" '
                f'({os.path.relpath(target, os.path.dirname(path))})')
        print(f'{path}: {line}')
        changes.append(line)
        if not dry_run:
            write_parameters_file(target, doc)
            os.remove(old)
    return changes


def upgrade_parameters(path, dry_run=False):
    '''Convert the parameters of ``path`` -- a parameters CSV, a model's parameters JSON with the
    older instance directory beside it, a ``*_modules_config.json``, or a directory of them --
    to nested JSON (see the module docstring). Returns the changes made.'''
    changes = []
    if os.path.isdir(path):
        configs, others = [], []
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            for name in sorted(files):
                full = os.path.join(root, name)
                if name.endswith('_modules_config.json'):
                    configs.append(full)
                else:
                    others.append(full)
        for config in configs:
            changes += _convert_config(config, dry_run)
        for other in others:
            if not os.path.isfile(other):
                continue                    # converted with its config
            if _is_parameters_csv(other):
                changes += _convert_parameters_csv(other, dry_run)
            elif other.endswith('_parameters.json') and os.path.isdir(
                    other[:-len('_parameters.json')] + '_supermodule_parameters'):
                changes += _fold_model_json(other, dry_run)
        return changes
    if path.endswith('_modules_config.json'):
        return _convert_config(path, dry_run)
    if path.lower().endswith('.json'):
        return _fold_model_json(path, dry_run)
    if not _is_parameters_csv(path):
        raise SystemExit(f'{path}: not a parameters CSV (a *_parameters*.csv with variable_name '
                         f'and value columns), module config or directory.')
    return _convert_parameters_csv(path, dry_run)


# ------------------------------------------------------------------------------- params_for_id

def upgrade_params_for_id(path, dry_run=False):
    '''Convert the params_for_id CSV ``path``, or every one below the directory ``path``, to
    JSON (``utilities/params_for_id_file.py``). Returns the changes made.'''
    from libcuflynx.utilities import params_for_id_file as pfi
    if os.path.isdir(path):
        paths = []
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            paths += [os.path.join(root, n) for n in sorted(files)
                      if pfi.is_params_for_id_csv(os.path.join(root, n))]
    elif pfi.is_params_for_id_csv(path):
        paths = [path]
    else:
        raise SystemExit(f'{path}: not a params_for_id CSV (a .csv with vessel_name and '
                         f'param_name columns) or a directory.')
    changes = []
    for csv_path in paths:
        try:
            out = pfi.convert_csv(csv_path, dry_run)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f'{csv_path}: written as {os.path.basename(out)}')
        changes.append(out)
    return changes


# ------------------------------------------------------------------------------------------ CLI

# ------------------------------------------------------------------ supermodule instances

def upgrade_supermodule_instances(path, dry_run=False, module_library_dirs=(), builtin_modules=True,
                                  external_modules_dir=None, instances=None, replace=False,
                                  parameters=None):
    '''Write the values of the supermodule instances of the model whose module array is ``path``
    into its parameters file (``supermodules.write_instance_parameters``), its supermodules
    looked for in the module sources given. Returns the changes made (empty if none).'''
    from libcuflynx.utilities.module_library import ModuleSources
    from libcuflynx.utilities.supermodules import write_instance_parameters
    sources = ModuleSources({'module_library_dirs': list(module_library_dirs),
                             'external_modules_dir': external_modules_dir,
                             'use_builtin_modules': builtin_modules})
    try:
        done = write_instance_parameters(path, sources.supermodule_registry(), instances, replace,
                                         dry_run, parameters)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    changes = [f'"{name}" {action} in '
               f'{os.path.relpath(written, os.path.dirname(os.path.abspath(path)))} from its version'
               for name, written, action in done if action != 'kept']
    for name, _, action in done:
        if action == 'kept':
            print(f'{path}: "{name}" has values in the model\'s parameters file; kept '
                  f'(--replace writes them again)')
    if changes:
        print(f'{path}: ' + '; '.join(changes))
    return changes


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['parameter-names']:
        from libcuflynx.scripts.migrate_parameter_names import main as migrate
        return migrate(argv[1:])
    parser = argparse.ArgumentParser(
        prog=COMMAND, description=__doc__.split('\n\n')[1],
        epilog='Not a pipeline stage: it takes file paths, not user_inputs.yaml (obs-data reads '
               "the user_inputs.yaml given with --user-inputs for its pre_time and sim_time).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (('module-config', 'convert module config files'),
                            ('module-array', 'convert module (or vessel) array files'),
                            ('obs-data', 'convert obs_data files'),
                            ('parameters', 'convert parameters CSVs (a model\'s, a '
                                           'parameterisation\'s, a module config\'s or a '
                                           'directory\'s) to nested JSON'),
                            ('supermodule-instances',
                             "write each supermodule instance's values into a model's "
                             "parameters file, from its version"),
                            ('params-for-id', 'convert params_for_id CSVs (or a directory\'s) '
                                              'to JSON')):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument('paths', nargs='+')
        sub.add_argument('--dry-run', action='store_true', help='print, do not write')
        if name in ('module-config', 'supermodule-instances'):
            sub.add_argument('--module-library-dir', dest='module_library_dirs', action='append',
                             default=[], metavar='DIR',
                             help="a module library (repeatable): where a supermodule's "
                                  "submodule versions are, whose parameterisations give its "
                                  "instances' values" if name == 'module-config' else
                                  "a module library (repeatable), as the model's "
                                  "module_library_dirs")
            sub.add_argument('--no-builtin-modules', action='store_true',
                             help="leave libcuflynx's built-in modules out")
        if name == 'supermodule-instances':
            sub.add_argument('--external-modules-dir', metavar='DIR',
                             help="the model's external_modules_dir")
            sub.add_argument('--instance', dest='instances', action='append', metavar='NAME',
                             help='only this supermodule instance (repeatable)')
            sub.add_argument('--replace', action='store_true',
                             help="write the values again from the version, over the model's")
            sub.add_argument('--parameters', metavar='FILE',
                             help="the model's parameters file (default: "
                                  "<prefix>_parameters.json beside the module array)")
        if name == 'obs-data':
            sub.add_argument('--user-inputs', help="the user_inputs.yaml whose pre_time and "
                                                   "sim_time the files were run with")
            sub.add_argument('--pre-time', type=float)
            sub.add_argument('--sim-time', type=float)
    commands.add_parser('parameter-names', help='convert parameter names (see '
                                                f'{COMMAND} parameter-names --help)')
    args = parser.parse_args(argv)
    changed = 0
    for path in args.paths:
        if args.command == 'module-config':
            changed += bool(upgrade_module_config(path, args.dry_run, args.module_library_dirs,
                                                  not args.no_builtin_modules))
        elif args.command == 'module-array':
            changed += bool(upgrade_module_array(path, args.dry_run))
        elif args.command == 'supermodule-instances':
            changed += bool(upgrade_supermodule_instances(
                path, args.dry_run, args.module_library_dirs, not args.no_builtin_modules,
                args.external_modules_dir, args.instances, args.replace, args.parameters))
        elif args.command == 'parameters':
            changed += bool(upgrade_parameters(path, args.dry_run))
        elif args.command == 'params-for-id':
            changed += bool(upgrade_params_for_id(path, args.dry_run))
        else:
            changed += bool(upgrade_obs_data(path, *_times(args), dry_run=args.dry_run))
    print(f'{changed} of {len(args.paths)} file(s) {"would be " if args.dry_run else ""}changed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
