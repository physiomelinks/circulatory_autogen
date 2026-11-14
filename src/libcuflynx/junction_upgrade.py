'''
``cuflynx-upgrade junctions``: move a model off the junction module types.

The module library replaced its junction module types (``Min_junction``, ``Nout_junction``,
``split_junction``, ``merge_junction``, ``2in2out_junction``, their ``_simple``/``venous_``
variants, the microvasculature's ``<vessel>_Min``/``_Nout``/... and the massless ``flow_merge``/
``flow_split``) by ordinary vessels whose compliant end has a summing port
(``"multi_port": ["sum", "True"]``): such an end sums the flows of every vessel on the other side
of its node, however many there are (see ``generators/vessel_nodes.py``). The library's
``renames.json`` says what replaces each junction version, in entries marked
``"by": "cuflynx-upgrade junctions"``:

* ``"to": [type, version]`` with ``"variables"`` (old -> new name on the same instance, e.g.
  ``v_in_sum`` -> ``v_in``) and ``"neighbour_variables"`` (``{"exit": ["v_out_1", "v_out_2"]}``:
  the k-th name is the flow into the k-th downstream neighbour, so ``<j>/v_out_k`` becomes that
  neighbour's own inflow variable): the instance is retyped;
* ``"split": [[old, new], ...]`` (``inlet_flow constant_2``): the instance becomes one instance of
  ``to`` per pair, ``<name>_1``, ``<name>_2``, ..., each with ``old`` renamed ``new``, all feeding
  what it fed;
* ``"to": null`` with ``"join"`` (``flow_merge``, ``flow_split``): the massless node module is
  removed and its neighbours joined directly; ``join.variables`` maps its names onto the variables
  of the port of its one neighbour on ``join.side`` (index 0 the flow, 1 the pressure);
* ``"to": null`` without ``join``: refused, with the entry's ``note``.

For each module array given, it rewrites in place: the array (JSON or CSV), the model's
``<prefix>_parameters.json`` (renamed and split values; values the new version does not declare
are dropped and listed), and ``<prefix>_obs_data*.json`` and ``<prefix>_params_for_id*.json``
beside it (every ``<instance>/<variable>`` name that changed). Run it after
``cuflynx-upgrade module-names``, so the array names the library's versions. The model's
equations do not change.
'''
import glob
import json
import os
import re

from libcuflynx.utilities.config_schemas import module_array_csv_to_records

BY = 'cuflynx-upgrade junctions'
RENAMES_FILE = 'renames.json'


# ------------------------------------------------------------------------------------- library

def library_entries(module_library_dirs):
    '''{(module_type, version): entry} of every junction entry in the libraries' renames.json.'''
    entries = {}
    for directory in module_library_dirs:
        path = os.path.join(directory, RENAMES_FILE)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        for entry in doc.get('renames', []):
            if entry.get('by') == BY:
                entries.setdefault(tuple(entry['from']), entry)
    if not entries:
        raise SystemExit(f'No junction entries ("by": "{BY}") in {RENAMES_FILE} of '
                         f'{", ".join(module_library_dirs) or "any module library"}. Give the '
                         'module library with --module-library-dir (its modules/ directory).')
    return entries


def config_index(directories):
    '''{(module_type, module_subtype): config entry} of every *_modules_config.json below.'''
    index = {}
    for directory in directories:
        for path in glob.glob(os.path.join(directory, '**', '*modules_config.json'), recursive=True):
            try:
                with open(path, encoding='utf-8') as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                continue
            for entry in doc if isinstance(doc, list) else [doc]:
                if isinstance(entry, dict) and 'module_type' in entry and 'module_subtype' in entry:
                    index.setdefault((entry['module_type'], entry['module_subtype']), entry)
    return index


def _vessel_port(config, side):
    '''The first vessel_port of ``config`` on ``side`` ("entrance"/"exit"), or None.'''
    if not config:
        return None
    for port in config.get(f'{side}_ports') or []:
        if isinstance(port, dict) and port.get('port_type') == 'vessel_port':
            return port
    return None


def _declared(config):
    return {v[0] for v in config.get('variables_and_units', [])} if config else None


# ------------------------------------------------------------------------------------- arrays

def _read_records(path):
    if path.lower().endswith('.json'):
        with open(path, encoding='utf-8-sig') as f:
            return json.load(f)
    return module_array_csv_to_records(path)


def _write_records(path, records, dry_run):
    if dry_run:
        return
    if path.lower().endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('[\n' + ',\n'.join(' ' + json.dumps(r) for r in records) + '\n]\n')
        return
    import pandas as pd
    columns = list(pd.read_csv(path, dtype=str, nrows=0).columns.str.strip())
    rows = []
    for record in records:
        rows.append({c: ' '.join(record.get(c, [])) if isinstance(record.get(c), list)
                     else record.get(c, '') for c in columns})
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _replace_in(lst, old, new):
    '''``lst`` with ``old`` replaced by the names in ``new``, in its place.'''
    out = []
    for item in lst:
        out.extend(new if item == old else [item])
    return out


def convert_records(records, entries, configs):
    '''Convert ``records``; returns (records, names, dropped_instances, notes):
    ``names`` maps old "<instance>/<variable>" names to new ones (None: no single counterpart),
    ``split`` maps a split instance to its new instances.'''
    by_name = {r['name']: r for r in records}
    names, notes, removed, split_into = {}, [], set(), {}
    retyped = {}

    def neighbours(record, side):
        key = 'out_instances' if side == 'exit' else 'inp_instances'
        return [n for n in record.get(key, []) if n in by_name]

    for record in records:
        key = (record.get('module_type'), record.get('module_subtype'))
        entry = entries.get(key)
        if entry is None:
            continue
        name = record['name']
        if entry.get('to') is None and not entry.get('join'):
            raise SystemExit(f'"{name}" is a {key[0]} {key[1]}, which the module library removed '
                             f'with no replacement: {entry.get("note", "")}')
        if entry.get('join'):
            removed.add(name)
            continue
        retyped[name] = (key, tuple(entry['to']))
        if entry.get('split'):
            split_into[name] = [f'{name}_{k}' for k in range(1, len(entry['split']) + 1)]

    # names on the converted instances
    for record in records:
        name = record['name']
        key = (record.get('module_type'), record.get('module_subtype'))
        entry = entries.get(key)
        if entry is None:
            continue
        if name in removed:
            join = entry['join']
            side = join['side']
            others = neighbours(record, side)
            if len(others) != 1:
                raise SystemExit(f'"{name}" ({key[0]} {key[1]}) has {len(others)} neighbours on its '
                                 f'{side} side; the upgrade joins its neighbours only when it has one.')
            other = by_name[others[0]]
            other_cfg = configs.get(retyped.get(other['name'], (None, (other.get('module_type'),
                                                                        other.get('module_subtype'))))[1])
            port = _vessel_port(other_cfg, 'entrance' if side == 'exit' else 'exit')
            for old, index in join['variables'].items():
                names[f'{name}/{old}'] = f'{other["name"]}/{port["variables"][index]}' if port else None
            continue
        for old, new in (entry.get('variables') or {}).items():
            names[f'{name}/{old}'] = f'{name}/{new}'
        for side, numbered in (entry.get('neighbour_variables') or {}).items():
            others = [n for n in neighbours(record, side)
                      if _vessel_port(configs.get(retyped.get(n, (None, (by_name[n].get('module_type'),
                                                                         by_name[n].get('module_subtype'))))[1]),
                                      'entrance' if side == 'exit' else 'exit')]
            for k, old in enumerate(numbered):
                if k < len(others):
                    other = by_name[others[k]]
                    cfg = configs.get(retyped.get(other['name'], (None, (other.get('module_type'),
                                                                         other.get('module_subtype'))))[1])
                    port = _vessel_port(cfg, 'entrance' if side == 'exit' else 'exit')
                    names[f'{name}/{old}'] = f'{other["name"]}/{port["variables"][0]}'
                else:
                    names[f'{name}/{old}'] = None
        for k, (old, new) in enumerate(entry.get('split') or [], 1):
            names[f'{name}/{old}'] = f'{name}_{k}/{new}'

    # the array
    out = []
    for record in records:
        name = record['name']
        if name in removed:
            continue
        record = dict(record)
        for list_key, side in (('inp_instances', 'entrance'), ('out_instances', 'exit')):
            lst = list(record.get(list_key, []))
            for gone in [n for n in lst if n in removed]:
                gone_record = by_name[gone]
                # what the removed node joined on the far side
                far = gone_record['inp_instances'] if list_key == 'inp_instances' else gone_record['out_instances']
                lst = _replace_in(lst, gone, [n for n in far if n != name])
            for old, new in split_into.items():
                lst = _replace_in(lst, old, new)
            if list_key in record or lst:
                record[list_key] = lst
        if name in retyped:
            (old_type, old_version), (new_type, new_version) = retyped[name]
            record['module_type'], record['module_subtype'] = new_type, new_version
            if name in split_into:
                for new_name in split_into[name]:
                    out.append(dict(record, name=new_name))
                notes.append(f'{name}: {old_type} {old_version} -> {len(split_into[name])} x '
                             f'{new_type} {new_version} ({", ".join(split_into[name])})')
                continue
            notes.append(f'{name}: {old_type} {old_version} -> {new_type} {new_version}')
        out.append(record)
    for name in sorted(removed):
        record = by_name[name]
        notes.append(f'{name}: {record["module_type"]} {record["module_subtype"]} removed, its '
                     f'neighbours joined')
    return out, names, removed, split_into, retyped, notes


# ------------------------------------------------------------------------------- other files

def _rename_strings(node, names, changed):
    if isinstance(node, dict):
        return {(_rename(k, names, changed) if isinstance(k, str) else k): _rename_strings(v, names, changed)
                for k, v in node.items()}
    if isinstance(node, list):
        return [_rename_strings(v, names, changed) for v in node]
    if isinstance(node, str):
        return _rename(node, names, changed)
    return node


def _rename(text, names, changed):
    def sub(match):
        old = match.group(0)
        new = names.get(old, old)
        if new is None:
            changed.setdefault('unmapped', set()).add(old)
            return old
        if new != old:
            changed.setdefault('renamed', set()).add(f'{old} -> {new}')
        return new
    if not names:
        return text
    pattern = '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.sub(rf'(?<![\w/])(?:{pattern})(?![\w/])', sub, text)


def convert_parameters(doc, names, removed, split_into, retyped, configs):
    '''The model's nested parameters ``doc`` converted; returns (doc, notes).'''
    notes = []
    for name in removed:
        if name in doc:
            notes.append(f'{name}: values dropped with the removed module ({", ".join(doc[name])})')
            doc.pop(name)
    for name, new_names in split_into.items():
        values = doc.pop(name, {})
        for k, new_name in enumerate(new_names, 1):
            part = {}
            for var, leaf in values.items():
                target = names.get(f'{name}/{var}')
                if target is None:
                    part[var] = leaf
                elif target.startswith(f'{new_name}/'):
                    part[target.split('/', 1)[1]] = leaf
            doc[new_name] = part
        notes.append(f'{name}: values split into {", ".join(new_names)}')
    for name, (_, new_key) in retyped.items():
        if name in split_into or name not in doc or not isinstance(doc[name], dict):
            continue
        renamed = {}
        for var, leaf in doc[name].items():
            target = names.get(f'{name}/{var}', f'{name}/{var}')
            if target and target.startswith(f'{name}/'):
                renamed[target.split('/', 1)[1]] = leaf
        declared = _declared(configs.get(new_key))
        if declared is not None:
            dropped = [v for v in renamed if v not in declared]
            for var in dropped:
                renamed.pop(var)
            if dropped:
                notes.append(f'{name}: {", ".join(dropped)} dropped ({new_key[0]} {new_key[1]} has no '
                             f'such variable)')
        doc[name] = renamed
    return doc, notes


def _model_files(array_path, pattern):
    directory = os.path.dirname(os.path.abspath(array_path))
    prefix = re.sub(r'_(module|vessel)_array\.(json|csv)$', '', os.path.basename(array_path))
    return sorted(glob.glob(os.path.join(directory, f'{prefix}{pattern}'))), prefix


def upgrade_junctions(path, module_library_dirs, external_modules_dir=None, dry_run=False,
                      entries=None, configs=None):
    '''Convert the module array at ``path`` and the model files beside it; returns the changes.'''
    from libcuflynx.utilities.parameters_file import write_parameters_file
    entries = entries if entries is not None else library_entries(module_library_dirs)
    if configs is None:
        configs = config_index(list(module_library_dirs) + ([external_modules_dir] if external_modules_dir else []))
    records = _read_records(path)
    if not any((r.get('module_type'), r.get('module_subtype')) in entries for r in records):
        return []
    records, names, removed, split_into, retyped, notes = convert_records(records, entries, configs)
    changes = list(notes)
    _write_records(path, records, dry_run)
    print(f'{path}:\n  ' + '\n  '.join(notes))

    params, prefix = _model_files(path, '_parameters.json')
    for p in params:
        with open(p, encoding='utf-8') as f:
            doc = json.load(f)
        doc, param_notes = convert_parameters(doc, names, removed, split_into, retyped, configs)
        if param_notes:
            print(f'{p}:\n  ' + '\n  '.join(param_notes))
            changes += param_notes
            if not dry_run:
                write_parameters_file(p, doc)

    others, _ = _model_files(path, '_obs_data*.json')
    others += _model_files(path, '_params_for_id*.json')[0]
    for p in others:
        with open(p, encoding='utf-8') as f:
            text = f.read()
        changed = {}
        new_doc = _rename_strings(json.loads(text), names, changed)
        if changed.get('renamed'):
            lines = sorted(changed['renamed'])
            print(f'{p}:\n  ' + '\n  '.join(lines))
            changes += lines
            if not dry_run:
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(new_doc, f, indent=1)
                    f.write('\n')
        if changed.get('unmapped'):
            lines = [f'{n} has no single counterpart (the flows of a junction port are now one '
                     f'sum); edit it by hand' for n in sorted(changed['unmapped'])]
            print(f'{p}:\n  ' + '\n  '.join(lines))
            changes += lines
    return changes
