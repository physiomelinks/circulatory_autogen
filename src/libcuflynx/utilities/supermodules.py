'''
Expands supermodule instances in a vessel array into their prefixed submodules.

A supermodule is a module config entry with ``"module_format": "supermodule"`` and a
``submodules`` list (see ``utilities/config_schemas.py``). An instance of it in a vessel
array, e.g.::

    {"name": "heart", "module_type": "heart", "module_subtype": "supermodule",
     "per_submodule_inputs": {"ra": ["venous_svc"], "la": ["pvn"]},
     "per_submodule_outputs": {"aov": ["aortic_root"], "puv": ["par"]}}

is replaced, at its position in the array, by one record per submodule, in the
supermodule's order:

* each submodule ``sub`` becomes ``<instance>_<sub>``, and its (sibling) inp/out names are
  prefixed the same way;
* the hosts in ``per_submodule_inputs[sub]`` are prepended to that submodule's inputs, and
  the hosts in ``per_submodule_outputs[sub]`` appended to its outputs, in order;
* in each host, the instance name in its out list is replaced, at the same position, by
  every ``<instance>_<sub>`` whose ``per_submodule_inputs`` lists that host; the instance
  name in its inp list by every ``<instance>_<sub>`` whose ``per_submodule_outputs`` lists
  it (in the order of the ``per_submodule_*`` object).

A submodule may itself be a supermodule instance (with ``per_submodule_*`` naming its
siblings); it is expanded in turn, and a supermodule that contains itself, directly or not,
is an error. Links between a host and a nested supermodule go through that nested
supermodule's own ``per_submodule_*``, so an instance's ``per_submodule_*`` may not name a
submodule that is itself a supermodule instance.

The supermodule's ``default_parameters`` rows are renamed from ``{var}_{sub}`` to
``{var}_{instance}_{sub}`` (the suffix is matched against the submodule names, longest
first); any other row is a global and keeps its name, and is added only once.

After expansion no record is a supermodule instance.
'''

import copy
import os

import pandas as pd

from libcuflynx.utilities.config_schemas import (PARAMETER_COLUMNS, PER_SUBMODULE_KEYS,
                                                 SUPERMODULE_FORMAT)

# a supermodule nested deeper than this is taken to be a cycle the ancestry check missed
_MAX_DEPTH = 64


def _key(record):
    return (record['vessel_type'], record['BC_type'])


def _is_instance(record, registry, source):
    '''Whether ``record`` is a supermodule instance; errors for an unknown supermodule.'''
    if _key(record) in registry:
        return True
    looks_like_one = (record['BC_type'] == SUPERMODULE_FORMAT
                      or any(k in record for k in PER_SUBMODULE_KEYS))
    if looks_like_one:
        known = sorted(registry) or 'none'
        raise ValueError(
            f'{source}: "{record["name"]}" is a supermodule instance of type '
            f'({record["vessel_type"]}, {record["BC_type"]}), but no supermodule entry of that '
            f'type was found in the module configs (known supermodules: {known}). Add it to a '
            f'*_modules_config.json in module_library_dirs or external_modules_dir.')
    return False


def _splice(names, target, replacement):
    '''``names`` with every ``target`` replaced, in place, by the names in ``replacement``.'''
    out = []
    for name in names:
        if name == target:
            out += replacement
        else:
            out.append(name)
    return out


def rename_default_parameter(variable_name, instance, submodule_names):
    '''``{var}_{sub}`` -> ``{var}_{instance}_{sub}``; a global name is returned unchanged.'''
    for sub in sorted(submodule_names, key=len, reverse=True):
        suffix = '_' + sub
        if variable_name.endswith(suffix) and len(variable_name) > len(suffix):
            return f'{variable_name[:-len(suffix)]}_{instance}_{sub}'
    return variable_name


def read_default_parameters(supermodule, instance, where):
    '''The supermodule's default_parameters rows, renamed for ``instance``.'''
    file_name = supermodule.get('default_parameters')
    if not file_name:
        return []
    config_dir = os.path.dirname(supermodule.get('config_path', '') or '')
    path = file_name if os.path.isabs(file_name) else os.path.join(config_dir, file_name)
    if not os.path.isfile(path):
        raise ValueError(f'{where}: default_parameters file {path} of supermodule '
                         f'({supermodule["vessel_type"]}, {supermodule["BC_type"]}) not found '
                         f'(it is resolved against {supermodule.get("config_path")}).')
    df = pd.read_csv(path, dtype=str, na_filter=False)
    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in PARAMETER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{where}: default_parameters file {path} is missing the columns '
                         f'{missing}; it needs {list(PARAMETER_COLUMNS)}.')
    submodule_names = [s['name'] for s in supermodule['submodules']]
    rows = []
    for row in df.itertuples(index=False):
        values = {c: str(getattr(row, c)).strip() for c in PARAMETER_COLUMNS}
        if not values['variable_name']:
            continue
        values['variable_name'] = rename_default_parameter(values['variable_name'], instance,
                                                           submodule_names)
        rows.append(values)
    return rows


def _expand_one(records, index, registry, source, ancestry):
    instance = records[index]
    name = instance['name']
    key = _key(instance)
    supermodule = registry[key]
    where = f'{source}: supermodule instance "{name}" {key}'

    chain = ancestry.get(name, ())
    if key in chain or len(chain) >= _MAX_DEPTH:
        path = ' -> '.join(f'{k[0]}/{k[1]}' for k in chain + (key,))
        raise ValueError(f'{where}: supermodules nest in a cycle ({path}).')

    submodules = supermodule['submodules']
    sub_names = [s['name'] for s in submodules]
    nested = {s['name'] for s in submodules if _key(s) in registry}
    per_inputs = instance.get('per_submodule_inputs', {})
    per_outputs = instance.get('per_submodule_outputs', {})
    for per_key, per in (('per_submodule_inputs', per_inputs),
                         ('per_submodule_outputs', per_outputs)):
        unknown = [s for s in per if s not in sub_names]
        if unknown:
            raise ValueError(f'{where}: {per_key} names {unknown}, which are not submodules '
                             f'of {key}; its submodules are {sub_names}.')
        inner = [s for s in per if s in nested]
        if inner:
            raise ValueError(f'{where}: {per_key} names {inner}, which are themselves '
                             f'supermodule instances; connect the host to one of their '
                             f'submodules through that supermodule\'s own per_submodule_*.')

    by_name = {r['name']: r for i, r in enumerate(records) if i != index}
    for per_key, per, host_key, host_list in (
            ('per_submodule_inputs', per_inputs, 'out', 'out_vessels'),
            ('per_submodule_outputs', per_outputs, 'inp', 'inp_vessels')):
        for sub, hosts in per.items():
            for host in hosts:
                if host not in by_name:
                    raise ValueError(f'{where}: {per_key}["{sub}"] names host "{host}", which '
                                     f'is not in the vessel array.')
                if name not in by_name[host][host_list]:
                    raise ValueError(
                        f'{where}: {per_key}["{sub}"] names host "{host}", but "{host}" does '
                        f'not list "{name}" in its {host_key} list '
                        f'({host_list}/{host_key}_instances).')

    input_hosts = {h for hosts in per_inputs.values() for h in hosts}
    output_hosts = {h for hosts in per_outputs.values() for h in hosts}
    for other in by_name.values():
        if name in other['out_vessels'] and other['name'] not in input_hosts:
            raise ValueError(
                f'{where}: "{other["name"]}" lists "{name}" in its out list, but no '
                f'per_submodule_inputs entry of "{name}" names "{other["name"]}", so it is not '
                f'known which submodule it feeds.')
        if name in other['inp_vessels'] and other['name'] not in output_hosts:
            raise ValueError(
                f'{where}: "{other["name"]}" lists "{name}" in its inp list, but no '
                f'per_submodule_outputs entry of "{name}" names "{other["name"]}", so it is not '
                f'known which submodule feeds it.')
    for list_key, hosts, per_key in (('inp_vessels', input_hosts, 'per_submodule_inputs'),
                                     ('out_vessels', output_hosts, 'per_submodule_outputs')):
        stray = [h for h in instance[list_key] if h not in hosts]
        if stray:
            raise ValueError(f'{where}: its {list_key.split("_")[0]} list names {stray}, which '
                             f'no {per_key} entry names.')

    def prefixed(sub):
        return f'{name}_{sub}'

    new_records = []
    for sub in submodules:
        record = copy.deepcopy(sub)
        record['name'] = prefixed(sub['name'])
        record['inp_vessels'] = (list(per_inputs.get(sub['name'], []))
                                 + [prefixed(n) for n in sub['inp_vessels']])
        record['out_vessels'] = ([prefixed(n) for n in sub['out_vessels']]
                                 + list(per_outputs.get(sub['name'], [])))
        for per_key in PER_SUBMODULE_KEYS:
            if per_key in record:
                record[per_key] = {s: [prefixed(h) for h in hosts]
                                   for s, hosts in record[per_key].items()}
        new_records.append(record)

    clashes = [r['name'] for r in new_records if r['name'] in by_name]
    if clashes:
        raise ValueError(f'{where}: expanding it gives the names {clashes}, which are already '
                         f'in the vessel array. Rename the instance or the clashing records.')

    for other in by_name.values():
        if name in other['out_vessels']:
            other['out_vessels'] = _splice(
                other['out_vessels'], name,
                [prefixed(s) for s, hosts in per_inputs.items() if other['name'] in hosts])
        if name in other['inp_vessels']:
            other['inp_vessels'] = _splice(
                other['inp_vessels'], name,
                [prefixed(s) for s, hosts in per_outputs.items() if other['name'] in hosts])

    for record in new_records:
        ancestry[record['name']] = chain + (key,)
    param_rows = read_default_parameters(supermodule, name, where)
    return records[:index] + new_records + records[index + 1:], param_rows


def expand_supermodules(records, registry, source=None):
    '''
    ``(records, extra_param_rows)``: ``records`` (normalised vessel records, see
    ``config_schemas.normalise_vessel_record``) with every supermodule instance -- a record
    whose (vessel_type, BC_type) is in ``registry`` -- replaced by its prefixed submodules,
    recursively, and the instances' renamed default parameters (each name once, the first
    occurrence kept). ``records`` is not modified.

    Raises ValueError, naming ``source`` and the instance, for an unknown supermodule type,
    an unknown submodule in per_submodule_*, a host that does not exist or does not name the
    instance back, a record that names the instance without a per_submodule_* entry linking
    them, an expanded name that clashes with an existing record, or nesting in a cycle.
    '''
    source = source or 'vessel array'
    records = copy.deepcopy(list(records))
    ancestry = {}
    extra_param_rows = []
    seen = set()
    while True:
        index = next((i for i, r in enumerate(records) if _is_instance(r, registry, source)), None)
        if index is None:
            break
        records, rows = _expand_one(records, index, registry, source, ancestry)
        for row in rows:
            if row['variable_name'] not in seen:
                seen.add(row['variable_name'])
                extra_param_rows.append(row)
    return records, extra_param_rows
