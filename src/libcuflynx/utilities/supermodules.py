'''
Expands supermodule instances in a module array into their submodules.

A supermodule is a module config entry with ``"module_format": "supermodule"`` and a
``submodules`` list (see ``utilities/config_schemas.py``). An instance of it in a module
array, e.g.::

    {"name": "heart", "module_type": "heart", "module_subtype": "supermodule",
     "per_submodule_inputs": {"ra": ["venous_svc"], "la": ["pvn"]},
     "per_submodule_outputs": {"aov": ["aortic_root"], "puv": ["par"]}}

is replaced, at its position in the array, by one record per submodule, in the
supermodule's order:

* each submodule ``sub`` becomes ``<instance>__<sub>`` (levels are joined with ``__``,
  ``parameter_names.LEVEL_SEPARATOR``), and its (sibling) inp/out names are prefixed the same
  way;
* the hosts in ``per_submodule_inputs[sub]`` are prepended to that submodule's inputs, and
  the hosts in ``per_submodule_outputs[sub]`` appended to its outputs, in order;
* in each host, the instance name in its out list is replaced, at the same position, by
  every ``<instance>__<sub>`` whose ``per_submodule_inputs`` lists that host; the instance
  name in its inp list by every ``<instance>__<sub>`` whose ``per_submodule_outputs`` lists
  it (in the order of the ``per_submodule_*`` object).

A host may instead name a submodule by its path, ``heart/ra`` (``neuron/soma/membrane`` two
levels down), in its own inp/out list. The reference becomes ``heart__ra`` where it stands,
so each host keeps its own order, and the submodule takes the host as it would from
``per_submodule_*``. A ``per_submodule_*`` key may be a path too, ``{"soma/membrane": [...]}``:
it is passed down to the nested instance ``soma``. A path is allowed only there and in a
host's lists, not in an instance's own lists or a supermodule's submodule records; a path left
over once everything is expanded names nothing and is an error.

A supermodule entry may declare an ordered ``interface``,
``[{"name", "submodule": <path>, "port_type", "label"}]``. A host that lists the instance with
no ``per_submodule_*`` entry for it is linked to the first interface entry whose
``port_type`` is one of the host's own ports facing it (its exit and general ports when it
feeds the instance, its entrance and general ports when it is fed); with none, it is an error
as before.

A submodule may itself be a supermodule instance (with ``per_submodule_*`` naming its
siblings); it is expanded in turn, and a supermodule that contains itself, directly or not,
is an error. Links between a host and a nested supermodule go through that nested
supermodule's own ``per_submodule_*`` (or a path), so an instance's ``per_submodule_*`` may not
name a submodule that is itself a supermodule instance, except by a path into it.

The supermodule's parameters -- those of its *parameterisation* (the record's
``"parameterisation"``, or the entry's ``default_parameterisation``; see
``utilities/parameterisations.py``), then its legacy ``default_parameters`` file -- name a
submodule's parameter ``<sub>/<var>`` (one nested in a submodule ``<sub>/<subsub>/<var>``), and
are renamed to ``{var}_{instance}__{sub}``, ``instance`` being the supermodule record's name;
``global/<var>`` or a plain name is a global and keeps its name, added only once. The older
``{var}_{sub}`` form is still read, matching the suffix against the submodule paths longest
first, with a FutureWarning (see ``utilities/parameter_names.py``). A submodule record may name
its own parameterisation; it stays on the expanded record, whose parameterisation is read with
the components' (``parameterisations.component_parameterisation_rows``), after the
supermodule's, so the
supermodule's values win.

After expansion no record is a supermodule instance.
'''

import copy
import os

from libcuflynx.utilities.config_schemas import PER_SUBMODULE_KEYS, SUPERMODULE_FORMAT
from libcuflynx.utilities.parameterisations import (first_rows_win, parameterisation_rows,
                                                    parameterisation_parameters_path,
                                                    warn_conflicting_globals,
                                                    read_parameter_rows, record_parameterisation)
from libcuflynx.utilities.parameter_names import (SEPARATOR, flat_name, supermodule_row_name,
                                                  warn_old_supermodule_names)

# a supermodule nested deeper than this is taken to be a cycle the ancestry check missed
_MAX_DEPTH = 64


def _key(record):
    return (record['module_type'], record['module_subtype'])


def _is_instance(record, registry, source):
    '''Whether ``record`` is a supermodule instance; errors for an unknown supermodule.'''
    if _key(record) in registry:
        return True
    looks_like_one = (record['module_subtype'] == SUPERMODULE_FORMAT
                      or any(k in record for k in PER_SUBMODULE_KEYS))
    if looks_like_one:
        known = sorted(registry) or 'none'
        raise ValueError(
            f'{source}: "{record["name"]}" is a supermodule instance of type '
            f'({record["module_type"]}, {record["module_subtype"]}), but no supermodule entry of that '
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
    '''``{var}_{sub}`` -> ``{var}_{instance}__{sub}``; a global name is returned unchanged.

    The older form only (the suffix is matched against ``submodule_names``, longest first);
    rows are read through ``parameter_names.supermodule_row_name``, which also takes
    ``<sub>/<var>``.'''
    for sub in sorted(submodule_names, key=len, reverse=True):
        suffix = '_' + sub
        if variable_name.endswith(suffix) and len(variable_name) > len(suffix):
            return f'{variable_name[:-len(suffix)]}_{flat_name((instance, sub))}'
    return variable_name


def _rename_rows(rows, instance, path_parts, source):
    '''``rows`` of a supermodule's parameters file (``source``) renamed for the record
    ``instance``, warning once when the file uses the older ``{var}_{sub}`` form.'''
    renamed, old = [], []
    for row in rows:
        name, old_form = supermodule_row_name(row['variable_name'], instance, path_parts,
                                              where=f'{source}: ')
        if old_form:
            old.append(row['variable_name'])
        renamed.append(dict(row, variable_name=name))
    warn_old_supermodule_names(old, source)
    return renamed


def submodule_path_parts(supermodule, registry=None, _depth=0):
    '''Every submodule of ``supermodule`` as a tuple of names: ``('sub',)``, and for a
    submodule that is itself a supermodule ``('sub', 'subsub')`` ... (its expansion names a
    nested submodule ``<instance>__<sub>__<subsub>``).'''
    paths = []
    for sub in supermodule['submodules']:
        paths.append((sub['name'],))
        nested = (registry or {}).get(_key(sub))
        if nested is not None and _depth < _MAX_DEPTH:
            paths += [(sub['name'],) + p
                      for p in submodule_path_parts(nested, registry, _depth + 1)]
    return paths


def submodule_paths(supermodule, registry=None):
    '''``submodule_path_parts`` as the names an expanded vessel carries after the instance:
    ``sub``, ``sub__subsub`` ...'''
    return [flat_name(p) for p in submodule_path_parts(supermodule, registry)]


def read_default_parameters(supermodule, instance, where, registry=None):
    '''The supermodule's default_parameters rows, renamed for ``instance``: a row
    ``<sub>/<var>`` (or the older ``{var}_{sub}``) names a submodule, ``<sub>/<subsub>/<var>``
    one nested in a submodule that is a supermodule, and becomes ``{var}_{instance}__{path}``;
    any other row is a global.'''
    file_name = supermodule.get('default_parameters')
    if not file_name:
        return []
    config_dir = os.path.dirname(supermodule.get('config_path', '') or '')
    path = file_name if os.path.isabs(file_name) else os.path.join(config_dir, file_name)
    if not os.path.isfile(path):
        raise ValueError(f'{where}: default_parameters file {path} of supermodule '
                         f'({supermodule["module_type"]}, {supermodule["module_subtype"]}) not found '
                         f'(it is resolved against {supermodule.get("config_path")}).')
    rows = read_parameter_rows(path, where, 'default_parameters file')
    return _rename_rows(rows, instance, submodule_path_parts(supermodule, registry), path)


def read_supermodule_parameters(supermodule, record, where, registry=None):
    '''
    The parameters of the supermodule instance ``record`` (its name is the prefix), renamed:
    those of its parameterisation (the record's, or the entry's default) first, then its
    default_parameters, each name once.
    '''
    name = record['name']
    parameterisation, own_rows = parameterisation_rows(
        supermodule, record_parameterisation(record, where), where)
    source = (parameterisation_parameters_path(supermodule['config_path'], parameterisation)
              if parameterisation else where)
    own_rows = _rename_rows(own_rows, name, submodule_path_parts(supermodule, registry), source)
    return first_rows_win(own_rows + read_default_parameters(supermodule, name, where, registry))


def _head(key):
    """``('soma', 'membrane')`` for the per_submodule_* key or path ``soma/membrane``."""
    head, _, tail = key.partition(SEPARATOR)
    return head, tail


def _unique(names):
    return list(dict.fromkeys(names))


def check_interface(supermodule, registry, where):
    """ValueError unless ``supermodule``'s ``interface`` has unique names and existing paths."""
    interface = supermodule.get('interface')
    if interface is None:
        return []
    if not isinstance(interface, list):
        raise ValueError(f'{where}: "interface" must be a list of '
                         f'{{"name", "submodule", "port_type", "label"}} objects.')
    paths = {SEPARATOR.join(p) for p in submodule_path_parts(supermodule, registry)}
    names = []
    for i, entry in enumerate(interface):
        if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) and entry[k]
                                                  for k in ('name', 'submodule', 'port_type')):
            raise ValueError(f'{where}: interface[{i}] needs the strings "name", "submodule" '
                             f'and "port_type" (got {entry!r}).')
        if entry['name'] in names:
            raise ValueError(f'{where}: interface name "{entry["name"]}" is used twice.')
        names.append(entry['name'])
        if entry['submodule'] not in paths:
            raise ValueError(f'{where}: interface "{entry["name"]}" names submodule '
                             f'"{entry["submodule"]}", which is not one of {sorted(paths)}.')
    return interface


def _host_port_types(host, components, facing_out):
    """The port types of ``host``'s module facing an instance it feeds (``facing_out``) or is
    fed by; empty for a host whose component is unknown."""
    entry = (components or {}).get((host['module_type'], host['module_subtype']))
    if entry is None:
        return set()
    keys = ('exit_ports', 'general_ports') if facing_out else ('entrance_ports', 'general_ports')
    return {port.get('port_type') for key in keys for port in (entry.get(key) or [])
            if isinstance(port, dict)}


def _bind_to_interface(name, interface, by_name, per_inputs, per_outputs, components):
    """``per_inputs``/``per_outputs`` with every host that lists ``name`` without a
    per_submodule_* entry bound to the first interface entry matching one of its ports."""
    per_inputs = {k: list(v) for k, v in per_inputs.items()}
    per_outputs = {k: list(v) for k, v in per_outputs.items()}
    for other in by_name.values():
        for list_key, per, facing_out in (('out_instances', per_inputs, True),
                                          ('inp_instances', per_outputs, False)):
            if name not in other[list_key] or any(other['name'] in h for h in per.values()):
                continue
            ports = _host_port_types(other, components, facing_out)
            entry = next((e for e in interface if e['port_type'] in ports), None)
            if entry is not None:
                per.setdefault(entry['submodule'], []).append(other['name'])
    return per_inputs, per_outputs


def _expand_one(records, index, registry, source, ancestry, components=None):
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
    interface = check_interface(supermodule, registry, where)
    for list_key in ('inp_instances', 'out_instances'):
        paths = [n for n in instance[list_key] if SEPARATOR in n]
        if paths:
            raise ValueError(f'{where}: its own {list_key} names the paths {paths}; an '
                             f'instance links to its hosts through per_submodule_*, and a path '
                             f'may only be written in a host\'s list.')
    by_name = {r['name']: r for i, r in enumerate(records) if i != index}
    per_inputs, per_outputs = _bind_to_interface(
        name, interface, by_name, instance.get('per_submodule_inputs', {}),
        instance.get('per_submodule_outputs', {}), components)
    for per_key, per in (('per_submodule_inputs', per_inputs),
                         ('per_submodule_outputs', per_outputs)):
        unknown = [s for s in per if _head(s)[0] not in sub_names]
        if unknown:
            raise ValueError(f'{where}: {per_key} names {unknown}, which are not submodules '
                             f'of {key}; its submodules are {sub_names}.')
        inner = [s for s in per if s in nested]
        if inner:
            raise ValueError(f'{where}: {per_key} names {inner}, which are themselves '
                             f'supermodule instances; connect the host to one of their '
                             f'submodules by a path (<submodule>/<its submodule>) or through '
                             f'that supermodule\'s own per_submodule_*.')
        flat = [s for s in per if _head(s)[1] and _head(s)[0] not in nested]
        if flat:
            raise ValueError(f'{where}: {per_key} names the paths {flat}, but their first '
                             f'submodule is not a supermodule instance.')

    # hosts that name a submodule by path: heart/ra, or heart/soma/membrane two levels down
    path_inputs, path_outputs, renamed = {}, {}, {}
    for other in by_name.values():
        for list_key, links in (('out_instances', path_inputs), ('inp_instances', path_outputs)):
            for entry in other[list_key]:
                if not entry.startswith(name + SEPARATOR):
                    continue
                head, tail = _head(entry[len(name) + 1:])
                if head not in sub_names:
                    raise ValueError(f'{where}: "{other["name"]}" names "{entry}", but "{head}" '
                                     f'is not a submodule of {key} ({sub_names}).')
                if head in nested and not tail:
                    raise ValueError(f'{where}: "{other["name"]}" names "{entry}", which is a '
                                     f'supermodule instance; name one of its submodules '
                                     f'("{entry}/<submodule>").')
                if tail and head not in nested:
                    raise ValueError(f'{where}: "{other["name"]}" names "{entry}", but "{head}" '
                                     f'is not a supermodule instance.')
                renamed[entry] = flat_name((name, head)) + (SEPARATOR + tail if tail else '')
                if not tail:
                    links.setdefault(head, []).append(other['name'])

    for per_key, per in (('per_submodule_inputs', per_inputs),
                         ('per_submodule_outputs', per_outputs)):
        for sub, hosts in per.items():
            repeated = sorted({h for h in hosts if hosts.count(h) > 1})
            if repeated:
                raise ValueError(f'{where}: {per_key}["{sub}"] names {repeated} more than once; '
                                 f'each host is linked once per submodule.')
    for other in by_name.values():
        for list_key in ('out_instances', 'inp_instances'):
            if other[list_key].count(name) > 1:
                raise ValueError(f'{where}: "{other["name"]}" lists "{name}" more than once in its '
                                 f'{list_key.split("_")[0]} list. List it once; the instance\'s '
                                 f'per_submodule_* entries say which of its submodules it links to.')
    for per_key, per, host_key, host_list in (
            ('per_submodule_inputs', per_inputs, 'out', 'out_instances'),
            ('per_submodule_outputs', per_outputs, 'inp', 'inp_instances')):
        for sub, hosts in per.items():
            for host in hosts:
                if host not in by_name:
                    raise ValueError(f'{where}: {per_key}["{sub}"] names host "{host}", which '
                                     f'is not in the module array.')
                if name not in by_name[host][host_list]:
                    raise ValueError(
                        f'{where}: {per_key}["{sub}"] names host "{host}", but "{host}" does '
                        f'not list "{name}" in its {host_key} list '
                        f'({host_list}/{host_key}_instances).')

    input_hosts = {h for hosts in per_inputs.values() for h in hosts}
    output_hosts = {h for hosts in per_outputs.values() for h in hosts}
    for other in by_name.values():
        if name in other['out_instances'] and other['name'] not in input_hosts:
            raise ValueError(
                f'{where}: "{other["name"]}" lists "{name}" in its out list, but no '
                f'per_submodule_inputs entry of "{name}" names "{other["name"]}"'
                + (' and none of its interface entries matches its ports' if interface else '')
                + ', so it is not known which submodule it feeds.')
        if name in other['inp_instances'] and other['name'] not in output_hosts:
            raise ValueError(
                f'{where}: "{other["name"]}" lists "{name}" in its inp list, but no '
                f'per_submodule_outputs entry of "{name}" names "{other["name"]}"'
                + (' and none of its interface entries matches its ports' if interface else '')
                + ', so it is not known which submodule feeds it.')
    for list_key, hosts, per_key in (('inp_instances', input_hosts, 'per_submodule_inputs'),
                                     ('out_instances', output_hosts, 'per_submodule_outputs')):
        stray = [h for h in instance[list_key] if h not in hosts]
        if stray:
            raise ValueError(f'{where}: its {list_key.split("_")[0]} list names {stray}, which '
                             f'no {per_key} entry names.')

    def prefixed(sub):
        return flat_name((name, sub))

    # per_submodule_* keys that are paths go down to the nested instance they start with
    pushed = {}
    for per_key, per in (('per_submodule_inputs', per_inputs),
                         ('per_submodule_outputs', per_outputs)):
        for sub, hosts in per.items():
            head, tail = _head(sub)
            if tail:
                pushed.setdefault(head, {}).setdefault(per_key, {})[tail] = list(hosts)

    new_records = []
    for sub in submodules:
        record = copy.deepcopy(sub)
        record['name'] = prefixed(sub['name'])
        record['inp_instances'] = (list(per_inputs.get(sub['name'], []))
                                   + path_inputs.get(sub['name'], [])
                                   + [prefixed(n) for n in sub['inp_instances']])
        record['out_instances'] = ([prefixed(n) for n in sub['out_instances']]
                                   + list(per_outputs.get(sub['name'], []))
                                   + path_outputs.get(sub['name'], []))
        for per_key in PER_SUBMODULE_KEYS:
            if per_key in record:
                record[per_key] = {s: [prefixed(h) for h in hosts]
                                   for s, hosts in record[per_key].items()}
            for s, hosts in pushed.get(sub['name'], {}).get(per_key, {}).items():
                record.setdefault(per_key, {})
                record[per_key][s] = record[per_key].get(s, []) + hosts
        new_records.append(record)

    clashes = [r['name'] for r in new_records if r['name'] in by_name]
    if clashes:
        raise ValueError(f'{where}: expanding it gives the names {clashes}, which are already '
                         f'in the module array. Rename the instance or the clashing records.')

    for other in by_name.values():
        for list_key in ('out_instances', 'inp_instances'):
            other[list_key] = [renamed.get(n, n) for n in other[list_key]]
        if name in other['out_instances']:
            other['out_instances'] = _splice(
                other['out_instances'], name,
                _unique(prefixed(_head(s)[0]) for s, hosts in per_inputs.items()
                        if other['name'] in hosts))
        if name in other['inp_instances']:
            other['inp_instances'] = _splice(
                other['inp_instances'], name,
                _unique(prefixed(_head(s)[0]) for s, hosts in per_outputs.items()
                        if other['name'] in hosts))
        # another instance not expanded yet that names this one as a host: its per_submodule_*
        # lists now name the submodules that link to it (whichever of the two expands first)
        for per_key, links in (('per_submodule_inputs', per_outputs),
                               ('per_submodule_outputs', per_inputs)):
            if per_key in other:
                replacement = _unique(prefixed(_head(s)[0]) for s, hosts in links.items()
                                      if other['name'] in hosts)
                other[per_key] = {s: _splice(hosts, name, replacement)
                                  for s, hosts in other[per_key].items()}

    for record in new_records:
        ancestry[record['name']] = chain + (key,)
    param_rows = read_supermodule_parameters(supermodule, instance, where, registry)
    return records[:index] + new_records + records[index + 1:], param_rows, \
        _global_settings(param_rows, instance, supermodule, registry)


def _global_settings(rows, instance, supermodule, registry):
    '''The rows of ``instance``'s parameters that set a global (those the renaming left
    alone), as ``warn_conflicting_globals`` settings.'''
    name = instance['name']
    local = tuple(f'_{flat_name((name, path))}' for path in submodule_paths(supermodule, registry))
    used = record_parameterisation(instance, name) or supermodule.get('default_parameterisation')
    who = f'supermodule "{name}"' + (f' (parameterisation "{used}")' if used else '')
    return [(row['variable_name'], row['value'], row['units'], who, name) for row in rows
            if not row['variable_name'].endswith(local)]



def expand_supermodules(records, registry, source=None, settings=None, components=None):
    '''
    ``(records, extra_param_rows)``: ``records`` (normalised vessel records, see
    ``config_schemas.normalise_vessel_record``) with every supermodule instance -- a record
    whose (module_type, module_subtype) is in ``registry`` -- replaced by its prefixed submodules,
    recursively, and the instances' renamed parameters -- parameterisation, then
    default_parameters -- each name once, the first occurrence kept (so an outer supermodule's
    values win over a nested one's). ``records`` is not modified.

    Raises ValueError, naming ``source`` and the instance, for an unknown supermodule type,
    an unknown submodule in per_submodule_*, a host that does not exist or does not name the
    instance back, a record that names the instance without a per_submodule_* entry linking
    them, an expanded name that clashes with an existing record, nesting in a cycle, or a
    path that names nothing. ``components`` (the component registry) lets a host bind to an
    ``interface`` entry by its ports.
    '''
    source = source or 'module array'
    records = copy.deepcopy(list(records))
    ancestry = {}
    extra_param_rows = []
    check_here = settings is None
    settings = [] if check_here else settings
    while True:
        index = next((i for i, r in enumerate(records) if _is_instance(r, registry, source)), None)
        if index is None:
            break
        records, rows, sets = _expand_one(records, index, registry, source, ancestry, components)
        extra_param_rows += rows
        settings += sets
    for record in records:
        for list_key in ('inp_instances', 'out_instances'):
            paths = [n for n in record[list_key] if SEPARATOR in n]
            if paths:
                raise ValueError(f'{source}: "{record["name"]}" names {paths} in its '
                                 f'{list_key}, which are not submodules of a supermodule '
                                 f'instance in the module array.')
    if check_here:
        warn_conflicting_globals(settings, source)
    return records, first_rows_win(extra_param_rows)
