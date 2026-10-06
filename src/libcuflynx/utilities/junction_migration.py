'''
Move a module array off the junction module types, onto ordinary vessels whose ports sum.

The junction types (``Min_junction``, ``Nout_junction``, ``MinNout_junction``, their ``_2``
variants, ``split_junction``, ``merge_junction``, ``2in2out_junction``, the ``_simple`` and
``venous_`` versions of these, and the microvasculature's ``<vessel>_Min``/``_Nout``/``_Minlet``/
``_Noutlet``/``_MinNout``) are ordinary vessels whose flow at a shared node was summed by name
or through a fixed number of ports. An ordinary vessel whose compliant end has a ``["sum", "True"]``
vessel_port does the same for any number of neighbours, on either side of the node (see
``libcuflynx.generators.port_nodes``), so each junction version maps to the ordinary version with
the same equations once its port flows are collapsed (``v_in_sum`` -> ``v_in``,
``v_out_1 + v_out_2`` -> ``v_out``).

Usage::

    python -m libcuflynx.utilities.junction_migration <module_array.json|.csv> [...]

rewrites each array in place and prints the variables whose names change, so obs_data,
params_for_id and plotting files that name them can be updated.
'''
import argparse
import json
import os
import re

from libcuflynx.utilities.config_schemas import module_array_csv_to_records

# (junction module_type, version) -> (ordinary module_type, version), names as in
# circulatory-autogen-modules. Each pair has the same equations, compared symbolically after the
# junction's port flows are collapsed; the junctions' extra outputs (du_dt, du_C_dt) are on the
# ordinary pv versions too.
JUNCTION_TWINS = {}
for _junction in ('Min_junction', 'Nout_junction', 'MinNout_junction'):
    for _version, _twin in {
            'vv': ('arterial', 'vv'),
            'vv_nonlinear': ('arterial', 'vv_nonlinear'),
            'vv_nonlinear_constR': ('arterial', 'vv_nonlinear_constR'),
            'vv_nonlinear_visco': ('arterial', 'vv_nonlinear_visco'),
            'vv_simple': ('arterial_simple', 'vv'),
            'vv_noI': ('simple', 'vv_noI')}.items():
        JUNCTION_TWINS[(_junction, _version)] = _twin
JUNCTION_TWINS.update({
    ('Min_junction', 'vp'): ('arterial', 'vp'),
    ('Min_junction', 'vp_nonlinear'): ('arterial', 'vp_nonlinear'),
    ('Min_junction', 'vp_nonlinear_constR'): ('arterial', 'vp_nonlinear_constR'),
    ('Min_junction', 'vp_nonlinear_visco'): ('arterial', 'vp_nonlinear_visco'),
    ('Min_junction', 'vp_simple'): ('arterial_simple', 'vp'),
    ('Min_junction', 'vp_simple_novisco'): ('venous_novisco', 'vp_woCont'),
    ('Min_junction', 'vp_noI'): ('simple', 'vp_noI'),
    ('Min_junction_usV_C_ctl', 'vp_simple'): ('arterial_simple', 'vp_usV_C_ctl'),
    ('Nout_junction', 'pv'): ('arterial', 'pv'),
    ('Nout_junction', 'pv_nonlinear'): ('arterial', 'pv_nonlinear'),
    ('Nout_junction', 'pv_nonlinear_constR'): ('arterial', 'pv_nonlinear_constR'),
    ('Nout_junction', 'pv_nonlinear_visco'): ('arterial', 'pv_nonlinear_visco'),
    ('Nout_junction', 'pv_simple'): ('arterial_simple', 'pv'),
    ('Nout_junction', 'pv_noI'): ('simple', 'pv_noI'),
    ('Min_junction_2', 'vp_noI'): ('simple_2', 'vp_noI'),
    ('Min_junction_2', 'vv_noI'): ('simple_2', 'vv_noI'),
    ('Nout_junction_2', 'pv_noI'): ('simple_2', 'pv_noI'),
    ('Nout_junction_2', 'vv_noI'): ('simple_2', 'vv_noI'),
    ('MinNout_junction_2', 'vv_noI'): ('simple_2', 'vv_noI'),
    ('2in2out_junction', 'vv'): ('arterial', 'vv'),
    ('split_junction_simple', 'pv'): ('arterial_simple', 'pv'),
    ('split_junction_simple', 'vv'): ('arterial_simple', 'vv'),
    ('merge_junction_simple', 'vp'): ('arterial_simple', 'vp'),
    ('merge_junction_simple', 'vv'): ('arterial_simple', 'vv'),
    ('venous_merge_junction', 'vp_nonlinear'): ('venous', 'vp_nonlinear'),
    ('venous_merge_junction_simple', 'vp'): ('venous', 'vp_woCont'),
    ('venous_merge_junction_simple', 'vv'): ('venous', 'vv'),
    ('venous_split_junction_simple', 'vv'): ('venous', 'vv'),
    ('split_junction', 'vv_novisco'): ('arterial', 'vv_novisco'),
})
for _version in ('pv', 'pv_nonlinear', 'pv_nonlinear_constR', 'pv_nonlinear_visco',
                 'vv', 'vv_nonlinear', 'vv_nonlinear_constR', 'vv_nonlinear_visco'):
    JUNCTION_TWINS[('split_junction', _version)] = ('arterial', _version)
for _version in ('vp', 'vp_nonlinear', 'vp_nonlinear_constR', 'vp_nonlinear_visco',
                 'vv', 'vv_nonlinear', 'vv_nonlinear_constR', 'vv_nonlinear_visco'):
    JUNCTION_TWINS[('merge_junction', _version)] = ('arterial', _version)

# circulatory-autogen-modules' microvasculature: the artery variants (and artery_inlet/_outlet)
# are artery's own versions; the arteriole, capillary, venule and vein variants are a resistive
# model (geometric R and C, no inertance) that is now each type's vp_micro_noI / pv_micro_noI
for _vessel in ('arteriole', 'capillary', 'venule', 'vein'):
    for _variant in ('Min', 'Noutlet'):
        JUNCTION_TWINS[(f'{_vessel}_{_variant}', 'vp_micro')] = (_vessel, 'vp_micro_noI')
    for _variant in ('Nout', 'Minlet'):
        JUNCTION_TWINS[(f'{_vessel}_{_variant}', 'pv_micro')] = (_vessel, 'pv_micro_noI')
for _variant, _versions in (('Min', ('vp_micro', 'vv_micro')), ('Noutlet', ('vp_micro',)),
                            ('Nout', ('pv_micro', 'vv_micro')), ('Minlet', ('pv_micro',)),
                            ('MinNout', ('vv_micro',)), ('inlet', ('pp_micro', 'pv_micro')),
                            ('outlet', ('pp_micro', 'vp_micro'))):
    for _version in _versions:
        JUNCTION_TWINS[(f'artery_{_variant}', _version)] = ('artery', _version)

# the junctions' port flows, and what each becomes on the ordinary vessel
RENAMED_VARIABLES = {'v_in_sum': 'v_in', 'v_out_sum': 'v_out',
                     'v_in_1': 'v_in', 'v_in_2': 'v_in', 'v_out_1': 'v_out', 'v_out_2': 'v_out'}

_KEYS = (('module_type', 'module_subtype'), ('vessel_type', 'BC_type'))


def _key(record):
    for type_key, version_key in _KEYS:
        if type_key in record and version_key in record:
            return type_key, version_key
    return None, None


def migrate_records(records):
    '''``records`` with every junction module replaced by its ordinary twin.

    An ``inlet_flow``/``nn_constant_2`` source (two constant inflows on one port, which only a
    junction's two inlets could take) feeding a converted junction becomes two ``nn_constant``
    sources, ``<name>_1`` and ``<name>_2``, which the ordinary vessel's inlet sums.

    Returns (records, renamed, parameters): ``renamed`` maps each converted module's name to its
    (old type, new type), ``parameters`` maps old parameter names to new ones. Raises KeyError for
    a junction version with no known twin.
    '''
    out, renamed, parameters = [], {}, {}
    for record in records:
        record = dict(record)
        type_key, version_key = _key(record)
        if type_key is not None:
            key = (record[type_key], record[version_key])
            if key in JUNCTION_TWINS:
                record[type_key], record[version_key] = JUNCTION_TWINS[key]
                renamed[record['name']] = (key, JUNCTION_TWINS[key])
            elif (re.search(r'junction', str(record[type_key])) and record[type_key] not in ('flow_merge', 'flow_split')) or \
                    re.fullmatch(r'(artery|arteriole|capillary|venule|vein)_(Min|Nout|Minlet|Noutlet|MinNout|inlet|outlet)',
                                 str(record[type_key])):
                raise KeyError(f'{record["name"]}: no ordinary vessel is known for the junction '
                               f'{record[type_key]}/{record[version_key]}')
        out.append(record)

    out_key = 'out_instances' if any('out_instances' in r for r in out) else 'out_vessels'
    inp_key = 'inp_instances' if out_key == 'out_instances' else 'inp_vessels'
    split = []
    for record in out:
        type_key, version_key = _key(record)
        if type_key and (record[type_key], record[version_key]) == ('inlet_flow', 'nn_constant_2') and \
                any(o in renamed for o in record.get(out_key, [])):
            name = record['name']
            for i in (1, 2):
                split.append(dict(record, name=f'{name}_{i}', **{version_key: 'nn_constant'}))
                parameters[f'v_{i}_{name}'] = f'v_{name}_{i}'
            for other in out:
                for list_key in (inp_key, out_key):
                    if name in other.get(list_key, []):
                        lst = list(other[list_key])
                        i = lst.index(name)
                        other[list_key] = lst[:i] + [f'{name}_1', f'{name}_2'] + lst[i + 1:]
            renamed.setdefault(name, ((record[type_key], 'nn_constant_2'), ('inlet_flow', 'nn_constant x2')))
        else:
            split.append(record)
    return split, renamed, parameters


def renamed_outputs(renamed):
    '''The output names that change: {"<vessel>/<old variable>": "<vessel>/<new variable>"}.'''
    return {f'{name}/{old}': f'{name}/{new}' for name in renamed for old, new in RENAMED_VARIABLES.items()}


def migrate_file(path, parameters_path=None):
    '''Rewrite the module array at ``path`` (JSON or CSV) without junction types, and the
    parameter names it changes in ``parameters_path`` (a parameters CSV), if given.

    Records are rewritten as they are, keeping their key style, ``instance`` and any other keys:
    a JSON array one record per line, a CSV array in its own columns. Returns the ``renamed``
    map of ``migrate_records``.
    '''
    if path.lower().endswith('.json'):
        with open(path, encoding='utf-8-sig') as f:
            records = json.load(f)
    else:
        records = module_array_csv_to_records(path)
    migrated, renamed, parameters = migrate_records(records)
    if not renamed:
        return renamed
    if path.lower().endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('[\n' + ',\n'.join(' ' + json.dumps(r) for r in migrated) + '\n]\n')
    elif len(migrated) == len(records):
        _retype_csv(path, migrated)
    else:
        _write_csv(path, migrated)
    if parameters and parameters_path and os.path.exists(parameters_path):
        with open(parameters_path) as f:
            lines = f.read().split('\n')
        for i, line in enumerate(lines):
            name = line.split(',', 1)[0].strip()
            if name in parameters:
                lines[i] = parameters[name] + line[len(line.split(',', 1)[0]):]
        with open(parameters_path, 'w') as f:
            f.write('\n'.join(lines))
    return renamed


def _retype_csv(path, records):
    '''Rewrite only the vessel_type and BC_type cells of a CSV array, keeping its layout.'''
    with open(path) as f:
        lines = f.read().split('\n')
    header = [c.strip() for c in lines[0].split(',')]
    columns = {'vessel_type': None, 'BC_type': None}
    for key in columns:
        for alias in ((key,) if key == 'vessel_type' else (key,)) + (('module_type',) if key == 'vessel_type' else ('module_subtype',)):
            if alias in header:
                columns[key] = header.index(alias)
    by_name = {r['name']: r for r in records}
    for i, line in enumerate(lines[1:], 1):
        cells = line.split(',')
        if len(cells) < len(header) or not cells[0].strip():
            continue
        record = by_name.get(cells[0].strip())
        if record is None:
            continue
        for key, column in columns.items():
            old, new = cells[column].strip(), str(record[key])
            if old != new:
                cells[column] = cells[column].replace(old, new, 1)
        lines[i] = ','.join(cells)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _write_csv(path, records):
    with open(path) as f:
        header = [c.strip() for c in f.readline().split(',')]
    lines = [', '.join(header)]
    for record in records:
        cells = []
        for column in header:
            value = record.get(column, '')
            if isinstance(value, (list, tuple)):
                value = ' '.join(value)
            cells.append(str(value))
        lines.append(', '.join(cells))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('module_arrays', nargs='+',
                        help='module arrays; <prefix>_parameters.csv next to each is updated too')
    args = parser.parse_args(argv)
    for path in args.module_arrays:
        parameters = re.sub(r'_(module|vessel)_array\.(json|csv)$', '_parameters.csv', path)
        renamed = migrate_file(os.path.abspath(path), os.path.abspath(parameters))
        print(f'{path}: {len(renamed)} junction module(s) converted')
        for name, (old, new) in renamed.items():
            print(f'  {name}: {old[0]}/{old[1]} -> {new[0]}/{new[1]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
