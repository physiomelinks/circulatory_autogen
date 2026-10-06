'''
Move a model onto the lumped vessels of a module library: every vessel whose version has a
lumped twin (a supermodule version whose config entry says what it ``replaces``, e.g.
circulatory-autogen-modules' ``arterial/vp_lumped``, built from compliance, resistance and
inertance submodules) uses the twin instead.

What changes, and what does not:

* the module array: the record's ``BC_type`` (module_subtype), nothing else; the supermodule's
  ``routes`` connect its neighbours to the right submodules;
* the parameters file: a parameter the twin keeps by name (a ``shared_parameters`` entry, e.g.
  ``C_aortic_root``, ``R_T_systemic_T``) is unchanged. A value the twin computes from the
  monolithic parameters (``replaces.parameters``, e.g. ``q_C_init_C = q_init - q_us``) is written
  as the submodule's parameter ``<var>_<vessel>_<submodule>``, from the file's values (or the
  monolithic default instance's), and the rows only it used are dropped;
* outputs: unchanged. The twin's ``outputs`` expose the monolithic outputs under the vessel's
  own name (``aortic_root/u`` is its compliance's pressure), so obs_data, params_for_id and
  plotting files keep working; a ``<vessel>/<var>`` they name that the twin does not expose is
  reported.

Usage::

    python -m libcuflynx.utilities.lumped_migration --library <modules dir> <module_array> \\
        [--parameters P] [--check F ...] [--out-dir D --prefix NEW]

rewrites the module array and parameters file in place, or, with ``--out-dir``/``--prefix``,
writes ``<D>/<NEW>_<suffix>`` copies of them and of every ``--check`` file (obs_data,
params_for_id, prediction variables: copied unchanged, after checking their outputs).
'''
import argparse
import ast
import csv
import json
import math
import os
import re

from libcuflynx.utilities.config_schemas import (load_component_registry, load_supermodule_registry,
                                                 read_module_array_records, record_to_style,
                                                 module_array_csv_to_records)
from libcuflynx.utilities.module_instances import instance_parameter_rows
from libcuflynx.utilities.module_library import ModuleSources


class Library(object):
    '''A module library's lumped twins: ``twins[(vessel_type, monolithic BC_type)]`` is the
    supermodule entry that replaces it.'''

    def __init__(self, library_dirs):
        sources = ModuleSources({'module_library_dirs': list(library_dirs), 'use_builtin_modules': False})
        self.components = load_component_registry(sources.config_files)
        self.supermodules = load_supermodule_registry(sources.config_files)
        self.twins = {}
        for (vessel_type, _bc), entry in self.supermodules.items():
            replaces = entry.get('replaces')
            if replaces:
                self.twins[(vessel_type, replaces['module_subtype'])] = entry

    def monolithic_defaults(self, key):
        entry = self.components.get(key)
        if entry is None:
            return {}
        _, rows = instance_parameter_rows(entry, None, f'monolithic {key}')
        return {r['variable_name']: r['value'] for r in rows}

    def submodule_units(self, twin, sub_name, variable):
        sub = next(s for s in twin['submodules'] if s['name'] == sub_name)
        entry = self.components.get((sub['vessel_type'], sub['BC_type'])) or {}
        return next((u for v, u, *_ in entry.get('variables_and_units') or [] if v == variable), '')


def _shared_names(twin):
    return {e if isinstance(e, str) else e['name'] for e in twin.get('shared_parameters') or []}


def _split_target(target, sub_names):
    '''"q_C_init_C" -> ("q_C_init", "C"): a submodule-level instance row name.'''
    for sub in sorted(sub_names, key=len, reverse=True):
        if target.endswith('_' + sub) and len(target) > len(sub) + 1:
            return target[:-len(sub) - 1], sub
    raise ValueError(f'{target} names none of the submodules {sub_names}')


_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.Name, ast.Load, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Gt, ast.GtE, ast.Lt, ast.LtE,
            ast.Eq, ast.NotEq)


def evaluate(expr, values):
    '''A replaces.parameters expression (arithmetic of monolithic parameter names) at ``values``;
    None if it names a parameter with no value.'''
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f'{expr}: {type(node).__name__} is not allowed in a replaces expression')
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not names <= set(values):
        return None
    return float(eval(compile(tree, '<replaces>', 'eval'), {'__builtins__': {}}, dict(values)))


def _float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def migrate(records, parameter_rows, library):
    '''
    (records, parameter_rows, outputs, notes) for a model on the lumped twins: ``outputs`` the
    ``<vessel>/<var>`` names the replaced vessels still have. ``records`` are normalised vessel
    records; ``parameter_rows`` dicts with variable_name, units, value, data_reference (other keys
    kept).
    '''
    out_records, outputs, notes = [], set(), []
    by_name = {r['variable_name']: r for r in parameter_rows}
    drop, add = set(), []
    for record in records:
        key = (record['vessel_type'], record['BC_type'])
        twin = library.twins.get(key)
        if twin is None:
            out_records.append(record)
            continue
        vessel = record['name']
        out_records.append(dict(record, BC_type=twin['BC_type']))
        mono = library.components.get(key) or {}
        mono_vars = [v for v, *_ in mono.get('variables_and_units') or []]
        values = {k: v for k, v in ((k, _float(v)) for k, v in library.monolithic_defaults(key).items())
                  if v is not None}
        for var in mono_vars:
            row = by_name.get(f'{var}_{vessel}')
            if row is not None and _float(row['value']) is not None:
                values[var] = _float(row['value'])
        sub_names = [s['name'] for s in twin['submodules']]
        for target, expr in (twin['replaces'].get('parameters') or {}).items():
            var, sub = _split_target(target, sub_names)
            value = evaluate(expr, values)
            if value is None:
                notes.append(f'{vessel}: {target} = {expr} has no value here; the twin\'s default instance gives it')
                continue
            add.append({'variable_name': f'{var}_{vessel}_{sub}', 'units': library.submodule_units(twin, sub, var),
                        'value': repr(value), 'data_reference': f'lumped_migration: {expr} of {vessel}'})
        kept = _shared_names(twin)
        for var in mono_vars:
            if var not in kept and f'{var}_{vessel}' in by_name:
                drop.add(f'{var}_{vessel}')
        outputs |= {f'{vessel}/{o}' for o in twin.get('outputs') or {}}
    rows = [r for r in parameter_rows if r['variable_name'] not in drop]
    existing = {r['variable_name'] for r in rows}
    rows += [r for r in add if r['variable_name'] not in existing]
    return out_records, rows, outputs, notes


def unexposed_outputs(text, records, library, outputs):
    '''``<vessel>/<var>`` tokens of replaced vessels in ``text`` that their twins do not expose.'''
    replaced = {r['name'] for r in records if (r['vessel_type'], r['BC_type']) in library.twins}
    found = set(re.findall(r'(?<![\w/])([A-Za-z0-9_]+/[A-Za-z_][A-Za-z0-9_]*)(?![\w/])', text))
    return sorted(t for t in found if t.split('/')[0] in replaced and t not in outputs)


# ---- files ---------------------------------------------------------------------------------------

def _read_records(path):
    return module_array_csv_to_records(path) if path.endswith('.csv') else read_module_array_records(path)


def _write_records(path, records, template_path):
    if path.endswith('.json'):
        with open(path, 'w') as f:
            f.write('[\n' + ',\n'.join(' ' + json.dumps(record_to_style(r)) for r in records) + '\n]\n')
        return
    # a CSV module array keeps its layout: only BC_type cells change
    with open(template_path, newline='') as f:
        rows = list(csv.reader(f))
    header = [c.strip() for c in rows[0]]
    name_col, bc_col = header.index('name'), header.index('BC_type')
    bc = {r['name']: r['BC_type'] for r in records}
    for row in rows[1:]:
        if len(row) > max(name_col, bc_col) and row[name_col].strip() in bc:
            old = row[bc_col]
            new = bc[row[name_col].strip()]
            row[bc_col] = old.replace(old.strip(), new) if old.strip() else new
    with open(path, 'w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)


def _read_parameters(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        fields = [c.strip() for c in reader.fieldnames]
        rows = [{(k or '').strip(): (v or '').strip() for k, v in r.items() if k} for r in reader]
    return fields, [r for r in rows if r.get('variable_name')]


def _write_parameters(path, fields, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def migrate_files(library, module_array, parameters=None, check_files=(), out_dir=None, prefix=None,
                  old_prefix=None):
    '''Migrates the files; returns {'replaced', 'outputs', 'notes', 'unexposed', 'written'}.'''
    def target(path):
        if out_dir is None:
            return path
        base = os.path.basename(path)
        if old_prefix and prefix and base.startswith(old_prefix + '_'):
            base = prefix + base[len(old_prefix):]
        return os.path.join(out_dir, base)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    records = _read_records(module_array)
    fields, rows = _read_parameters(parameters) if parameters else ([], [])
    new_records, new_rows, outputs, notes = migrate(records, rows, library)
    written = []
    _write_records(target(module_array), new_records, module_array)
    written.append(target(module_array))
    if parameters:
        _write_parameters(target(parameters), fields, new_rows)
        written.append(target(parameters))
    unexposed = []
    for path in check_files:
        with open(path) as f:
            text = f.read()
        unexposed += [f'{os.path.basename(path)}: {t}' for t in unexposed_outputs(text, records, library, outputs)]
        if target(path) != path:
            with open(target(path), 'w') as f:
                f.write(text)
            written.append(target(path))
    replaced = sorted({r['name'] for r in records if (r['vessel_type'], r['BC_type']) in library.twins})
    return {'replaced': replaced, 'outputs': sorted(outputs), 'notes': notes, 'unexposed': unexposed,
            'written': written}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('module_array')
    parser.add_argument('--library', action='append', required=True, help='a module library modules/ directory')
    parser.add_argument('--parameters')
    parser.add_argument('--check', action='append', default=[],
                        help='an obs_data, params_for_id or prediction-variables file naming outputs')
    parser.add_argument('--out-dir')
    parser.add_argument('--prefix', help='the new file prefix (with --out-dir)')
    args = parser.parse_args(argv)
    old_prefix = re.sub(r'_(module|vessel)_array\.(csv|json)$', '', os.path.basename(args.module_array))
    result = migrate_files(Library(args.library), args.module_array, args.parameters, args.check,
                           args.out_dir, args.prefix, old_prefix)
    for path in result['written']:
        print('wrote', path)
    for note in result['notes'] + [f'not exposed by its lumped twin: {u}' for u in result['unexposed']]:
        print('NOTE', note)


if __name__ == '__main__':
    main()
