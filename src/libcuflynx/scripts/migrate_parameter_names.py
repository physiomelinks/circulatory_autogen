'''
``cuflynx-migrate-parameter-names``: rewrite parameter names written ``{variable}_{vessel}`` as
``<vessel>/<variable>`` (see ``utilities/parameter_names.py``).

Two kinds of file:

* a host parameters file, with ``--module-array <prefix>_module_array.json|csv``: each name
  ``{var}_{vessel}`` of a vessel in the (expanded) module array becomes ``<vessel>/<var>``,
  written ``<record>/<submodule>/<var>`` for a supermodule's expanded vessel;
* a supermodule's instance files and default_parameters, with ``--supermodule-config
  <its *_modules_config.json>``: ``{var}_{sub}`` becomes ``<sub>/<var>``, and
  ``<sub>/<subsub>/<var>`` for a nested submodule.

A name is rewritten only when exactly one reading of it names a real variable of that
module, so the guess the old form forced is checked rather than repeated; a global constant
keeps its plain name. A name with more than one such reading is left as it is and reported,
for a person to settle. Every other column, and the row order, is kept.
'''

import argparse
import csv
import io
import os
import sys

from libcuflynx.utilities.config_schemas import (is_supermodule_entry, load_component_registry,
                                                 load_expanded_vessel_records,
                                                 load_module_config, load_supermodule_registry,
                                                 read_module_array_records)
from libcuflynx.utilities.module_instances import available_instances, instance_parameters_path
from libcuflynx.utilities.module_library import ModuleSources
from libcuflynx.utilities.parameter_names import SEPARATOR
from libcuflynx.utilities.supermodules import submodule_path_parts

EPILOG = (
    "Not a pipeline stage: it does not read user_inputs.yaml. Give the files' module array or\n"
    "supermodule config, and the module libraries their modules come from.\n"
    "\n"
    "  cuflynx-migrate-parameter-names --module-array resources/m_module_array.json \\\n"
    "      --module-library-dir ../circulatory-autogen-modules/modules\n"
    "  cuflynx-migrate-parameter-names --supermodule-config \\\n"
    "      modules/heart/versions/v1/heart_v1_modules_config.json --module-library-dir modules\n"
    "\n"
    "--dry-run prints the changes without writing them."
)


def _variables(entry):
    '''``{variable: kind}`` of a component entry.'''
    return {v[0]: v[3] for v in entry.get('variables_and_units') or [] if len(v) > 3}


def _candidates(name, places, variables_of):
    '''The readings of ``name`` as ``{var}_{place}`` whose ``var`` is a non-global variable of
    that place: ``[(written place, var)]``. ``places`` is ``{joined name: written}``.'''
    found = []
    for joined, written in places.items():
        suffix = '_' + joined
        if not name.endswith(suffix) or len(name) <= len(suffix):
            continue
        variable = name[:-len(suffix)]
        kind = variables_of(joined).get(variable)
        if kind is not None and kind != 'global_constant':
            found.append((written, variable))
    return found


def _rewrite_rows(path, places, variables_of):
    '''``(new text, changes, ambiguous)`` for the parameters CSV at ``path``.'''
    with open(path, newline='', encoding='utf-8-sig') as f:
        text = f.read()
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if 'variable_name' not in fields:
        raise ValueError(f'{path}: no variable_name column')
    rows, changes, ambiguous = [], [], []
    for row in reader:
        name = (row.get('variable_name') or '').strip()
        if name and SEPARATOR not in name:
            found = _candidates(name, places, variables_of)
            if len(found) == 1:
                written, variable = found[0]
                new = f'{written}{SEPARATOR}{variable}'
                changes.append((name, new))
                row['variable_name'] = new
            elif len(found) > 1:
                ambiguous.append((name, [f'{w}{SEPARATOR}{v}' for w, v in found]))
        rows.append(row)
    newline = '\r\n' if '\r\n' in text else '\n'
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator=newline,
                            extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue(), changes, ambiguous


def _sources(args):
    return ModuleSources({'module_library_dirs': args.module_library_dirs or None,
                          'external_modules_dir': args.external_modules_dir,
                          'use_builtin_modules': not args.no_builtin_modules}).config_files


def host_places(module_array, config_files):
    '''``({joined vessel name: written}, variables_of)`` for a host file's module array.'''
    supermodules = load_supermodule_registry(config_files)
    components = load_component_registry(config_files)
    written = {}
    for record in read_module_array_records(module_array):
        entry = supermodules.get((record['vessel_type'], record['BC_type']))
        if entry is None:
            written[record['name']] = record['name']
            continue
        for parts in submodule_path_parts(entry, supermodules):
            written['_'.join((record['name'],) + parts)] = SEPARATOR.join(
                (record['name'],) + parts)
    records, _ = load_expanded_vessel_records(module_array, supermodules, components)
    variables = {}
    for record in records:
        entry = components.get((record['vessel_type'], record['BC_type']))
        variables[record['name']] = _variables(entry) if entry else {}
        written.setdefault(record['name'], record['name'])
    places = {name: written[name] for name in variables}
    return places, (lambda name: variables.get(name, {}))


def supermodule_places(entry, config_files):
    '''``({joined submodule path: written}, variables_of)`` for a supermodule's rows.'''
    supermodules = load_supermodule_registry(config_files)
    components = load_component_registry(config_files)
    variables, places = {}, {}
    for parts in submodule_path_parts(entry, supermodules):
        container, leaf = entry, None
        for part in parts:
            leaf = next(s for s in container['submodules'] if s['name'] == part)
            container = supermodules.get((leaf['vessel_type'], leaf['BC_type']), {})
        component = components.get((leaf['vessel_type'], leaf['BC_type']))
        joined = '_'.join(parts)
        variables[joined] = _variables(component) if component else {}
        places[joined] = SEPARATOR.join(parts)
    return places, (lambda name: variables.get(name, {}))


def supermodule_files(config_path):
    '''The supermodule entry in ``config_path`` and the parameters files to rewrite.'''
    entries = [e for e in load_module_config(config_path, include_supermodules=True)
               if is_supermodule_entry(e)]
    if len(entries) != 1:
        raise ValueError(f'{config_path} holds {len(entries)} supermodule entries; give a config '
                         f'with exactly one.')
    entry = dict(entries[0], config_path=os.path.abspath(config_path))
    files = [instance_parameters_path(config_path, i) for i in available_instances(config_path)]
    if entry.get('default_parameters'):
        files.append(os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                  entry['default_parameters']))
    return entry, files


def build_parser():
    parser = argparse.ArgumentParser(
        description='Rewrite {variable}_{vessel} parameter names as <vessel>/<variable>, in a '
                    'host parameters file or in a supermodule\'s instance and default '
                    'parameters files.',
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument('--module-array', help='the host model\'s module array')
    kind.add_argument('--supermodule-config', help='a supermodule\'s *_modules_config.json')
    parser.add_argument('--parameters', help='the host parameters file (default: '
                                             '<prefix>_parameters.csv beside the module array)')
    parser.add_argument('--module-library-dir', dest='module_library_dirs', action='append',
                        default=[], metavar='DIR', help='a module library (repeatable)')
    parser.add_argument('--external-modules-dir', default=None, metavar='DIR')
    parser.add_argument('--no-builtin-modules', action='store_true',
                        help='leave libcuflynx\'s built-in modules out (as a module library '
                             'with its own copies of them does)')
    parser.add_argument('--dry-run', action='store_true', help='print, do not write')
    return parser


def run(args):
    config_files = _sources(args)
    if args.module_array:
        places, variables_of = host_places(args.module_array, config_files)
        files = [args.parameters or _host_parameters(args.module_array)]
    else:
        if not args.module_library_dirs and not args.external_modules_dir:
            raise SystemExit('--supermodule-config needs --module-library-dir (the library its '
                             'submodules come from), to check each name against them.')
        entry, files = supermodule_files(args.supermodule_config)
        places, variables_of = supermodule_places(entry, config_files)
    status = 0
    for path in files:
        text, changes, ambiguous = _rewrite_rows(path, places, variables_of)
        print(f'{path}: {len(changes)} name(s) rewritten'
              + (f', {len(ambiguous)} left for you' if ambiguous else ''))
        for old, new in changes:
            print(f'    {old} -> {new}')
        for old, readings in ambiguous:
            print(f'    {old}: could be {" or ".join(readings)}; left unchanged')
            status = 1
        if changes and not args.dry_run:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                f.write(text)
    return status


def _host_parameters(module_array):
    directory, name = os.path.split(module_array)
    for suffix in ('_module_array.json', '_module_array.csv', '_vessel_array.json',
                   '_vessel_array.csv'):
        if name.endswith(suffix):
            return os.path.join(directory, name[:-len(suffix)] + '_parameters.csv')
    raise ValueError(f'{module_array}: not a <prefix>_module_array file; give --parameters')


def main(argv=None):
    """Entry point for the ``cuflynx-migrate-parameter-names`` command."""
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
