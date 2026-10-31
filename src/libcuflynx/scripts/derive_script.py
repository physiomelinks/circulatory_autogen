'''
``cuflynx-derive``: evaluate a parameter derivations file and compare each derived parameter
with the value cached in the parameters file; ``--write`` updates the cached values.

Exits 0 when every cached value matches its derivation (or after ``--write``), 1 when one does
not.
'''

import argparse
import sys

EPILOG = (
    "Does not read user_inputs.yaml: give the derivations file, the obs_data whose input_items\n"
    "and data_items it reads, and the parameters file it sets (default: <name>_parameters.json\n"
    "beside <name>_derivations.json).\n"
    "\n"
    "  cuflynx-derive BVC_derivations.json --obs BVC_obs_data.json\n"
    "  cuflynx-derive BVC_derivations.json --obs BVC_obs_data.json --write"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate parameter derivations (CellML models or Python functions of '
                    'experimental inputs) and check or update the cached parameter values.',
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('derivations', help='a <name>_derivations.json')
    parser.add_argument('--obs', action='append', default=[], metavar='OBS_DATA',
                        help='an obs_data file whose input_items and data_items the derivations '
                             'read (repeat for several)')
    parser.add_argument('--parameters', help='the parameters file the derivations set '
                                             '(default: beside the derivations file)')
    parser.add_argument('--write', action='store_true',
                        help='write the derived values into the parameters file')
    return parser


def run(args):
    from libcuflynx.derivations import (CACHE_RTOL, check_derivations, evaluate_derivations,
                                        load_derivations, parameters_file_for, read_parameters,
                                        write_derived_values)
    parameters = args.parameters or parameters_file_for(args.derivations)
    doc = read_parameters(parameters)
    results = evaluate_derivations(load_derivations(args.derivations), doc, args.obs)
    for result in results:
        print(f'{result.derivation.name} ({result.derivation.method})')
        for symbol, out in result.outputs.items():
            print(f"  {out['parameter']}: cached {out['cached']}, derived {out['value']!r} "
                  f"{out['units'] or ''}".rstrip())
    if args.write:
        changed = write_derived_values(parameters, results)
        print(f'wrote {parameters}: {changed} value(s) changed')
        return 0
    problems = check_derivations(results, doc)
    for problem in problems:
        print(f'  {problem}')
    if problems:
        print(f'{len(problems)} problem(s): a cached value must match its derivation within '
              f'rtol {CACHE_RTOL}; rerun with --write to update {parameters}.')
        return 1
    print(f'every cached value in {parameters} matches its derivation.')
    return 0


def main(argv=None):
    """Entry point for the ``cuflynx-derive`` command."""
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
