'''
``cuflynx-derive``: evaluate a parameter derivations file and compare each derived parameter
with the value cached in the parameters file; ``--write`` updates the cached values.

Exits 0 when every cached value matches its derivation (or after ``--write``), 1 when one does
not (or, after ``--write``, when a parameter names a derivation that does not set it), and 2 with
the message when a file or derivation cannot be read or evaluated. A calibrated parameter (origin
``calibrated``) is printed with its derived value but neither checked nor written.
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
    from libcuflynx.derivations import (CACHE_RTOL, calibrated_outputs, check_derivations,
                                        evaluate_derivations, load_derivations,
                                        parameters_file_for, read_parameters,
                                        write_derived_values)
    parameters = args.parameters or parameters_file_for(args.derivations)
    doc = read_parameters(parameters)
    results = evaluate_derivations(load_derivations(args.derivations), doc, args.obs)
    calibrated = {id(out) for _, out in calibrated_outputs(results, doc)}
    for result in results:
        print(f'{result.derivation.name} ({result.derivation.method})')
        for name, fixed in result.fixed_constants.items():
            print(f"  fixed constant {name} = {fixed['value']!r} {fixed['units']}".rstrip()
                  + ' (from the CellML file)')
        for symbol, out in result.outputs.items():
            held = 'calibrated' if id(out) in calibrated else 'cached'
            note = ' (calibrated: neither checked nor written)' if id(out) in calibrated else ''
            print(f"  {out['parameter']}: {held} {out['cached']}, derived {out['value']!r} "
                  f"{out['units'] or ''}".rstrip() + note)
    if args.write:
        changed = write_derived_values(parameters, results)
        print(f'wrote {parameters}: {changed} value(s) changed')
        for result in results:
            for out in result.outputs.values():
                out['cached'] = out['value']
        # what --write cannot fix: a parameter naming a derivation that does not set it
        problems = check_derivations(results, read_parameters(parameters))
        for problem in problems:
            print(f'  {problem}')
        if problems:
            print(f'{len(problems)} problem(s) --write cannot fix: remove the derivation key, '
                  f'or add the derivation that sets the parameter.')
            return 1
        return 0
    problems = check_derivations(results, doc)
    for problem in problems:
        print(f'  {problem}')
    if problems:
        print(f'{len(problems)} problem(s): a cached value must match its derivation within '
              f'rtol {CACHE_RTOL}; rerun with --write to update {parameters} (a parameter '
              f'naming a derivation that does not set it needs its derivation key removed).')
        return 1
    skipped = (f' ({len(calibrated)} calibrated value{"s" if len(calibrated) > 1 else ""} '
               f'not checked)' if calibrated else '')
    print(f'every cached value in {parameters} matches its derivation{skipped}.')
    return 0


def main(argv=None):
    """Entry point for the ``cuflynx-derive`` command."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as e:
        # a derivation, schema or file error: its message, not a traceback
        print(f'cuflynx-derive: error: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
