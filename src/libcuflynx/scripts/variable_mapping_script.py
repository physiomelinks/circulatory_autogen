'''
``cuflynx-variable-mapping``: write (or refresh) a model's ``<prefix>_variable_mapping.csv``,
the LaTeX symbol of every variable and parameter, for ``cuflynx-methods-latex``.

Rerunning keeps every symbol already in the file, adds new variables with the default rule
and drops variables the model no longer has.
'''

import argparse
import sys

EPILOG = (
    "Does not read user_inputs.yaml: give the CellML model (a generated *_flat.cellml or any\n"
    "CellML), or a calibration_workflow.json, whose target model is used and whose mapping\n"
    "goes in the target instance's directory by default.\n"
    "\n"
    "Columns: variable_name (component/variable, as obs_data and CUFLynx name it), latex\n"
    "(edit these), kind, component, units."
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Write a model's variable mapping: every variable and parameter with the "
                    "LaTeX symbol it is written as in generated methods.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('model', nargs='?', help='a CellML model')
    source.add_argument('--workflow', help="a calibration_workflow.json (maps its target model)")
    parser.add_argument('-o', '--output', help='where to write the mapping (default: beside '
                                               'the model, <prefix>_variable_mapping.csv)')
    parser.add_argument('--module-library-dir', dest='module_library_dirs', action='append',
                        default=[], metavar='DIR', help='module library (with --workflow)')
    parser.add_argument('--run-dir', help="the workflow's run directory (with --workflow)")
    return parser


def run(args):
    from libcuflynx.reporting.variable_mapping import (check_variable_mapping,
                                                       default_mapping_path,
                                                       update_variable_mapping_file)
    if args.workflow:
        from libcuflynx.reporting.methods import update_workflow_variable_mapping
        path, rows = update_workflow_variable_mapping(args.workflow, args.output,
                                                      args.module_library_dirs, args.run_dir)
    else:
        path = args.output or default_mapping_path(args.model)
        rows = update_variable_mapping_file(args.model, path)
    problems = check_variable_mapping(rows)
    edited = sum(1 for r in rows if r.get('edited'))
    print(f'wrote {path}: {len(rows)} variables ({edited} with edited symbols)')
    for latex, names in problems['duplicates'].items():
        print(f'  warning: {", ".join(names)} are all written {latex}')
    for name in problems['empty']:
        print(f'  warning: {name} has no symbol')
    return 0


def main(argv=None):
    """Entry point for the ``cuflynx-variable-mapping`` command."""
    args = build_parser().parse_args(argv)
    if args.model is None and not args.workflow:
        build_parser().error('give a model or --workflow')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
