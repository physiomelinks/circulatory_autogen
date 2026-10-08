'''
``cuflynx-methods-latex``: write a methods section in LaTeX -- the model's equations module
by module, a flow chart of the calibration, and what was done in each of its steps.
'''

import argparse
import os
import shutil
import subprocess
import sys

EPILOG = (
    "Two sources:\n"
    "  --workflow calibration_workflow.json [--run-dir DIR]   a calibration workflow\n"
    "  --model M.cellml --obs-data O.json --params-for-id P.csv [--settings user_inputs.yaml]\n"
    "                                                          one calibration; its method,\n"
    "                                                          solver and UQ settings are read\n"
    "                                                          from the user_inputs.yaml given\n"
    "With a run directory the calibrated values are reported too. Symbols come from the\n"
    "variable mapping (cuflynx-variable-mapping); without one, from the default rule.\n"
    "\n"
    "The output is a complete document (compile twice, or with latexmk, for the figure\n"
    "reference); --body-only writes just the sections, to \\input into a paper."
)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Write the methods of a calibration in LaTeX: the model equations per '
                    'module, a flow chart of the calibration steps, and the methods of each.',
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workflow', help='a calibration_workflow.json')
    parser.add_argument('--module-library-dir', dest='module_library_dirs', action='append',
                        default=[], metavar='DIR', help='module library (with --workflow)')
    parser.add_argument('--model', help='a CellML model (one calibration)')
    parser.add_argument('--obs-data', help='its obs_data.json')
    parser.add_argument('--params-for-id', help='its params_for_id (.csv or .json)')
    parser.add_argument('--settings', metavar='YAML',
                        help='a user_inputs.yaml with the param_id settings that were used')
    parser.add_argument('--run-dir', help="a workflow's output directory, or a param_id run "
                                          'directory, to report results from')
    parser.add_argument('--mapping', help='the variable mapping CSV')
    parser.add_argument('-o', '--output', required=True, help='the .tex to write')
    parser.add_argument('--title', help='document title')
    parser.add_argument('--body-only', action='store_true',
                        help='no preamble: sections to \\input into another document')
    parser.add_argument('--flowchart-only', action='store_true',
                        help='only the flow chart (with --workflow)')
    parser.add_argument('--pdf', action='store_true', help='also compile it with latexmk')
    return parser


def _settings(path):
    if not path:
        return {}
    import yaml
    with open(path) as f:
        inputs = yaml.safe_load(f) or {}
    return inputs


def run(args, parser):
    from libcuflynx.reporting.methods import (calibration_methods_latex,
                                              workflow_methods_latex, write_latex)
    standalone = not args.body_only
    if args.workflow:
        text = workflow_methods_latex(args.workflow, run_dir=args.run_dir, mapping=args.mapping,
                                      module_library_dirs=args.module_library_dirs,
                                      standalone=standalone, title=args.title,
                                      flowchart_only=args.flowchart_only)
    else:
        if not (args.model and args.obs_data and args.params_for_id):
            parser.error('give --workflow, or --model, --obs-data and --params-for-id')
        inputs = _settings(args.settings)
        text = calibration_methods_latex(
            args.model, args.obs_data, args.params_for_id, settings=inputs,
            run_dir=args.run_dir, mapping=args.mapping, standalone=standalone,
            title=args.title, do_uq=bool(inputs.get('do_uq')), do_ia=bool(inputs.get('do_ia')),
            use_emulator=bool(inputs.get('use_emulator')))
    path = write_latex(text, args.output)
    print(f'wrote {path}')
    if args.pdf:
        if not standalone:
            parser.error('--pdf needs a complete document (drop --body-only)')
        latexmk = shutil.which('latexmk')
        if latexmk is None:
            print('latexmk not found; compile the .tex yourself')
            return 1
        directory = os.path.dirname(os.path.abspath(path))
        done = subprocess.run([latexmk, '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                               os.path.basename(path)], cwd=directory,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if done.returncode != 0:
            print(done.stdout[-3000:])
            return done.returncode
        print(f'wrote {os.path.splitext(path)[0]}.pdf')
    return 0


def main(argv=None):
    """Entry point for the ``cuflynx-methods-latex`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args, parser)


if __name__ == '__main__':
    sys.exit(main())
