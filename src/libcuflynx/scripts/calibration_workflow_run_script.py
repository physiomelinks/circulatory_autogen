'''
``cuflynx-calibration-workflow``: run a calibration workflow (a calibration_workflow.json)
with one command -- each step's calibration in order, then the merge into the target.

See ``libcuflynx.calibration_workflow`` and the "Calibration workflows" section of
tutorial/docs/parameter-identification.md.
'''

import argparse
import json
import sys

from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI

MPI = _get_MPI()
from libcuflynx.scripts import _cli  # noqa: E402

EPILOG = (
    "Unlike the other stages, this command does not read user_inputs.yaml: each step names\n"
    "its model, obs_data and params_for_id in the workflow file, and its settings come from\n"
    "the file's \"settings\" (and the step's own).\n"
    "\n"
    "Outputs go to --output-dir (default ./workflow_output/<workflow_name>): one directory\n"
    "per step and workflow_result.json with the merged parameter set.\n"
    "\n"
    "Run under a launcher to use more than one rank, e.g.\n"
    "`mpiexec -n 4 cuflynx-calibration-workflow <workflow.json>`; every step's calibration\n"
    "then uses all ranks."
)


def build_parser():
    # not _cli.build_parser: its --user-inputs option means nothing here
    parser = argparse.ArgumentParser(
        description='Run the steps of a calibration_workflow.json in order -- each '
                    'a calibration, sensitivity analysis, reduction, emulator training or '
                    'identifiability analysis of a model, an obs_data and a params_for_id, '
                    'later steps taking earlier steps\' values, emulators or chosen '
                    'parameters -- and merge the calibrations into the target model.',
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('workflow', help='the calibration_workflow.json to run')
    parser.add_argument('--module-library-dir', dest='module_library_dirs', action='append',
                        default=[], metavar='DIR',
                        help='a module library to find the library modules in '
                             '(repeatable; added to the file\'s own module_library_dirs)')
    parser.add_argument('--output-dir', default=None, metavar='DIR',
                        help='where to write the run (default ./workflow_output/<name>)')
    which = parser.add_mutually_exclusive_group()
    which.add_argument('--from-step', default=None, metavar='ID',
                       help='run this step and the ones after it, reusing earlier results')
    which.add_argument('--only', default=None, metavar='ID',
                       help='run only this step, reusing the results it depends on')
    parser.add_argument('--dry-run', action='store_true',
                        help='check the workflow against the library and print the plan; '
                             'run nothing')
    parser.add_argument('--write-calibrated', action='store_true',
                        help='also write the merged set beside the target\'s parameters '
                             'file, as <name>_calibrated_parameters.json')
    parser.add_argument('--pack', default=None, metavar='DIR',
                        help='copy every file the workflow needs to run (its inputs, module '
                             'versions and settings files, none of its results) into the new '
                             'directory DIR, with the workflow rewritten to them; run nothing')
    return parser


def _model_label(model):
    if model is None:
        return 'no model'
    if 'module_array' in model:
        return model['module_array']
    return f"{model['module_type']}/{model['version']}/{model['parameterisation']}"


def _print_progress(event):
    kind = event['event']
    if kind == 'step_started':
        print(f"[workflow] step {event['index'] + 1}/{event['total']}: {event['step']} "
              f"({event.get('kind', 'calibration')}, {_model_label(event['model'])})",
              flush=True)
    elif kind == 'step_finished':
        cost = (f"best cost {event['best_cost']:.6g} " if event.get('best_cost') is not None
                else '')
        print(f"[workflow] step {event['step']} done: {cost}in {event['duration_s']:.1f} s",
              flush=True)
    elif kind == 'step_failed':
        print(f"[workflow] step {event['step']} FAILED: {event['error']}", flush=True)
    elif kind == 'step_skipped':
        print(f"[workflow] step {event['step']}: reusing its earlier result", flush=True)


def run(args):
    from libcuflynx.calibration_workflow import pack_workflow, run_calibration_workflow
    if args.pack:
        if MPI.COMM_WORLD.Get_rank() != 0:
            return None
        packed = pack_workflow(args.workflow, args.pack,
                               module_library_dirs=args.module_library_dirs)
        print(f"[workflow] packed {len(packed['files'])} files ({len(packed['modules'])} "
              f"module versions) into {args.pack}; run {packed['workflow']}")
        return packed
    result = run_calibration_workflow(
        args.workflow, output_dir=args.output_dir, module_library_dirs=args.module_library_dirs,
        from_step=args.from_step, only=args.only, dry_run=args.dry_run,
        write_calibrated=args.write_calibrated, progress=_print_progress)
    if MPI.COMM_WORLD.Get_rank() != 0:
        return result
    if args.dry_run:
        print(json.dumps(result, indent=1))
        return result
    print(f"[workflow] {result['workflow_name']}: "
          f"{'complete' if result['complete'] else 'partial'}; merged parameters:")
    for row in result['merged_parameters']:
        print(f"    {row['parameterisation_name']:<32} {row['value']:<.10g}   "
              f"(step {row['from_step']})")
    if result.get('calibrated_parameters_file'):
        print(f"[workflow] wrote {result['calibrated_parameters_file']}")
    return result


def main(argv=None):
    """Entry point for the ``cuflynx-calibration-workflow`` command."""
    args = build_parser().parse_args(argv)
    return _cli.run_stage(lambda: run(args), MPI)


if __name__ == '__main__':
    sys.exit(main())
