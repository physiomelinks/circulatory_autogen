"""``cuflynx-run-predictions``: an obs_data's prediction items on a model, without a calibration.

    cuflynx-run-predictions MODEL OBS_DATA --dt 0.1 [--solver CVODE_myokit] [--model-type cellml]
        [--solver-info '{"MaximumStep": 0.01}'] [--operation-funcs my_funcs.py]
        [--params params.csv] [--output-dir predictions]

Writes to the output directory:

- ``traces.npz``: per item without an operation, ``<data_item_name>`` = rows ``[time, values]``
  over its segment, time from the end of the experiment's warm-up;
- ``features.csv``: ``data_item_name,value`` per scalar feature item;
- ``validation_results.json``: the items that carry a value against the model;
- ``<name>.png`` per ``prediction_plots`` entry of the obs_data.

``--params`` is a CSV of ``name,value`` rows (``vessel/param`` names) applied at the start of
each experiment, for example a calibrated parameter set.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        prog='cuflynx-run-predictions',
        description="Run an obs_data's protocol on a model and evaluate its prediction items "
                    "(traces, features, validation), with no calibration.",
        epilog='Configured by these arguments alone; user_inputs.yaml is not read.')
    parser.add_argument('model', help='the model file (CellML, or a generated Python model)')
    parser.add_argument('obs_data', help='the obs_data.json whose prediction_items to run')
    parser.add_argument('--dt', type=float, required=True, help='output sampling step')
    parser.add_argument('--solver', default='CVODE_myokit')
    parser.add_argument('--model-type', default=None,
                        help='needed for a model that is not CellML (python, casadi_python ...)')
    parser.add_argument('--solver-info', default='{}', help='solver_info as a JSON object')
    parser.add_argument('--operation-funcs', default=None,
                        help='a file of user operation funcs, as operation_funcs_external_path')
    parser.add_argument('--params', default=None,
                        help='CSV of name,value rows applied at the start of each experiment')
    parser.add_argument('--output-dir', default='predictions')
    return parser


def read_params(path):
    names, values = [], []
    with open(path, newline='') as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().startswith('#') or row[0].strip() == 'name':
                continue
            names.append(row[0].strip())
            values.append(float(row[1]))
    return names, values


def write_outputs(result, output_dir):
    """traces.npz, features.csv and validation_results.json; returns their paths."""
    from libcuflynx.param_id import validation

    os.makedirs(output_dir, exist_ok=True)
    paths = [os.path.join(output_dir, 'traces.npz'), os.path.join(output_dir, 'features.csv')]
    np.savez(paths[0], **{name: np.vstack([t, values])
                          for name, (t, values) in result.traces.items()})
    with open(paths[1], 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['data_item_name', 'value'])
        for name, value in result.features.items():
            writer.writerow([name, repr(float(value))])
    written = validation.write_validation_results(result.validation, output_dir)
    return paths + ([written] if written else [])


def run(args):
    from libcuflynx.protocol_runners import run_prediction_items

    names, values = read_params(args.params) if args.params else (None, None)
    result = run_prediction_items(
        args.model, args.obs_data, args.dt, json.loads(args.solver_info), solver=args.solver,
        operation_funcs_path=args.operation_funcs, param_names=names, param_vals=values,
        model_type=args.model_type, plot_dir=args.output_dir)
    for path in write_outputs(result, args.output_dir) + result.plots:
        print(f'wrote {path}')
    return 0


def main(argv=None):
    """Entry point for the ``cuflynx-run-predictions`` command."""
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
