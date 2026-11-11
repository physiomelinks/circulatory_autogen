"""Coverage and speed of casadi_python + casadi_integrator/cvodes against CVODE_myokit.

For every model in resources/ that generates as CellML (a ``<prefix>_module_array.csv`` with a
``<prefix>_parameters.json``, plus the standalone ``*.cellml`` files there), this:

1. generates the CellML and runs it on CVODE_myokit, the reference;
2. generates the casadi_python module (PythonGenerator, casadi_compat);
3. runs it on casadi_integrator / cvodes with jit off and with jit on;

over the same timeline from the model's initial state (casadi at reltol 1e-8 and ``--abstol``,
Myokit at rtol = atol = 1e-8, both with a 1 ms maximum step), and records whether each step worked
(the error if not), the wall times (generation, helper load, first run -- which for jit includes
compiling -- and a repeat run) and the worst state error against Myokit:
``max over states of max_t |casadi - myokit| / max_t |myokit|``.

Each model runs in its own subprocess with a time limit, so one hang or crash does not stop the
rest. This is a report, not a test: it decides nothing by itself.

Run (from the repository root, in a venv with casadi and myokit):

    OMP_NUM_THREADS=1 python benchmarks/casadi_coverage.py --workers 4 \
        --out benchmarks/results/casadi_coverage.json --markdown benchmarks/results/casadi_coverage.md

``--models 3compartment SN_simple`` runs a subset; ``--model-file path.cellml`` adds a CellML file
from elsewhere (e.g. a large model kept outside the repository).
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES = os.path.join(REPO, 'resources')


def resource_models():
    """Model names in resources/: module arrays with parameters, and standalone CellML files."""
    names = []
    for f in sorted(os.listdir(RESOURCES)):
        if f.endswith('_module_array.csv'):
            prefix = f[:-len('_module_array.csv')]
            if os.path.exists(os.path.join(RESOURCES, f'{prefix}_parameters.json')):
                names.append(prefix)
        elif f.endswith('.cellml'):
            names.append(os.path.join(RESOURCES, f))
    return names


def _generate_cellml(name, workdir):
    if name.endswith('.cellml'):
        return name
    import yaml
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    with open(os.path.join(REPO, 'user_run_files', 'user_inputs.yaml')) as f:
        cfg = yaml.safe_load(f)
    for key in ('user_inputs_path_override', 'param_id_output_dir', 'param_id_obs_path'):
        cfg.pop(key, None)
    # a copy of the model's resources, since generation writes <name>_parameters_unfinished.json
    # next to them when parameters are missing
    resources = os.path.join(workdir, 'resources')
    os.makedirs(resources, exist_ok=True)
    for f in os.listdir(RESOURCES):
        if f.startswith(name):
            shutil.copy(os.path.join(RESOURCES, f), resources)
    cfg.update(dict(DEBUG=True, file_prefix=name, input_param_file=f'{name}_parameters.json',
                    model_type='cellml', solver='CVODE_myokit', pre_time=0.0, sim_time=0.1,
                    dt=0.01, resources_dir=resources, generated_models_dir=workdir,
                    solver_info={'MaximumStep': 0.001}))
    if not generate_with_new_architecture(False, cfg):
        raise RuntimeError('CellML generation failed')
    return os.path.join(workdir, name, f'{name}.cellml')


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def _states(helper, names):
    out = {}
    for name in names:
        try:
            out[name] = np.asarray(helper.get_results([name], flatten=True)[0],
                                   dtype=float).ravel()
        except Exception:
            pass
    return out


def _worst_error(reference, other):
    worst, worst_name = 0.0, None
    for name, ref in reference.items():
        if name not in other:
            continue
        n = min(len(ref), len(other[name]))
        scale = max(float(np.max(np.abs(ref[:n]))), 1e-12)
        err = float(np.max(np.abs(other[name][:n] - ref[:n]))) / scale
        if not np.isfinite(err):
            return float('inf'), name
        if err > worst:
            worst, worst_name = err, name
    return worst, worst_name


def run_one(name, sim_time, dt, jit_modes, abstol=1e-10):
    """One model's row of the report."""
    from libcuflynx.generators.PythonGenerator import PythonGenerator
    from libcuflynx.solver_wrappers import get_simulation_helper

    row = {'model': os.path.basename(name).replace('.cellml', ''), 'sim_time': sim_time,
           'dt': dt, 'abstol': abstol}
    workdir = tempfile.mkdtemp(prefix=f'casadi_cov_{row["model"]}_')
    step = 'generate CellML'
    try:
        cellml, row['cellml_generate_s'] = _timed(lambda: _generate_cellml(name, workdir))
        step = 'CVODE_myokit'
        myokit, row['myokit_load_s'] = _timed(lambda: get_simulation_helper(
            model_path=cellml, model_type='cellml', solver='CVODE_myokit', dt=dt,
            sim_time=sim_time, pre_time=0.0,
            solver_info={'MaximumStep': 0.001, 'rtol': 1e-8, 'atol': 1e-8}))
        _, row['myokit_first_run_s'] = _timed(myokit.run)
        step = 'casadi generation'
        module, row['casadi_generate_s'] = _timed(lambda: PythonGenerator(
            cellml, output_dir=workdir, module_name=f'{row["model"]}_casadi',
            casadi_compat=True).generate())
        row['casadi_generates'] = True
    except Exception as exc:
        row['failed_at'] = step
        row['error'] = f'{type(exc).__name__}: {exc}'[:400]
        return row

    reference = None
    for jit in jit_modes:
        tag = 'jit' if jit else 'nojit'
        try:
            helper, row[f'{tag}_load_s'] = _timed(lambda: get_simulation_helper(
                model_path=module, model_type='casadi_python', solver='casadi_integrator',
                dt=dt, sim_time=sim_time, pre_time=0.0,
                solver_info={'method': 'cvodes', 'max_step_size': 0.001,
                             'max_num_steps': 100000, 'abstol': abstol, 'jit': jit}))
            _, row[f'{tag}_first_run_s'] = _timed(helper.run)
            casadi_states = _states(helper, list(helper.state_name_to_idx))
            _, row[f'{tag}_repeat_run_s'] = _timed(helper.run)
            if reference is None:
                # read before Myokit's repeat run, which continues from where the first stopped
                reference = _states(myokit, list(casadi_states))
            err, state = _worst_error(reference, casadi_states)
            row[f'{tag}_max_rel_error'] = err
            row[f'{tag}_worst_state'] = state
            row[f'{tag}_states_compared'] = len(reference)
            row[f'{tag}_simulates'] = True
        except Exception as exc:
            row[f'{tag}_simulates'] = False
            row[f'{tag}_error'] = f'{type(exc).__name__}: {exc}'[-400:]
    row['n_states'] = len(reference) if reference else None
    try:
        _, row['myokit_repeat_run_s'] = _timed(myokit.run)
    except Exception:
        pass
    return row


def _child(args):
    try:
        row = run_one(args.one, args.sim_time, args.dt, [False] + ([True] if args.jit else []),
                      abstol=args.abstol)
    except Exception:
        row = {'model': args.one, 'error': traceback.format_exc()[-400:]}
    print('CASADI_COVERAGE_ROW ' + json.dumps(row), flush=True)


def _spawn(name, args):
    cmd = [sys.executable, os.path.abspath(__file__), '--one', name, '--sim-time',
           str(args.sim_time), '--dt', str(args.dt), '--abstol', str(args.abstol)] + (
               [] if args.jit else ['--no-jit'])
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '1'))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        return {'model': os.path.basename(name), 'error': f'timed out after {args.timeout} s'}
    for line in proc.stdout.splitlines():
        if line.startswith('CASADI_COVERAGE_ROW '):
            return json.loads(line[len('CASADI_COVERAGE_ROW '):])
    return {'model': os.path.basename(name),
            'error': f'exit {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}'}


def _fmt(value, digits=2):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.{digits}g}' if abs(value) < 1e-2 or abs(value) >= 1e4 else \
            f'{value:.{digits}f}'
    return str(value)


def markdown(rows):
    head = ('| model | states | casadi gen (s) | Myokit run (s) | cvodes first / repeat (s) | '
            'cvodes err | jit first / repeat (s) | jit err | notes |')
    lines = [head, '|' + '---|' * 9]
    for r in rows:
        notes = r.get('error') or r.get('nojit_error') or r.get('jit_error') or ''
        if r.get('failed_at'):
            notes = f"fails at {r['failed_at']}: {notes}"
        notes = notes.replace('|', '/').replace('\n', ' ')[:160]
        lines.append('| ' + ' | '.join([
            r.get('model', ''), _fmt(r.get('n_states')), _fmt(r.get('casadi_generate_s')),
            _fmt(r.get('myokit_repeat_run_s'), 3),
            f"{_fmt(r.get('nojit_first_run_s'), 3)} / {_fmt(r.get('nojit_repeat_run_s'), 3)}"
            if r.get('nojit_simulates') else 'no',
            _fmt(r.get('nojit_max_rel_error')),
            f"{_fmt(r.get('jit_first_run_s'), 3)} / {_fmt(r.get('jit_repeat_run_s'), 3)}"
            if r.get('jit_simulates') else ('no' if 'jit_simulates' in r else ''),
            _fmt(r.get('jit_max_rel_error')), notes]) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--models', nargs='*', help='resource model names (default: all)')
    parser.add_argument('--model-file', action='append', default=[],
                        help='extra CellML file to include')
    parser.add_argument('--sim-time', type=float, default=2.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--abstol', type=float, default=1e-10,
                        help="casadi_integrator abstol (CA's default 1e-10; Myokit runs at 1e-8)")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--no-jit', dest='jit', action='store_false')
    parser.add_argument('--out', help='JSON results file')
    parser.add_argument('--markdown', help='Markdown table file')
    parser.add_argument('--one', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        return _child(args)

    names = args.models or resource_models()
    names = [n if n.endswith('.cellml') else n for n in names] + args.model_file
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_spawn, n, args): n for n in names}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda r: r.get('model', ''))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=1)
    table = markdown(rows)
    if args.markdown:
        with open(args.markdown, 'w') as f:
            f.write(table)
    print(table)


if __name__ == '__main__':
    main()
