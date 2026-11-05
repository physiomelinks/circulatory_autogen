'''Run a generated C++ 0D model coupled to its external Python models.

    cuflynx-couple <generated model folder> [--t-end T] [--out DIR] [--check]

Each coupling step from t to t + dt (explicit, staggered):

1. every external model receives the 0D values it reads at t, advances to t + dt and returns
   the values it sets;
2. those are set in the 0D model (as libCellML external variables), which steps to t + dt.

With ``subiterations`` > 0 (in the api block) the step is repeated, from snapshots of both
models, until the external outputs change by less than ``tol`` (relative): each side then holds
the average of the other's values at the start and end of the step, a trapezoidal coupling that
is second order in dt (the explicit step is first order). Use it for stiff exchanges or when the
coupling step can't be small.

The 0D model writes its usual output files (sol0D_states.txt, sol0D_variables.txt) every
``dt`` of the user inputs from ``pre_time`` on; the steps are shortened to land on those times.
'''

import argparse
import importlib.util
import os
import sys
import time as _time
from dataclasses import dataclass, field

import numpy as np

from libcuflynx.coupling.model0d_lib import Model0dError, Model0dLibrary, load_info

TOL_TIME = 1e-12


@dataclass
class CoupledResult:
    '''What a coupled run returns (on every rank).'''
    times: np.ndarray                       # output times
    exchange: dict                          # {"row/variable": array (n_times, n_values)}
    records: dict                           # {"vessel/variable": array (n_times,)} of record=[...]
    timings: dict = field(default_factory=dict)  # wall-clock seconds: total, zero_d, external, setup
    n_steps: int = 0
    output_dir: str = None


def _get_comm(comm):
    if comm is not None:
        return comm
    from libcuflynx.utilities.mpi_utils import get_MPI
    MPI = get_MPI()
    return MPI.COMM_WORLD if MPI is not None else None


def _rank_size(comm):
    if comm is None:
        return 0, 1
    return comm.Get_rank(), comm.Get_size()


def load_external_class(path, class_name, row):
    '''The class ``class_name`` from the Python file ``path`` (loaded under a unique module name,
    so two models' files can both be called e.g. model.py).'''
    if not os.path.isfile(path):
        raise Model0dError(f"External model file {path} (row '{row}') not found.")
    module_name = f'cuflynx_external_{row}_{abs(hash(path))}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise Model0dError(f"{path} has no class {class_name!r} (row '{row}'; api.python.class).")
    return getattr(module, class_name)


class _External:
    '''One external model and its exchange variables.'''

    def __init__(self, entry, model0d, comm, output_dir):
        self.row = entry['row']
        self.entry = entry
        self.variables = entry['variables']
        self.inputs = [v for v in self.variables if v['direction'] == 'to_external']
        self.outputs = [v for v in self.variables if v['direction'] == 'from_external']
        self.model0d = model0d
        cls = load_external_class(entry['file'], entry['class'], self.row)
        info = {
            'row': self.row,
            'units': {v['variable']: v['units'] for v in self.variables},
            'directions': {v['variable']: v['direction'] for v in self.variables},
            'coupling_dt': entry['coupling_dt'],
            'output_dir': output_dir,
        }
        neighbours = {v['variable']: list(v['neighbours']) for v in self.variables}
        self.obj = cls(params=dict(entry['parameters']), neighbours=neighbours, comm=comm, info=info)
        for method in ('initial_outputs', 'step'):
            if not callable(getattr(self.obj, method, None)):
                raise Model0dError(f"External model {entry['class']} (row '{self.row}') has no {method}() method.")
        if entry.get('subiterations', 0) > 0:
            for method in ('snapshot', 'restore'):
                if not callable(getattr(self.obj, method, None)):
                    raise Model0dError(f"Row '{self.row}' asks for subiterations, so {entry['class']} needs "
                                       f"snapshot() and restore().")

    def _name(self, v):
        return f"{self.row}/{v['variable']}"

    def read_inputs(self):
        return {v['variable']: self.model0d.get(self._name(v)) for v in self.inputs}

    def check_outputs(self, outputs, where):
        if not isinstance(outputs, dict):
            raise Model0dError(f"{self.entry['class']}.{where} must return a dict {{variable: values}}, "
                               f"got {type(outputs).__name__}")
        missing = [v['variable'] for v in self.outputs if v['variable'] not in outputs]
        if missing:
            raise Model0dError(f"{self.entry['class']}.{where} did not return {missing} (row '{self.row}' "
                               f"sets them in the 0D model).")
        out = {}
        for v in self.outputs:
            arr = np.asarray(outputs[v['variable']], dtype=float).reshape(-1)
            n = len(v['neighbours'])
            if arr.size == 1 and n > 1:
                arr = np.full(n, float(arr[0]))
            if arr.size != n:
                raise Model0dError(f"{self.entry['class']}.{where}: '{v['variable']}' needs {n} value(s), one per "
                                   f"connected module {v['neighbours']}, got {arr.size}.")
            if not np.all(np.isfinite(arr)):
                raise Model0dError(f"{self.entry['class']}.{where}: '{v['variable']}' is not finite: {arr}")
            out[v['variable']] = arr
        return out

    def set_outputs(self, outputs):
        for v in self.outputs:
            self.model0d.set(self._name(v), outputs[v['variable']])


def run_coupled(model_dir, t_end=None, output_dir=None, comm=None, solver=None, build=True,
                record=(), verbose=True):
    '''Run the generated model in ``model_dir`` coupled to its external Python models.

    t_end: end time (default pre_time + sim_time of the user inputs); output_dir: where the 0D
    output files go (default: as main0d); record: extra 0D states/variables ("vessel/variable")
    to return at the output times. Returns a CoupledResult; under MPI only rank 0 writes files.
    '''
    t_wall0 = _time.perf_counter()
    comm = _get_comm(comm)
    rank, size = _rank_size(comm)
    model_dir = os.path.abspath(model_dir)
    info = load_info(model_dir)

    # build once (rank 0), then everyone loads the library
    if rank == 0:
        lib = Model0dLibrary(model_dir, build=build, verbose=False)
    if comm is not None and size > 1:
        comm.Barrier()
    if rank != 0:
        lib = Model0dLibrary(model_dir, build=False)
    model = lib.create(solver)

    output_dir = os.path.abspath(output_dir or info.get('output_dir') or os.path.join(model_dir, 'coupled_outputs'))
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    externals = [_External(entry, model, comm, output_dir) for entry in info['external_models']]
    if not externals:
        raise Model0dError(f'{model_dir}: no external models in external_models.json')
    record_refs = {name: model.lookup(name) for name in record}

    pre_time = float(info.get('pre_time') or 0.0)
    sim_time = info.get('sim_time')
    if t_end is None:
        if sim_time is None:
            raise Model0dError('No sim_time in external_models.json: pass t_end.')
        t_end = pre_time + float(sim_time)
    dt_out = float(info['dt_output'])
    dt_couple = min(float(e.entry['coupling_dt']) for e in externals)
    subiterations = max(int(e.entry.get('subiterations', 0)) for e in externals)
    tol = min(float(e.entry.get('tol', 1e-8)) for e in externals)
    relaxation = min(float(e.entry.get('relaxation', 1.0)) for e in externals)

    if verbose and rank == 0:
        print(f"cuflynx-couple :: {info['model_name']}: {len(externals)} external model(s), coupling dt "
              f"{dt_couple:g} s, output every {dt_out:g} s from {pre_time:g} s to {t_end:g} s"
              + (f', up to {subiterations} subiterations' if subiterations else ''))

    timings = {'setup': _time.perf_counter() - t_wall0, 'zero_d': 0.0, 'external': 0.0}

    # initial values from the external models
    t0 = _time.perf_counter()
    current = {}
    for e in externals:
        current[e.row] = e.check_outputs(e.obj.initial_outputs(), 'initial_outputs()')
        e.set_outputs(current[e.row])
    timings['external'] += _time.perf_counter() - t0

    times, exch_trace, rec_trace = [], {x: [] for x in model.exchange}, {n: [] for n in record}

    def save(t):
        times.append(t)
        for name in model.exchange:
            exch_trace[name].append(model.get(name))
        for name, ref in record_refs.items():
            rec_trace[name].append(model.value(ref))
        if rank == 0:
            model.write_output()
        for e in externals:
            if callable(getattr(e.obj, 'write', None)):
                e.obj.write(t)

    if rank == 0:
        model.open_output(output_dir)
    n_out = 0
    next_out = pre_time
    t = model.time
    if t >= next_out - TOL_TIME:
        save(t)
        n_out += 1
        next_out = pre_time + n_out * dt_out
    n_steps = 0
    while t < t_end - TOL_TIME:
        dt = min(dt_couple, t_end - t)
        if next_out > t + TOL_TIME:
            dt = min(dt, next_out - t)
        inputs = {e.row: e.read_inputs() for e in externals}
        if subiterations == 0:
            t0 = _time.perf_counter()
            for e in externals:
                current[e.row] = e.check_outputs(e.obj.step(t, dt, inputs[e.row]), 'step()')
                e.set_outputs(current[e.row])
            t1 = _time.perf_counter()
            model.step(dt)
            timings['external'] += t1 - t0
            timings['zero_d'] += _time.perf_counter() - t1
        else:
            # Trapezoidal fixed point: each side holds the average of the other's values at the
            # start and the end of the step (second order in dt), iterated from snapshots.
            model.snapshot()
            for e in externals:
                e.obj.snapshot()
            inputs_start, inputs_use = inputs, inputs
            previous = None
            for it in range(subiterations + 1):
                if it > 0:
                    model.restore(keep=True)
                    for e in externals:
                        e.obj.restore()
                t0 = _time.perf_counter()
                outs = {}
                for e in externals:
                    o = e.check_outputs(e.obj.step(t, dt, inputs_use[e.row]), 'step()')
                    if previous is not None and relaxation < 1.0:
                        o = {k: relaxation * v + (1.0 - relaxation) * previous[e.row][k] for k, v in o.items()}
                    outs[e.row] = o
                    e.set_outputs({k: 0.5 * (current[e.row][k] + v) for k, v in o.items()})
                t1 = _time.perf_counter()
                model.step(dt)
                timings['external'] += t1 - t0
                timings['zero_d'] += _time.perf_counter() - t1
                # the 0D values at the end of the step, computed with the external values there
                for e in externals:
                    e.set_outputs(outs[e.row])
                inputs_end = {e.row: e.read_inputs() for e in externals}
                if previous is not None:
                    change = max([np.max(np.abs(outs[r][k] - previous[r][k]) / (np.abs(previous[r][k]) + 1e-30))
                                  for r in outs for k in outs[r]] or [0.0])
                    if change <= tol:
                        break
                previous = outs
                inputs_use = {r: {k: 0.5 * (inputs_start[r][k] + inputs_end[r][k]) for k in inputs_start[r]}
                              for r in inputs_start}
            model.drop_snapshot()
            # the values at the end of the step, for the outputs and the next step
            for e in externals:
                e.set_outputs(outs[e.row])
            current = outs
        t = model.time
        n_steps += 1
        if t >= next_out - TOL_TIME:
            save(t)
            n_out += 1
            next_out = pre_time + n_out * dt_out

    for e in externals:
        if callable(getattr(e.obj, 'close', None)):
            e.obj.close()
    model.close()
    timings['total'] = _time.perf_counter() - t_wall0
    result = CoupledResult(times=np.array(times),
                           exchange={k: np.array(v) for k, v in exch_trace.items()},
                           records={k: np.array(v) for k, v in rec_trace.items()},
                           timings=timings, n_steps=n_steps, output_dir=output_dir)
    if rank == 0:
        _write_exchange_csv(result)
        if verbose:
            print(f"cuflynx-couple :: done: {n_steps} coupling steps in {timings['total']:.2f} s "
                  f"(0D {timings['zero_d']:.2f} s, external {timings['external']:.2f} s); "
                  f"outputs in {output_dir}")
    return result


def _write_exchange_csv(result):
    cols, data = ['t'], [result.times.reshape(-1, 1)]
    for name, arr in result.exchange.items():
        arr = arr.reshape(len(result.times), -1)
        for j in range(arr.shape[1]):
            cols.append(f'{name}[{j}]' if arr.shape[1] > 1 else name)
        data.append(arr)
    np.savetxt(os.path.join(result.output_dir, 'coupling_exchange.csv'), np.hstack(data), delimiter=',',
               header=','.join(cols), comments='')


def describe(model_dir):
    '''Text listing what a coupled run would exchange (``cuflynx-couple --check``).'''
    info = load_info(model_dir)
    lines = [f"Model {info['model_name']} ({info['solver']}), output every {info['dt_output']} s, "
             f"pre_time {info['pre_time']} s, sim_time {info['sim_time']} s"]
    for e in info['external_models']:
        lines.append(f"\nExternal model '{e['row']}': class {e['class']} in {e['file']}"
                     f"{'' if os.path.isfile(e['file']) else '  (FILE NOT FOUND)'}")
        lines.append(f"  coupling_dt {e['coupling_dt']} s, subiterations {e['subiterations']}")
        lines.append(f"  parameters: {e['parameters']}")
        for v in e['variables']:
            arrow = '0D -> external' if v['direction'] == 'to_external' else 'external -> 0D'
            lines.append(f"  {v['variable']:<16} {arrow:<15} [{v['units']}]  connected to {', '.join(v['neighbours'])}")
    return '\n'.join(lines)


def main(argv=None):
    '''Entry point for the ``cuflynx-couple`` command.'''
    parser = argparse.ArgumentParser(
        prog='cuflynx-couple',
        description='Run a generated C++ 0D model coupled to its external Python models (api transport '
                    '"python"). Builds the model\'s shared library first if needed. Under MPI, run with '
                    'mpiexec -n N cuflynx-couple ... Its configuration is the model folder\'s '
                    'external_models.json, which cuflynx-generate writes from user_inputs.yaml (model_type: cpp) '
                    'and the module configs.')
    parser.add_argument('model_dir', help='the generated model folder (with external_models.json)')
    parser.add_argument('--t-end', type=float, default=None, help='end time [s] (default pre_time + sim_time)')
    parser.add_argument('--out', default=None, help='output folder for the 0D results')
    parser.add_argument('--solver', default=None, help='0D solver (default: the generated one)')
    parser.add_argument('--no-build', action='store_true', help="don't build the shared library")
    parser.add_argument('--check', action='store_true',
                        help='only list the external models and what they exchange; run nothing')
    args = parser.parse_args(argv)
    if args.check:
        print(describe(args.model_dir))
        return 0
    try:
        run_coupled(args.model_dir, t_end=args.t_end, output_dir=args.out, solver=args.solver,
                    build=not args.no_build)
    except Model0dError as e:
        print(f'cuflynx-couple :: error: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
