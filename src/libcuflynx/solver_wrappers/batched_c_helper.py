"""
BE_batched_c: CellML models integrated by generated C, block by block, on several threads.

The model is split into blocks of states that do not depend on one another (each cell of a module array of
unconnected cells, say; a circulation is one block). Each block's rates and exact Jacobian are generated as C
from libCellML's analysis (:mod:`libcuflynx.generators.kernel_ir`, :mod:`libcuflynx.generators.cpp.batched`),
and each block is integrated with its own adaptive backward Euler steps (modified Newton, LU factors kept
across steps), so quiet blocks take long steps and the cost grows linearly with the number of blocks. Blocks
run on OpenMP threads with results identical for any number of threads. A block whose inputs did not change
between runs (sub-experiments) continues its step; one whose inputs changed restarts from its current state.

``method`` (in solver_info): 'backward_euler' (the default; first order, the cheapest step) or 'trbdf2' (second
order: fewer steps at tight tolerances, and much less phase error on oscillations). Both are L-stable. Both suit
stiff models at moderate tolerances (rtol ~1e-3 to 1e-6) and many blocks; CVODE_myokit is the better choice
for one tightly coupled model at tight tolerances.

solver_info keys: ``rtol`` (default 1e-4), ``atol`` (1e-8), ``MaximumStep`` (unbounded),
``MaximumNumberOfSteps`` (steps per block per run, default 10 000 000), ``threads`` (default 1) and
``jacobian`` ('exact', the default, or 'finite_difference').

The model sees time as CVODE_myokit does: each sub-experiment's clock starts at 0 (Myokit's
Simulation.reset()). Only blocks that read time notice; the integrator keeps one clock across sub-experiments.

CVODES_batched_c (:class:`CVODESSimulationHelper`) runs the same generated kernels and blocks with one CVODES per
block (BDF, Newton, dense LU, the exact Jacobian), on threads in the same way; it needs SUNDIALS 6 or later
(``SUNDIALS_DIR``). Its defaults are CVODE_myokit's tolerances (rtol 1e-6, atol 1e-8); ``jacobian:
'finite_difference'`` uses CVODES's difference quotients. MaximumNumberOfSteps counts as BE_batched_c's does.
"""
import time as _time
import warnings

import numpy as np

from libcuflynx.generators.cpp import batched
from libcuflynx.solver_wrappers.name_resolver import VariableNameResolver
from libcuflynx.solver_wrappers.param_grouping import pair_names_with_values
from libcuflynx.utilities.protocol_shapes import materialise_shapes, validate_trace_references

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-8
DEFAULT_MAX_STEPS = 10_000_000
CVODES_DEFAULT_RTOL = 1e-6
CVODES_DEFAULT_ATOL = 1e-8
JACOBIANS = ('exact', 'finite_difference')
STATUS = {1: 'step size below its minimum', 2: 'too many steps (MaximumNumberOfSteps)', 3: 'non-finite state',
          4: 'CVODES failed'}


class SimulationHelper:
    """BE_batched_c with the SimulationHelper surface of the CellML backends (no sensitivities)."""

    SOLVER = 'BE_batched_c'
    DEFAULT_RTOL = DEFAULT_RTOL
    DEFAULT_ATOL = DEFAULT_ATOL
    #: newton_tol, max_newton and max_jacobian_age for BatchedSimulation.set_options
    NEWTON_OPTIONS = {'newton_tol': 0.01, 'max_newton': 6, 'max_jacobian_age': 50}

    def __init__(self, cellml_path, dt, sim_time, solver_info=None, pre_time=0.0):
        self.cellml_path = cellml_path
        self.solver_info = dict(solver_info or {})
        self.dt = dt
        self.protocol_info = None
        self._configure(self.solver_info)

        t0 = _time.perf_counter()
        self.info, path, self.build_seconds, self.cached = batched.load_or_build(
            cellml_path, self.solver_info.get('build_dir'), solver=self.SOLVER)
        self.setup_seconds = _time.perf_counter() - t0
        info = self.info
        self.library = batched.BatchedLibrary(path)
        self.sim = batched.BatchedSimulation(self.library, info['blocks'], info['state_count'],
                                             info['variable_count'])
        self.sim.set_options(rtol=self.rtol, atol=self.atol, max_step=self.max_step, max_steps=self.max_steps,
                             finite_difference_jacobian=(self.jacobian == 'finite_difference'), method=self.method,
                             **self.NEWTON_OPTIONS)

        self.STATE_COUNT = info['state_count']
        self.state_names = info['state_names']
        self.variable_names = info['variable_names']
        self.state_index = info['state_index']
        self.var_index = info['var_index']
        self._kind_of_var = info['variable_kinds']
        self.state_init_ref = info['state_init_ref']
        self._init_values()
        self._traced = {}           # one-array index -> trace name
        self._trace_targets = []    # constants the protocol traces somewhere: reported as time series
        self.clock = 0.0
        self.last_states = None
        self._variables_rows = None
        if sim_time is not None and pre_time is not None:
            self._setup_time(dt, sim_time, pre_time)

    # ------------------------------------------------------------------ setup

    def _configure(self, si):
        self.rtol = float(si.get('rtol', self.DEFAULT_RTOL))
        self.atol = float(si.get('atol', self.DEFAULT_ATOL))
        self.max_step = float(si['MaximumStep']) if si.get('MaximumStep') else 0.0
        self.max_steps = int(si.get('MaximumNumberOfSteps', DEFAULT_MAX_STEPS))
        self.threads = int(si.get('threads', 1))
        self.jacobian = si.get('jacobian', 'exact')
        if self.jacobian not in JACOBIANS:
            raise ValueError(f"solver_info jacobian must be one of {JACOBIANS}, not {self.jacobian!r}")
        methods = batched.SOLVERS[self.SOLVER]['methods']
        self.method = si.get('method') or methods[0]
        if self.method not in methods:
            raise ValueError(f"{self.SOLVER} method must be one of {methods}, not {self.method!r}")

    def _init_values(self):
        self.states = np.zeros(self.info['state_count'])
        self.variables = np.zeros(self.info['variable_count'])
        rates = np.zeros(self.info['state_count'])
        self.library.bk_initialise(batched.ptr(self.states), batched.ptr(rates), batched.ptr(self.variables))
        self.library.bk_compute_computed_constants(batched.ptr(self.states), batched.ptr(rates),
                                                   batched.ptr(self.variables))
        self._literal_initial_states = self.states.copy()
        self._var_overrides = {}
        self._state_overrides = {}
        self.original_states = self._initial_states()
        self.states[:] = self.original_states
        self.default_states = list(self.original_states)
        self.default_variables = self.variables.copy()
        self.sim.invalidate()

    def _initial_states(self):
        out = self._literal_initial_states.copy()
        for i, ref in enumerate(self.state_init_ref):
            if ref is not None:
                out[i] = self.variables[ref]
        for i, v in self._state_overrides.items():
            out[i] = v
        return out

    def _setup_time(self, dt, sim_time, pre_time, start_time=0.0):
        self.dt = dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.start_time = start_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time / dt)
        self.n_steps = int(sim_time / dt)
        self.tSim = np.linspace(start_time + pre_time, self.stop_time, self.n_steps + 1)

    # ------------------------------------------------------------------ running

    def update_times(self, dt, start_time, sim_time, pre_time):
        """A new segment, continuing from the current state; the model's clock restarts at 0, as CVODE_myokit's
        does."""
        self._setup_time(dt, sim_time, pre_time, start_time=start_time)
        self.sim.set_time_origin(self.clock)

    def run(self):
        times = self.clock + self.pre_time + (self.tSim - self.tSim[0])
        failures = []
        if self.pre_time > 0:
            failures = self.sim.run(self.states, self.variables, self.clock, [self.clock + self.pre_time],
                                    None, self.threads)
        out = np.empty((len(times), self.STATE_COUNT))
        if not failures:
            failures = self.sim.run(self.states, self.variables, self.clock + self.pre_time, times, out,
                                    self.threads)
        if failures:
            b, status, t = failures[0]
            names = [self.state_names[i] for i in self.info['blocks'][b]['states'][:3]]
            print(f'{self.SOLVER} simulation failed: {len(failures)} block(s); block {b} ({", ".join(names)}, ...) '
                  f'stopped at t={t:g}: {STATUS.get(status, status)}.')
            warnings.warn(f'{self.SOLVER} simulation failed (rtol {self.rtol:g}, atol {self.atol:g}, '
                          f'MaximumStep {self.max_step or "unbounded"}): {STATUS.get(status, status)}')
            self.sim.invalidate()
            return False
        self.clock = float(times[-1])
        self.last_states = out
        self._last_times = times
        self._last_variables = self.variables.copy()
        self._variables_rows = None
        return True

    def stats(self):
        """Integrator totals since construction: steps, rejected steps, rate evaluations, Jacobians, ..."""
        return self.sim.stats()

    def _rows(self):
        if self._variables_rows is None:
            rows = np.empty((len(self._last_times), self.info['variable_count']))
            self.sim.outputs(np.ascontiguousarray(self._last_times), self.last_states, self._last_variables, rows,
                             self.threads)
            self._variables_rows = rows
        return self._variables_rows

    def _restart_clock(self):
        self.clock = 0.0
        self.sim.set_time_origin(0.0)

    def reset_and_clear(self, only_one_exp=-1):
        self._state_overrides = {}
        self._clear_traces()
        self.states[:] = self.original_states
        self._restart_clock()
        self.last_states = None
        self.sim.invalidate()

    def reset_states(self):
        self.states[:] = self._initial_states()
        self.default_states = list(self.states)
        self._restart_clock()
        self.last_states = None
        self.sim.invalidate()

    def close_simulation(self):
        self.sim.close()

    # ------------------------------------------------------------------ names and values

    def _resolve_name(self, name):
        """(kind, index): kind 'state' (into the states) or 'var' (into the one variables array)."""
        if '/' not in name and name.count('.') == 1:
            name = name.replace('.', '/')
        return VariableNameResolver.resolve_key(name, [('state', self.state_index), ('var', self.var_index)])

    def _index(self, name):
        kind, key = self._resolve_name(name)
        if kind == 'state':
            return kind, self.state_index[key]
        if kind == 'var':
            return kind, self.var_index[key]
        raise ValueError(f'variable {name} not found')

    def get_time(self, include_pre_time=False):
        return self.tSim if include_pre_time else self.tSim - self.pre_time

    def get_all_variable_names(self):
        """States and algebraic variables, as component.variable (the names CVODE_myokit logs)."""
        names = [n.replace('/', '.', 1) for n in self.state_names]
        names += [n.replace('/', '.', 1) for n, k in zip(self.variable_names, self._kind_of_var)
                  if k in ('algebraic', 'external')]
        names += [self.variable_names[k].replace('/', '.', 1) for k in self._trace_targets]   # as CVODE_myokit
        return names

    def _extract(self, name):
        if name == 'time':
            return self.tSim - self.pre_time
        if self.last_states is None:
            raise RuntimeError('Simulation has not been run yet.')
        kind, index = self._index(name)
        if kind == 'state':
            return self.last_states[:, index].copy()
        kind = self._kind_of_var[index]
        varies = kind in ('algebraic', 'external') or index in self._traced or index in self._trace_targets
        if varies or (self._traced and kind == 'computed_constant'):    # a trace moves computed constants too
            return self._rows()[:, index].copy()
        return np.asarray([self.variables[index]])

    def get_results(self, variables_list_of_lists, flatten=False):
        if self.last_states is None:
            raise RuntimeError('Simulation has not been run yet.')
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]
        results = [[self._extract(name) for name in names] for names in variables_list_of_lists]
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_all_results(self, flatten=False):
        return self.get_results(self.get_all_variable_names(), flatten=flatten)

    def get_all_results_dict(self):
        out = {'environment.time': self.tSim - self.pre_time}
        for name in self.get_all_variable_names():
            out[name] = self._extract(name)
        return out

    def get_init_param_vals(self, param_names):
        values = []
        for name_or_list in param_names:
            names = name_or_list if isinstance(name_or_list, list) else [name_or_list]
            vals = []
            for name in names:
                kind, index = self._index(name)
                vals.append(self.default_states[index] if kind == 'state' else self.default_variables[index])
            values.append(vals if len(vals) > 1 else vals[0])
        return values

    def get_default_param_vals(self, param_names):
        return self.get_init_param_vals(param_names)

    def set_protocol_info(self, protocol_info):
        """The protocol to run: its shapes become traces, and every trace a params_to_change entry names must
        exist."""
        if protocol_info is not None:
            materialise_shapes(protocol_info)
            validate_trace_references(protocol_info)
        self.protocol_info = protocol_info
        self._trace_targets = []
        for name, values in (protocol_info or {}).get('params_to_change', {}).items():
            if any(isinstance(v, str) for sub in values for v in (sub if isinstance(sub, list) else [sub])):
                kind, index = self._index(name)
                if kind == 'var' and index not in self._trace_targets:
                    self._trace_targets.append(index)

    def _trace(self, name):
        traces = (self.protocol_info or {}).get('protocol_traces')
        if traces is None:
            raise ValueError('params_to_change entry is a string trace key, but protocol_traces not found in '
                             'protocol_info.')
        if name not in traces:
            raise ValueError(f"Protocol trace '{name}' not found in protocol_traces.")
        trace = traces[name]
        if 'values' not in trace or 't' not in trace:
            raise ValueError(f"Protocol trace '{name}' needs 't' and 'values'.")
        return trace

    def _clear_traces(self):
        for index in list(self._traced):
            self.sim.clear_trace(index)
        self._traced = {}

    def set_param_vals(self, param_names, param_vals, change_states=True):
        """Set constants (and computed constants), or states; by default states initialised by a changed
        variable are re-initialised. ``change_states=False`` (mid-protocol) refuses to set states."""
        before = self.variables.copy()
        states_set = False
        for idx, name_or_list in enumerate(param_names):
            for name, val in pair_names_with_values(name_or_list, param_vals[idx], 'set_param_vals'):
                kind, index = self._index(name)
                if isinstance(val, str):
                    # a protocol trace: the variable follows it on the model's clock from the next run
                    if kind == 'state' or self._kind_of_var[index] != 'constant':
                        raise ValueError(f'{name} cannot follow a trace: only constants can')
                    trace = self._trace(val)
                    self.sim.set_trace(index, trace['t'], trace['values'])
                    self._traced[index] = val
                    continue
                if kind != 'state' and index in self._traced:
                    self.sim.clear_trace(index)
                    del self._traced[index]
                if kind == 'state':
                    if not change_states:
                        raise ValueError(
                            f'set_param_vals(change_states=False) cannot set states directly, but was given: '
                            f'{name} (state {self.state_names[index]}).')
                    self._state_overrides[index] = float(val)
                    self.states[index] = float(val)
                    states_set = True
                else:
                    if self._kind_of_var[index] == 'algebraic':
                        raise ValueError(f'{name} is an algebraic variable: it cannot be set')
                    self._var_overrides[index] = float(val)
                    self.variables[index] = float(val)
        # into scratch states: the states it initialises are refreshed below only when change_states
        scratch = np.zeros(self.STATE_COUNT)
        self.library.bk_compute_computed_constants(batched.ptr(scratch), batched.ptr(np.zeros(self.STATE_COUNT)),
                                                   batched.ptr(self.variables))
        for index, val in self._var_overrides.items():
            self.variables[index] = val
        changed = set(np.flatnonzero(~((self.variables == before) | (np.isnan(self.variables) & np.isnan(before)))))
        if change_states and changed:
            for i, ref in enumerate(self.state_init_ref):
                if ref in changed and i not in self._state_overrides:
                    self.states[i] = self.variables[ref]
                    states_set = True
        self.default_states = list(self._initial_states())
        if states_set:
            self.sim.invalidate()

    def modify_params_and_run_and_get_results(self, param_names, mod_factors, obs_names, absolute=False):
        if absolute:
            new_param_vals = mod_factors
        else:
            init_param_vals = self.get_init_param_vals(param_names)
            new_param_vals = [a * b for a, b in zip(init_param_vals, mod_factors)]
        self.set_param_vals(param_names, new_param_vals)
        if not self.run():
            raise RuntimeError('simulation failed')
        pred_obs_new = self.get_results(obs_names)
        self.reset_and_clear()
        return pred_obs_new


class CVODESSimulationHelper(SimulationHelper):
    """CVODES_batched_c: BE_batched_c's kernels, blocks and threads, each block integrated by its own CVODES."""

    SOLVER = 'CVODES_batched_c'
    DEFAULT_RTOL = CVODES_DEFAULT_RTOL
    DEFAULT_ATOL = CVODES_DEFAULT_ATOL
    NEWTON_OPTIONS = {'newton_tol': 0.0, 'max_newton': 0, 'max_jacobian_age': 0}     # CVODES's own
