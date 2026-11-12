import atexit
import contextlib
import gc
import glob
import importlib.util
import inspect
import os
import tempfile
import shutil
import threading
import warnings

import numpy as np
import copy
import sys
try:
    import casadi as ca
except ImportError:
    ca = None
from .name_resolver import VariableNameResolver
from libcuflynx.solver_wrappers.computed_states import computed_initial_states
from libcuflynx.solver_wrappers.experiment_time import ExperimentTimeMixin
from libcuflynx.utilities.protocol_shapes import materialise_shapes

#: The SUNDIALS methods, integrated once over the whole output grid (see _run_sundials_grid).
_GRID_METHODS = ('cvodes', 'idas')

#: Integrators built by _run_sundials_grid, shared by every helper in the process:
#: {(model path, model mtime, integrator key): integrator}. A jit integrator compiles its C
#: when it is built, so this is what makes it compile once per model, grid and options.
_INTEGRATORS = {}


_JIT_DIR = None
_JIT_LOCK = threading.RLock()
#: Prefix of the C source files CasADi writes for this process's jit builds.
_JIT_NAME = f'libcuflynx_jit_{os.getpid()}_'


def _jit_dir():
    """The private temporary directory (removed at exit) holding the objects and shared
    libraries of this process's jit integrators."""
    global _JIT_DIR
    with _JIT_LOCK:
        if _JIT_DIR is None or not os.path.isdir(_JIT_DIR):
            _JIT_DIR = tempfile.mkdtemp(prefix='libcuflynx_casadi_jit_')
            atexit.register(_remove_jit_dir, _JIT_DIR)
        return _JIT_DIR


def _remove_jit_dir(path):
    """Free the jit integrators first: CasADi deletes each one's objects and library when it
    is destroyed."""
    _INTEGRATORS.clear()
    gc.collect()
    shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def _jit_build(enabled):
    """Around a step that builds or first calls a jit integrator. CasADi writes the generated C
    source to the working directory or, in newer versions, to the private jit directory (its name
    can only be a prefix, not a path), so this serialises jit builds and deletes that source from
    both afterwards. The working directory is never changed."""
    if not enabled:
        yield
        return
    with _JIT_LOCK:
        cwd = os.getcwd()
        try:
            yield
        finally:
            for folder in {cwd, _jit_dir()}:
                for path in glob.glob(os.path.join(glob.escape(folder), _JIT_NAME + '*.c')):
                    with contextlib.suppress(OSError):
                        os.remove(path)


def jit_options(solver_info):
    """The CasADi Function options for ``solver_info['jit']``: generate C for the integrator's
    right-hand side and its derivatives and compile it, with ``compiler`` (CasADi's plugin,
    'shell' by default) and ``jit_options`` (default flags -O2). CasADi's objects and shared
    libraries go to a private temporary directory (``directory``); its C source goes to the
    working directory under a ``libcuflynx_jit_<pid>_`` name, which _jit_build deletes.
    Raises RuntimeError when the
    'shell' plugin's C compiler is not on PATH or the working directory is not writable."""
    compiler = solver_info.get('compiler') or 'shell'
    options = {'flags': ['-O2'], **dict(solver_info.get('jit_options') or {})}
    if compiler == 'shell':
        options.setdefault('directory', os.path.join(_jit_dir(), ''))
    if compiler == 'shell':
        executable = options.get('compiler', 'gcc')
        if shutil.which(executable) is None:
            raise RuntimeError(
                f"casadi_integrator compiles the model with a C compiler (solver_info jit, "
                f"on by default), but {executable!r} is not on PATH. Install gcc or clang, "
                f"name one with solver_info jit_options: {{compiler: <path>}}, or set "
                f"solver_info jit: false.")
        if not os.access(os.getcwd(), os.W_OK):
            raise RuntimeError(
                f"casadi_integrator jit: CasADi writes its generated C source to the working "
                f"directory, {os.getcwd()!r}, which is not writable. Run from a writable "
                f"directory or set solver_info jit: false.")
    # jit_cleanup would look for the source under ``directory``; _jit_build removes it.
    return {'jit': True, 'compiler': compiler, 'jit_options': options,
            'jit_name': _JIT_NAME, 'jit_cleanup': False}

class SimulationHelper(ExperimentTimeMixin):
    """
    CasADi-based solver for libCellML-generated Python modules.

    Matches the key interface of the OpenCOR SimulationHelper:
    - run()
    - update_times(dt, start_time, sim_time, pre_time)
    - get_results / get_all_results / get_all_variable_names
    - get_init_param_vals / set_param_vals
    """

    def __init__(self, model_path, dt, sim_time, solver_info=None, pre_time=0.0):
        if ca is None:
            raise RuntimeError("CasADi solver requested but CasADi is not available")
        self.model_path = model_path
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.solver_info = solver_info or {}
        solver_method = self.solver_info.get('method')
        self.solve_ivp_method = solver_method
        # End state of the previous sub-experiment, used to carry state across sub-experiment
        # boundaries in the forward (non-AD) protocol. None means "start fresh" (cleared by
        # reset_states at each experiment boundary). Set before update_times, which reads it.
        self._sub_carry_state = None
        # Symbolic counterpart, used under do_ad: re-roots the next sub's x0_symb at the
        # previous sub's symbolic end state so the sub-experiments chain inside the graph.
        self._sub_carry_symb = None
        # Must precede update_times below, which branches on it to choose the symbolic or the
        # numeric carry. Set True by _create_param_subset; cleared by reset_and_clear.
        self._do_ad = False
        self._state_traj_symb = None
        self._state_traj_symb_builder = None
        self._var_traj_symb = None
        self._var_traj_symb_builder = None
        # Inputs driven by a protocol trace: {VARIABLE_INFO index: (t, values)}, t in seconds
        # from the end of the run's pre_time, as for CVODE_myokit. Set by set_param_vals.
        self._traces = {}
        self.protocol_info = None
        self._load_model()
        self.update_times(dt, 0.0, sim_time, pre_time)
        self._init_state()
        self._has_run = False

    def set_protocol_info(self, protocol_info):
        """Store protocol metadata for a common helper API, with protocol_shapes expanded into
        protocol_traces."""
        if protocol_info is not None:
            materialise_shapes(protocol_info)
        self.protocol_info = protocol_info

    def _jit_options(self):
        """jit_options(solver_info) when jit is on (solver_info 'jit', default
        CASADI_JIT_DEFAULT), else {}. An unset jit with no C compiler warns once and runs
        without jit; an explicit jit: true raises."""
        from libcuflynx.parsers.PrimitiveParsers import CASADI_JIT_DEFAULT
        explicit = 'jit' in self.solver_info
        if not self.solver_info.get('jit', CASADI_JIT_DEFAULT):
            return {}
        try:
            return jit_options(self.solver_info)
        except RuntimeError as exc:
            if explicit:
                raise
            if not getattr(self, '_warned_no_compiler', False):
                warnings.warn(f"{exc} Running without jit.")
                self._warned_no_compiler = True
            return {}

    def _build_integrator_opts(self):
        """Build CasADi integrator options from validated solver_info.

        ``reltol``/``abstol`` are SUNDIALS (``cvodes``/``idas``) options; the fixed-step
        plugins (``rk``, ``collocation``) reject them ("Unknown option: abstol"), so only
        pass them to the adaptive SUNDIALS integrators.
        """
        integrator_opts = {}
        if self.solve_ivp_method in ('cvodes', 'idas'):
            integrator_opts['reltol'] = self.solver_info.get('reltol', self.solver_info.get('rtol', 1e-8))
            integrator_opts['abstol'] = self.solver_info.get('abstol', self.solver_info.get('atol', 1e-10))
            # CasADi's default sparse QR mis-solves the ill-conditioned Newton matrix I - gamma*J of
            # stiff circulatory models; sparse LU with pivoting (as CVODE_myokit uses) does not.
            # Override with ``options: {linear_solver: ...}``.
            integrator_opts['linear_solver'] = 'csparse'
        for key in ('max_num_steps', 'max_step_size'):
            if key in self.solver_info:
                integrator_opts[key] = self.solver_info[key]
        integrator_opts.update(self._jit_options())
        integrator_opts.update(self.solver_info.get('options', {}))
        return integrator_opts

    # ---- setup helpers ----
    def _load_model(self):
        spec = importlib.util.spec_from_file_location("generated_model", self.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module

        self.STATE_COUNT = module.STATE_COUNT
        self.VARIABLE_INFO = module.VARIABLE_INFO
        self.STATE_INFO = module.STATE_INFO

        self._resolver = VariableNameResolver(self.STATE_INFO, self.VARIABLE_INFO)

        # Convenience maps (derived from resolver, kept for compatibility)
        self.state_name_to_idx = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "state"}
        self.var_name_to_idx   = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "var"}
        self.state_idx_to_name = {idx: name for name, idx in self.state_name_to_idx.items()}
        self.var_idx_to_name   = {idx: name for name, idx in self.var_name_to_idx.items()}

        # identify constants and algebraics
        self.constant_indices = [i for i, info in enumerate(self.VARIABLE_INFO) if info["type"].name in ["CONSTANT", "COMPUTED_CONSTANT"]]
        self.algebraic_indices = [i for i, info in enumerate(self.VARIABLE_INFO) if info["type"].name == "ALGEBRAIC"]

    @staticmethod
    def _as_float(value):
        """Cast numeric or CasADi DM values to Python float for numpy arrays."""
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        return float(ca.DM(value))

    def _init_state(self):
        # Save numeric initial values before symbolic patching
        _s0 = self.model.create_states_array()
        _r0 = self.model.create_states_array()
        _v0 = self.model.create_variables_array()
        self.model.initialise_variables(_s0, _r0, _v0)
        for idx, val in computed_initial_states(self.model, _v0).items():
            _s0[idx] = val
        self._numeric_x0 = np.array([self._as_float(v) for v in _s0], dtype=float)
        self._numeric_variables_all = np.array(
            [self._as_float(v) for v in _v0], dtype=float
        )

        self.states = self.model.create_states_array()
        self.rates = self.model.create_states_array()
        self.variables = self.model.create_variables_array()

        self._patch_math_functions()

        self.states_symb = self._compute_states_symb()
        self.variables_all_symb = self._compute_all_variables_symb()
        self._init_var_idx_to_state_idx = {}
        self._discover_init_var_state_links()
        self.variables_all_symb, self.rates_symb = self._compute_rates_symb()
        self._build_integrator_symbols()

        self.model.initialise_variables(self.states, self.rates, self.variables)
        for idx, val in computed_initial_states(self.model, self.variables).items():
            self.states[idx] = val
        self._store_constants_defaults()
        self._store_variable_defaults()

    def _store_constants_defaults(self):
        self.default_constants = [self.variables[i] for i in self.constant_indices]
        self.default_state_inits = [s for s in self.states]

    def _store_variable_defaults(self):
        self.variables_symb = ca.vertcat(
            *[self.variables_all_symb[i] for i in self.constant_indices]
        )
        # Use pre-saved numeric values — symbolic self.variables cannot be cast to float
        self.variables = np.array(
            [self._numeric_variables_all[i] for i in self.constant_indices],
            dtype=float,
        )
        # Full-length array for model function calls (compute_computed_constants, etc.)
        self.variables_model = list(self._numeric_variables_all)
        # A frozen copy of the constants array as the model was loaded, taken here rather than in
        # _store_constants_defaults because self.variables only becomes the constants-only array
        # (the one _var_idx_to_const_pos indexes) on the line above.
        #
        # get_init_param_vals reads the *live* array, so once set_param_vals has written it
        # reports the current value, not the model default -- unlike the Myokit backend, which
        # keeps default_values. Anything needing a stable baseline (a scale modifier's
        # theta * baseline_i) must read this, or the factor compounds across iterations.
        self.default_variables = list(self.variables)

    def _compute_states_symb(self):
        states = self.states.copy()
        for i, info in enumerate(self.model.STATE_INFO):
            states[i] = ca.SX.sym(info["name"])
        self.states = states
        return ca.vertcat(*states)

    def _compute_all_variables_symb(self):
        variables = self.variables.copy()
        for i, info in enumerate(self.model.VARIABLE_INFO):
            variables[i] = ca.SX.sym(info["name"])
        self.variables = variables
        # Keep the per-variable primitives: variables_all_symb is a vertcat, and the algebraic
        # map needs the element list to substitute computed constants into (#389).
        self._primitive_variables_symb = list(variables)
        return ca.vertcat(*variables)

    def _variables_all_symb_list(self):
        """The primitive symbol per model variable, as a list."""
        return getattr(self, '_primitive_variables_symb', None) or [
            self.variables_all_symb[i] for i in range(self.variables_all_symb.numel())]

    def _with_computed_constants(self, variables, states_out=None):
        """A copy of ``variables`` with every COMPUTED_CONSTANT replaced by its expression.

        libCellML splits constants into CONSTANT (a literal initial value) and
        COMPUTED_CONSTANT (an expression over other constants, evaluated once by
        ``compute_computed_constants``). This helper gives each variable its own SX symbol, so
        a computed constant was an *independent* symbol and the relation defining it -- e.g.
        ``k = theta * c`` -- never entered the symbolic graph. A parameter reaching the
        dynamics only through such a constant was therefore disconnected from the rates: its
        AD gradient came out identically zero and the symbolic cost was flat in it, while the
        numeric path (which re-runs compute_computed_constants) moved normally. Issue #389.

        Running ``compute_computed_constants`` on the *symbolic* array fixes that: each
        computed slot becomes an expression of the primitive symbols, so everything built from
        the result depends on the true constants.

        The primitives are left untouched, because ``variables_symb`` is used as a
        ``ca.Function`` **input** and CasADi requires inputs to be symbolic primitives, not
        expressions. Computed constants simply stop appearing in the graph; their primitive
        symbols remain in the input vector, unused, which CasADi allows.

        If ``states_out`` is a dict, it receives ``{state index: expression}`` for the states
        the model computes from constants (``computed_states``), so an initial state follows
        the constants it is computed from.
        """
        substituted = list(variables)
        try:
            if "states" in inspect.signature(self.model.compute_computed_constants).parameters:
                states = self.model.create_states_array()
                self.model.compute_computed_constants(substituted, states)
                if states_out is not None:
                    states_out.update({i: v for i, v in enumerate(states)
                                       if not (isinstance(v, float) and v != v)})
            else:
                self.model.compute_computed_constants(substituted)
        except Exception as exc:
            # A model whose computed constants are not symbolically evaluable keeps the old
            # behaviour rather than failing to load; say so, because the cost of it is a
            # silently zero gradient for anything that only acts through one.
            warnings.warn(
                f"could not evaluate compute_computed_constants symbolically ({exc}); "
                f"parameters acting only through a computed constant will have a zero "
                f"AD gradient on this model (issue #389).")
            return list(variables)
        return substituted

    def _compute_rates_symb(self):
        # Computed constants first, so a parameter that only feeds one still reaches the
        # rates symbolically (#389). variables_all_symb stays primitive -- see
        # _with_computed_constants -- so every ca.Function signature is unchanged.
        self._computed_initial_states_symb = {}
        self._symbolic_variables_with_computed = self._with_computed_constants(
            self.variables, self._computed_initial_states_symb)
        # The rates are built against a time symbol, so the SUNDIALS grid integrator can give a
        # model that reads the variable of integration (voi) the true time. rates_symb is the
        # same expression at t = start_time: the fixed-step methods and the diagnostics take an
        # autonomous (x, p) right-hand side.
        self.t_symb = ca.SX.sym('t')
        self.model.compute_rates(self.t_symb, self.states, self.rates,
                                 self._symbolic_variables_with_computed)
        self.rates_symb_t = ca.vertcat(*self.rates)
        self.rates_depend_on_time = bool(ca.depends_on(self.rates_symb_t, self.t_symb))
        rates_autonomous = ca.substitute(self.rates_symb_t, self.t_symb,
                                         ca.SX(float(self.start_time)))
        return ca.vertcat(*self.variables), rates_autonomous

    def _discover_init_var_state_links(self):
        """Map *_init parameters to the state they initialise (see initialise_variables)."""
        constant_types = {"CONSTANT", "COMPUTED_CONSTANT"}
        for var_idx, info in enumerate(self.model.VARIABLE_INFO):
            vtype = info["type"]
            type_name = vtype.name if hasattr(vtype, "name") else str(vtype)
            if type_name not in constant_types:
                continue
            if not info["name"].endswith("_init"):
                continue
            state_kind, state_idx = self._resolver.resolve(info["name"][:-5])
            if state_kind != "state":
                continue
            self._init_var_idx_to_state_idx[var_idx] = state_idx

    def _build_integrator_symbols(self):
        """Build x0 and integrator parameter vectors with disjoint CasADi symbols.

        *_init parameters set state ICs only. They must appear in x0 but not also in the
        integrator p vector, otherwise CasADi reports non-independent inputs.
        """
        state_idx_to_var_idx = {s: v for v, s in self._init_var_idx_to_state_idx.items()}
        x0_parts = []
        for state_idx, state_sym in enumerate(self.states):
            init_var_idx = state_idx_to_var_idx.get(state_idx)
            if init_var_idx is not None:
                x0_parts.append(self.variables_all_symb[init_var_idx])
            else:
                x0_parts.append(state_sym)
        for state_idx, expr in self._computed_initial_states_symb.items():
            x0_parts[state_idx] = expr
        self.x0_symb = ca.vertcat(*x0_parts)
        # Pristine root of the symbolic trajectory. In AD mode a sub-experiment boundary
        # re-roots x0_symb at the previous sub's symbolic end state (see update_times), so the
        # experiment boundary needs the original back. Built once -- this method is only called
        # from __init__ -- so caching it here is safe.
        self._x0_symb_pristine = self.x0_symb

        self._integrator_const_indices = [
            i for i in self.constant_indices if i not in self._init_var_idx_to_state_idx
        ]
        self.variables_symb_integrator = ca.vertcat(
            *[self.variables_all_symb[i] for i in self._integrator_const_indices]
        )

    def _x0_numeric(self):
        return np.array([self._as_float(v) for v in self.states], dtype=float)

    def _integrator_p_numeric(self):
        return np.array(
            [self._as_float(self.variables_model[i]) for i in self._integrator_const_indices],
            dtype=float,
        )

    def _sync_numeric_state_for_init_var(self, var_idx, val):
        """Keep numeric state ICs aligned with *_init parameter values for AD evaluation."""
        state_idx = self._init_var_idx_to_state_idx.get(var_idx)
        if state_idx is not None:
            self.states[state_idx] = float(val)
            self.default_state_inits[state_idx] = float(val)
    
    # Patch math functions to use CasADi versions for symbolic compatibility. Add more functions as needed.
    def _patch_math_functions(self):
        cellml_math_map = {
            "log": ca.log,
            "exp": ca.exp,
            "sin": ca.sin,
            "cos": ca.cos,
            "tan": ca.tan,
            "sqrt": ca.sqrt,
            "floor": ca.floor,
            "pow": ca.power,
            "fabs": ca.fabs,
        }
        for name, func in cellml_math_map.items():
            setattr(self.model, name, func)

    # ---- name resolution ----
    def _resolve_name(self, name: str):
        """Delegate to VariableNameResolver. Returns (kind, index) or (None, None)."""
        return self._resolver.resolve(name)

    def _var_idx_to_const_pos(self, var_idx: int) -> int:
        """Convert a VARIABLE_INFO index to its position in self.variables (constants array)."""
        return self.constant_indices.index(var_idx)

    # ---- timing helpers ----
    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.start_time = start_time
        self._track_experiment_start(start_time, pre_time)
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time/dt)
        self.n_steps = int(sim_time/dt)
        self.t_eval = np.arange(start_time, self.stop_time + dt/2, dt)
        # stored portion excludes pre_time
        self.tSim = self.t_eval[self.pre_steps:]
        # Carry state across a sub-experiment boundary: the previous run() in this experiment
        # recorded its end state; continue the next sub-experiment from there. reset_states
        # clears it at each experiment boundary, so an experiment's first sub-experiment starts
        # from the (possibly overridden) default state, not a stale carry.
        #
        # In AD mode the carry must be *symbolic*. self.states is the numeric point the whole
        # composed graph is evaluated at, so it has to stay at the experiment's initial
        # condition; re-rooting x0_symb at the previous sub's symbolic end state is what chains
        # the sub-experiments inside the graph. That makes the gradient the gradient of the
        # protocol actually being simulated. Previously the carry was skipped entirely under
        # do_ad, so every sub restarted from the default state while the forward path chained --
        # meaning do_ad silently calibrated a different protocol from the one that was later
        # plotted (reset_and_clear turns _do_ad off, so simulate_with_best_param_vals does
        # carry).
        if self._do_ad:
            if self._sub_carry_symb is not None:
                self.x0_symb = self._sub_carry_symb
        elif self._sub_carry_state is not None:
            self.states = list(self._sub_carry_state)

    # ---- parameter helpers ----
    def get_default_param_vals(self, param_names):
        """The model's values as loaded, regardless of what has been written since.

        Same shape as get_init_param_vals, but read from the frozen snapshot rather than the live
        arrays. See the note in _store_constants_defaults.
        """
        vals = []
        for name_or_list in param_names:
            if not isinstance(name_or_list, list):
                name_or_list = [name_or_list]
            sub = []
            for name in name_or_list:
                kind, idx = self._resolver.resolve(name)
                if kind == "state":
                    sub.append(self.default_state_inits[idx])
                elif kind == "var":
                    sub.append(self.default_variables[self._var_idx_to_const_pos(idx)])
                else:
                    raise ValueError(f"Parameter {name!r} not found (resolved kind={kind!r})")
            vals.append(sub[0] if len(sub) == 1 else sub)
        return vals

    def get_init_param_vals(self, param_names):
        vals = []
        for name_or_list in param_names:
            if not isinstance(name_or_list, list):
                name_or_list = [name_or_list]
            sub = []
            for name in name_or_list:
                kind, idx = self._resolver.resolve(name)
                if kind == "state":
                    sub.append(self.states[idx])
                elif kind == "var":
                    sub.append(self.variables[self._var_idx_to_const_pos(idx)])
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
            vals.append(sub if len(sub) > 1 else sub[0])
        return vals

    def set_param_vals(self, param_names, param_vals, change_states=True):
        """Set parameter values, by default including any state initial values they drive.

        ``change_states=False`` is for mid-protocol updates that must preserve the state the
        previous sub-experiment evolved into; naming a state there is an error.
        """
        if not change_states:
            offenders = []
            for _n_or_l in param_names:
                for _n in (_n_or_l if isinstance(_n_or_l, (list, tuple)) else [_n_or_l]):
                    try:
                        _kind, _ = self._resolver.resolve(_n)
                    except Exception:
                        continue
                    if _kind == "state":
                        offenders.append(str(_n))
            if offenders:
                raise ValueError(
                    "set_param_vals(change_states=False) cannot set states directly, but was "
                    f"given: {', '.join(offenders)}. change_states=False exists for mid-protocol "
                    "updates that must preserve the evolved state.")
        set_states = set()  # states named directly: not refreshed from computed constants
        for idx, name_or_list in enumerate(param_names):
            vals = param_vals[idx]

            def _to_list(x):
                if isinstance(x, (list, tuple)):
                    return list(x)
                try:
                    import numpy as _np
                    if isinstance(x, _np.ndarray):
                        return x.tolist()
                except Exception:
                    pass
                return [x]

            name_or_list = _to_list(name_or_list)
            vals = _to_list(vals)

            for name, val in zip(name_or_list, vals):
                if isinstance(val, str):
                    self._set_trace(name, val)
                    continue
                kind, idx_res = self._resolver.resolve(name)
                if kind == "state":
                    self.states[idx_res] = val
                    set_states.add(idx_res)
                elif kind == "var":
                    self._traces.pop(idx_res, None)
                    self.variables[self._var_idx_to_const_pos(idx_res)] = val
                    self.variables_model[idx_res] = val
                    if change_states:
                        self._sync_numeric_state_for_init_var(idx_res, val)
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
        if change_states:
            # states computed from constants follow them, except those named in this call
            for idx, val in computed_initial_states(self.model, self.variables_model).items():
                if idx not in set_states:
                    self.states[idx] = val
                    self.default_state_inits[idx] = val
        else:
            self.model.compute_computed_constants(self.variables_model)

    def _set_trace(self, name, trace_name):
        """Drive the constant ``name`` from the protocol trace ``trace_name`` (linear between
        the trace's points, its end values held outside them), from the next run() on."""
        if self.solve_ivp_method not in _GRID_METHODS:
            raise NotImplementedError(
                f"'{name}' was given the protocol trace name '{trace_name}', but solver_info "
                f"method {self.solve_ivp_method!r} cannot drive a variable from a time series. "
                f"Use method 'cvodes' or 'idas', or solver 'CVODE_myokit'.")
        traces = (self.protocol_info or {}).get('protocol_traces') or {}
        if trace_name not in traces:
            raise ValueError(f"Protocol trace '{trace_name}' not found in protocol_traces.")
        trace = traces[trace_name]
        if 'values' not in trace:
            raise ValueError(f"Protocol trace '{trace_name}' is missing 'values' key.")
        kind, idx = self._resolver.resolve(name)
        if kind != "var" or idx not in self.constant_indices:
            raise ValueError(f"protocol trace '{trace_name}' must drive a constant, not {name!r}")
        if len(trace['t']) != len(trace['values']) or len(trace['t']) == 0:
            raise ValueError(f"Protocol trace '{trace_name}' needs matching, non-empty 't' and "
                             f"'values'.")
        # The (t, value) pairs sorted as myokit.TimeSeriesProtocol sorts them, so a repeated
        # time (a step) is read the same way on both backends.
        pairs = sorted(zip((float(x) for x in trace['t']), (float(x) for x in trace['values'])))
        t = np.array([pt for pt, _ in pairs])
        v = np.array([pv for _, pv in pairs])
        self._traces[idx] = (t, v)

    def _trace_pieces(self, idx, t_left, t_right):
        """(intercept, slope) of trace ``idx`` on [t_left, t_right], times from the end of
        pre_time, which lie within one linear piece of the trace."""
        t, v = self._traces[idx]
        mid = 0.5 * (t_left + t_right)
        k = int(np.searchsorted(t, mid, side='right')) - 1
        if k < 0:
            return v[0], 0.0
        if k >= len(t) - 1:
            return v[-1], 0.0
        slope = (v[k + 1] - v[k]) / (t[k + 1] - t[k])
        return v[k] - slope * t[k], slope

    def _trace_values(self, idx, times):
        """Trace ``idx`` at ``times`` (from the end of pre_time), each from its own piece."""
        out = np.empty(len(times))
        for i, tt in enumerate(times):
            a, b = self._trace_pieces(idx, tt, tt)
            out[i] = a + b * tt
        return out

    def _post_process(self):
        """Build the algebraic-variable trajectories over the sim-time window.

        The algebraic map ``(t, x, p) -> vars`` is the same function at every output
        time, so it is built **once** symbolically and then evaluated across the whole
        time grid with ``ca.Function.map``. Evaluating the model symbolically once per
        output step instead (a Python loop over ``tSim``) meant O(n_times) traversals of
        the model equations and an SX graph of n_vars x n_times distinct expressions —
        on 3compartment (143 algebraic vars, 1000 steps) that was ~9.1s, i.e. >99% of
        ``run()``, dwarfing the 0.09s ODE solve, and made CasADi look an order of
        magnitude slower than it is. Mapping the single function is ~80x faster and
        produces bit-identical values.

        ``var_traj_dm`` is the map evaluated on the numeric state trajectory.
        ``var_traj_symb`` is the same map applied to ``state_traj_symb``, an SX expression in
        ``(states_symb, variables_symb)`` for AD; it is built on first access.
        """
        var_names = list(self.var_name_to_idx.keys())
        n_times = len(self.tSim)

        if n_times == 0:
            self._var_traj_symb = ca.SX(len(var_names), 0)
            self.var_traj_dm = np.zeros((len(var_names), 0))
            return

        # --- Algebraic map, built once: (t, x, p) -> all algebraic variables ---
        t_symb = ca.SX.sym('t_alg')
        rates = [0.0] * self.STATE_COUNT
        # Same substitution as the rates (#389): an algebraic variable -- and therefore an
        # observable built on one -- that depends on a computed constant would otherwise be
        # differentiated against an independent symbol and report a zero sensitivity.
        vars_symb_copy = self._with_computed_constants(list(self._variables_all_symb_list()))
        self.model.compute_rates(t_symb, self.states_symb, rates, vars_symb_copy)
        self.model.compute_variables(t_symb, self.states_symb, rates, vars_symb_copy)
        alg_vec = ca.vertcat(*[vars_symb_copy[self.var_name_to_idx[name]] for name in var_names])
        alg_func = ca.Function('alg_map', [t_symb, self.states_symb, self.variables_symb], [alg_vec])
        alg_map = alg_func.map(n_times)

        # The state trajectory spans the full pre_time+sim_time horizon, but tSim is the
        # sim-time window (t_eval[pre_steps:]). Align them by dropping the pre_time
        # warmup columns — otherwise the algebraic-variable trajectory is built from
        # the initial-transient states (t≈0) instead of the settled sim window, so a
        # nonzero pre_time returns the wrong (pre-warmup) values. States are already
        # sliced with [pre_steps:] in _extract; this makes the algebraic vars match.
        pre_steps = self.pre_steps
        t_row = np.asarray(self.tSim, dtype=float).reshape(1, n_times)

        # A traced input is a row of values over time rather than one constant.
        traced_rows = {
            self._var_idx_to_const_pos(idx): self._trace_values(
                idx, np.asarray(self.tSim, dtype=float) - (self.start_time + self.pre_time))
            for idx in self._traces}

        # --- Numeric pass (for get_results / get_all_results) ---
        p_num = np.repeat(np.asarray(self.variables, dtype=float).reshape(-1, 1), n_times, 1)
        for pos, row in traced_rows.items():
            p_num[pos, :] = row
        self.var_traj_dm = np.array(alg_map(
            t_row, ca.DM(self.state_traj_dm[:, pre_steps:]), ca.DM(p_num)))  # (n_vars, n_times)

        # --- Symbolic pass (for AD), on first access ---
        def _symbolic():
            p_symb = ca.repmat(self.variables_symb, 1, n_times)
            for pos, row in traced_rows.items():
                p_symb[pos, :] = ca.DM(row).T
            return alg_map(t_row, self.state_traj_symb[:, pre_steps:], p_symb)
        self._var_traj_symb = None
        self._var_traj_symb_builder = _symbolic  # SX: shape (n_vars, n_times)

    @property
    def state_traj_symb(self):
        """The state trajectory of the last run() as SX in (states_symb, variables_symb)."""
        if self._state_traj_symb is None and self._state_traj_symb_builder is not None:
            self._state_traj_symb = self._state_traj_symb_builder()
        return self._state_traj_symb

    @state_traj_symb.setter
    def state_traj_symb(self, value):
        self._state_traj_symb = value
        self._state_traj_symb_builder = None

    @property
    def var_traj_symb(self):
        """The algebraic-variable trajectory of the last run() as SX, over the sim window."""
        if self._var_traj_symb is None and self._var_traj_symb_builder is not None:
            self._var_traj_symb = self._var_traj_symb_builder()
        return self._var_traj_symb

    @var_traj_symb.setter
    def var_traj_symb(self, value):
        self._var_traj_symb = value
        self._var_traj_symb_builder = None

    # ---- simulation ----
    def _run_semi_implicit_euler(self, total_steps):
        """Fixed-step semi-implicit (linearly-implicit) Euler with diagonal damping.

            x_{n+1} = x_n + dt * f(x_n, p) / (1 - dt * d f_i/d x_i)

        The damping term ``d f_i/d x_i`` (diagonal of the rates Jacobian) is the
        automatic generalisation of the hand-coded ``lam`` damping used for the
        standalone cardiovascular model: for a stable/stiff state it is negative,
        so the denominator ``1 - dt*J_ii = 1 + dt*|J_ii|`` damps the stiff mode and
        keeps the explicit-looking update stable at the model dt.

        Unlike ``cvodes``, the whole integrator is one symbolic ``mapaccum`` graph,
        so CasADi differentiates the cost by ordinary reverse-mode AD. This avoids
        the adjoint-sensitivity solver (``CVodeF -> CV_ERR_FAILURE``) that fails on
        stiff, discontinuous models such as 3compartment.
        """
        jac_diag = ca.diag(ca.jacobian(self.rates_symb, self.states_symb))
        x_next = self.states_symb + self.dt * self.rates_symb / (1.0 - self.dt * jac_diag)
        step = ca.Function("step", [self.states_symb, self.variables_symb_integrator], [x_next])
        self.F_map = step.mapaccum(total_steps)
        # Constant integrator params are broadcast across all steps (single column).
        return self.F_map(self.x0_symb, self.variables_symb_integrator)

    def _run_symbolic_bdf(self, total_steps):
        """Fixed-step implicit BDF (order 2, with a BDF1 startup step), built as a
        symbolic CasADi graph so it supports automatic differentiation.

        Each step solves its implicit update equation with a CasADi ``rootfinder``
        (Newton). CasADi differentiates a rootfinder analytically via the implicit-
        function theorem, so — unlike the former scipy ``solve_ivp`` BDF — the whole
        trajectory is one differentiable graph: ``state_traj_symb`` is populated and
        AD works (this method is the AD-capable BDF, returning ``res_xf`` exactly like
        ``_run_semi_implicit_euler`` so ``run()`` handles it uniformly).

        The implicit BDF is stable for stiff systems where an explicit step would
        blow up. BDF2 needs two history points, so:

          * BDF1 (backward Euler) first step:  ``x1 - x0 - dt f(x1) = 0``
          * BDF2 remaining steps:  ``x_{n+1} - 4/3 x_n + 1/3 x_{n-1} - 2/3 dt f(x_{n+1}) = 0``

        with the BDF2 steps chained over the ``[x_n; x_{n-1}]`` history via ``mapaccum``.

        **Sub-stepping for robustness.** Models like 3compartment have non-smooth valve
        switches (``if_else`` / ``fmax``); when an implicit step is large enough to jump
        across a switch the residual has no root on the assumed branch and the Newton
        solve diverges (``rootfinder process failed``) — erratically with the output dt.
        So the implicit solve is taken on an internal step capped at ``solver_info['max_step']``
        (default 1e-3) and the trajectory subsampled back onto the output grid. Internal
        steps are ordinary differentiable rootfinder solves, so AD is unaffected. A good
        Newton guess (explicit-Euler / linear-extrapolation predictor) further helps
        convergence.
        """
        n = self.STATE_COUNT
        p_sym = self.variables_symb_integrator
        f_func = ca.Function('bdf_rhs', [self.states_symb, p_sym], [self.rates_symb])

        if total_steps <= 0:
            return ca.SX(n, 0)

        # Internal-step cap for the implicit solve. Default 1e-3; a falsy value
        # (None / 0, e.g. an unset UI field) falls back to the default rather than
        # disabling sub-stepping.
        _ms = self.solver_info.get('max_step')
        max_step = float(_ms) if _ms else 1e-3
        n_sub = max(1, int(np.ceil(self.dt / max_step)))
        idt = self.dt / n_sub                 # internal (sub-)step
        internal_total = total_steps * n_sub  # number of internal steps over the horizon

        # ---- BDF1 startup step (explicit-Euler predictor as the Newton guess) ----
        x_unk = ca.SX.sym('x_unk', n)
        xn = ca.SX.sym('xn', n)
        g1 = x_unk - xn - idt * f_func(x_unk, p_sym)
        G1 = ca.Function('bdf1_res', [x_unk, ca.vertcat(xn, p_sym)], [g1])
        # fast_newton converges where plain 'newton' diverges, and is differentiable
        # (implicit-function theorem), so AD through it matches FD exactly.
        solve1 = ca.rootfinder('bdf1_solve', 'fast_newton', G1)
        step1 = ca.Function('bdf1_step', [xn, p_sym],
                            [solve1(xn + idt * f_func(xn, p_sym), ca.vertcat(xn, p_sym))])

        x1 = step1(self.x0_symb, p_sym)
        if internal_total == 1:
            full = ca.horzcat(self.x0_symb, x1)
        else:
            # ---- BDF2 over the [x_n; x_{n-1}] history (linear-extrapolation predictor) ----
            H = ca.SX.sym('H', 2 * n)
            x_curr, x_prev = H[:n], H[n:]
            x_unk2 = ca.SX.sym('x_unk2', n)
            g2 = (x_unk2 - (4.0 / 3.0) * x_curr + (1.0 / 3.0) * x_prev
                  - (2.0 / 3.0) * idt * f_func(x_unk2, p_sym))
            G2 = ca.Function('bdf2_res', [x_unk2, ca.vertcat(H, p_sym)], [g2])
            solve2 = ca.rootfinder('bdf2_solve', 'fast_newton', G2)
            H_next = ca.vertcat(solve2(2.0 * x_curr - x_prev, ca.vertcat(H, p_sym)), x_curr)
            step2 = ca.Function('bdf2_step', [H, p_sym], [H_next])

            H0 = ca.vertcat(x1, self.x0_symb)  # history entering the first BDF2 step (produces x2)
            H_traj = step2.mapaccum(internal_total - 1)(H0, p_sym)
            full = ca.horzcat(self.x0_symb, x1, H_traj[:n, :])  # internal grid (internal_total+1 cols)

        # Subsample the internal trajectory back onto the output grid (drop x0).
        out_idx = [k * n_sub for k in range(1, total_steps + 1)]
        return full[:, out_idx]  # x at output steps 1..total_steps

    def _run_sundials_grid(self, total_steps):
        """One CVODES/IDAS integration over the whole output grid; sets state_traj_dm.

        The integrator is built with the output times as its grid, so SUNDIALS keeps its step
        size and history from one output time to the next and reports the state at each.
        Parameters are constant within a run(), which is one sub-experiment, so nothing
        between two output times needs a restart; a sub-experiment boundary is a new run().
        The result is a CasADi call node in (x0, p), so a jacobian of anything built from it
        differentiates through the integrator.

        A constant driven by a protocol trace enters as a control, linear in time between the
        trace's points, which join the grid. Where a trace steps (a repeated time) the run is
        split and CVODES restarts from the state at the step; nowhere else.

        Each integrator runs in time since the start of its segment, with the segment's start
        time as an extra parameter (the model sees ``t = start + tau``), so it serves every
        segment with the same grid, traced inputs and options -- each sub-experiment of a
        protocol included. They are cached on those.
        """
        x0_symb = self.x0_symb
        x0_p = ca.Function('x0_p', [self.states_symb, self.variables_symb],
                           [x0_symb, self.variables_symb_integrator])
        x0_num, p_num = x0_p(ca.DM(self._x0_numeric()), ca.DM(self.variables))
        if total_steps <= 0:
            self.state_traj_dm = np.array(x0_num)
            self.state_traj_symb = x0_symb
            return
        t0 = float(self.t_eval[0])
        dt = float(self.dt)
        out_grid = [k * dt for k in range(1, total_steps + 1)]
        traced = sorted(self._traces)
        # Trace times, in tau (time since t0): from the end of pre_time.
        sim_start = float(self.pre_time) + (t0 - float(self.start_time))
        tol = 1e-9 * dt

        def on_grid(tau):
            """tau, or the output time it coincides with."""
            k = int(round(tau / dt))
            return k * dt if 1 <= k <= total_steps and abs(k * dt - tau) <= tol else tau

        knots, steps = set(), set()
        for idx in traced:
            t_trace = self._traces[idx][0]
            for i, tt in enumerate(t_trace):
                tau = on_grid(float(tt) + sim_start)
                if 0.0 < tau < out_grid[-1]:
                    knots.add(tau)
                    if (i > 0 and t_trace[i - 1] == tt) or (
                            i + 1 < len(t_trace) and t_trace[i + 1] == tt):
                        steps.add(tau)
        grid = sorted(set(out_grid) | knots)
        col = {g: i for i, g in enumerate(grid)}
        out_cols = [col[g] for g in out_grid]
        bounds = [0.0] + sorted(steps) + [grid[-1]]

        integrator_opts = self._build_integrator_opts()
        model_key = (os.path.abspath(self.model_path), os.path.getmtime(self.model_path))
        segments = []  # (integrator, start tau, control matrix or None)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            seg_grid = [g for g in grid if lo < g <= hi]
            local = [g - lo for g in seg_grid]
            key = model_key + (self.solve_ivp_method, tuple(np.round(np.asarray(local) / dt, 9)),
                               tuple(traced))
            # The symbolic trajectory (AD) calls an integrator without jit: CasADi 3.7 does not
            # return from ca.forward through a jit-compiled CVODES integrator.
            plain_opts = {k: v for k, v in integrator_opts.items()
                          if k not in ('jit', 'compiler', 'jit_options', 'jit_name', 'jit_cleanup')}
            for opts in (integrator_opts, plain_opts):
                if key + (repr(opts),) not in _INTEGRATORS:
                    with _jit_build(opts.get('jit', False)):
                        _INTEGRATORS[key + (repr(opts),)] = self._sundials_integrator(
                            local, len(traced), opts)
            U = None
            if traced:
                # Rows (intercept, slope) per grid interval, in the segment's own time.
                lefts = [lo] + seg_grid[:-1]
                U = np.zeros((2 * len(traced), len(seg_grid)))
                for j, idx in enumerate(traced):
                    for i, (left, right) in enumerate(zip(lefts, seg_grid)):
                        a, b = self._trace_pieces(idx, left - sim_start, right - sim_start)
                        U[2 * j, i] = a + b * (lo - sim_start)
                        U[2 * j + 1, i] = b
                U = ca.DM(U)
            segments.append((_INTEGRATORS[key + (repr(integrator_opts),)],
                             _INTEGRATORS[key + (repr(plain_opts),)], lo, U))
        self.F = segments[0][0]
        self.F_map = None

        def integrate(x0, p, symbolic=False):
            columns, x = [], x0
            for F_numeric, F_symbolic, lo, U in segments:
                F = F_symbolic if symbolic else F_numeric
                call = {} if U is None else {"u": U}
                xf = F(x0=x, p=ca.vertcat(p, t0 + lo), **call)["xf"]
                columns.append(xf)
                x = xf[:, -1]
            return ca.horzcat(*columns)[:, out_cols]

        # Numeric pass: the initial state and integrator parameters at the current values,
        # then the integrator on numbers. x0_symb may be the symbolic end state of the
        # previous sub-experiment (AD mode), so it is evaluated rather than read.
        with _jit_build(integrator_opts.get('jit', False)):
            self.state_traj_dm = np.array(ca.horzcat(x0_num, integrate(x0_num, p_num)))

        # Symbolic pass, built on first access of state_traj_symb: calling the integrator on SX
        # makes one output node per state and output time, which costs more than the solve.
        p_symb = self.variables_symb_integrator
        self._state_traj_symb = None
        self._state_traj_symb_builder = lambda: ca.horzcat(
            x0_symb, integrate(x0_symb, p_symb, symbolic=True))

    def _sundials_integrator(self, grid, n_traced, integrator_opts):
        """A CVODES/IDAS integrator over ``grid`` (time since its start, tau), with parameters
        (constants, start time) and, for ``n_traced`` traced inputs, a control
        (intercept, slope) per input per grid interval: input = intercept + slope * tau."""
        tau = ca.SX.sym('tau')
        t_start = ca.SX.sym('t_start')
        rates = ca.substitute(self.rates_symb_t, self.t_symb, t_start + tau)
        ode = {"x": self.states_symb, "p": ca.vertcat(self.variables_symb_integrator, t_start),
               "t": tau}
        if n_traced:
            u = ca.SX.sym('u', 2 * n_traced)
            traced = sorted(self._traces)
            inputs = ca.vertcat(*[u[2 * j] + u[2 * j + 1] * tau for j in range(n_traced)])
            prims = ca.vertcat(*[self.variables_all_symb[idx] for idx in traced])
            rates = ca.substitute(rates, prims, inputs)
            ode["u"] = u
        ode["ode"] = rates
        return ca.integrator("F", self.solve_ivp_method, ode, 0.0, grid, integrator_opts)

    def _warn_if_time_dependent(self):
        """The fixed-step methods integrate the autonomous rates_symb (t = start_time)."""
        if self.rates_depend_on_time:
            warnings.warn(
                f"this model's rates depend on the variable of integration, but solver_info "
                f"method {self.solve_ivp_method!r} integrates them with time held at "
                f"{self.start_time}. Use method 'cvodes' or 'idas', which pass the time to "
                f"the model.")

    def run(self):
        if self._traces and self.solve_ivp_method not in _GRID_METHODS:
            raise NotImplementedError(
                f"solver_info method {self.solve_ivp_method!r} cannot drive a variable from a "
                f"protocol trace; use method 'cvodes' or 'idas'.")
        # Integrate full pre_time + sim_time horizon so slicing by pre_steps
        # returns the expected sim-time segment.
        total_steps = int(max(0, len(self.t_eval) - 1))

        if self.solve_ivp_method == 'semi_implicit_euler':
            self._warn_if_time_dependent()
            res_xf = self._run_semi_implicit_euler(total_steps)
        elif self.solve_ivp_method in ('bdf', 'BDF'):
            self._warn_if_time_dependent()
            # Symbolic implicit BDF (rootfinder per step) — supports CasADi AD.
            res_xf = self._run_symbolic_bdf(total_steps)
        elif self.solve_ivp_method in _GRID_METHODS:
            res_xf = None
            self._run_sundials_grid(total_steps)
        else:
            self._warn_if_time_dependent()
            ode = {
                "x": self.states_symb,
                "p": self.variables_symb_integrator,
                "ode": self.rates_symb,
            }
            integrator_opts = self._build_integrator_opts()
            with _jit_build(integrator_opts.get('jit', False)):
                self.F = ca.integrator("F", self.solve_ivp_method, ode, 0, self.dt,
                                       integrator_opts)
                self.F_map = self.F.mapaccum(total_steps)
            res_xf = self.F_map(x0=self.x0_symb, p=self.variables_symb_integrator)["xf"]

        if res_xf is not None:
            # Symbolic trajectory — SX function of (states_symb, variables_symb); required for AD
            self.state_traj_symb = ca.horzcat(self.x0_symb, res_xf)

            # Numeric trajectory — evaluate symbolic graph at current param values for get_results()
            x0 = self._x0_numeric()
            traj_func = ca.Function('state_traj', [self.states_symb, self.variables_symb], [self.state_traj_symb])
            self.state_traj_dm = np.array(traj_func(ca.DM(x0), ca.DM(self.variables)))  # (n_states, total_steps+1)

        self._has_run = True
        self._post_process()

        # Record the integration end state so the next sub-experiment can continue from here
        # (matching the Myokit/OpenCOR backends). The carry is applied in update_times, not by
        # mutating self.states here: self.states must stay at this run's initial condition so
        # post-run inspection (get_init_param_vals, a re-run, _post_process reference checks)
        # sees the same state it integrated from.
        self._sub_carry_state = list(self.state_traj_dm[:, -1])
        if self._do_ad:
            # Symbolic end state. Re-rooting the next sub at this expression composes the
            # sub-experiments into one graph, so the numeric evaluation point (self.states)
            # stays the experiment's initial condition and the gradient chains through every
            # sub.
            #
            # Every sub shares ONE symbolic variable vector, and the numeric evaluation
            # substitutes today's values throughout it. So the carried expression must have
            # this sub's constants baked in as literals -- otherwise the next sub, which calls
            # set_param_vals with its own params_to_change values first, would re-evaluate this
            # sub's segment at *those* values. (Verified: with the same params_to_change value
            # in both subs the carry matches the forward path exactly; with different values it
            # does not.) The calibrated parameters are deliberately left symbolic -- they are
            # what the gradient is taken with respect to, and they do not change between subs.
            calibrated = set(getattr(self, '_calib_const_positions', None) or [])
            frozen = ca.vertcat(*[
                self.variables_symb[j] if j in calibrated
                else ca.DM(float(self.variables[j]))
                for j in range(self.variables_symb.numel())
            ])
            self._sub_carry_symb = ca.substitute(
                self.state_traj_symb[:, -1], self.variables_symb, frozen)

        return True

    # ---- results ----
    def get_all_variable_names(self):
        return list(self.state_name_to_idx.keys()) + list(self.var_name_to_idx.keys())

    def _extract(self, name):
        """Return numeric trajectory slice for get_results / get_all_results.

        When in AD mode (_do_ad=True), returns symbolic SX so the cost computed
        from these values remains differentiable w.r.t. variables_symb.
        """
        if self._do_ad:
            return self._extract_symb(name)
        if name == 'time':
            return self.get_time()
        if name in self.state_name_to_idx:
            idx = self.state_name_to_idx[name]
            return self.state_traj_dm[idx, self.pre_steps:]
        if name in self.var_name_to_idx:
            idx = self.var_name_to_idx[name]
            # var_traj_dm columns are already sim-time only (built from tSim in _post_process)
            return self.var_traj_dm[idx, :]
        kind, idx_res = self._resolve_name(name)
        if kind == "state":
            return self.state_traj_dm[idx_res, self.pre_steps:]
        if kind == "var":
            return self.var_traj_dm[idx_res, :]
        raise ValueError(f"variable {name} not found")

    def _extract_symb(self, name):
        """Return symbolic (SX) trajectory slice for AD cost building."""
        if name in self.state_name_to_idx:
            idx = self.state_name_to_idx[name]
            return self.state_traj_symb[idx, self.pre_steps:]
        if name in self.var_name_to_idx:
            idx = self.var_name_to_idx[name]
            # var_traj_symb columns are already sim-time only (built from tSim in _post_process)
            return self.var_traj_symb[idx, :]
        kind, idx_res = self._resolve_name(name)
        if kind == "state":
            return self.state_traj_symb[idx_res, self.pre_steps:]
        if kind == "var":
            return self.var_traj_symb[idx_res, :]
        raise ValueError(f"variable {name} not found (symbolic)")

    def get_results(self, variables_list_of_lists, flatten=False):
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]
        results = []
        for variables_list in variables_list_of_lists:
            row = [self._extract(name) for name in variables_list]
            results.append(row)
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_all_results(self, flatten=False):
        return self.get_results(self.get_all_variable_names(), flatten=flatten)
    
    def get_all_results_dict(self):
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
            return {name: val for name, val in self._last_results_dict.items()}
        if self._last_results_dict is not None:
            return {name: val for name, val in self._last_results_dict.items()}
        raise RuntimeError("Simulation has not been run yet.")

    def _collect_all_results_dict(self):
        variable_names = self.get_all_variable_names()
        # Use symbolic extracts so var_func is a differentiable CasADi function (for AD)
        values = [self._extract_symb(name) for name in variable_names]
        values = [
            ca.reshape(v, -1, 1) if isinstance(v, ca.SX) else v
            for v in values
        ]
        self.var_func = ca.Function('var_func', [self.states_symb, self.variables_symb], values)
        values = self.var_func(self.states, self.variables)
        return {name: val for name, val in zip(variable_names, values)}

    def _create_param_subset(self, param_names, param_vals=None):
        """Build the symbolic parameter subset the cost jacobian is taken against.

        ``param_names`` is **flat**: one name per symbol, one value per name. A grouped
        params_for_id row or a modifier entry names several model constants, and the caller
        (``param_id.casadi_backend``) flattens them to their members and expands the values,
        then folds the per-member derivatives back into one per calibrated variable with the
        entry's chain-rule weights. Doing the fold there rather than here keeps this helper a
        plain "symbols for these names" service, and reuses the same weights the Myokit/FSA
        arm applies (``modifier_weights_by_index``), so the two backends cannot disagree
        about what d/dtheta means.

        A single-member list is accepted and unwrapped: ``[['a/C'], ['b/R']]`` is the canonical
        ``param_id_info["param_names"]`` shape and names one constant per entry, so there is
        nothing to flatten and no ambiguity. Only a *multi-member* group is refused -- taking
        its first member is the pre-#380 bug (the gradient tracks one constant while the cost
        moves all of them), and it is the caller's job to expand it.
        """
        param_names = [x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x
                       for x in param_names]
        grouped = [x for x in param_names if isinstance(x, (list, tuple))]
        if grouped:
            raise ValueError(
                f"_create_param_subset expects one name per symbol, got multi-member "
                f"{grouped}. Flatten grouped/modifier entries to their members (and expand "
                f"their values) before calling; see param_id.casadi_backend.flatten_entries.")

        # Resolve each param to its VARIABLE_INFO index, then find its SX symbol
        var_indices = []   # VARIABLE_INFO indices
        const_positions = []  # positions in self.variables (constants-only array)
        symb_list = []

        for name in param_names:
            kind, var_idx = self._resolver.resolve(name)
            if kind != "var":
                raise ValueError(f"Parameter {name!r} not found as a variable (resolved kind={kind!r})")
            const_pos = self._var_idx_to_const_pos(var_idx)
            var_indices.append(var_idx)
            const_positions.append(const_pos)
            symb_list.append(self.variables_all_symb[var_idx])

        self.variables_symb_subset = ca.vertcat(*symb_list)
        # Positions (into self.variables / self.variables_symb) of the parameters being
        # calibrated. The sub-experiment carry needs them: everything *else* must be frozen at
        # its current value when a sub's end state is carried forward -- see run().
        self._calib_const_positions = list(const_positions)

        if param_vals is not None:
            param_vals = np.asarray(param_vals, dtype=float)
            for const_pos, val in zip(const_positions, param_vals):
                self.variables[const_pos] = val
            for var_idx, val in zip(var_indices, param_vals):
                self.variables_model[var_idx] = val
                self._sync_numeric_state_for_init_var(var_idx, val)
            self.variables_subset = param_vals
        else:
            self.variables_subset = np.array(
                [self.variables[pos] for pos in const_positions], dtype=float
            )

        self._do_ad = True  # switch get_results to symbolic mode for AD

    # ---- reset helpers ----
    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        """Run unlogged warmup once; use end state as default for reset_states()."""
        offline_pre_time = float(offline_pre_time)
        if offline_pre_time <= 0:
            return
        self._do_ad = False
        self.update_times(self.dt, 0.0, offline_pre_time, 0.0)
        success = self.run()
        if not success:
            raise RuntimeError("Offline pre-time simulation failed")
        self.states = list(self.state_traj_dm[:, -1])
        self.default_state_inits = copy.copy(self.states)
        self._has_run = False
        self.states = copy.copy(self.default_state_inits)
        self.model.compute_computed_constants(self.variables_model)

    def reset_and_clear(self, only_one_exp=-1):
        self._do_ad = False
        self._traces = {}
        self._init_state()

    def reset_states(self):
        self.states = copy.copy(self.default_state_inits)
        self.model.compute_computed_constants(self.variables_model)
        # Experiment boundary: drop any carried sub-experiment end state so the next
        # experiment's first sub-experiment starts from the default (reset) state. The symbolic
        # carry is dropped with it, and x0_symb is re-rooted at the pristine initial state --
        # otherwise the next experiment's graph would still be composed on top of the previous
        # experiment's trajectory.
        self._sub_carry_state = None
        self._sub_carry_symb = None
        self.x0_symb = self._x0_symb_pristine

    def close_simulation(self):
        # no-op for scipy solver
        pass

