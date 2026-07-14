import importlib.util
import numpy as np
import copy
import sys
try:
    import casadi as ca
except ImportError:
    ca = None
from .name_resolver import VariableNameResolver

class SimulationHelper:
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
        self._load_model()
        self.update_times(dt, 0.0, sim_time, pre_time)
        self._init_state()
        self._has_run = False
        self._do_ad = False  # set True by _create_param_subset; cleared by reset_and_clear

    def set_protocol_info(self, protocol_info):
        """Store protocol metadata for a common helper API."""
        self.protocol_info = protocol_info

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
        for key in ('max_num_steps', 'max_step_size'):
            if key in self.solver_info:
                integrator_opts[key] = self.solver_info[key]
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
        self.model.compute_computed_constants(_v0)
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
        self.model.compute_computed_constants(self.variables)
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
        return ca.vertcat(*variables)

    def _compute_rates_symb(self):
        self.model.compute_rates(self.start_time, self.states, self.rates, self.variables)
        return ca.vertcat(*self.variables), ca.vertcat(*self.rates)

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
        self.x0_symb = ca.vertcat(*x0_parts)

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
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time/dt)
        self.n_steps = int(sim_time/dt)
        self.t_eval = np.arange(start_time, self.stop_time + dt/2, dt)
        # stored portion excludes pre_time
        self.tSim = self.t_eval[self.pre_steps:]

    # ---- parameter helpers ----
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

    def set_param_vals(self, param_names, param_vals):
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
                kind, idx_res = self._resolver.resolve(name)
                if kind == "state":
                    self.states[idx_res] = val
                elif kind == "var":
                    self.variables[self._var_idx_to_const_pos(idx_res)] = val
                    self.variables_model[idx_res] = val
                    self._sync_numeric_state_for_init_var(idx_res, val)
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
        self.model.compute_computed_constants(self.variables_model)

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

        The result is still an SX expression in ``(states_symb, variables_symb)``, so AD
        through the algebraic outputs is unaffected.
        """
        var_names = list(self.var_name_to_idx.keys())
        n_times = len(self.tSim)

        if n_times == 0:
            self.var_traj_symb = ca.SX(len(var_names), 0)
            self.var_traj_dm = np.zeros((len(var_names), 0))
            return

        # --- Algebraic map, built once: (t, x, p) -> all algebraic variables ---
        t_symb = ca.SX.sym('t_alg')
        rates = [0.0] * self.STATE_COUNT
        vars_symb_copy = copy.copy(self.variables_all_symb)
        self.model.compute_rates(t_symb, self.states_symb, rates, vars_symb_copy)
        self.model.compute_variables(t_symb, self.states_symb, rates, vars_symb_copy)
        alg_vec = ca.vertcat(*[vars_symb_copy[self.var_name_to_idx[name]] for name in var_names])
        alg_func = ca.Function('alg_map', [t_symb, self.states_symb, self.variables_symb], [alg_vec])

        # state_traj_symb spans the full pre_time+sim_time horizon, but tSim is the
        # sim-time window (t_eval[pre_steps:]). Align them by dropping the pre_time
        # warmup columns — otherwise the algebraic-variable trajectory is built from
        # the initial-transient states (t≈0) instead of the settled sim window, so a
        # nonzero pre_time returns the wrong (pre-warmup) values. States are already
        # sliced with [pre_steps:] in _extract; this makes the algebraic vars match.
        state_cols = self.state_traj_symb[:, self.pre_steps:]

        # --- Symbolic pass (preserved for AD) ---
        t_row = np.asarray(self.tSim, dtype=float).reshape(1, n_times)
        self.var_traj_symb = alg_func.map(n_times)(
            t_row, state_cols, ca.repmat(self.variables_symb, 1, n_times)
        )  # SX: shape (n_vars, n_times)

        # --- Numeric pass (for get_results / get_all_results) ---
        # Evaluate the symbolic trajectory at current numeric param values
        x0 = self._x0_numeric()
        var_func = ca.Function('var_traj', [self.states_symb, self.variables_symb], [self.var_traj_symb])
        self.var_traj_dm = np.array(var_func(ca.DM(x0), ca.DM(self.variables)))  # (n_vars, n_times)

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

    def run(self):
        # Integrate full pre_time + sim_time horizon so slicing by pre_steps
        # returns the expected sim-time segment.
        total_steps = int(max(0, len(self.t_eval) - 1))

        if self.solve_ivp_method == 'semi_implicit_euler':
            res_xf = self._run_semi_implicit_euler(total_steps)
        elif self.solve_ivp_method in ('bdf', 'BDF'):
            # Symbolic implicit BDF (rootfinder per step) — supports CasADi AD.
            res_xf = self._run_symbolic_bdf(total_steps)
        else:
            ode = {
                "x": self.states_symb,
                "p": self.variables_symb_integrator,
                "ode": self.rates_symb,
            }
            integrator_opts = self._build_integrator_opts()
            self.F = ca.integrator("F", self.solve_ivp_method, ode, 0, self.dt, integrator_opts)
            self.F_map = self.F.mapaccum(total_steps)
            res_xf = self.F_map(x0=self.x0_symb, p=self.variables_symb_integrator)["xf"]

        # Symbolic trajectory — SX function of (states_symb, variables_symb); required for AD
        self.state_traj_symb = ca.horzcat(self.x0_symb, res_xf)

        # Numeric trajectory — evaluate symbolic graph at current param values for get_results()
        x0 = self._x0_numeric()
        p_int = self._integrator_p_numeric()
        traj_func = ca.Function('state_traj', [self.states_symb, self.variables_symb], [self.state_traj_symb])
        self.state_traj_dm = np.array(traj_func(ca.DM(x0), ca.DM(self.variables)))  # (n_states, total_steps+1)

        self._has_run = True
        self._post_process()
        
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
            return self.tSim
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
        param_names = [x[0] for x in param_names]

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
        self._init_state()

    def reset_states(self):
        self.states = copy.copy(self.default_state_inits)
        self.model.compute_computed_constants(self.variables_model)

    def close_simulation(self):
        # no-op for scipy solver
        pass

