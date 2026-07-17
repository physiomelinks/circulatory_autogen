"""
AADC-based solver backend for circulatory_autogen.

Peer to casadi_python_solver_helper.py. Implements the same SimulationHelper
interface so it plugs into paramID, sensitivity analysis, etc.

This backend is already wired in: it is imported and selected by
src/solver_wrappers/__init__.py for solver 'aadc_semi_implicit', registered in
parsers/PrimitiveParsers.py (model_type 'aadc_python'), and dispatched by
scripts/script_generate_with_new_architecture.py. To use it, set in your config:
  model_type: aadc_python
  solver: aadc_semi_implicit
Requires: pip install aadc

Key differences from CasADI backend:
  - No symbolic graph — AADC records actual execution on idouble tape
  - Conditionals (if/else) handled via aadc.iif() — no crash
  - Stiff ODEs via semi-implicit Euler with diagonal damping
  - Kernel recorded once (~3s), then gradient eval ~6ms
"""
import importlib.util
import math
import copy
import warnings
import numpy as np

try:
    import aadc
except ImportError:
    aadc = None

# Reuse the shared name resolver from circulatory_autogen
try:
    from .name_resolver import VariableNameResolver
except ImportError:
    # Standalone usage outside circulatory_autogen package
    VariableNameResolver = None


_THIRD_PARTY_NOTICE = (
    "You have selected the AADC backend (model_type='aadc_python'). AADC is THIRD-PARTY "
    "PROPRIETARY software from Matlogica. It is NOT part of Circulatory Autogen, is not "
    "covered by Circulatory Autogen's Apache-2.0 licence, and is not open source; its terms "
    "restrict it to academic / non-commercial use. Use of AADC is governed solely by your "
    "own licence agreement with Matlogica. No feature of Circulatory Autogen requires it: "
    "the default, supported, fully open-source AD backend is CasADi "
    "(model_type='casadi_python')."
)
_notice_emitted = False


def _emit_third_party_notice():
    """Warn once per process that AADC is optional third-party proprietary software."""
    global _notice_emitted
    if not _notice_emitted:
        warnings.warn(_THIRD_PARTY_NOTICE, UserWarning, stacklevel=3)
        _notice_emitted = True


class SimulationHelper:
    """
    AADC-based solver for libCellML-generated Python modules.

    Matches the key interface of the other SimulationHelpers:
    - run()
    - update_times(dt, start_time, sim_time, pre_time)
    - get_results / get_all_results / get_all_variable_names
    - get_init_param_vals / set_param_vals
    """

    def __init__(self, model_path, dt, sim_time, solver_info=None, pre_time=0.0):
        if aadc is None:
            raise RuntimeError("AADC solver requested but aadc is not installed")
        _emit_third_party_notice()
        self.model_path = model_path
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.solver_info = solver_info or {}
        self._load_model()
        self.update_times(dt, 0.0, sim_time, pre_time)
        self._init_state()
        self._has_run = False
        self._do_ad = False
        self._aad_funcs = None  # AAD kernel recorded by _record_rhs_aad
        self._rk_data = None    # adaptive-step trajectory stored by _integrate
        self._n_threads = int(self.solver_info.get('threads', 4))
        self._aad_workers = aadc.ThreadPool(self._n_threads)

    def set_protocol_info(self, protocol_info):
        self.protocol_info = protocol_info

    # ---- setup helpers ----
    def _load_model(self):
        spec = importlib.util.spec_from_file_location("generated_model", self.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module

        self.STATE_COUNT = module.STATE_COUNT
        self.VARIABLE_INFO = module.VARIABLE_INFO
        self.STATE_INFO = module.STATE_INFO

        if VariableNameResolver is not None:
            self._resolver = VariableNameResolver(self.STATE_INFO, self.VARIABLE_INFO)
        else:
            self._resolver = None

        self.state_name_to_idx = {}
        self.var_name_to_idx = {}
        if self._resolver:
            self.state_name_to_idx = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "state"}
            self.var_name_to_idx = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "var"}
        self.state_idx_to_name = {idx: name for name, idx in self.state_name_to_idx.items()}
        self.var_idx_to_name = {idx: name for name, idx in self.var_name_to_idx.items()}

        self.constant_indices = [i for i, info in enumerate(self.VARIABLE_INFO)
                                 if info["type"].name in ["CONSTANT", "COMPUTED_CONSTANT"]]
        self.algebraic_indices = [i for i, info in enumerate(self.VARIABLE_INFO)
                                  if info["type"].name == "ALGEBRAIC"]

    def _init_state(self):
        _s0 = self.model.create_states_array()
        _r0 = self.model.create_states_array()
        _v0 = self.model.create_variables_array()
        self.model.initialise_variables(_s0, _r0, _v0)
        self.model.compute_computed_constants(_v0)
        self._numeric_x0 = np.array(_s0, dtype=float)
        self._numeric_variables_all = np.array(_v0, dtype=float)

        self.states = list(_s0)
        self.rates = list(_r0)
        self.variables = np.array([_v0[i] for i in self.constant_indices], dtype=float)

        self.default_constants = list(self.variables)
        self.default_state_inits = list(self.states)

    def _patch_math_functions(self):
        """Replace math functions in the model module with AADC-compatible versions.

        Saves originals so _unpatch_math_functions() can restore them.
        Safe for multi-instance use: each instance restores before the next patches.
        """
        aadc_math_map = {
            "log": aadc.math.log,
            "exp": aadc.math.exp,
            "sin": aadc.math.sin,
            "cos": aadc.math.cos,
            "tan": aadc.math.tan,
            "sqrt": aadc.math.sqrt,
            "pow": aadc.math.pow,
        }
        self._math_originals = {}
        for name, func in aadc_math_map.items():
            self._math_originals[name] = getattr(self.model, name, None)
            setattr(self.model, name, func)

        # floor: extract passive value (not differentiable, but needed for cardiac phase)
        def aadc_floor(x):
            return math.floor(float(x))
        self._math_originals["floor"] = getattr(self.model, "floor", None)
        setattr(self.model, "floor", aadc_floor)

        # Replace comparison functions with aadc.iif versions
        def leq_func(a, b):
            return a <= b  # returns idouble comparison for aadc.iif
        def geq_func(a, b):
            return a >= b
        def lt_func(a, b):
            return a < b
        def gt_func(a, b):
            return a > b
        def and_func(a, b):
            return aadc.iand(a, b)
        def aadc_max(a, b):
            return aadc.iif(a >= b, a, b)

        for name, func in [("leq_func", leq_func), ("geq_func", geq_func),
                            ("lt_func", lt_func), ("gt_func", gt_func),
                            ("and_func", and_func), ("max", aadc_max)]:
            self._math_originals[name] = getattr(self.model, name, None)
            setattr(self.model, name, func)

    def _unpatch_math_functions(self):
        """Restore model module to its original math functions after AAD recording."""
        for name, orig in getattr(self, "_math_originals", {}).items():
            if orig is None:
                if hasattr(self.model, name):
                    delattr(self.model, name)
            else:
                setattr(self.model, name, orig)
        self._math_originals = {}

    # ---- name resolution ----
    def _resolve_name(self, name):
        if self._resolver:
            return self._resolver.resolve(name)
        return (None, None)

    def _var_idx_to_const_pos(self, var_idx):
        return self.constant_indices.index(var_idx)

    # ---- timing ----
    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.start_time = start_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time / dt)
        self.n_steps = int(sim_time / dt)
        self.t_eval = np.arange(start_time, self.stop_time + dt / 2, dt)
        self.tSim = self.t_eval[self.pre_steps:]

    # ---- parameter helpers ----
    def get_init_param_vals(self, param_names):
        vals = []
        for name_or_list in param_names:
            if not isinstance(name_or_list, list):
                name_or_list = [name_or_list]
            sub = []
            for name in name_or_list:
                kind, idx = self._resolve_name(name)
                if kind == "state":
                    sub.append(self.states[idx])
                elif kind == "var":
                    sub.append(self.variables[self._var_idx_to_const_pos(idx)])
                else:
                    raise ValueError(f"parameter name {name} not found")
            vals.append(sub if len(sub) > 1 else sub[0])
        return vals

    def set_param_vals(self, param_names, param_vals):
        for idx, name_or_list in enumerate(param_names):
            vals = param_vals[idx]
            if not isinstance(name_or_list, (list, tuple)):
                name_or_list = [name_or_list]
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            for name, val in zip(name_or_list, vals):
                kind, idx_res = self._resolve_name(name)
                if kind == "state":
                    self.states[idx_res] = val
                elif kind == "var":
                    self.variables[self._var_idx_to_const_pos(idx_res)] = val
                    var_name = self.var_idx_to_name.get(idx_res, "")
                    var_part = var_name.split("/")[-1] if "/" in var_name else var_name
                    if var_part.endswith("_init"):
                        state_var = var_part[:-5]
                        state_kind, state_idx = self._resolve_name(state_var)
                        if state_kind == "state":
                            self.states[state_idx] = val
                            self.default_state_inits[state_idx] = val
                else:
                    raise ValueError(f"parameter name {name} not found")

    # ---- ODE integration ----
    def _integrate_implicit_euler_ift(self, states, variables_all, total_steps, dt):
        """Implicit Euler via aadc.least_squares + Implicit Function Theorem.

        Each step solves: y_{n+1} - y_n - dt * f(y_{n+1}) = 0
        using Levenberg-Marquardt. IFT gives exact AD gradients automatically.
        Works both in passive mode (forward sim) and on AADC tape (gradient).

        Returns state trajectory as list of state arrays (post pre-time).
        """
        n = self.STATE_COUNT
        self._patch_math_functions()

        # Identify zeta indices (valve states to clamp to [0,1])
        zeta_indices = [i for i, info in enumerate(self.model.STATE_INFO)
                        if 'zeta' in info.get('name', '').lower()]

        y = [aadc.idouble(float(states[i])) for i in range(n)]
        id_dt = aadc.idouble(float(dt))
        vars_rec = list(variables_all)

        traj = [[float(yi) for yi in y]]

        for step in range(total_steps):
            t_step = step * dt

            def residual(y_next, _y_n=list(y), _vars=list(vars_rec), _t=t_step, _dt=id_dt):
                rates = [aadc.idouble(0.0) for _ in range(n)]
                self.model.compute_rates(_t, list(y_next), rates, list(_vars))
                return [y_next[i] - _y_n[i] - _dt * rates[i] for i in range(n)]

            result = aadc.least_squares(residual, list(y), ftol=1e-12, xtol=1e-12)
            y = list(result.x)

            # Clamp zeta to [0, 1]
            for z in zeta_indices:
                y[z] = aadc.iif(y[z] >= 0.0, y[z], aadc.idouble(0.0))
                y[z] = aadc.iif(y[z] <= 1.0, y[z], aadc.idouble(1.0))

            traj.append([float(yi) if not hasattr(yi, 'val') else yi.val() for yi in y])

        self._unpatch_math_functions()
        return traj[self.pre_steps:]

    def _integrate_implicit_euler_ift_on_tape(self, st, vars_rec, total_steps, dt, n):
        """Implicit Euler on AADC tape — same algorithm as forward, but with idouble.

        Parameters are already idouble from tape recording context.
        Returns final state as list of idouble for cost evaluation.
        """
        zeta_indices = [i for i, info in enumerate(self.model.STATE_INFO)
                        if 'zeta' in info.get('name', '').lower()]

        id_dt = aadc.idouble(float(dt))

        for step in range(total_steps):
            t_step = step * dt

            def residual(y_next, _y_n=list(st), _vars=list(vars_rec), _t=t_step, _dt=id_dt):
                rates = [aadc.idouble(0.0) for _ in range(self.STATE_COUNT)]
                self.model.compute_rates(_t, list(y_next), rates, list(_vars))
                return [y_next[i] - _y_n[i] - _dt * rates[i] for i in range(self.STATE_COUNT)]

            result = aadc.least_squares(residual, list(st), ftol=1e-12, xtol=1e-12)
            st = list(result.x)

            for z in zeta_indices:
                st[z] = aadc.iif(st[z] >= 0.0, st[z], aadc.idouble(0.0))
                st[z] = aadc.iif(st[z] <= 1.0, st[z], aadc.idouble(1.0))

        return st

    def _integrate_bdf(self, states, variables_all, total_steps, dt):
        """BDF (implicit, adaptive) via scipy + AADC Jacobian.

        Uses VectorFunctionWithJacobian from the inner RHS kernel to provide
        exact dense Jacobian to scipy's BDF solver. Accurate for stiff models.

        Returns state trajectory as list of state arrays (post pre-time).
        """
        import math as _math
        from scipy.integrate import solve_ivp

        n = self.STATE_COUNT
        x0 = np.array(states[:n], dtype=float)
        T = total_steps * dt

        # Record inner RHS kernel for VectorFunctionWithJacobian
        if not hasattr(self, '_bdf_vfj') or self._bdf_vfj is None:
            rhs_f = aadc.Functions()
            rhs_f.start_recording()
            id_si = [aadc.idouble(float(x0[i])) for i in range(n)]
            a_si = [s.mark_as_input() for s in id_si]
            id_vi = [aadc.idouble(float(v) if not (isinstance(v, float) and _math.isnan(v)) else 0.0)
                     for v in variables_all]
            # Mark AD params as inputs on inner kernel too
            a_params_inner = []
            if hasattr(self, '_ad_param_var_indices'):
                for idx in self._ad_param_var_indices:
                    id_vi[idx] = aadc.idouble(float(variables_all[idx]))
                    a_params_inner.append(id_vi[idx].mark_as_input())
            id_ri = [aadc.idouble(0.0) for _ in range(n)]
            self.model.compute_rates(0.0, id_si, id_ri, list(id_vi))
            r_ri = []
            for i in range(n):
                r = id_ri[i] if hasattr(id_ri[i], 'mark_as_output') else aadc.idouble(float(id_ri[i]))
                r_ri.append(r.mark_as_output())
            rhs_f.stop_recording()
            self._bdf_vfj = aadc.VectorFunctionWithJacobian(rhs_f, a_si, a_params_inner, r_ri)
            self._bdf_rhs_funcs = rhs_f

        # Set current parameter values
        if hasattr(self, '_ad_param_var_indices') and self._ad_param_var_indices:
            param_vals = np.array([float(variables_all[idx]) for idx in self._ad_param_var_indices])
            self._bdf_vfj.set_params(param_vals)

        # Solve with scipy BDF
        vfj = self._bdf_vfj
        sol = solve_ivp(
            lambda t, y: vfj.func(y),
            (0, T), x0,
            method='BDF',
            jac=lambda t, y: vfj.jac(y).reshape(n, n),
            rtol=float(self.solver_info.get('tol', 1e-6)),
            atol=float(self.solver_info.get('tol', 1e-6)) * 1e-3,
            max_step=dt,
            t_eval=np.linspace(0, T, total_steps + 1),
        )

        if sol.status != 0:
            import warnings
            warnings.warn(f"BDF solver: {sol.message}")

        # Convert to trajectory list, drop pre-time
        traj = [sol.y[:, i].copy() for i in range(sol.y.shape[1])]
        return traj[self.pre_steps:]

    def _integrate_semi_implicit(self, states, variables_all, total_steps, dt):
        """Semi-implicit Euler with numerical diagonal damping.

        Stable for stiff models. Damping lam[i] = |df_i/dy_i| computed via
        FD at each step. Integration: y[i] += dt * f[i] / (1 + dt * lam[i]).

        Returns state trajectory as list of state arrays.
        """
        n = self.STATE_COUNT
        x = np.array(states[:n], dtype=float)
        vars_all = list(variables_all)
        eps_fd = 1e-8

        # Identify zeta indices (valve states to clamp to [0,1])
        zeta_indices = [i for i, info in enumerate(self.model.STATE_INFO)
                        if 'zeta' in info.get('name', '').lower()]

        traj = [x.copy()]

        for step in range(total_steps):
            t = step * dt
            rates = [0.0] * n
            self.model.compute_rates(t, list(x), rates, list(vars_all))

            # Numerical diagonal Jacobian for damping
            lam = np.zeros(n)
            for i in range(n):
                x_bump = list(x)
                h_i = max(abs(x[i]) * eps_fd, eps_fd)
                x_bump[i] += h_i
                r_bump = [0.0] * n
                self.model.compute_rates(t, x_bump, r_bump, list(vars_all))
                lam[i] = abs((r_bump[i] - rates[i]) / h_i)

            # Semi-implicit step
            for i in range(n):
                x[i] = x[i] + dt * rates[i] / (1.0 + dt * lam[i])

            # Clamp zeta to [0, 1]
            for z in zeta_indices:
                x[z] = max(0.0, min(1.0, x[z]))

            traj.append(x.copy())

        # Drop the pre_time spin-up region and align with self.tSim (the post-pre
        # output grid), matching the adaptive-RK45 path. The uniform grid here uses
        # the same dt as update_times, so traj[pre_steps:] lines up with self.tSim.
        # Do NOT overwrite self.tSim — that would re-introduce the pre-time region
        # and produce a different output length/origin than the RK45 integrator.
        return traj[self.pre_steps:]

    def _integrate_rk4(self, states, variables_all, total_steps, dt):
        """Fixed-step RK4.

        Deliberately step-for-step identical to the RK4 loop that compute_cost_and_gradient_tape
        records, so that a forward solve and the tape integrate the *same* discrete system. The
        AD gradient is the exact gradient of whatever the tape integrates; if the forward solve
        integrates something else (the adaptive RK45 in _integrate, say) then the gradient is
        the gradient of a different function than the cost, and a gradient-based optimiser is
        being fed inconsistent information. Any change here must be mirrored there.
        """
        n = self.STATE_COUNT
        st = [float(s) for s in states[:n]]
        vars_all = list(variables_all)

        traj = [list(st)]
        for step in range(total_steps):
            t = step * dt
            k1 = [0.0] * n
            self.model.compute_rates(t, st, k1, vars_all)
            st2 = [st[i] + 0.5 * dt * k1[i] for i in range(n)]
            k2 = [0.0] * n
            self.model.compute_rates(t + 0.5 * dt, st2, k2, vars_all)
            st3 = [st[i] + 0.5 * dt * k2[i] for i in range(n)]
            k3 = [0.0] * n
            self.model.compute_rates(t + 0.5 * dt, st3, k3, vars_all)
            st4 = [st[i] + dt * k3[i] for i in range(n)]
            k4 = [0.0] * n
            self.model.compute_rates(t + dt, st4, k4, vars_all)
            for i in range(n):
                st[i] = st[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            if step >= self.pre_steps:
                traj.append(list(st))

        self.states = np.array(st, dtype=float)
        self._rk_data = None
        return traj

    def _integrate(self, states, variables_all, total_steps, dt):
        """
        Adaptive RK45 (Dormand-Prince) integration.

        Uses the algorithm from arXiv:2410.01911 (Martins & Lakshtanov).
        Adaptive step size for accuracy; stores stages for discrete adjoint.

        Returns state trajectory: list of state arrays at each sim-time step.
        Also stores self._rk_data for adjoint computation.
        """
        n = self.STATE_COUNT
        x = np.array(states[:n], dtype=float)
        vars_all = list(variables_all)

        # Dormand-Prince 4(5) Butcher tableau
        a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ])
        b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        b_hat = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        s = 7

        tol = float(self.solver_info.get('tol', 1e-8))
        safety = 0.9
        h_min = 1e-12
        h_max = dt * 10

        t0 = 0.0
        tf = total_steps * dt
        t = t0
        h = dt

        def rhs(x_in, t_in):
            rates = [0.0] * n
            self.model.compute_rates(t_in, list(x_in), rates, list(vars_all))
            return np.array(rates, dtype=float)

        # Full trajectory storage
        all_t = [t]
        all_x = [x.copy()]
        all_h = []
        all_k = []

        steps = 0
        while t < tf - 1e-14:
            if t + h > tf:
                h = tf - t

            # Compute stages
            k = [None] * s
            for i in range(s):
                xi = x.copy()
                for j in range(i):
                    xi += h * a[i, j] * k[j]
                k[i] = rhs(xi, t + c[i] * h)

            # Higher-order solution
            x_new = x.copy()
            for i in range(s):
                x_new += h * b[i] * k[i]

            # Error estimate
            err = np.zeros(n)
            for i in range(s):
                err += h * (b[i] - b_hat[i]) * k[i]
            err_norm = np.linalg.norm(err / (1.0 + np.abs(x_new))) / max(np.sqrt(n), 1)

            if err_norm <= tol or h <= h_min:
                if h <= h_min and err_norm > tol:
                    print(f"WARNING: RK45 step accepted at h_min={h_min} with err_norm={err_norm:.2e} > tol={tol:.2e} at t={t:.4g}. Gradient may be inaccurate.")
                t += h
                x = x_new
                all_t.append(t)
                all_x.append(x.copy())
                all_h.append(h)
                all_k.append([ki.copy() for ki in k])
                steps += 1

            # Adjust step size
            if err_norm > 0:
                h_new = safety * h * (tol / err_norm) ** 0.2
            else:
                h_new = h * 2.0
            h = max(h_min, min(h_max, h_new))

            if steps > 10 * total_steps:
                break

        # Store for adjoint
        self._rk_data = {
            't': all_t, 'x': all_x, 'h': all_h, 'k': all_k,
            'n_states': n, 'vars_all': vars_all
        }

        # Interpolate onto uniform grid for get_results compatibility
        # Linear interpolation between adaptive-step trajectory points
        traj = []
        j = 0
        for ti in self.tSim:
            # Advance j to bracket ti
            while j < len(all_t) - 2 and all_t[j + 1] < ti:
                j += 1
            if j >= len(all_t) - 1:
                traj.append(list(all_x[-1]))
            elif abs(all_t[j + 1] - all_t[j]) < 1e-15:
                traj.append(list(all_x[j]))
            else:
                # Linear interpolation
                alpha = (ti - all_t[j]) / (all_t[j + 1] - all_t[j])
                xi = [(1 - alpha) * all_x[j][i] + alpha * all_x[j + 1][i]
                      for i in range(n)]
                traj.append(xi)

        return traj

    # ---- simulation ----
    def _warn_if_stiff(self, variables_all):
        """Probe the first second of dynamics and warn loudly if the model is stiff.

        AADC's tape-consistent integrators are all fixed-step, and fixed-step schemes are
        inaccurate or outright unstable on stiff systems -- on the stiff 3compartment model
        the fixed-step implicit-Euler tape solve deviates from CVODE by ~2e4%. Stiffness is
        therefore the single best predictor that an AADC run (forward or gradient) will be
        wrong, so we surface it before doing any work.

        Two cheap, independent signals over the first second of simulated time:
          * the spectral abscissa of the RHS Jacobian, max|Re(lambda)|: if
            max|Re(lambda)| * dt >> 1 the fastest mode is far below the step, i.e. stiff;
          * an explicit forward-Euler probe: if it blows up to non-finite / huge values, the
            model is stiff for an explicit fixed step (the class AADC's rk4 belongs to).
        Diagnostics only -- any failure here is swallowed so it can never break a run.
        """
        if getattr(self, '_stiffness_checked', False):
            return
        self._stiffness_checked = True
        try:
            n = self.STATE_COUNT
            vars_all = list(variables_all)

            def rhs(x):
                rates = [0.0] * n
                self.model.compute_rates(0.0, list(x), rates, list(vars_all))
                return np.array([float(r) for r in rates], dtype=float)

            def spectral_abscissa(x):
                f0 = rhs(x)
                if not np.all(np.isfinite(f0)):
                    return np.inf
                J = np.zeros((n, n))
                for j in range(n):
                    dxj = 1e-6 * max(abs(float(x[j])), 1e-8)
                    xp = np.array(x, dtype=float); xp[j] += dxj
                    J[:, j] = (rhs(xp) - f0) / dxj
                eig = np.linalg.eigvals(J)
                return float(np.max(np.abs(eig.real))) if eig.size else 0.0

            x0 = np.array([float(s) for s in self.states[:n]], dtype=float)
            dt = float(self.dt)
            worst_abscissa = spectral_abscissa(x0)

            # Explicit forward-Euler probe over the first second (or the sim horizon if shorter).
            probe_T = min(1.0, dt * (self.pre_steps + self.n_steps))
            n_probe = max(20, int(np.ceil(probe_T / dt)))
            h = probe_T / n_probe if n_probe else dt
            x = x0.copy()
            blew_up = False
            scale = 1.0 + float(np.max(np.abs(x0)))
            for k in range(n_probe):
                fx = rhs(x)
                if not np.all(np.isfinite(fx)):
                    blew_up = True
                    break
                x = x + h * fx
                if not np.all(np.isfinite(x)) or float(np.max(np.abs(x))) > 1e6 * scale:
                    blew_up = True
                    break
                if k % max(1, n_probe // 5) == 0:
                    worst_abscissa = max(worst_abscissa, spectral_abscissa(x))

            product = worst_abscissa * dt
            if blew_up or product > 1.0:
                fastest = (1.0 / worst_abscissa) if worst_abscissa > 0 else float('inf')
                warnings.warn(
                    "AADC stiffness check: the model appears STIFF over the first second of "
                    f"simulation (fastest mode timescale ~{fastest:.2e}s vs dt={dt:g}s; "
                    f"max|Re(eig(J))|*dt~{product:.1f}"
                    + ("; an explicit forward-Euler probe diverged" if blew_up else "")
                    + "). AADC's tape-consistent integrators are all fixed-step and are "
                    "inaccurate or unstable on stiff systems (e.g. ~2e4% error vs CVODE on "
                    "3compartment), so both the forward solve and the AADC gradient are likely "
                    "wrong here. Use model_type 'casadi_python' with solver_info method 'bdf', "
                    "or a Myokit CVODES forward-sensitivity run (model_type 'cellml_only', "
                    "solver 'CVODE_myokit', do_ad: true), for stiff models.")
        except Exception:
            # A diagnostic must never break the actual simulation.
            pass

    def run(self):
        """Run simulation. If _do_ad is True, records onto AADC tape."""
        total_steps = self.pre_steps + self.n_steps

        # Build full variables array from constants
        variables_all = list(self._numeric_variables_all)
        for const_pos, const_idx in enumerate(self.constant_indices):
            variables_all[const_idx] = self.variables[const_pos]

        # Stiffness is what breaks AADC's fixed-step tape methods (measured up to ~2e4% error
        # and unstable rootfinds on the stiff 3compartment valve dynamics), so probe the model
        # once and warn loudly before integrating.
        self._warn_if_stiff(variables_all)

        # Choose integrator based on method
        method = self.solver_info.get('method', 'adaptive_rk45')
        if method == 'implicit_euler_ift':
            traj = self._integrate_implicit_euler_ift(self.states, variables_all, total_steps, self.dt)
        elif method == 'bdf':
            self._patch_math_functions()
            traj = self._integrate_bdf(self.states, variables_all, total_steps, self.dt)
            self._unpatch_math_functions()
        elif method == 'semi_implicit':
            traj = self._integrate_semi_implicit(self.states, variables_all, total_steps, self.dt)
        elif method == 'rk4':
            # fixed-step, and identical to what the tape records -- the method to use when the
            # cost and the AD gradient have to be of the same function
            traj = self._integrate_rk4(self.states, variables_all, total_steps, self.dt)
        else:
            traj = self._integrate(self.states, variables_all, total_steps, self.dt)
        self.state_traj = np.array(traj).T  # (n_states, n_sim_steps)

        # Compute algebraic variables at each time point
        self._compute_var_traj(traj, variables_all)

        self._has_run = True
        return True

    def _compute_var_traj(self, state_traj_list, variables_all):
        """Compute algebraic variable trajectories from state trajectory."""
        var_names = list(self.var_name_to_idx.keys())
        n_vars = len(var_names)
        n_times = len(state_traj_list)
        self.var_traj = np.zeros((n_vars, n_times))

        for ti_idx, st in enumerate(state_traj_list):
            t = self.tSim[ti_idx] if ti_idx < len(self.tSim) else 0.0
            rates = [0.0] * self.STATE_COUNT
            vars_copy = list(variables_all)
            self.model.compute_rates(t, st, rates, vars_copy)
            try:
                self.model.compute_variables(t, st, rates, vars_copy)
            except AttributeError:
                pass
            for vi, name in enumerate(var_names):
                idx = self.var_name_to_idx[name]
                self.var_traj[vi, ti_idx] = float(vars_copy[idx])

    # ---- time ----
    def get_time(self, include_pre_time=False):
        if include_pre_time:
            return self.tSim
        else:
            return self.tSim - self.pre_time

    # ---- results ----
    def get_all_variable_names(self):
        return list(self.state_name_to_idx.keys()) + list(self.var_name_to_idx.keys())

    def _extract(self, name):
        if name == 'time':
            return self.tSim
        if name in self.state_name_to_idx:
            idx = self.state_name_to_idx[name]
            return self.state_traj[idx, :]
        if name in self.var_name_to_idx:
            var_names = list(self.var_name_to_idx.keys())
            idx = var_names.index(name)
            return self.var_traj[idx, :]
        kind, idx_res = self._resolve_name(name)
        if kind == "state":
            return self.state_traj[idx_res, :]
        raise ValueError(f"variable {name} not found")

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
        """Return all results as a dict keyed by variable name."""
        if not self._has_run:
            raise RuntimeError("Simulation has not been run yet.")
        names = self.get_all_variable_names()
        values = self.get_results(names, flatten=True)
        return {name: np.asarray(val) for name, val in zip(names, values)}

    # ---- AADC AD: discrete adjoint (arXiv:2410.01911) ----
    def _create_param_subset(self, param_names, param_vals=None):
        """Mark parameters for AD. Called by paramID before run()."""
        self._ad_param_names = [x[0] if isinstance(x, list) else x for x in param_names]
        self._ad_param_var_indices = []
        for name in self._ad_param_names:
            kind, idx = self._resolve_name(name)
            if kind == "var":
                self._ad_param_var_indices.append(idx)
            else:
                raise ValueError(f"AD parameter {name} must be a variable, got {kind}")
        if param_vals is not None:
            param_vals = np.asarray(param_vals, dtype=float)
            for i, name in enumerate(self._ad_param_names):
                kind, idx = self._resolve_name(name)
                if kind == "var":
                    self.variables[self._var_idx_to_const_pos(idx)] = param_vals[i]
        self._do_ad = True

        # Record AAD kernel immediately (needs fresh model state)
        variables_all = list(self._numeric_variables_all)
        for const_pos, const_idx in enumerate(self.constant_indices):
            variables_all[const_idx] = self.variables[const_pos]
        self._record_rhs_aad(variables_all)

    def _record_rhs_aad(self, variables_all):
        """Record the ODE RHS with AAD for vector-Jacobian products.

        Uses the same pattern as the verified standalone AadRhs:
        record compute_rates(t, x, rates, vars) with idouble x, p, t.
        """
        n = self.STATE_COUNT
        m = len(self._ad_param_names)

        # Use rk-adjoint-python's AadRhs which is already verified
        vars_list = list(variables_all)
        param_var_indices = list(self._ad_param_var_indices)
        model = self.model

        self._patch_math_functions()

        def rhs_for_aad(x, p, t):
            v = list(vars_list)
            for i, var_idx in enumerate(param_var_indices):
                v[var_idx] = p[i]
            rates = [aadc.idouble(0.0) for _ in range(n)]
            model.compute_rates(t, x, rates, v)
            return rates

        p0 = np.array([float(variables_all[idx]) for idx in param_var_indices])
        x0 = np.zeros(n)

        funcs = aadc.Functions()
        funcs.start_recording()

        id_x = [aadc.idouble(float(x0[i])) for i in range(n)]
        a_x = [xi.mark_as_input() for xi in id_x]

        id_p = [aadc.idouble(float(p0[i])) for i in range(m)]
        a_p = [pi.mark_as_input() for pi in id_p]

        id_t = aadc.idouble(0.0)
        a_t = id_t.mark_as_input()

        dxdt = rhs_for_aad(id_x, id_p, id_t)

        r_f = [fi.mark_as_output() for fi in dxdt]

        funcs.stop_recording()
        self._unpatch_math_functions()

        self._aad_funcs = funcs
        self._aad_a_x = a_x
        self._aad_a_p = a_p
        self._aad_a_t = a_t
        self._aad_r_f = r_f

    def _record_vjp_kernel(self):
        """Record a second AAD kernel: given adjoint seed v, compute v^T df/dx and v^T df/dp.

        This is the efficient approach from VectorizedAdjoint (arXiv:2410.01911):
        one forward + one reverse pass gives the full VJP, instead of n separate passes.
        """
        n = self.STATE_COUNT
        m = len(self._ad_param_names)

        funcs = aadc.Functions()
        funcs.start_recording()

        # Inputs: x, p, t (same as RHS kernel)
        id_x = [aadc.idouble(0.0) for _ in range(n)]
        a_x = [xi.mark_as_input() for xi in id_x]
        id_p = [aadc.idouble(0.0) for _ in range(m)]
        a_p = [pi.mark_as_input() for pi in id_p]
        id_t = aadc.idouble(0.0)
        a_t = id_t.mark_as_input()

        # Adjoint seed v (also input)
        id_v = [aadc.idouble(0.0) for _ in range(n)]
        a_v = [vi.mark_as_input() for vi in id_v]

        # Evaluate RHS
        vars_list = list(self._numeric_variables_all)
        for const_pos, const_idx in enumerate(self.constant_indices):
            vars_list[const_idx] = self.variables[const_pos]
        param_var_indices = list(self._ad_param_var_indices)

        self._patch_math_functions()
        v_copy = list(vars_list)
        for i, var_idx in enumerate(param_var_indices):
            v_copy[var_idx] = id_p[i]
        rates = [aadc.idouble(0.0) for _ in range(n)]
        self.model.compute_rates(id_t, id_x, rates, v_copy)
        self._unpatch_math_functions()

        # Compute v^T f as a scalar: sum_i v[i] * f[i]
        vtf = id_v[0] * rates[0]
        for i in range(1, n):
            vtf = vtf + id_v[i] * rates[i]

        # Output: v^T f (scalar). Gradients w.r.t. x and p give v^T df/dx and v^T df/dp.
        r_vtf = vtf.mark_as_output()

        funcs.stop_recording()

        self._vjp_funcs = funcs
        self._vjp_a_x = a_x
        self._vjp_a_p = a_p
        self._vjp_a_t = a_t
        self._vjp_a_v = a_v
        self._vjp_r_vtf = r_vtf

    def _vjp(self, x, p_vals, t, v):
        """Vector-Jacobian product via single AAD reverse pass.

        Computes v^T df/dx (shape n) and v^T df/dp (shape m) efficiently:
        one forward + one reverse, not n separate evaluations.
        """
        n = self.STATE_COUNT
        m = len(self._ad_param_names)

        if not hasattr(self, '_vjp_funcs') or self._vjp_funcs is None:
            self._record_vjp_kernel()

        inputs = {}
        for i in range(n):
            inputs[self._vjp_a_x[i]] = float(x[i])
        for i in range(m):
            inputs[self._vjp_a_p[i]] = float(p_vals[i])
        inputs[self._vjp_a_t] = float(t)
        for i in range(n):
            inputs[self._vjp_a_v[i]] = float(v[i])

        all_xp = list(self._vjp_a_x) + list(self._vjp_a_p)
        request = {self._vjp_r_vtf: all_xp}

        res = aadc.evaluate(self._vjp_funcs, request, inputs, self._aad_workers)

        vjp_x = np.array([float(np.asarray(res[1][self._vjp_r_vtf][self._vjp_a_x[j]]).flat[0])
                          for j in range(n)])
        vjp_p = np.array([float(np.asarray(res[1][self._vjp_r_vtf][self._vjp_a_p[j]]).flat[0])
                          for j in range(m)])

        return vjp_x, vjp_p

    def _record_rhs_inner_kernel(self, variables_all, n):
        """Record a small AADC kernel for the RHS only — used to compute
        Jacobian diagonal (damping) at each integration step via reverse mode."""
        import math as _math
        rhs_funcs = aadc.Functions()
        rhs_funcs.start_recording()
        id_si = [aadc.idouble(float(self.states[i])) for i in range(n)]
        a_si = [s.mark_as_input() for s in id_si]
        id_vi = [aadc.idouble(float(v) if not (isinstance(v, float) and _math.isnan(v)) else 0.0)
                 for v in variables_all]
        a_pi = None
        if self._ad_param_var_indices:
            idx0 = self._ad_param_var_indices[0]
            id_vi[idx0] = aadc.idouble(float(variables_all[idx0]))
            a_pi = id_vi[idx0].mark_as_input()
        id_ri = [aadc.idouble(0.0) for _ in range(n)]
        self.model.compute_rates(0.0, id_si, id_ri, list(id_vi))
        r_rates = []
        for i in range(n):
            r = id_ri[i] if hasattr(id_ri[i], 'mark_as_output') else aadc.idouble(float(id_ri[i]))
            r_rates.append(r.mark_as_output())
        rhs_funcs.stop_recording()

        self._rhs_funcs = rhs_funcs
        self._rhs_a_st = a_si
        self._rhs_a_param = a_pi
        self._rhs_r_rates = r_rates
        self._rhs_workers = aadc.ThreadPool(1)

    def compute_gradient_tape(self, cost_func_idouble):
        """dJ/dp only. See compute_cost_and_gradient_tape, which this delegates to.

        Prefer compute_cost_and_gradient_tape when the cost is also needed: the tape produces
        J(p) and dJ/dp in the same evaluation, and taking the cost from anywhere else risks
        differentiating one function while minimising another.
        """
        return self.compute_cost_and_gradient_tape(cost_func_idouble)[1]

    def compute_cost_and_gradient_tape(self, cost_func_idouble):
        """
        Compute J(p) and dJ/dp by recording the full ODE + cost on AADC tape.

        This is the fast path (~6ms for 27 states, 2200 steps).
        Records the entire forward integration with idouble, then
        one reverse pass gives all gradients. Memory: O(total_steps).

        Parameters
        ----------
        cost_func_idouble : callable
            J(states_idouble, params_idouble) -> idouble scalar.
            Must work with aadc.idouble arithmetic.

        Returns
        -------
        dJdp : np.array
            Gradient of J w.r.t. the AD parameters.
        """
        if not hasattr(self, '_ad_param_names'):
            raise RuntimeError("Call _create_param_subset() first")

        n = self.STATE_COUNT
        m = len(self._ad_param_names)

        variables_all = list(self._numeric_variables_all)
        for const_pos, const_idx in enumerate(self.constant_indices):
            variables_all[const_idx] = self.variables[const_pos]

        total_steps = self.pre_steps + self.n_steps
        dt = self.dt

        # Record full ODE integration on tape
        if not hasattr(self, '_tape_funcs') or self._tape_funcs is None:
            self._patch_math_functions()

            tape_method = self.solver_info.get('method', 'adaptive_rk45')

            funcs = aadc.Functions()
            funcs.start_recording()

            # Parameter inputs
            id_p = [aadc.idouble(float(variables_all[idx]))
                    for idx in self._ad_param_var_indices]
            a_p = [pi.mark_as_input() for pi in id_p]

            # Build variables with idouble params
            vars_rec = list(variables_all)
            for i, var_idx in enumerate(self._ad_param_var_indices):
                vars_rec[var_idx] = id_p[i]

            # Initial state seeded ON the tape via initialise_variables, so a parameter that
            # sets a state's initial value (e.g. an *_init constant) propagates to the
            # gradient. Seeding from frozen floats (self.states) would leave such parameters
            # disconnected from the tape and give them a spurious zero gradient column. A copy
            # of vars_rec is passed so initialise_variables cannot clobber the idouble
            # parameter entries used by the integration below.
            st = [aadc.idouble(0.0) for _ in range(n)]
            rates_seed = [aadc.idouble(0.0) for _ in range(n)]
            self.model.initialise_variables(st, rates_seed, list(vars_rec))

            if tape_method == 'implicit_euler_ift':
                st = self._integrate_implicit_euler_ift_on_tape(
                    st, vars_rec, total_steps, dt, n)
            elif tape_method == 'semi_implicit':
                zeta_idx = [i for i, info in enumerate(self.model.STATE_INFO)
                            if 'zeta' in info.get('name', '').lower()]

                # Record semi-implicit Euler with damping ON tape.
                # Jacobian diagonal df_i/dy_i computed via on-tape finite
                # differences: perturb state i, recompute rates, take ratio.
                # This puts lam on the same tape as rates → exact AD gradient.
                # (Previous version used a separate inner kernel → passive
                # floats → AD/FD ≈ 0.79 on stiff models.)
                FD_EPS = 1e-6

                self._tape_trajectory = [list(st)]
                for step in range(total_steps):
                    t_step = step * dt
                    # Rates at current state (on tape)
                    rates_id = [aadc.idouble(0.0) for _ in range(n)]
                    self.model.compute_rates(t_step, st, rates_id, list(vars_rec))

                    # Jacobian diagonal via on-tape FD: lam[i] = |df_i/dy_i|
                    lam = [aadc.idouble(0.0)] * n
                    for i in range(n):
                        h_i = aadc.iif(st[i] >= aadc.idouble(0.0), st[i], -st[i])
                        h_i = aadc.iif(h_i >= aadc.idouble(1e-8),
                                       h_i * aadc.idouble(FD_EPS),
                                       aadc.idouble(FD_EPS))
                        st_bump = list(st)
                        st_bump[i] = st[i] + h_i
                        rates_bump = [aadc.idouble(0.0) for _ in range(n)]
                        self.model.compute_rates(t_step, st_bump, rates_bump, list(vars_rec))
                        dlam = (rates_bump[i] - rates_id[i]) / h_i
                        lam[i] = aadc.iif(dlam >= aadc.idouble(0.0), dlam, -dlam)

                    # Semi-implicit step (all idouble — on tape)
                    for i in range(n):
                        st[i] = st[i] + aadc.idouble(dt) * rates_id[i] / (aadc.idouble(1.0) + aadc.idouble(dt) * lam[i])
                    for z in zeta_idx:
                        st[z] = aadc.iif(st[z] >= aadc.idouble(0.0), st[z], aadc.idouble(0.0))
                        st[z] = aadc.iif(st[z] <= aadc.idouble(1.0), st[z], aadc.idouble(1.0))

                    if step >= self.pre_steps:
                        self._tape_trajectory.append(list(st))
            else:
                # RK4 on tape (for non-stiff models)
                # Collect trajectory for trajectory-based cost functions (max, min, mean)
                self._tape_trajectory = [list(st)]  # initial state
                for step in range(total_steps):
                    t = aadc.idouble(step * dt)
                    k1 = [aadc.idouble(0.0) for _ in range(n)]
                    self.model.compute_rates(t, st, k1, list(vars_rec))
                    st2 = [st[i] + 0.5 * dt * k1[i] for i in range(n)]
                    k2 = [aadc.idouble(0.0) for _ in range(n)]
                    self.model.compute_rates(t + 0.5 * dt, st2, k2, list(vars_rec))
                    st3 = [st[i] + 0.5 * dt * k2[i] for i in range(n)]
                    k3 = [aadc.idouble(0.0) for _ in range(n)]
                    self.model.compute_rates(t + 0.5 * dt, st3, k3, list(vars_rec))
                    st4 = [st[i] + dt * k3[i] for i in range(n)]
                    k4 = [aadc.idouble(0.0) for _ in range(n)]
                    self.model.compute_rates(t + dt, st4, k4, list(vars_rec))
                    for i in range(n):
                        st[i] = st[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
                    # Store post-pre_time trajectory on tape
                    if step >= self.pre_steps:
                        self._tape_trajectory.append(list(st))

            # Cost — pass trajectory if cost function accepts it, else final state only
            import inspect
            sig = inspect.signature(cost_func_idouble)
            if len(sig.parameters) >= 3:
                # cost_func(final_state, params, trajectory)
                cost = cost_func_idouble(st, id_p, getattr(self, '_tape_trajectory', None))
            else:
                cost = cost_func_idouble(st, id_p)
            r_cost = cost.mark_as_output()

            funcs.stop_recording()
            self._unpatch_math_functions()

            self._tape_funcs = funcs
            self._tape_a_p = a_p
            self._tape_r_cost = r_cost

        # Evaluate
        p_vals = [float(variables_all[idx]) for idx in self._ad_param_var_indices]
        inputs = {self._tape_a_p[i]: p_vals[i] for i in range(m)}
        request = {self._tape_r_cost: list(self._tape_a_p)}

        res = aadc.evaluate(self._tape_funcs, request, inputs, self._aad_workers)

        # J(p) and dJ/dp come out of the *same* tape evaluation, so they are necessarily the
        # cost and the gradient of the cost of one and the same function.
        cost = float(np.asarray(res[0][self._tape_r_cost]).flat[0])
        dJdp = np.array([float(np.asarray(res[1][self._tape_r_cost][self._tape_a_p[j]]).flat[0])
                         for j in range(m)])
        return cost, dJdp

    def compute_gradient_batch(self, param_array, cost_func_idouble=None):
        """
        Batch gradient evaluation for multiple parameter sets.

        Uses multi-threading + AVX vectorization for throughput.
        Requires tape-based kernel (recorded by compute_gradient_tape).

        Parameters
        ----------
        param_array : np.array, shape (N, m)
            N sets of m parameters.
        cost_func_idouble : callable or None
            If tape not yet recorded, uses this to record.

        Returns
        -------
        costs : np.array, shape (N,)
        grads : np.array, shape (N, m)
        """
        if not hasattr(self, '_tape_funcs') or self._tape_funcs is None:
            if cost_func_idouble is not None:
                self.compute_gradient_tape(cost_func_idouble)
            else:
                raise RuntimeError("Tape kernel not recorded. Call compute_gradient_tape or pass cost_func_idouble.")

        m = len(self._ad_param_names)
        N = param_array.shape[0]

        inputs = {self._tape_a_p[j]: param_array[:, j] for j in range(m)}
        request = {self._tape_r_cost: list(self._tape_a_p)}

        res = aadc.evaluate(self._tape_funcs, request, inputs, self._aad_workers)

        costs = np.asarray(res[0][self._tape_r_cost]).flatten()[:N]
        grads = np.zeros((N, m))
        for j in range(m):
            grads[:, j] = np.asarray(res[1][self._tape_r_cost][self._tape_a_p[j]]).flatten()[:N]

        return costs, grads

    def compute_hessian(self, cost_func, eps=1e-5):
        """
        Compute Hessian via FD of tape gradient. Uses batch evaluation.

        Returns
        -------
        H : np.array, shape (m, m)
        """
        m = len(self._ad_param_names)
        variables_all = list(self._numeric_variables_all)
        for const_pos, const_idx in enumerate(self.constant_indices):
            variables_all[const_idx] = self.variables[const_pos]
        p0 = np.array([float(variables_all[idx]) for idx in self._ad_param_var_indices])

        # Build 2m parameter sets (up/down for each param)
        param_sets = []
        for i in range(m):
            h = p0[i] * eps if p0[i] != 0 else eps
            p_up = p0.copy(); p_up[i] += h
            p_dn = p0.copy(); p_dn[i] -= h
            param_sets.append(p_up)
            param_sets.append(p_dn)
        param_array = np.array(param_sets)

        # Batch evaluate all 2m gradient evaluations at once
        def cost_id(st, p):
            return cost_func(st)
        _, grads = self.compute_gradient_batch(param_array, cost_func_idouble=cost_id)

        # Compute Hessian from gradient differences
        H = np.zeros((m, m))
        for i in range(m):
            h = p0[i] * eps if p0[i] != 0 else eps
            grad_up = grads[2*i, :]
            grad_dn = grads[2*i+1, :]
            H[i, :] = (grad_up - grad_dn) / (2 * h)

        return H

    def compute_gradient(self, cost_func, dJdx_T=None, method='auto'):
        """
        Compute dJ/dp. Automatically selects the fastest method.

        Methods:
          'tape'    — record full ODE on tape, one reverse pass (~6ms).
                      Fast but memory O(total_steps).
          'adjoint' — discrete adjoint (arXiv:2410.01911), per-stage VJP.
                      Slower (~65ms) but memory O(1 step).
          'auto'    — tape if total_steps < 50000, else adjoint.

        Parameters
        ----------
        cost_func : callable
            J(x_T) -> scalar. Cost function of final state.
        dJdx_T : np.array or None
            If provided, gradient of cost w.r.t. final state (avoids FD).
        method : str
            'auto', 'tape', or 'adjoint'.

        Returns
        -------
        dJdp : np.array
            Gradient of J w.r.t. the AD parameters.
        """
        # Auto-select method
        total_steps = self.pre_steps + self.n_steps
        if method == 'auto':
            method = 'tape' if total_steps < 50000 else 'adjoint'

        if method == 'tape':
            # Fast path: record full ODE on tape
            n = self.STATE_COUNT
            if cost_func is not None:
                # Wrap the user cost for tape recording (end-point cost of final state).
                def cost_idouble(st, p, _cf=cost_func):
                    return _cf(st)
            else:
                # Generic fallback: sum of squares of the final state.
                def cost_idouble(st, p):
                    result = st[0] * st[0]
                    for i in range(1, n):
                        result = result + st[i] * st[i]
                    return result
            return self.compute_gradient_tape(cost_idouble)

        # Discrete adjoint path
        if not hasattr(self, '_rk_data') or self._rk_data is None:
            raise RuntimeError("Must call run() before compute_gradient()")
        if not hasattr(self, '_ad_param_names'):
            raise RuntimeError("Must call _create_param_subset() before compute_gradient()")

        rk = self._rk_data
        all_x = rk['x']
        all_h = rk['h']
        all_k = rk['k']
        all_t = rk['t']
        n = rk['n_states']
        N = len(all_h)

        # Get current parameter values
        p_vals = np.array([self._numeric_variables_all[idx]
                           for idx in self._ad_param_var_indices], dtype=float)
        # Update from self.variables
        for i, var_idx in enumerate(self._ad_param_var_indices):
            const_pos = self._var_idx_to_const_pos(var_idx)
            p_vals[i] = float(self.variables[const_pos])

        if not hasattr(self, '_aad_funcs') or self._aad_funcs is None:
            raise RuntimeError("AAD kernel not recorded. Call _create_param_subset() first.")

        # Terminal condition
        if dJdx_T is not None:
            wbarend = np.array(dJdx_T, dtype=float)
        else:
            x_T = np.array(all_x[-1])
            J0 = cost_func(x_T)
            wbarend = np.zeros(n)
            eps = 1e-7
            for i in range(n):
                x_up = x_T.copy(); x_up[i] += eps
                wbarend[i] = (cost_func(x_up) - J0) / eps

        # Butcher tableau (Dormand-Prince) — used by backward sweep below
        # (arXiv:2410.01911, Algorithm 1; ported from C++ backpropagation.hpp)
        a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ])
        b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        s = 7

        alphabar = np.zeros(len(p_vals))

        for step in range(N - 1, -1, -1):
            h = all_h[step]
            x_n = np.array(all_x[step])
            k = all_k[step]
            t_n = all_t[step]

            # Initialize stage adjoints (C++ back_prop_step lines 171-176)
            w_bar = np.zeros((n, s + 2))
            w_bar[:, s + 1] = wbarend

            # Distribute incoming adjoint to stages (C++ lines 179-184)
            for i in range(n):
                w_bar[i, 0] += w_bar[i, s + 1]
                for mm in range(1, s + 1):
                    w_bar[i, mm] += b[mm - 1] * h * w_bar[i, s + 1]

            # Backward through stages s to 1 (C++ lines 196-224)
            for mm in range(s, 0, -1):
                t_mn = t_n + c[mm - 1] * h

                # Reconstruct intermediate state (C++ get_intermediate_state)
                x_mn = x_n.copy()
                for kk in range(1, mm):
                    x_mn += h * a[mm - 1, kk - 1] * k[kk - 1]

                w_bar_m = w_bar[:, mm].copy()
                vjp_x, vjp_p = self._vjp(x_mn, p_vals, t_mn, w_bar_m)

                # Update stage 0 adjoint (C++ line 211)
                for i in range(n):
                    w_bar[i, 0] += vjp_x[i]

                # Update earlier stage adjoints (C++ lines 214-216)
                for kk in range(1, mm):
                    for i in range(n):
                        w_bar[i, kk] += vjp_x[i] * a[mm - 1, kk - 1] * h

                # Accumulate parameter sensitivity (C++ lines 220-222)
                alphabar += vjp_p

            # Update wbarend for next step (C++ lines 227-228)
            wbarend[:] = w_bar[:, 0]

        return alphabar

    # ---- reset helpers ----
    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        offline_pre_time = float(offline_pre_time)
        if offline_pre_time <= 0:
            return
        self._do_ad = False
        self.update_times(self.dt, 0.0, offline_pre_time, 0.0)
        self.run()
        self.states = list(self.state_traj[:, -1])
        self.default_state_inits = list(self.states)
        self._has_run = False

    def reset_and_clear(self, only_one_exp=-1):
        self._do_ad = False
        self._aad_funcs = None  # kernel recorded by _record_rhs_aad
        self._rk_data = None    # trajectory stored by _integrate
        self._init_state()

    def reset_states(self):
        self.states = list(self.default_state_inits)

    def close_simulation(self):
        pass
