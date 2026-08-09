"""AADC (Matlogica) tape-based cost and gradient for the param-id classes.

The AADC backend records the forward integration on a tape and replays it, so the cost has
to be re-implemented on that tape rather than reusing the generic cost-assembly path -- both
the cost and its gradient then come out of one tape evaluation and cannot drift apart. That
is why this module, unlike casadi_backend and fsa_backend, calls no generic method of the
param-id object at all: it needs only ``pid.sim_helper`` and the raw obs/protocol/param
dicts.

``cost_and_grad(pid, param_vals)`` takes the param-id object as its first argument. Nothing
here imports ``param_id.paramID``; the module imports standalone.

AADC is optional third-party proprietary software. It is imported lazily inside
``cost_on_tape`` (i.e. only once a tape is actually being recorded), exactly as before the
extraction -- hoisting it to module scope would change when ImportError surfaces on a
machine without AADC.
"""
import warnings

import numpy as np

# The methods whose forward integration the tape can record step-for-step. Defined in
# parsers.PrimitiveParsers alongside SOLVER_SCHEMA['ad_suitable_methods'], which is derived from
# it, so the advertised menu and the check enforced here cannot drift (issue #336). Re-exported
# under the original name for callers that already import it from this module.
from parsers.PrimitiveParsers import (AADC_TAPE_CONSISTENT_METHODS as TAPE_CONSISTENT_METHODS,
                                      AADC_LEGACY_METHOD_ALIASES)
BDF_NEWTON_METHOD = 'bdf_newton'
SEMI_IMPLICIT_SIGNED_METHOD = 'semi_implicit_signed'
# Legacy names kept so configs written before the split still run; both now select a
# gradient_strategy of the one method rather than naming an integrator (issue #346).
BDF_TAPE_METHOD = 'bdf_tape'
BDF_KERNEL_METHOD = 'bdf_kernel'
GRADIENT_STRATEGIES = ('tape', 'kernel')
DEFAULT_GRADIENT_STRATEGY = 'tape'


def resolve_gradient_strategy(method, solver_info):
    """Resolve a configured method to ``(canonical_method, strategy)`` for the signed scheme.

    Returns ``None`` for any method that is not ``semi_implicit_signed`` or one of its legacy
    aliases, so the caller can fall through to the other gradient paths.

    The alias table lives in PrimitiveParsers next to the schema that advertises the split, and
    is read here rather than restated: a second copy of the mapping is exactly how 'bdf_tape'
    and 'bdf_kernel' drifted into looking like two integrators in the first place (issue #346).
    """
    if method in AADC_LEGACY_METHOD_ALIASES:
        canonical, strategy = AADC_LEGACY_METHOD_ALIASES[method]
        configured = (solver_info or {}).get('gradient_strategy')
        if configured is not None and configured != strategy:
            # The legacy name is the more specific statement of intent, so it wins -- but say so,
            # rather than dropping a setting the user deliberately wrote.
            warnings.warn(
                f"solver_info method {method!r} already fixes the gradient strategy to "
                f"{strategy!r}, so gradient_strategy={configured!r} is ignored. Use method "
                f"{canonical!r} to choose the strategy explicitly.", stacklevel=2)
        return canonical, strategy

    if method != SEMI_IMPLICIT_SIGNED_METHOD:
        return None

    strategy = (solver_info or {}).get('gradient_strategy') or DEFAULT_GRADIENT_STRATEGY
    if strategy not in GRADIENT_STRATEGIES:
        raise ValueError(
            f"solver_info['gradient_strategy'] must be one of {list(GRADIENT_STRATEGIES)}, got "
            f"{strategy!r}. It selects how '{SEMI_IMPLICIT_SIGNED_METHOD}' evaluates its "
            f"gradient; the integration is the same either way.")
    return SEMI_IMPLICIT_SIGNED_METHOD, strategy


def cost_and_grad(pid, param_vals):
    """Compute J(p) and ∇J(p) via AADC.

    For tape-consistent methods (rk4, semi_implicit, etc.): records the entire
    ODE integration + cost on an AADC tape and gets the gradient via one reverse pass.

    For bdf_newton: uses Newton forward solve with VFJ Jacobian + accumulated IFT
    sensitivity. No full tape — just a small kernel for compute_rates, replayed per step.
    This is the AADC analogue of CasADi's symbolic BDF with rootfinder + IFT.
    """
    # Common setup: resolve param names and set values
    param_names_raw = pid.param_id_info["param_names"]
    param_names = []
    for pn in param_names_raw:
        if isinstance(pn, (list, tuple)):
            param_names.append(pn[0])
        else:
            param_names.append(pn)

    pid.sim_helper.set_param_vals(param_names_raw, param_vals)
    pid.sim_helper._ad_param_names = list(param_names)

    ad_indices = []
    for pname in param_names:
        kind, idx = pid.sim_helper._resolve_name(pname)
        if kind == "var":
            ad_indices.append(idx)
        elif kind == "state":
            raise ValueError(f"Param '{pname}' resolves to a state, not a variable.")
        else:
            raise ValueError(f"Param '{pname}' not found by name resolver.")
    pid.sim_helper._ad_param_var_indices = ad_indices

    method = pid.sim_helper.solver_info.get('method', 'adaptive_rk45')
    if method == BDF_NEWTON_METHOD:
        return _cost_and_grad_bdf_newton(pid, param_vals)
    # One method, two execution strategies. The legacy names carried the strategy in the method
    # itself; they still select it, so old configs keep working (issue #346).
    signed = resolve_gradient_strategy(method, pid.sim_helper.solver_info)
    if signed is not None:
        if signed[1] == 'kernel':
            return _cost_and_grad_bdf_kernel(pid, param_vals)
        return _cost_and_grad_bdf_tape(pid, param_vals)
    if method not in TAPE_CONSISTENT_METHODS:
        raise ValueError(
            f"solver method '{method}' cannot be recorded on an AADC tape, so the forward "
            f"solve and the gradient would integrate different systems. With do_ad, use one "
            f"of {list(TAPE_CONSISTENT_METHODS)} or '{BDF_NEWTON_METHOD}' (for stiff models).")

    # Build cost function that works with idouble on tape.
    # Receives: final state (list of idouble), params (list of idouble),
    # and optionally the full trajectory (list of lists of idouble).
    obs_info = pid.obs_info
    sim_helper = pid.sim_helper
    operation_funcs = pid.operation_funcs_dict
    cost_funcs = pid.cost_funcs_dict
    cost_types = pid.cost_type

    # The tape records one straight-line integration, so it cannot express a protocol with
    # several experiments / sub-experiments (each of which resets the state and changes
    # parameters). Refuse rather than silently differentiate the wrong thing.
    num_experiments = pid.protocol_info["num_experiments"] if pid.protocol_info else 1
    num_sub_per_exp = pid.protocol_info["num_sub_per_exp"] if pid.protocol_info else [1]
    if num_experiments > 1 or any(n > 1 for n in num_sub_per_exp):
        raise NotImplementedError(
            f'the AADC tape cannot represent a protocol with {num_experiments} experiment(s) '
            f'and sub-experiment counts {list(num_sub_per_exp)}: it records a single '
            f'straight-line integration. Use a single-experiment obs_data, or turn off do_ad.')

    # The tape cost (cost_on_tape below) can only reproduce observables whose operand is a
    # STATE and whose operation the tape re-implements (max/min/mean, or a plain final
    # value; series). An operand that is an *algebraic* variable (not a state) resolves to
    # no state index and cannot be put on the tape, and operations such as max_minus_min are
    # not reproduced either. Such an observable would be silently dropped from the tape cost,
    # making the tape cost -- and therefore its gradient -- a different function than the one
    # being minimised, so the optimiser would descend the wrong cost. Refuse rather than
    # silently mislead. (Fully supporting algebraic observables needs the algebraic variables
    # recomputed on the tape from the state trajectory, tracked in issue #258.)
    def _operand_is_tapeable(op):
        """Check if an operand can be represented on the AADC tape.

        Returns True for states (directly on tape) and algebraic variables
        (computable from states via compute_variables on tape).
        """
        kind, _ = pid.sim_helper._resolve_name(op)
        return kind in ('state', 'var')

    supported_const_ops = (None, 'max', 'min', 'mean', 'max_minus_min')
    operand_names_o = pid.obs_info.get("operands", []) if pid.obs_info else []
    operations_o = pid.obs_info.get("operations", []) if pid.obs_info else []
    data_types_o = pid.obs_info.get("data_types", []) if pid.obs_info else []

    # Collect which algebraic variable indices are needed on tape
    needed_var_indices = set()
    for jj in range(len(operand_names_o)):
        op = operand_names_o[jj][0] if isinstance(operand_names_o[jj], (list, tuple)) \
            else operand_names_o[jj]
        kind, idx = pid.sim_helper._resolve_name(op)
        if kind == 'var':
            needed_var_indices.add(idx)
    needed_var_indices = sorted(needed_var_indices)
    # Store on sim_helper so the tape recording can use it
    pid.sim_helper._needed_var_indices = needed_var_indices

    untaped = []
    for jj in range(len(operand_names_o)):
        op = operand_names_o[jj][0] if isinstance(operand_names_o[jj], (list, tuple)) \
            else operand_names_o[jj]
        dtype = data_types_o[jj] if jj < len(data_types_o) else 'constant'
        oper = operations_o[jj] if jj < len(operations_o) else None
        if dtype == 'constant':
            if not _operand_is_tapeable(op) or oper not in supported_const_ops:
                untaped.append(f"{op} (op={oper})")
        elif dtype == 'series':
            if not _operand_is_tapeable(op):
                untaped.append(f"{op} (series)")
        else:
            untaped.append(f"{op} (data_type={dtype})")
    if untaped:
        raise NotImplementedError(
            f"AADC is not usable with this observable set: {len(untaped)} of "
            f"{len(operand_names_o)} observable(s) cannot be represented on the AADC tape: "
            f"{untaped}. Supported: state or algebraic-variable operands with "
            f"max/min/mean/max_minus_min operations (or series).")

    weighted_obs_denominator = 0
    if pid._num_weighted_obs_by_exp_sub is not None:
        for exp_idx in range(num_experiments):
            for sub_idx in range(num_sub_per_exp[exp_idx]):
                weighted_obs_denominator += pid._num_weighted_obs_by_exp_sub[exp_idx][sub_idx]

    def cost_on_tape(states_idouble, params_idouble, trajectory=None, var_trajectory=None):
        import aadc as _aadc
        from param_id.math_backend import make_math_backend
        mb = make_math_backend("aadc")

        cost = _aadc.idouble(0.0)
        if obs_info is None:
            return cost

        def _cost_scale(obs_idx):
            """The constant in front of the normalised squared residual, for the cost type
            configured on this observable.

            The tape re-implements the cost by hand, so it has to reproduce the *configured*
            cost function exactly -- if it does not, the gradient is the gradient of some
            other function than the one being minimised. gaussian_MLE is
            ``0.5 * mean(((x - mu)/std)^2 * w)`` and MSE is twice that. The hand-rolled form
            below was missing the 0.5, which made every tape cost exactly 2x the real one.
            """
            name = cost_types[obs_idx] if obs_idx < len(cost_types) else 'gaussian_MLE'
            if name == 'gaussian_MLE':
                return 0.5
            if name == 'MSE':
                return 1.0
            raise NotImplementedError(
                f"cost_type '{name}' cannot be recorded on an AADC tape yet; the tape cost "
                f"would not match the cost the optimiser minimises. Use gaussian_MLE or MSE, "
                f"or turn off do_ad.")

        gt_const = obs_info.get("ground_truth_const", [])
        std_const = obs_info.get("std_const_vec", [])
        operations = obs_info.get("operations", [])
        operand_names = obs_info.get("operands", [])
        data_types = obs_info.get("data_types", [])
        weights_const = pid.protocol_info["scaled_weight_const_from_exp_sub"][0][0] \
            if pid.protocol_info else np.ones(len(gt_const))

        # Helper: resolve operand name to state index. _resolve_name is authoritative --
        # see _operand_is_state for why the leaf-name fallback that used to live here was
        # removed (it could bind the tape to a same-leaf state in an unrelated component,
        # consistently in both cost and gradient, so nothing downstream could detect it).
        def _resolve_obs_idx(op_name):
            """Resolve operand to (source, index) where source is 'state' or 'var'.

            For states: index into states array.
            For vars: index into the var_trajectory (position in needed_var_indices).
            """
            kind, resolved_idx = sim_helper._resolve_name(op_name)
            if kind == "state":
                return ('state', resolved_idx)
            if kind == "var" and resolved_idx in needed_var_indices:
                return ('var', needed_var_indices.index(resolved_idx))
            return (None, None)

        gt_series = obs_info.get("ground_truth_series", [])
        std_series = obs_info.get("std_series_vec", [])
        obs_dts = obs_info.get("obs_dt", [])
        sim_dt = float(sim_helper.dt)
        weights_series = pid.protocol_info["scaled_weight_series_from_exp_sub"][0][0] \
            if pid.protocol_info and "scaled_weight_series_from_exp_sub" in pid.protocol_info \
            else np.ones(len(gt_series)) if gt_series else np.array([])

        const_idx = 0
        series_idx = 0
        for jj in range(len(operand_names)):
            op_name = operand_names[jj][0] if isinstance(operand_names[jj], (list, tuple)) else operand_names[jj]
            operation = operations[jj]
            source, si = _resolve_obs_idx(op_name)

            if data_types[jj] == 'constant':
                if const_idx >= len(gt_const) or source is None:
                    const_idx += 1
                    continue

                # Apply operation to trajectory
                if trajectory is not None and operation in ('max', 'min', 'mean', 'max_minus_min'):
                    if source == 'state':
                        series_vals = [trajectory[t][si] for t in range(len(trajectory))]
                    elif source == 'var' and var_trajectory is not None:
                        series_vals = [var_trajectory[t][si] for t in range(len(var_trajectory))]
                    else:
                        const_idx += 1
                        continue
                    if operation == 'max':
                        obs_val = mb.max(series_vals)
                    elif operation == 'min':
                        obs_val = mb.min(series_vals)
                    elif operation == 'mean':
                        obs_val = mb.mean(series_vals)
                    elif operation == 'max_minus_min':
                        obs_val = mb.max(series_vals) - mb.min(series_vals)
                else:
                    if source == 'state':
                        obs_val = states_idouble[si]
                    elif source == 'var' and var_trajectory is not None and len(var_trajectory) > 0:
                        obs_val = var_trajectory[-1][si]  # final value
                    else:
                        const_idx += 1
                        continue

                gt_val = _aadc.idouble(float(gt_const[const_idx]))
                std_val = _aadc.idouble(float(std_const[const_idx]))
                w = _aadc.idouble(float(weights_const[const_idx]))
                diff = (obs_val - gt_val) / std_val
                cost = cost + diff * diff * w * _aadc.idouble(_cost_scale(jj))
                const_idx += 1

            elif data_types[jj] == 'series':
                # Series: compare trajectory at each time point
                if trajectory is None or source is None:
                    series_idx += 1
                    continue
                if series_idx >= len(gt_series):
                    series_idx += 1
                    continue

                gt_s = gt_series[series_idx]
                # std may be one value for the whole series or one per sample
                std_raw = np.asarray(std_series[series_idx], dtype=float) \
                    if series_idx < len(std_series) else np.asarray(1.0)
                if std_raw.ndim == 0:
                    std_raw = np.full(len(gt_s), float(std_raw))
                w_s = float(weights_series[series_idx]) if series_idx < len(weights_series) else 1.0

                if w_s > 0 and gt_s is not None:
                    # trajectory[k] is the state at time k*dt, and ground-truth sample k is
                    # at k*obs_dt. Those grids only coincide when dt == obs_dt, so line the
                    # simulated series up with the observation times by linear interpolation
                    # -- the weights depend only on the two grids, never on the parameters,
                    # so this stays on tape and stays differentiable. Indexing
                    # trajectory[t_idx] directly (as this did) silently compared the
                    # simulation against the wrong times whenever dt != obs_dt.
                    obs_dt_s = float(obs_dts[series_idx]) if series_idx < len(obs_dts) \
                        else sim_dt
                    n_traj = len(trajectory)
                    n_pts = 0
                    terms = []
                    traj_src = trajectory if source == 'state' else var_trajectory
                    if traj_src is None:
                        series_idx += 1
                        continue
                    for k in range(len(gt_s)):
                        pos = k * obs_dt_s / sim_dt
                        lower = int(np.floor(pos))
                        if lower >= n_traj - 1:
                            if lower == n_traj - 1 and abs(pos - lower) < 1e-9:
                                sim_val = traj_src[lower][si]
                            else:
                                break  # past the end of the simulation: no data to compare
                        else:
                            frac = pos - lower
                            sim_val = (traj_src[lower][si] * _aadc.idouble(1.0 - frac)
                                       + traj_src[lower + 1][si] * _aadc.idouble(frac))
                        terms.append((sim_val, float(gt_s[k]), float(std_raw[k])))
                        n_pts += 1

                    for sim_val, gt_val_k, std_k in terms:
                        diff = (sim_val - _aadc.idouble(gt_val_k)) / _aadc.idouble(std_k)
                        cost = cost + diff * diff * _aadc.idouble(
                            _cost_scale(jj) * w_s / n_pts)

                series_idx += 1

        # get_cost_obs_and_pred_from_params divides the summed sub-costs by the total
        # number of weighted observable slots, so the tape must do the same or its cost is
        # a constant multiple of the real one -- and a constant multiple of the cost has a
        # constant multiple of the gradient, which is exactly as wrong for a line search.
        # (This was the factor of 2 in the measured AD/FD = [2, 2, 2, 2].)
        if weighted_obs_denominator > 0:
            cost = cost / _aadc.idouble(float(weighted_obs_denominator))

        return cost

    # Run forward + reverse on AADC tape. Both the cost and the gradient come out of the
    # same evaluation, so get_cost_aadc and get_jac_cost_aadc cannot drift apart.
    return pid.sim_helper.compute_cost_and_gradient_tape(cost_on_tape)


def _cost_and_grad_bdf_newton(pid, param_vals):
    """BDF Newton gradient via accumulated IFT sensitivity.

    Forward: Newton implicit Euler with VFJ Jacobian (Python compute_rates,
    AADC adjoint for df/dy). Gradient: forward sensitivity S = dx/dp
    accumulated per step via IFT: S_{n+1} = (I - dt*J)^{-1} * (S_n + dt*df/dp).
    Uses the same LU factorization as Newton → no extra ill-conditioning.

    Verified: AD/FD = 1.000 on 3compartment (27 states, 4 params, 22000 steps).
    """
    import aadc
    from scipy.linalg import lu_factor, lu_solve

    sim_helper = pid.sim_helper
    n = sim_helper.STATE_COUNT
    dt = sim_helper.dt
    total_steps = sim_helper.pre_steps + sim_helper.n_steps
    pre_steps = sim_helper.pre_steps

    # Set parameter values
    param_names_raw = pid.param_id_info["param_names"]
    param_names = []
    for pn in param_names_raw:
        param_names.append(pn[0] if isinstance(pn, (list, tuple)) else pn)

    variables_all = list(sim_helper._numeric_variables_all)
    for ci, idx in enumerate(sim_helper.constant_indices):
        variables_all[idx] = sim_helper.variables[ci]

    ad_indices = sim_helper._ad_param_var_indices
    n_p = len(ad_indices)
    for i, idx in enumerate(ad_indices):
        variables_all[idx] = float(param_vals[i])
    p_vals = [float(param_vals[i]) for i in range(n_p)]

    # Build VFJ kernel if needed
    sim_helper._integrate_bdf_newton(sim_helper.states, variables_all, 0, dt)  # init VFJ

    vfj = sim_helper._bdf_vfj
    vfj.set_params(np.array(p_vals))
    rhs_f = sim_helper._bdf_rhs_f
    a_p = sim_helper._bdf_a_p
    a_x = sim_helper._bdf_a_x
    r_r = sim_helper._bdf_r_r
    workers = sim_helper._aad_workers

    # Sub-stepping
    max_step = float(sim_helper.solver_info.get('max_step', 0.001))
    n_sub = max(1, int(np.ceil(dt / max_step)))
    idt = dt / n_sub
    max_newton = 4
    newton_tol = 1e-10

    zeta_indices = [i for i, info in enumerate(sim_helper.model.STATE_INFO)
                    if 'zeta' in info.get('name', '').lower()]

    # Forward + accumulated sensitivity
    x = np.array(sim_helper.states[:n], dtype=float)
    S = np.zeros((n, n_p))  # accumulated dx/dp

    # Initialize S for parameters that set initial state values (*_init pattern)
    state_name_to_idx = sim_helper.state_name_to_idx
    for k in range(n_p):
        pname = param_names[k]
        # Strip component prefix, e.g. 'global/q_lv_init' → 'q_lv_init'
        short = pname.split('/')[-1] if '/' in pname else pname
        if short.endswith('_init'):
            state_stem = short[:-5]  # 'q_lv_init' → 'q_lv'
            # Find matching state
            for sname, sidx in state_name_to_idx.items():
                s_short = sname.split('/')[-1] if '/' in sname else sname
                if s_short == state_stem:
                    S[sidx, k] = 1.0
                    break

    traj_sim = []
    S_history = []  # sensitivity at each sim trajectory point

    # Pre-build inputs template
    inputs_template = {a_p[k]: p_vals[k] for k in range(n_p)}
    request_dfdp = {r: a_p for r in r_r}
    need_sensitivity = True  # compute sensitivity from the start (warmup affects steady state)
    eye_n = np.eye(n)
    x_prev = None  # for BDF2
    S_prev = None  # sensitivity at step n-1 (for BDF2)

    for step in range(total_steps):

        for sub in range(n_sub):
            y = x.copy()
            lu_piv = None
            use_bdf2 = x_prev is not None

            for nit in range(max_newton):
                rates_arr = vfj.func(y)
                if use_bdf2:
                    F = y - (4.0/3.0) * x + (1.0/3.0) * x_prev - (2.0/3.0) * idt * rates_arr
                    jac_coeff = 2.0 / 3.0
                else:
                    F = y - x - idt * rates_arr
                    jac_coeff = 1.0
                if np.max(np.abs(F)) < newton_tol:
                    break
                J_rhs = vfj.jac(y)
                J_g = eye_n - jac_coeff * idt * J_rhs
                try:
                    lu_piv = lu_factor(J_g)
                    dy = lu_solve(lu_piv, -F)
                except np.linalg.LinAlgError:
                    break
                y += dy

            # IFT sensitivity: S_{n+1} = (I - c*idt*J)^{-1} * (rhs_S)
            if need_sensitivity and lu_piv is not None:
                inputs = dict(inputs_template)
                for i in range(n):
                    inputs[a_x[i]] = float(y[i])
                res_eval = aadc.evaluate(rhs_f, request_dfdp, inputs, workers)
                dfdp = np.zeros((n, n_p))
                for i in range(n):
                    for k in range(n_p):
                        dfdp[i, k] = float(np.asarray(res_eval[1][r_r[i]][a_p[k]]).flat[0])
                if use_bdf2 and S_prev is not None:
                    # BDF2: S_{n+1} = (I - (2/3)*idt*J)^{-1} * ((4/3)*S_n - (1/3)*S_{n-1} + (2/3)*idt*df/dp)
                    rhs_S = (4.0/3.0) * S - (1.0/3.0) * S_prev + jac_coeff * idt * dfdp
                else:
                    # Implicit Euler: S_{n+1} = (I - idt*J)^{-1} * (S_n + idt*df/dp)
                    rhs_S = S + jac_coeff * idt * dfdp
                S_new = np.zeros_like(S)
                for k in range(n_p):
                    S_new[:, k] = lu_solve(lu_piv, rhs_S[:, k])
                S_prev = S.copy()
                S = S_new

            for z in zeta_indices:
                y[z] = max(0.0, min(1.0, y[z]))
            x_prev = x.copy()
            x = y.copy()

        if step >= pre_steps:
            traj_sim.append(x.copy())
            S_history.append(S.copy())

    # Store trajectory and compute algebraic variables
    sim_helper.state_traj = np.array(traj_sim).T
    sim_helper._compute_var_traj(traj_sim, variables_all)
    # var_traj: (n_vars, n_sim_steps) — algebraic variables at each sim point
    var_traj = sim_helper.var_traj if hasattr(sim_helper, 'var_traj') else None
    sim_helper._has_run = True

    # S_history: (n_sim_steps, n_states, n_params) — sensitivity at each sim point
    S_history = np.array(S_history)  # already collected above
    n_sim = len(traj_sim)

    # Cost computed directly from stored trajectory (avoid re-running sim)
    # Matches the cost function in cost_on_tape / get_cost_obs_and_pred_from_params
    cost = 0.0

    # Gradient via chain rule through observables
    # cost = sum_obs w_obs * ((obs_val - gt) / std)^2
    # dcost/dp = sum_obs w_obs * 2*(obs_val - gt)/std^2 * d(obs_val)/dp
    # d(obs_val)/dp depends on operation (mean/max/min/max_minus_min)
    obs_info = pid.obs_info
    if obs_info is None:
        return cost, np.zeros(n_p)

    gt_const = obs_info.get("ground_truth_const", [])
    std_const = obs_info.get("std_const_vec", [])
    operations = obs_info.get("operations", [])
    operand_names = obs_info.get("operands", [])
    data_types = obs_info.get("data_types", [])
    cost_types = pid.cost_type if hasattr(pid, 'cost_type') else ['gaussian_MLE'] * len(gt_const)
    weights = pid.protocol_info["scaled_weight_const_from_exp_sub"][0][0] \
        if pid.protocol_info and "scaled_weight_const_from_exp_sub" in pid.protocol_info \
        else np.ones(len(gt_const))
    weighted_obs_denom = sum(1 for w in weights if w > 0) if len(weights) > 0 else 1

    grad = np.zeros(n_p)
    const_idx = 0
    traj_arr = sim_helper.state_traj  # (n_states, n_sim)

    for jj in range(len(operand_names)):
        if data_types[jj] != 'constant':
            continue
        op_name = operand_names[jj][0] if isinstance(operand_names[jj], (list, tuple)) else operand_names[jj]
        operation = operations[jj]
        kind, si = sim_helper._resolve_name(op_name)

        if const_idx >= len(gt_const) or kind is None:
            const_idx += 1
            continue

        gt = float(gt_const[const_idx])
        std = float(std_const[const_idx])
        w = float(weights[const_idx])
        scale = 0.5 if (const_idx < len(cost_types) and cost_types[const_idx] == 'gaussian_MLE') else 1.0

        if kind == 'var' and si is not None and var_traj is not None:
            # Algebraic variable: du/dp = (du/dx)*S + du/dp_direct
            series = var_traj[si, :]
            var_raw_idx = sim_helper.var_name_to_idx[op_name]
            S_series = np.zeros((len(traj_sim), n_p))
            h_fd = 1e-7
            for ti in range(len(traj_sim)):
                st = traj_sim[ti]
                # Reference: u at (st, p)
                rates0 = [0.0] * n
                v0 = list(variables_all)
                sim_helper.model.compute_rates(0.0, st, rates0, v0)
                try:
                    sim_helper.model.compute_variables(0.0, st, rates0, v0)
                except AttributeError:
                    pass
                u0 = v0[var_raw_idx]
                # du/dx via FD over states
                du_dx = np.zeros(n)
                for j in range(n):
                    st_p = list(st)
                    st_p[j] += h_fd
                    rates_p = [0.0] * n
                    v_p = list(variables_all)
                    sim_helper.model.compute_rates(0.0, st_p, rates_p, v_p)
                    try:
                        sim_helper.model.compute_variables(0.0, st_p, rates_p, v_p)
                    except AttributeError:
                        pass
                    du_dx[j] = (v_p[var_raw_idx] - u0) / h_fd
                # du/dp_direct via FD over params (relative step)
                du_dp_direct = np.zeros(n_p)
                for k in range(n_p):
                    p_val = variables_all[ad_indices[k]]
                    dp = h_fd * max(abs(p_val), 1e-15)
                    v_pk = list(variables_all)
                    v_pk[ad_indices[k]] = p_val + dp
                    rates_pk = [0.0] * n
                    sim_helper.model.compute_rates(0.0, st, rates_pk, v_pk)
                    try:
                        sim_helper.model.compute_variables(0.0, st, rates_pk, v_pk)
                    except AttributeError:
                        pass
                    du_dp_direct[k] = (v_pk[var_raw_idx] - u0) / dp
                # Full sensitivity: du/dp = du/dx @ S + du/dp_direct
                S_series[ti, :] = du_dx @ S_history[ti, :, :] + du_dp_direct

        elif kind == 'state' and si is not None:
            series = traj_arr[si, :]
            S_series = S_history[:, si, :]
        else:
            const_idx += 1
            continue

        if operation == 'mean':
            obs_val = np.mean(series)
            dobs_dp = np.mean(S_series, axis=0)  # (n_p,)
        elif operation == 'max':
            k_max = np.argmax(series)
            obs_val = series[k_max]
            dobs_dp = S_series[k_max, :]
        elif operation == 'min':
            k_min = np.argmin(series)
            obs_val = series[k_min]
            dobs_dp = S_series[k_min, :]
        elif operation == 'max_minus_min':
            k_max = np.argmax(series)
            k_min = np.argmin(series)
            obs_val = series[k_max] - series[k_min]
            dobs_dp = S_series[k_max, :] - S_series[k_min, :]
        else:
            obs_val = series[-1]
            dobs_dp = S_series[-1, :]

        # Cost contribution
        residual = (obs_val - gt) / std
        cost += scale * residual * residual * w / weighted_obs_denom

        # dcost/dp for this observable
        dcost_dobs = scale * 2.0 * (obs_val - gt) / (std * std) * w / weighted_obs_denom
        grad += dcost_dobs * dobs_dp

        const_idx += 1

    return cost, grad


# ---- Tape-based semi-implicit BDF ----
# In-memory cache for recorded tape (shared across calls for the same pid)
_tape_cache = {}


def _tape_cache_path(sim_helper):
    """Path for pickle-cached tape file, based on model path + solver config."""
    import hashlib
    model_path = getattr(sim_helper, 'model_path', '') or ''
    dt = sim_helper.dt
    max_step = float(sim_helper.solver_info.get('max_step', 0.001))
    pre_steps = sim_helper.pre_steps
    n_steps = sim_helper.n_steps
    key = f"{model_path}|{dt}|{max_step}|{pre_steps}|{n_steps}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    import os
    cache_dir = os.path.join(os.path.dirname(model_path) if model_path else '/tmp', '.aadc_tape_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"bdf_tape_{h}.pkl")


def _cost_and_grad_bdf_tape(pid, param_vals):
    """Cost and gradient via full-tape semi-implicit BDF.

    Records the ENTIRE integration + cost on one AADC tape (first call only).
    Subsequent calls replay the tape with new parameter values — no Python loop.

    Semi-implicit: x_{n+1} = x_n + dt * f(x_n) / (1 - dt * diag(J))
    Diagonal Jacobian via FD on tape. ~0.97% error vs Newton BDF at dt=0.001.

    Verified: AD/FD = 1.0000 on 3compartment (27 states, 4 params).
    Evaluate: ~2.2s for 22000 steps (vs 46s Python IFT, vs 2s CasADi).
    """
    import aadc

    sim_helper = pid.sim_helper
    n = sim_helper.STATE_COUNT
    model = sim_helper.model
    dt = sim_helper.dt
    total_steps = sim_helper.pre_steps + sim_helper.n_steps
    pre_steps = sim_helper.pre_steps
    n_sim = sim_helper.n_steps

    param_names_raw = pid.param_id_info["param_names"]
    param_names = [pn[0] if isinstance(pn, (list, tuple)) else pn for pn in param_names_raw]

    variables_all = list(sim_helper._numeric_variables_all)
    for ci, idx in enumerate(sim_helper.constant_indices):
        variables_all[idx] = sim_helper.variables[ci]

    ad_indices = sim_helper._ad_param_var_indices
    n_p = len(ad_indices)
    max_step = float(sim_helper.solver_info.get('max_step', 0.001))
    n_sub = max(1, int(np.ceil(dt / max_step)))
    idt = dt / n_sub
    total_subs = total_steps * n_sub

    # Observable info
    obs_info = pid.obs_info
    if obs_info is None:
        return 0.0, np.zeros(n_p)

    gt_const = obs_info.get("ground_truth_const", [])
    std_const = obs_info.get("std_const_vec", [])
    operations = obs_info.get("operations", [])
    operand_names = obs_info.get("operands", [])
    data_types = obs_info.get("data_types", [])
    cost_types = pid.cost_type if hasattr(pid, 'cost_type') else ['gaussian_MLE'] * len(gt_const)
    weights = pid.protocol_info["scaled_weight_const_from_exp_sub"][0][0] \
        if pid.protocol_info and "scaled_weight_const_from_exp_sub" in pid.protocol_info \
        else np.ones(len(gt_const))
    weighted_obs_denom = sum(1 for w in weights if w > 0) if len(weights) > 0 else 1

    # Resolve observables: which state/var indices to track
    # kind='state': read from x[si] directly
    # kind='var': read from id_v[var_raw_idx] after compute_variables
    obs_list = []  # (kind, idx, op_name, operation, gt, std, w, scale)
    const_idx = 0
    for jj in range(len(operand_names)):
        if data_types[jj] != 'constant':
            continue
        if const_idx >= len(gt_const):
            const_idx += 1
            continue
        op_name = operand_names[jj][0] if isinstance(operand_names[jj], (list, tuple)) else operand_names[jj]
        operation = operations[jj]
        kind, si = sim_helper._resolve_name(op_name)
        gt = float(gt_const[const_idx])
        std = float(std_const[const_idx])
        w = float(weights[const_idx])
        scale = 0.5 if (const_idx < len(cost_types) and cost_types[const_idx] == 'gaussian_MLE') else 1.0
        if kind in ('state', 'var') and si is not None:
            var_raw_idx = sim_helper.var_name_to_idx.get(op_name) if kind == 'var' else None
            obs_list.append((kind, si, var_raw_idx, operation, gt, std, w, scale))
        const_idx += 1
    needs_compute_variables = any(k == 'var' for k, *_ in obs_list)

    # Cache: in-memory first, then disk (pickle)
    import pickle, os
    cache_key = id(pid)
    cached = _tape_cache.get(cache_key)
    disk_path = _tape_cache_path(sim_helper)

    if cached is not None and cached['total_subs'] == total_subs:
        funcs = cached['funcs']
        p_args = cached['p_args']
        x_args = cached['x_args']
        cost_res = cached['cost_res']
    elif os.path.exists(disk_path):
        # Load from disk cache
        with open(disk_path, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('total_subs') == total_subs:
            funcs = cached['funcs']
            p_args = cached['p_args']
            x_args = cached['x_args']
            cost_res = cached['cost_res']
            _tape_cache[cache_key] = cached
        else:
            cached = None
    else:
        cached = None

    if cached is None:
        # Record tape
        funcs = aadc.Functions()
        funcs.start_recording()

        x = [aadc.idouble(float(sim_helper.states[i])) for i in range(n)]
        x_args = [x[i].mark_as_input() for i in range(n)]

        id_v = [aadc.idouble(float(v) if v == v else 0.0) for v in variables_all]
        p_args = []
        for k in range(n_p):
            id_v[ad_indices[k]] = aadc.idouble(float(param_vals[k]))
            p_args.append(id_v[ad_indices[k]].mark_as_input())

        h_fd = aadc.idouble(1e-7)

        # Initialize sim-time accumulators for observables
        obs_accum = {}
        for kind, si, var_raw_idx, op, gt, std, w, scale in obs_list:
            key = (kind, si, op)
            if key not in obs_accum:
                if op == 'mean':
                    obs_accum[key] = {'sum': aadc.idouble(0.0), 'count': 0}
                elif op == 'max':
                    obs_accum[key] = {'val': None}
                elif op == 'min':
                    obs_accum[key] = {'val': None}
                elif op == 'max_minus_min':
                    obs_accum[key] = {'max_val': None, 'min_val': None}

        jac_lag = int(sim_helper.solver_info.get('jac_lag', 10))
        diag_J = [aadc.idouble(0.0)] * n
        sub_counter = 0
        sim_step_counter = 0
        for step in range(total_steps):
            for sub in range(n_sub):
                rates = [aadc.idouble(0.0)] * n
                model.compute_rates(aadc.idouble(0.0), x, rates, list(id_v))

                # Diagonal Jacobian via FD (every jac_lag sub-steps)
                if sub_counter % jac_lag == 0:
                    for i in range(n):
                        x_pert = list(x)
                        x_pert[i] = x[i] + h_fd
                        r_pert = [aadc.idouble(0.0)] * n
                        model.compute_rates(aadc.idouble(0.0), x_pert, r_pert, list(id_v))
                        ri = rates[i] if isinstance(rates[i], aadc.idouble) else aadc.idouble(float(rates[i]))
                        rpi = r_pert[i] if isinstance(r_pert[i], aadc.idouble) else aadc.idouble(float(r_pert[i]))
                        diag_J[i] = (rpi - ri) / h_fd
                sub_counter += 1

                for i in range(n):
                    ri = rates[i] if isinstance(rates[i], aadc.idouble) else aadc.idouble(float(rates[i]))
                    x[i] = x[i] + aadc.idouble(idt) * ri / (aadc.idouble(1.0) - aadc.idouble(idt) * diag_J[i])

            # Accumulate observables during sim_time
            if step >= pre_steps:
                sim_step_counter += 1
                # Compute algebraic variables if needed
                if needs_compute_variables:
                    rates_cv = [aadc.idouble(0.0)] * n
                    id_v_cv = list(id_v)
                    model.compute_rates(aadc.idouble(0.0), x, rates_cv, id_v_cv)
                    try:
                        model.compute_variables(aadc.idouble(0.0), x, rates_cv, id_v_cv)
                    except AttributeError:
                        pass

                for kind, si, var_raw_idx, op, gt, std, w, scale in obs_list:
                    key = (kind, si, op)
                    if kind == 'state':
                        xi = x[si]
                    else:  # var
                        xi = id_v_cv[var_raw_idx]
                        if not isinstance(xi, aadc.idouble):
                            xi = aadc.idouble(float(xi))
                    if op == 'mean':
                        obs_accum[key]['sum'] = obs_accum[key]['sum'] + xi
                        obs_accum[key]['count'] += 1
                    elif op == 'max':
                        if obs_accum[key]['val'] is None:
                            obs_accum[key]['val'] = xi
                        else:
                            obs_accum[key]['val'] = aadc.iif(xi > obs_accum[key]['val'], xi, obs_accum[key]['val'])
                    elif op == 'min':
                        if obs_accum[key]['val'] is None:
                            obs_accum[key]['val'] = xi
                        else:
                            obs_accum[key]['val'] = aadc.iif(xi < obs_accum[key]['val'], xi, obs_accum[key]['val'])
                    elif op == 'max_minus_min':
                        if obs_accum[key]['max_val'] is None:
                            obs_accum[key]['max_val'] = xi
                            obs_accum[key]['min_val'] = xi
                        else:
                            obs_accum[key]['max_val'] = aadc.iif(xi > obs_accum[key]['max_val'], xi, obs_accum[key]['max_val'])
                            obs_accum[key]['min_val'] = aadc.iif(xi < obs_accum[key]['min_val'], xi, obs_accum[key]['min_val'])

        # Compute cost on tape
        tape_cost = aadc.idouble(0.0)
        for kind, si, var_raw_idx, op, gt, std, w, scale in obs_list:
            key = (kind, si, op)
            if op == 'mean':
                obs_val = obs_accum[key]['sum'] / aadc.idouble(float(obs_accum[key]['count']))
            elif op == 'max':
                obs_val = obs_accum[key]['val']
            elif op == 'min':
                obs_val = obs_accum[key]['val']
            elif op == 'max_minus_min':
                obs_val = obs_accum[key]['max_val'] - obs_accum[key]['min_val']
            else:
                continue
            if obs_val is None:
                continue
            residual = (obs_val - aadc.idouble(gt)) / aadc.idouble(std)
            tape_cost = tape_cost + aadc.idouble(scale * w / weighted_obs_denom) * residual * residual

        cost_res = tape_cost.mark_as_output()
        funcs.stop_recording()

        tape_data = {
            'funcs': funcs, 'p_args': p_args, 'x_args': x_args,
            'cost_res': cost_res, 'total_subs': total_subs,
        }
        _tape_cache[cache_key] = tape_data
        # Save to disk for future process launches
        try:
            with open(disk_path, 'wb') as f:
                pickle.dump(tape_data, f)
        except Exception:
            pass  # disk cache is best-effort

    # Evaluate
    workers = aadc.ThreadPool(1)
    inputs = {x_args[i]: np.array([float(sim_helper.states[i])]) for i in range(n)}
    for k in range(n_p):
        inputs[p_args[k]] = np.array([float(param_vals[k])])
    request = {cost_res: p_args}

    res = aadc.evaluate(funcs, request, inputs, workers)
    cost_val = float(np.asarray(res[0][cost_res]).flat[0])
    grad = np.zeros(n_p)
    for k in range(n_p):
        grad[k] = float(np.asarray(res[1][cost_res][p_args[k]]).flat[0])

    return cost_val, grad


def _cost_and_grad_bdf_kernel(pid, param_vals):
    """Cost and gradient via C++ kernel replay (fastest method for stiff ODE).

    Records compute_rates ONCE as AADC kernel, then replays it from C++
    in a semi-implicit BDF loop with ConstStateExtFunc (forward + reverse AD).
    First call: ~6s (kernel recording + tape creation). Subsequent: ~0.3s (cached).

    Requires AADC Python module built from source with bdf_loop.cpp.
    Falls back to bdf_tape (Python tape) if C++ function not available.

    Verified: AD/FD = 1.0000 on 3compartment (27 states, 4 params, 22000 steps).
    """
    import aadc

    if not hasattr(aadc._aadc_core, 'bdf_record_and_evaluate'):
        import warnings
        warnings.warn("bdf_kernel: C++ bdf_record_and_evaluate not available, "
                      "falling back to bdf_tape (Python tape). Build AADC from "
                      "source with bdf_loop.cpp for 10x speedup.")
        return _cost_and_grad_bdf_tape(pid, param_vals)

    sim_helper = pid.sim_helper
    n = sim_helper.STATE_COUNT
    dt = sim_helper.dt
    total_steps = sim_helper.pre_steps + sim_helper.n_steps
    pre_steps = sim_helper.pre_steps

    max_step = float(sim_helper.solver_info.get('max_step', 0.001))
    n_sub = max(1, int(np.ceil(dt / max_step)))
    idt = dt / n_sub
    jac_lag = int(sim_helper.solver_info.get('jac_lag', 10))

    param_names_raw = pid.param_id_info["param_names"]
    param_names = [pn[0] if isinstance(pn, (list, tuple)) else pn for pn in param_names_raw]

    variables_all = list(sim_helper._numeric_variables_all)
    for ci, idx in enumerate(sim_helper.constant_indices):
        variables_all[idx] = sim_helper.variables[ci]

    ad_indices = sim_helper._ad_param_var_indices
    n_p = len(ad_indices)
    for i, idx in enumerate(ad_indices):
        variables_all[idx] = float(param_vals[i])

    # Build observable list for C++ function
    # Format: (kind, state_idx, var_raw_idx, op_code, gt, std, weight, scale)
    # op_code: 0=mean, 1=max, 2=min, 3=max_minus_min
    obs_info = pid.obs_info
    if obs_info is None:
        return 0.0, np.zeros(n_p)

    gt_const = obs_info.get("ground_truth_const", [])
    std_const = obs_info.get("std_const_vec", [])
    operations = obs_info.get("operations", [])
    operand_names = obs_info.get("operands", [])
    data_types = obs_info.get("data_types", [])
    cost_types = pid.cost_type if hasattr(pid, 'cost_type') else ['gaussian_MLE'] * len(gt_const)
    weights = pid.protocol_info["scaled_weight_const_from_exp_sub"][0][0] \
        if pid.protocol_info and "scaled_weight_const_from_exp_sub" in pid.protocol_info \
        else np.ones(len(gt_const))
    weighted_obs_denom = sum(1 for w in weights if w > 0) if len(weights) > 0 else 1

    op_map = {'mean': 0, 'max': 1, 'min': 2, 'max_minus_min': 3}
    obs_list = []
    const_idx = 0
    for jj in range(len(operand_names)):
        if data_types[jj] != 'constant':
            continue
        if const_idx >= len(gt_const):
            const_idx += 1
            continue
        op_name = operand_names[jj][0] if isinstance(operand_names[jj], (list, tuple)) else operand_names[jj]
        operation = operations[jj]
        kind, si = sim_helper._resolve_name(op_name)
        gt = float(gt_const[const_idx])
        std = float(std_const[const_idx])
        w = float(weights[const_idx])
        scale = 0.5 if (const_idx < len(cost_types) and cost_types[const_idx] == 'gaussian_MLE') else 1.0
        op_code = op_map.get(operation, -1)
        if kind == 'state' and si is not None and op_code >= 0:
            obs_list.append((0, si, 0, op_code, gt, std, w * scale / weighted_obs_denom, 1.0))
        elif kind == 'var' and si is not None and op_code >= 0:
            var_raw_idx = sim_helper.var_name_to_idx.get(op_name, si)
            obs_list.append((1, si, var_raw_idx, op_code, gt, std, w * scale / weighted_obs_denom, 1.0))
        const_idx += 1

    states = list(sim_helper.states[:n])
    param_values = [float(param_vals[i]) for i in range(n_p)]

    compute_variables_fn = None
    if hasattr(sim_helper.model, 'compute_variables'):
        compute_variables_fn = sim_helper.model.compute_variables

    cost, grad_list = aadc._aadc_core.bdf_record_and_evaluate(
        sim_helper.model.compute_rates,
        states, variables_all,
        list(ad_indices), param_values,
        total_steps, pre_steps, n_sub, idt,
        obs_list, compute_variables_fn, jac_lag
    )

    grad = np.array([float(g) for g in grad_list])
    return float(cost), grad
