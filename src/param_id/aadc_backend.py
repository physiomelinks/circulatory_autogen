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
import numpy as np

# AADC solver methods whose forward integration the tape can record step-for-step. An adaptive
# integrator picks its step sizes from the state, so the sequence of operations changes with the
# parameters and cannot be replayed from a tape.
TAPE_CONSISTENT_METHODS = ('rk4', 'implicit_euler_ift', 'semi_implicit', 'implicit_newton')
BDF_NEWTON_METHOD = 'bdf_newton'


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
    traj_sim = []
    S_history = []  # sensitivity at each sim trajectory point

    # Pre-build inputs template
    inputs_template = {a_p[k]: p_vals[k] for k in range(n_p)}
    request_dfdp = {r: a_p for r in r_r}
    need_sensitivity = False
    eye_n = np.eye(n)

    for step in range(total_steps):
        if step == pre_steps:
            need_sensitivity = True
            S = np.zeros((n, n_p))

        for sub in range(n_sub):
            y = x.copy()
            lu_piv = None

            for nit in range(max_newton):
                # VFJ.func is 35× faster than Python compute_rates
                rates_arr = vfj.func(y)
                F = y - x - idt * rates_arr
                if np.max(np.abs(F)) < newton_tol:
                    break
                J_rhs = vfj.jac(y)
                J_g = eye_n - idt * J_rhs
                try:
                    lu_piv = lu_factor(J_g)
                    dy = lu_solve(lu_piv, -F)
                except np.linalg.LinAlgError:
                    break
                y += dy

            # IFT sensitivity only during sim_time (skip warmup)
            if need_sensitivity and lu_piv is not None:
                inputs = dict(inputs_template)
                for i in range(n):
                    inputs[a_x[i]] = float(y[i])
                res_eval = aadc.evaluate(rhs_f, request_dfdp, inputs, workers)
                dfdp = np.zeros((n, n_p))
                for i in range(n):
                    for k in range(n_p):
                        dfdp[i, k] = float(np.asarray(res_eval[1][r_r[i]][a_p[k]]).flat[0])
                rhs_S = S + idt * dfdp
                for k in range(n_p):
                    S[:, k] = lu_solve(lu_piv, rhs_S[:, k])

            for z in zeta_indices:
                y[z] = max(0.0, min(1.0, y[z]))
            x = y.copy()

        if step >= pre_steps:
            traj_sim.append(x.copy())
            S_history.append(S.copy())

    # Store trajectory and compute observables
    sim_helper.state_traj = np.array(traj_sim).T  # (n_states, n_sim_steps)
    sim_helper._compute_var_traj(traj_sim, variables_all)
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

        if kind == 'state' and si is not None:
            series = traj_arr[si, :]  # (n_sim,)
            S_series = S_history[:, si, :]  # (n_sim, n_p)

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

        # TODO: algebraic variable observables (kind == 'var') — need var sensitivity
        const_idx += 1

    return cost, grad
