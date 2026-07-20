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
TAPE_CONSISTENT_METHODS = ('rk4', 'implicit_euler_ift', 'semi_implicit')


def cost_and_grad(pid, param_vals):
    """Compute J(p) and ∇J(p) via AADC tape: forward solve + cost on tape, reverse pass.

    Uses sim_helper.compute_gradient_tape() which records the entire ODE
    integration + cost function on an AADC tape and gets the gradient via
    one reverse pass. Requires a tape-compatible solver (implicit_euler_ift
    or semi_implicit).
    """
    # The AADC tape must integrate the same discrete system as the forward solve, or the
    # gradient is the exact gradient of a *different* function than the cost. The tape has
    # to replay a fixed sequence of operations, so it can only record fixed-step schemes:
    # 'rk4' (its default), 'implicit_euler_ift' and 'semi_implicit'. sim_helper.run() with
    # 'adaptive_rk45' or 'bdf' integrates something else entirely, and the tape then quietly
    # falls back to RK4 -- measured on Lotka-Volterra, that gave AD/FD = [1.79, 1.96, 1.32,
    # -0.067], the last with the wrong sign. Refuse instead.
    method = pid.sim_helper.solver_info.get('method', 'adaptive_rk45')
    if method not in TAPE_CONSISTENT_METHODS:
        raise ValueError(
            f"solver method '{method}' cannot be recorded on an AADC tape, so the forward "
            f"solve and the gradient would integrate different systems. With do_ad, use one "
            f"of {list(TAPE_CONSISTENT_METHODS)} (fixed-step, and what the tape records) "
            f"-- an adaptive integrator chooses its steps from the state, so its step "
            f"sequence changes with the parameters and cannot be taped. Or turn off do_ad.")

    param_names_raw = pid.param_id_info["param_names"]
    # param_names may be lists of strings (e.g. [['alpha_Lotka_Volterra']]) — flatten
    param_names = []
    for pn in param_names_raw:
        if isinstance(pn, (list, tuple)):
            param_names.append(pn[0])
        else:
            param_names.append(pn)

    # Set up AD parameter tracking on the simulation helper
    pid.sim_helper.set_param_vals(param_names_raw, param_vals)
    pid.sim_helper._ad_param_names = list(param_names)

    # Map param names to variable indices using sim_helper's name resolver
    ad_indices = []
    for pname in param_names:
        kind, idx = pid.sim_helper._resolve_name(pname)
        if kind == "var":
            ad_indices.append(idx)
        elif kind == "state":
            raise ValueError(f"Param '{pname}' resolves to a state, not a variable. "
                             "AADC gradient currently supports variable parameters only.")
        else:
            raise ValueError(f"Param '{pname}' not found by name resolver.")
    pid.sim_helper._ad_param_var_indices = ad_indices

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
    def _operand_is_state(op):
        # _resolve_name is authoritative. There used to be a fallback here matching the
        # operand's leaf name (op.split('/')[-1]) against every state name, which could
        # declare an *algebraic* observable tapeable purely because some unrelated
        # component happened to own a state with the same leaf -- e.g. observable
        # 'pulmonary_artery/v' matching state 'heart/v'. That suppressed the untapeable
        # check below, and _resolve_state_idx's matching fallback then bound the tape to
        # the first state with that leaf: a different variable entirely. Because both the
        # tape cost and its gradient used the wrong variable they agreed with each other,
        # so AD-vs-FD checks passed and the optimiser converged cleanly onto a fit of a
        # variable the user never asked for.
        kind, _ = pid.sim_helper._resolve_name(op)
        return kind == 'state'

    supported_const_ops = (None, 'max', 'min', 'mean')
    operand_names_o = pid.obs_info.get("operands", []) if pid.obs_info else []
    operations_o = pid.obs_info.get("operations", []) if pid.obs_info else []
    data_types_o = pid.obs_info.get("data_types", []) if pid.obs_info else []
    untaped = []
    for jj in range(len(operand_names_o)):
        op = operand_names_o[jj][0] if isinstance(operand_names_o[jj], (list, tuple)) \
            else operand_names_o[jj]
        dtype = data_types_o[jj] if jj < len(data_types_o) else 'constant'
        oper = operations_o[jj] if jj < len(operations_o) else None
        if dtype == 'constant':
            if not _operand_is_state(op) or oper not in supported_const_ops:
                untaped.append(f"{op} (op={oper})")
        elif dtype == 'series':
            if not _operand_is_state(op):
                untaped.append(f"{op} (series)")
        else:
            untaped.append(f"{op} (data_type={dtype})")
    if untaped:
        raise NotImplementedError(
            f"AADC is not usable with this observable set: {len(untaped)} of "
            f"{len(operand_names_o)} observable(s) cannot be represented on the AADC tape "
            f"(operand is an algebraic variable rather than a state, or the operation is "
            f"unsupported such as max_minus_min): {untaped}. The current AADC wrapper can "
            f"only tape observables whose operand is a state with a max/min/mean operation "
            f"(or a state series). Taping these would silently minimise a reduced cost, not "
            f"the one the optimiser evaluates. Support for algebraic-variable observables and "
            f"max_minus_min on the tape is tracked in issue #258. For a correct gradient on "
            f"these observables now, use model_type 'casadi_python' (solver_info method "
            f"'bdf') or a Myokit CVODES FSA run (model_type 'cellml_only', solver "
            f"'CVODE_myokit', do_ad true).")

    weighted_obs_denominator = 0
    if pid._num_weighted_obs_by_exp_sub is not None:
        for exp_idx in range(num_experiments):
            for sub_idx in range(num_sub_per_exp[exp_idx]):
                weighted_obs_denominator += pid._num_weighted_obs_by_exp_sub[exp_idx][sub_idx]

    def cost_on_tape(states_idouble, params_idouble, trajectory=None):
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
        def _resolve_state_idx(op_name):
            kind, resolved_idx = sim_helper._resolve_name(op_name)
            if kind == "state":
                return resolved_idx
            return None

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
            si = _resolve_state_idx(op_name)

            if data_types[jj] == 'constant':
                if const_idx >= len(gt_const) or si is None:
                    const_idx += 1
                    continue

                # Apply operation to trajectory
                if trajectory is not None and operation in ('max', 'min', 'mean'):
                    series_vals = [trajectory[t][si] for t in range(len(trajectory))]
                    if operation == 'max':
                        obs_val = mb.max(series_vals)
                    elif operation == 'min':
                        obs_val = mb.min(series_vals)
                    elif operation == 'mean':
                        obs_val = mb.mean(series_vals)
                else:
                    obs_val = states_idouble[si]

                gt_val = _aadc.idouble(float(gt_const[const_idx]))
                std_val = _aadc.idouble(float(std_const[const_idx]))
                w = _aadc.idouble(float(weights_const[const_idx]))
                diff = (obs_val - gt_val) / std_val
                cost = cost + diff * diff * w * _aadc.idouble(_cost_scale(jj))
                const_idx += 1

            elif data_types[jj] == 'series':
                # Series: compare trajectory at each time point
                if trajectory is None or si is None:
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
                    for k in range(len(gt_s)):
                        pos = k * obs_dt_s / sim_dt
                        lower = int(np.floor(pos))
                        if lower >= n_traj - 1:
                            if lower == n_traj - 1 and abs(pos - lower) < 1e-9:
                                sim_val = trajectory[lower][si]
                            else:
                                break  # past the end of the simulation: no data to compare
                        else:
                            frac = pos - lower
                            sim_val = (trajectory[lower][si] * _aadc.idouble(1.0 - frac)
                                       + trajectory[lower + 1][si] * _aadc.idouble(frac))
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
