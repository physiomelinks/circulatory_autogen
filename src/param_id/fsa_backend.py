"""Myokit CVODES forward-sensitivity (FSA) gradient for the param-id classes.

This is the analytic-gradient path for stiff / long-warmup ``cellml_only`` models, which
neither CasADi nor AADC covers. Every public function takes the param-id object as its first
argument (``pid``); the ``_fsa_*`` bookkeeping stays as instance attributes on it, so
anything inspecting ``pid._fsa_ineligible_names`` keeps working.

Nothing here imports ``param_id.paramID`` -- the dependency runs one way only and this module
imports standalone, matching ``param_id.plot_outputs`` and ``param_id.casadi_backend``.

Generic methods this module calls back into on ``pid``:
  - ``get_cost_obs_and_pred_from_params(param_vals, reset=True, only_one_exp=k)``
  - ``get_cost_from_operands(operands, exp_idx=..., sub_idx=...)``
  - ``get_cost_from_params(param_vals)``  (the FD fallback for ineligible parameters)
  - ``_total_weighted_obs_denominator()``  (generic; deliberately left in paramID.py next to
    the cost-assembly layer that computes the same divisor inline in two other places)

There is no optional import to guard: FSA capability is probed at runtime through
``hasattr(pid.sim_helper, 'enable_fsa')``.
"""
import warnings

import numpy as np


def gradient_available(pid):
    """True when this run can produce an analytic gradient via Myokit CVODES FSA.

    Requires a cellml_only model run through the Myokit backend (whose SimulationHelper
    exposes enable_fsa) with do_ad requested. This is the gradient path for stiff /
    long-warmup models that neither CasADi nor AADC covers.
    """
    return (pid.model_type == 'cellml_only'
            and getattr(pid, 'do_ad', False)
            and hasattr(pid.sim_helper, 'enable_fsa'))


def ensure_setup(pid):
    """Enable CVODES forward sensitivities on the Myokit sim helper (once).

    Dependents are the unique observable-operand variables; independents are the AD
    parameters. A parameter that feeds a state's initial-value expression (and only that) is
    handled analytically by the chain rule through an ``init(state)`` sensitivity -- see
    ``myokit_helper.enable_fsa`` / ``_init_chain_rule_targets``. Only a parameter that also
    enters the dynamics (so the chain rule would be incomplete) stays FSA-ineligible and falls
    back to finite differences; a single warning reports how many and which.
    """
    if getattr(pid, '_fsa_setup_done', False):
        return
    # Flatten param names (a param may be shared across several vessels via a list).
    pid._fsa_param_names_flat = [
        n[0] if isinstance(n, list) else n for n in pid.param_id_info["param_names"]]
    # Unique operand variables across all observables, order-preserving.
    dep_names = []
    for operands in pid.obs_info["operands"]:
        for v in operands:
            if v not in dep_names:
                dep_names.append(v)
    pid._fsa_dependent_names = dep_names

    # enable_fsa rebuilds the Simulation; preserve any offline warmup default state.
    offline_state = getattr(pid.sim_helper, '_offline_default_state', None)
    ineligible = pid.sim_helper.enable_fsa(dep_names, pid._fsa_param_names_flat)
    if offline_state is not None:
        pid.sim_helper._offline_default_state = offline_state
        pid.sim_helper.default_states = list(offline_state)

    pid._fsa_ineligible_names = list(ineligible or [])
    if pid._fsa_ineligible_names:
        n_total = len(pid._fsa_param_names_flat)
        warnings.warn(
            f"FSA: {len(pid._fsa_ineligible_names)} of {n_total} parameters are "
            f"unsuitable for CVODES forward sensitivity (they enter the dynamics *and* a "
            f"state's initial-value expression, so the initial-value chain rule is "
            f"incomplete): {pid._fsa_ineligible_names}; these will use finite-difference "
            f"gradients (2 extra simulations each per gradient).")
    pid._fsa_setup_done = True


def get_jac_cost(pid, param_vals, return_cost=False):
    """Gradient dJ/dp via Myokit CVODES forward sensitivity + directional derivative.

    With ``return_cost=True`` this returns ``(cost, grad)`` instead of just ``grad``. The
    augmented FSA solve already produces the unperturbed operand traces, so the cost J(p)
    is reconstructed from them with ``get_cost_from_operands`` (cheap arithmetic, no extra
    solve) -- it is identical to ``get_cost_from_params(p)``. This lets L-BFGS-B get both
    the value and the gradient from one CVODES solve per point instead of two.

    FSA gives S = d(operand_trace)/dp for every eligible parameter. Rather than
    differentiate each observable operation and cost term by hand, we perturb the
    operand traces along S (operand + h*S ≈ operand(p+h)) and re-evaluate the *existing*
    cost path get_cost_from_operands; the finite difference over h is dJ/dp exactly (S is
    the true trace derivative), reusing every operation / weight / std / cost function
    with no duplication. FSA-ineligible parameters get a central finite difference over
    the full cost.

    Multi-sub-experiment protocols are supported: the Myokit helper carries dy/dp across
    sub-experiment boundaries (myokit_helper.update_times), so each sub's operand
    sensitivities already include the parameter's effect through earlier subs' end states
    (the cross-sub chain-rule term). Summing the per-sub directional derivatives is then
    exact; the helper retains each sub's sensitivities in _fsa_sensitivities_history.
    """
    ensure_setup(pid)
    param_vals = np.asarray(param_vals, dtype=float)
    n_params = len(param_vals)

    num_experiments = pid.protocol_info["num_experiments"]
    num_sub_per_exp = pid.protocol_info["num_sub_per_exp"]

    eligible_names = set(pid.sim_helper._fsa_eligible_param_names or [])
    # Params handled by the chain rule d(obs)/d(param) = sum_s d(obs)/d(init s)*d(init_s)/d(param):
    # they carry no column of their own, so their operand sensitivity is synthesised below from
    # the init(state) columns and then treated exactly like an eligible param (issue #270).
    chain_rule_map = getattr(pid.sim_helper, '_fsa_chain_rule_map', None) or {}
    chain_state_qnames = sorted({sq for tgts in chain_rule_map.values() for sq, _ in tgts})
    has_sensitivity = eligible_names | set(chain_rule_map)
    # Map each flattened param name to its column index in param_vals.
    flat_names = pid._fsa_param_names_flat
    grad = np.zeros(n_params)
    denom = float(pid._total_weighted_obs_denominator())
    raw_cost = 0.0  # unperturbed sub-costs, so we can also return J(p) from this same solve

    # ---- Eligible params: directional derivative via FSA, summed over (exp, sub) ----
    for exp_idx in range(num_experiments):
        _, operands_list, _ = pid.get_cost_obs_and_pred_from_params(
            param_vals, reset=True, only_one_exp=exp_idx)
        # Per-sub sensitivities captured during this experiment's protocol run, in sub order
        # (reset_states cleared the history at the experiment start).
        sens_history = list(pid.sim_helper._fsa_sensitivities_history)
        base = int(np.sum(num_sub_per_exp[:exp_idx]))
        for sub_idx in range(num_sub_per_exp[exp_idx]):
            subexp_count = base + sub_idx
            # A failed simulation makes get_cost_obs_and_pred_from_params return
            # `np.inf, [], []` -- an *empty* list, not a list of Nones. Indexing it
            # unguarded raises IndexError before the None check below can fire, which
            # propagates out of scipy.minimize and kills the whole calibration. The
            # non-AD path returns inf here and lets L-BFGS-B's line search back off, so
            # bounds-check and fall through to the same (inf, nan) result.
            operands = operands_list[subexp_count] \
                if subexp_count < len(operands_list) else None
            if operands is None:
                return (np.inf, np.full(n_params, np.nan)) if return_cost \
                    else np.full(n_params, np.nan)
            # Unperturbed cost of this sub from the operand traces the solve already gave us
            # (get_cost_from_operands is the same one get_cost_obs_and_pred_from_params uses,
            # so raw_cost / denom reproduces get_cost_from_params exactly -- no extra solve).
            raw_cost += float(pid.get_cost_from_operands(
                operands, exp_idx=exp_idx, sub_idx=sub_idx))
            sens_arr = sens_history[sub_idx] if sub_idx < len(sens_history) else None
            sens = pid.sim_helper.get_sensitivities(
                pid._fsa_dependent_names, flat_names, sensitivities=sens_arr)

            # Synthesise the operand sensitivity of each chain-rule param and inject it into
            # `sens` under the param's own name, so the directional-difference loop below treats
            # it identically to a param that had its own FSA column. For dependent var d:
            #   d(d)/d(param) = sum_s [ d(d)/d(init s) ] * [ d(init_s)/d(param) ]
            if chain_rule_map:
                init_sens = pid.sim_helper.get_init_state_sensitivities(
                    pid._fsa_dependent_names, chain_state_qnames, sensitivities=sens_arr)
                for pname, targets in chain_rule_map.items():
                    for dep_name in pid._fsa_dependent_names:
                        acc = None
                        for state_qname, dinit in targets:
                            s_trace = init_sens.get(dep_name, {}).get(state_qname)
                            if s_trace is None:
                                continue
                            term = np.asarray(s_trace, dtype=float) * dinit
                            acc = term if acc is None else acc + term
                        if acc is not None:
                            sens.setdefault(dep_name, {})[pname] = acc

            for j in range(n_params):
                pname = flat_names[j]
                if pname not in has_sensitivity:
                    continue
                pj = float(param_vals[j])
                # Central directional difference along the exact sensitivity S. The step acts
                # on fixed operand traces (not the solver), so it is immune to integration
                # noise; a moderate step avoids catastrophic cancellation in raw_p - raw_m
                # while staying small enough that argmax/argmin of max/min observables is
                # stable and linear operations (mean) are reproduced exactly.
                h = 1e-3 * abs(pj) if pj != 0.0 else 1e-4
                pert_p = perturb_operands_along_sensitivity(pid, operands, sens, pname, h)
                pert_m = perturb_operands_along_sensitivity(pid, operands, sens, pname, -h)
                raw_p = float(pid.get_cost_from_operands(pert_p, exp_idx=exp_idx, sub_idx=sub_idx))
                raw_m = float(pid.get_cost_from_operands(pert_m, exp_idx=exp_idx, sub_idx=sub_idx))
                grad[j] += (raw_p - raw_m) / (2.0 * h)

    grad /= denom

    # ---- Ineligible params: central finite difference over the full mean cost ----
    ineligible_idx = [j for j in range(n_params) if flat_names[j] not in has_sensitivity]
    if ineligible_idx:
        base_cost = float(pid.get_cost_from_params(param_vals))
        if not np.isfinite(base_cost):
            base_cost = None
        for j in ineligible_idx:
            pj = float(param_vals[j])
            # Real re-simulation FD, so the step must balance truncation against the
            # integrator noise floor: the central-difference optimum is ~tol^(1/3), i.e.
            # a ~1e-3 relative step for the ~1e-9 cost noise at rtol/atol 1e-8 (a
            # convergence study confirmed rel 1e-3 is well inside the flat region while
            # rel 1e-4 sits in the noise floor).
            h = 1e-3 * abs(pj) if pj != 0.0 else 1e-5
            p_plus = param_vals.copy(); p_plus[j] += h
            p_minus = param_vals.copy(); p_minus[j] -= h
            c_plus = float(pid.get_cost_from_params(p_plus))
            c_minus = float(pid.get_cost_from_params(p_minus))
            if np.isfinite(c_plus) and np.isfinite(c_minus):
                grad[j] = (c_plus - c_minus) / (2.0 * h)
            elif base_cost is not None and np.isfinite(c_plus):
                grad[j] = (c_plus - base_cost) / h
            elif base_cost is not None and np.isfinite(c_minus):
                grad[j] = (base_cost - c_minus) / h
            else:
                grad[j] = 0.0
    if return_cost:
        return raw_cost / denom, grad
    return grad


def get_cost_and_jac(pid, param_vals):
    """(cost, gradient) from a single Myokit CVODES FSA solve. See get_jac_cost."""
    return get_jac_cost(pid, param_vals, return_cost=True)


def perturb_operands_along_sensitivity(pid, operands, sens, pname, h):
    """Return a copy of one sub-experiment's operand traces stepped by h along dS/dp.

    `operands` is the list-of-operand-arrays for one observable-bearing sub-experiment,
    indexed [obs_entry][operand] to match pid.obs_info["operands"]; `sens[var][pname]`
    is the FSA sensitivity trace of operand variable `var` w.r.t. parameter `pname`.
    """
    pert = []
    for JJ, operand_arrays in enumerate(operands):
        operand_var_names = pid.obs_info["operands"][JJ]
        new_entry = []
        for k, arr in enumerate(operand_arrays):
            arr = np.asarray(arr, dtype=float)
            var_name = operand_var_names[k]
            s_trace = sens.get(var_name, {}).get(pname)
            if s_trace is None:
                new_entry.append(arr.copy())
                continue
            s_trace = np.asarray(s_trace, dtype=float)
            L = min(arr.shape[0], s_trace.shape[0])
            stepped = arr.copy()
            stepped[:L] = arr[:L] + h * s_trace[:L]
            new_entry.append(stepped)
        pert.append(new_entry)
    return pert
