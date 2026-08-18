"""Myokit CVODES forward-sensitivity (FSA) gradient for the param-id classes.

This is the analytic-gradient path for stiff / long-warmup ``cellml`` models, which
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

from libcuflynx.parsers.PrimitiveParsers import modifier_weights_by_index, param_entry_labels

# Dict key under which a combined (per-entry) sensitivity is handed to
# perturb_operands_along_sensitivity -- the perturbation machinery is unchanged and looks the
# trace up by name, so the combined trace travels under a name no model variable can have.
_THETA_KEY = '<theta>'


def operand_sensitivities(sim_helper, dependent_names, param_names, sensitivities=None):
    """d(operand_trace)/d(param) for every param that carries a sensitivity.

    Returns ``{dependent_name: {param_name: np.ndarray}}``. Wraps the Myokit helper's
    ``get_sensitivities`` (FSA-eligible params, whose own CVODES column *is* the sensitivity)
    and folds in the initial-value **chain-rule** params (issue #270): a constant that feeds a
    state's initial value has no column of its own, so its operand sensitivity is synthesised as
    ``sum_s d(operand)/d(init s) * d(init_s)/d(param)`` from the ``init(state)`` columns and the
    per-state factors in ``sim_helper._fsa_chain_rule_map``. Params that are neither eligible nor
    chain-ruled (true FD-fallback params) are simply absent from the inner dicts.

    Shared by the FSA gradient (``get_jac_cost``) and the local sensitivity analysis
    (``observable_feature_sensitivities``) so both see exactly the same d(operand)/d(param).
    """
    sens = sim_helper.get_sensitivities(dependent_names, param_names, sensitivities=sensitivities)
    chain_rule_map = getattr(sim_helper, '_fsa_chain_rule_map', None) or {}
    if chain_rule_map:
        chain_state_qnames = sorted({sq for tgts in chain_rule_map.values() for sq, _ in tgts})
        init_sens = sim_helper.get_init_state_sensitivities(
            dependent_names, chain_state_qnames, sensitivities=sensitivities)
        for pname, targets in chain_rule_map.items():
            for dep_name in dependent_names:
                acc = None
                for state_qname, dinit in targets:
                    s_trace = init_sens.get(dep_name, {}).get(state_qname)
                    if s_trace is None:
                        continue
                    term = np.asarray(s_trace, dtype=float) * dinit
                    acc = term if acc is None else acc + term
                if acc is not None:
                    sens.setdefault(dep_name, {})[pname] = acc
    return sens


def observable_feature_sensitivities(pid, param_vals):
    """d(observable feature)/d(param) for the scalar (const) observables, via the Myokit CVODES
    sensitivities and a directional derivative.

    Returns ``{observable_label: {param_label: d(feature)/d(param)}}`` -- the same shape as the
    CasADi arm -- so the two backends report the identical quantity. FSA gives the exact operand
    sensitivity S = d(operand)/d(param); the feature (max/min/mean/...) is re-evaluated on
    ``operands +/- h*S`` and central-differenced, so d(feature)/d(param) reuses the existing
    operation/observable code with no re-simulation, exactly as get_jac_cost does for the cost.
    Single sub-experiment only (the SA local path); the multi-sub carry stays in get_jac_cost.

    This is the Myokit arm of ``OpencorParamID.get_observable_sensitivities``. Sensitivities
    are per calibrated variable, keyed by ``param_entry_labels``: a grouped or modifier entry
    reports d(feature)/d(theta) = sum_i w_i * d(feature)/d(p_i) over its members (see
    ``combined_entry_sensitivities``). Only entries whose members all carry a CVODES
    sensitivity (FSA-eligible plus initial-value chain-rule, #270) are included; there is no
    finite-difference fallback here.
    """
    ensure_setup(pid)
    param_vals = np.asarray(param_vals, dtype=float)
    labels = param_entry_labels(pid.param_id_info)

    num_sub_total = sum(len(st) for st in pid.protocol_info["sim_times"])
    if num_sub_total != 1:
        raise NotImplementedError(
            "Local (CVODES) observable sensitivities currently support a single sub-experiment; "
            f"this protocol has {num_sub_total} sub-experiments.")

    _, operands_list, _ = pid.get_cost_obs_and_pred_from_params(
        param_vals, reset=True, only_one_exp=0)
    if not operands_list or operands_list[0] is None:
        raise RuntimeError("Local sensitivity nominal simulation failed to converge.")
    operands = operands_list[0]

    sens = operand_sensitivities(
        pid.sim_helper, pid._fsa_dependent_names, pid._fsa_param_names_flat)
    has_sensitivity = (set(pid.sim_helper._fsa_eligible_param_names or [])
                       | set(getattr(pid.sim_helper, '_fsa_chain_rule_map', None) or {}))

    const_to_obs = pid.obs_info["const_idx_to_obs_idx"]
    out = {pid._observable_label(obs_idx): {} for obs_idx in const_to_obs}
    for j, label in enumerate(labels):
        if not entry_has_sensitivity(pid, j, has_sensitivity):
            continue
        entry_sens = combined_entry_sensitivities(pid, sens, j)
        pj = float(param_vals[j])
        h = 1e-3 * abs(pj) if pj != 0.0 else 1e-4
        pert_p = perturb_operands_along_sensitivity(pid, operands, entry_sens, _THETA_KEY, h)
        pert_m = perturb_operands_along_sensitivity(pid, operands, entry_sens, _THETA_KEY, -h)
        const_p = np.asarray(pid.get_obs_output_dict(pert_p)['const'], dtype=float)
        const_m = np.asarray(pid.get_obs_output_dict(pert_m)['const'], dtype=float)
        d_feature = (const_p - const_m) / (2.0 * h)
        for k, obs_idx in enumerate(const_to_obs):
            out[pid._observable_label(obs_idx)][label] = float(d_feature[k])
    return out


def gradient_available(pid):
    """True when this run can produce an analytic gradient via Myokit CVODES FSA.

    Requires a cellml model run through the Myokit backend (whose SimulationHelper
    exposes enable_fsa) with do_ad requested. This is the gradient path for stiff /
    long-warmup models that neither CasADi nor AADC covers.
    """
    return (pid.model_type == 'cellml'
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
    # One calibrated variable (theta) may govern several model parameters: a grouped row
    # shares one value across all members, a scale modifier applies theta * baseline_i to
    # each. Since #376 set_param_vals moves every member, so the cost is a function of all of
    # them and the gradient must be too: d/dtheta = sum_i w_i * d/dp_i, with w_i = 1 for a
    # shared-value group and w_i = baseline_i for a scale modifier. CVODES therefore needs a
    # sensitivity column per *member*, and each entry keeps its (member, weight) list for the
    # combination in combined_entry_sensitivities.
    weights = modifier_weights_by_index(pid.param_id_info)
    entry_members = []
    for j, names in enumerate(pid.param_id_info["param_names"]):
        members = list(names) if isinstance(names, (list, tuple)) else [names]
        w = weights.get(j)
        entry_members.append(list(zip(members, w)) if w is not None
                             else [(m, 1.0) for m in members])
    pid._fsa_entry_members = entry_members
    flat = []
    for members in entry_members:
        for m, _ in members:
            if m not in flat:
                flat.append(m)
    pid._fsa_param_names_flat = flat
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
            f"incomplete): {pid._fsa_ineligible_names}; each calibrated variable containing "
            f"one falls back to finite-difference gradients (2 extra simulations per "
            f"gradient).")
    pid._fsa_setup_done = True


def entry_has_sensitivity(pid, entry_idx, has_sensitivity):
    """True when every member of entry ``entry_idx`` carries an analytic sensitivity.

    The combined derivative sum_i w_i * S_i needs every S_i; with one member missing the sum
    would silently drop that member's contribution, so a partially-covered entry falls back to
    finite differences as a whole.
    """
    return all(m in has_sensitivity for m, _ in pid._fsa_entry_members[entry_idx])


def combined_entry_sensitivities(pid, sens, entry_idx):
    """d(operand)/d(theta) for one params_for_id entry: sum_i w_i * S_i over its members.

    ``sens`` is ``{dependent_name: {member_name: trace}}`` from operand_sensitivities. Returns
    the same two-level shape keyed by ``_THETA_KEY``, so
    perturb_operands_along_sensitivity works on it unchanged -- all three cases (ungrouped,
    shared-value group, scale modifier) differ only in the weights fixed at ensure_setup.
    """
    members = pid._fsa_entry_members[entry_idx]
    out = {}
    for var, per_param in sens.items():
        acc = None
        for pname, w in members:
            s_trace = per_param.get(pname)
            if s_trace is None:
                continue
            term = np.asarray(s_trace, dtype=float) * float(w)
            acc = term if acc is None else acc + term
        if acc is not None:
            out[var] = {_THETA_KEY: acc}
    return out


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
    with no duplication. An entry containing any FSA-ineligible member gets a central
    finite difference over the full cost instead.

    Grouped and modifier entries are exact: theta's perturbation runs along the combined
    sensitivity sum_i w_i * S_i of all members (w_i = 1 shared-value, w_i = baseline_i
    scale modifier), matching how set_param_vals moves every member since #376.

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
    # they carry no FSA column of their own (issue #270). operand_sensitivities() covers both
    # these and the eligible params, so both are "has_sensitivity" here.
    chain_rule_map = getattr(pid.sim_helper, '_fsa_chain_rule_map', None) or {}
    has_sensitivity = eligible_names | set(chain_rule_map)
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
            # d(operand)/d(param) for every param that has a sensitivity -- FSA-eligible plus
            # init-value chain-rule params, synthesised identically (see operand_sensitivities).
            sens = operand_sensitivities(
                pid.sim_helper, pid._fsa_dependent_names, flat_names, sensitivities=sens_arr)

            for j in range(n_params):
                if not entry_has_sensitivity(pid, j, has_sensitivity):
                    continue
                # Perturbing theta by h moves member i by w_i * h, so the perturbation is
                # along the combined sensitivity sum_i w_i * S_i (identical to S itself for
                # an ungrouped parameter). The directional-difference machinery below is
                # unchanged; the three cases fall out of the weights fixed at ensure_setup.
                entry_sens = combined_entry_sensitivities(pid, sens, j)
                pj = float(param_vals[j])
                # Central directional difference along the exact sensitivity S. The step acts
                # on fixed operand traces (not the solver), so it is immune to integration
                # noise; a moderate step avoids catastrophic cancellation in raw_p - raw_m
                # while staying small enough that argmax/argmin of max/min observables is
                # stable and linear operations (mean) are reproduced exactly.
                h = 1e-3 * abs(pj) if pj != 0.0 else 1e-4
                pert_p = perturb_operands_along_sensitivity(
                    pid, operands, entry_sens, _THETA_KEY, h)
                pert_m = perturb_operands_along_sensitivity(
                    pid, operands, entry_sens, _THETA_KEY, -h)
                raw_p = float(pid.get_cost_from_operands(pert_p, exp_idx=exp_idx, sub_idx=sub_idx))
                raw_m = float(pid.get_cost_from_operands(pert_m, exp_idx=exp_idx, sub_idx=sub_idx))
                grad[j] += (raw_p - raw_m) / (2.0 * h)

    grad /= denom

    # ---- Entries without full analytic coverage: central FD over the full mean cost.
    # The step is in theta, and get_cost_from_params expands theta to every member (and
    # through a modifier's baselines), so this is correct for grouped entries too. ----
    ineligible_idx = [j for j in range(n_params)
                      if not entry_has_sensitivity(pid, j, has_sensitivity)]
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
