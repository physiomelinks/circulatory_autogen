"""Finite-difference observable sensitivities -- the backend-agnostic arm of
``OpencorParamID.get_observable_sensitivities`` (issue #338).

The analytic arms are better and stay the default: CasADi differentiates the
observable vector, and Myokit CVODES gives the exact operand sensitivity. But
they only exist for two backends. AADC has none, and neither does the plain
``python`` (scipy) backend, so local sensitivity analysis was simply unavailable
there and the user was told to run a global Sobol SA instead -- a different
analysis, costing ``num_samples*(2M+2)`` simulations rather than ``2M``.

This computes the same quantity by re-simulating: central differences on the
parameter vector, with the features re-evaluated through the ordinary observable
path. It works wherever a forward run does.

**Never a silent fallback.** ``get_observable_sensitivities`` still raises when
no analytic backend is available and none was asked for, because a result
quietly computed a different way is not the same result: FD costs ``2M``
simulations, and its accuracy depends on a step size the analytic arms do not
have. The caller opts in by name (``method='FD'``), so what produced a number is
always something they chose.
"""
import contextlib
import numpy as np

from libcuflynx.parsers.PrimitiveParsers import param_entry_labels


def _step(pj, pmin, pmax, h):
    """The central-difference step for one parameter.

    Relative to the parameter, because parameters here span many orders of
    magnitude and one absolute step cannot suit them all. A parameter sitting at
    exactly zero has no scale of its own, so its range supplies one; if that is
    degenerate too, fall back to the raw h rather than a zero step, which would
    divide by zero.
    """
    if pj != 0.0:
        return abs(pj) * h
    rng = float(pmax) - float(pmin)
    return h * rng if rng > 0 else h


@contextlib.contextmanager
def _evaluating_segment(pid, exp, sub):
    """Scope the segment when the object supports it, and do nothing when it does not.

    Only a real ``OpencorParamID`` needs telling which segment its operands came from -- it is
    what keeps a cross-segment ``operation_kwargs`` reference reading the right experiment
    (#466). A minimal stand-in that just answers ``get_obs_output_dict`` has no segments to
    confuse, and this module has never required anything more of what it is handed.
    """
    scope = getattr(pid, 'evaluating_segment', None)
    if scope is None:
        yield
        return
    with scope(exp, sub):
        yield


def observable_features(pid, param_vals):
    """The scalar (const) observable features at ``param_vals``.

    Evaluated through the same path the cost uses, so a feature here is the
    feature the calibration is fitting -- not a second implementation of it.

    Each observable is evaluated in **its own** experiment and sub-experiment. A
    data_item names both, and the cost scores it against that segment, so
    evaluating every observable against one segment differentiates the wrong
    trace. Reading experiment 0 for all of them gave every observable outside
    experiment 0 the same near-zero sensitivity -- wrong in a way visible only if
    you already knew what to expect.

    ``operands_list`` is flat over sub-experiments in CA's order, so the segment
    for (exp, sub) is ``sum(num_sub_per_exp[:exp]) + sub``.
    """
    _, operands_list, _ = pid.get_cost_obs_and_pred_from_params(
        np.asarray(param_vals, dtype=float), reset=True)
    if not operands_list:
        return None

    obs = pid.obs_info
    const_to_obs = obs["const_idx_to_obs_idx"]
    num_sub_per_exp = pid.protocol_info["num_sub_per_exp"]

    # One get_obs_output_dict call per distinct segment rather than per observable:
    # it evaluates every data item against whatever operands it is handed, so the
    # segment is what varies and the const index picks the observable out of it.
    #
    # The segments are visited in ascending order sharing one temp_results table, exactly as
    # the cost path does, so an item that references one in an earlier segment reads the same
    # value here as it does there (#466). Evaluating them in const-index order with a table
    # per segment would give the gradient a different feature than the cost was built from.
    wanted = []
    for obs_idx in const_to_obs:
        exp = int(obs["experiment_idxs"][obs_idx])
        sub = int(obs["subexperiment_idxs"][obs_idx])
        flat = sum(num_sub_per_exp[:exp]) + sub
        if flat >= len(operands_list) or operands_list[flat] is None:
            return None
        wanted.append((flat, exp, sub))

    by_segment = {}
    if hasattr(pid, 'temp_results'):
        pid.temp_results = {}
    for flat, exp, sub in sorted(set(wanted)):
        with _evaluating_segment(pid, exp, sub):
            by_segment[flat] = np.asarray(
                pid.get_obs_output_dict(operands_list[flat])['const'], dtype=float)

    out = np.full(len(const_to_obs), np.nan)
    for k, (flat, _exp, _sub) in enumerate(wanted):
        consts = by_segment[flat]
        if k < len(consts):
            out[k] = consts[k]
    return out


#: Public name kept as the internal one too, so existing call sites read unchanged. This is
#: also the function an emulator is trained against (issue #333): training targets and the
#: features the cost is computed from are then the same function, not two implementations of it.
_features = observable_features


def cost_gradient(pid, param_vals, h=1e-3):
    """dJ/dtheta by central differences on the cost -- the backend-agnostic gradient.

    Deliberately differentiates ``get_cost_from_params``, the same function the optimiser
    minimises, so the jacobian and the objective cannot describe different functions. The step
    is ``_step``'s per-parameter relative one, for the same reason it is there: CA parameters
    span many orders of magnitude and one absolute step cannot suit them all.

    Costs 2M evaluations. That is the wrong trade against a real solver -- which is why the
    analytic arms remain the default there -- and the right one against an emulator, where an
    evaluation is a matrix multiply.
    """
    param_vals = np.asarray(param_vals, dtype=float)
    mins = np.asarray(pid.param_id_info["param_mins"], dtype=float)
    maxs = np.asarray(pid.param_id_info["param_maxs"], dtype=float)

    grad = np.zeros_like(param_vals)
    for j in range(param_vals.size):
        step = _step(float(param_vals[j]), mins[j], maxs[j], h)
        # Kept inside [min, max], and the denominator is the span actually used. At a bound
        # the difference simply becomes one-sided. This matters for an emulator, which is only
        # valid inside its training box: without it, a gradient evaluated at a bound would ask
        # the emulator to extrapolate and (rightly) be refused mid-optimisation.
        upper = min(float(param_vals[j]) + step, float(maxs[j]))
        lower = max(float(param_vals[j]) - step, float(mins[j]))
        span = upper - lower
        if span <= 0:
            grad[j] = 0.0
            continue
        p_plus = param_vals.copy()
        p_plus[j] = upper
        p_minus = param_vals.copy()
        p_minus[j] = lower
        grad[j] = (pid.get_cost_from_params(p_plus) - pid.get_cost_from_params(p_minus)) / span
    return grad


def observable_feature_sensitivities(pid, param_vals, h=1e-3):
    """d(observable feature)/d(param) by central finite differences.

    Returns ``{observable_label: {param_label: d(feature)/d(param)}}`` -- the same
    shape and the same quantity as the CasADi and CVODES arms, keyed by
    ``param_entry_labels``, so a local sensitivity analysis is comparable across
    backends whichever computed it. The perturbation is in theta, which
    ``get_cost_obs_and_pred_from_params`` expands to every member of a grouped or
    modifier entry, so those derivatives are d(feature)/d(theta) already.

    Costs ``2M`` simulations for M parameters. A parameter whose perturbed runs
    do not both converge is reported as None rather than as a number derived
    from a failed solve; a caller can then see which parameter was unusable
    instead of reading a plausible-looking zero.
    """
    param_vals = np.asarray(param_vals, dtype=float)
    names = param_entry_labels(pid.param_id_info)
    mins = np.asarray(pid.param_id_info["param_mins"], dtype=float)
    maxs = np.asarray(pid.param_id_info["param_maxs"], dtype=float)

    const_to_obs = pid.obs_info["const_idx_to_obs_idx"]
    out = {pid._observable_label(obs_idx): {} for obs_idx in const_to_obs}

    if _features(pid, param_vals) is None:
        raise RuntimeError("Local sensitivity nominal simulation failed to converge.")

    for j, pname in enumerate(names):
        step = _step(float(param_vals[j]), mins[j], maxs[j], h)

        p_plus = param_vals.copy()
        p_plus[j] += step
        p_minus = param_vals.copy()
        p_minus[j] -= step

        f_plus = _features(pid, p_plus)
        f_minus = _features(pid, p_minus)
        for k, obs_idx in enumerate(const_to_obs):
            label = pid._observable_label(obs_idx)
            if f_plus is None or f_minus is None:
                out[label][pname] = None
                continue
            d = (f_plus[k] - f_minus[k]) / (2.0 * step)
            out[label][pname] = float(d) if np.isfinite(d) else None
    return out
