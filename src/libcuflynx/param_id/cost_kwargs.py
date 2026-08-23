"""The ``cost_kwargs`` contract for user-defined cost functions (issue #84).

Cost funcs were called with a fixed positional signature ``(output, ground_truth, std, weight)``
at every call site, which forced two things on every cost func ever written:

* it had to accept ``std`` and ``weight`` even when the cost does not use them -- and a cost that
  genuinely has no notion of a standard deviation (an absolute bound, a barrier, a
  distribution-fit) had to take the argument and drop it;
* it could take nothing else, so a user cost with a tolerance, an exponent or a reference scale
  had no way to receive it short of a module-level global.

This module is the single place that decides what a given cost func is actually called with. It
deliberately mirrors the ``operation_kwargs`` contract in ``param_id.operation_funcs`` (issue
#304) -- same shape, same validation stance, same reserved-name handling -- so the two
user-extension points behave the same way rather than each having their own rules.

Framework-supplied arguments (``std``, ``weight``) are passed only when the func's signature has
somewhere to put them. User arguments come from a data_item's ``cost_kwargs`` mapping.
"""
import inspect
import math
from libcuflynx.utilities.obs_data_helpers import obs_item_names


# Supplied by circulatory_autogen itself from the obs data. A data_item's `cost_kwargs` must not
# set these -- doing so would either shadow the real std/weight (silently calibrating against the
# wrong thing) or raise a bare "got multiple values for keyword argument".
RESERVED_COST_KWARGS = frozenset({"std", "weight"})

# The two values every call site has available to offer. Kept here rather than at the call sites
# so adding a third framework-supplied argument is one edit, not six.
FRAMEWORK_COST_KWARGS = ("std", "weight")


def get_cost_kwarg_spec(func):
    """Introspect a cost func's signature.

    Returns ``(accepted, positional, accepts_any)``:

    - ``accepted``: parameter names that may be supplied by keyword, framework or user.
    - ``positional``: parameters with no default, i.e. the ones filled positionally
      (``output`` and the ground truth), excluding anything the framework supplies by name.
    - ``accepts_any``: True when the func declares ``**kwargs`` (e.g. ``MSE``), in which case any
      key is accepted and every framework argument is offered.

    A func whose signature cannot be introspected is treated as ``accepts_any``, so validation
    never blocks a legitimate call -- the same fallback ``get_operation_kwarg_spec`` uses.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return [], [], True

    accepted = []
    positional = []
    accepts_any = False
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            accepts_any = True
        elif param.kind == param.POSITIONAL_ONLY:
            positional.append(name)
        else:
            accepted.append(name)
            if param.default is param.empty and name not in RESERVED_COST_KWARGS:
                positional.append(name)
    return accepted, positional, accepts_any


def ground_truth_param_name(func):
    """The name of the ground truth this cost func takes, from its signature.

    ``gaussian_MLE(output, desired_mean, std, weight)`` is scored against a number;
    ``kernel_density_estimation(output, prob_dist_params, weight)`` against a distribution. Both
    are scalar observables -- only what they are compared *to* differs, which is a property of
    the cost, not of the data. Reading it off the signature is the same rule
    ``framework_kwargs_for`` already applies to the keyword arguments, extended to the positional
    ground truth so the two no longer need separate call sites (issue #421).

    Returns ``'desired_mean'`` when the signature says nothing useful, which is the shape every
    cost took before distributions existed.
    """
    _, positional, _ = get_cost_kwarg_spec(func)
    return positional[1] if len(positional) >= 2 else 'desired_mean'


def framework_kwargs_for(func, std=None, weight=None):
    """The framework-supplied kwargs this particular cost func can actually receive.

    ``gaussian_MLE(output, desired_mean, std, weight)`` gets both; ``multimodal_gaussian(output,
    prob_dist_params, weight)`` gets only ``weight``, because it has nowhere to put a std -- which
    is the point of this function, and previously meant it could not share a call site.
    """
    accepted, _, accepts_any = get_cost_kwarg_spec(func)
    available = {"std": std, "weight": weight}
    if accepts_any:
        return {k: v for k, v in available.items() if v is not None}
    return {k: v for k, v in available.items() if k in accepted and v is not None}


def _coerce_cost_kwarg(value, default):
    """Nudge a JSON-sourced number towards the type of the func's default.

    JSON has one number type, so a GUI or hand-edited obs_data.json easily writes ``2.0`` where
    the default is the int ``2``, or ``1`` where it is a float. Only those two integral<->float
    conversions happen; everything else passes through. Same rule as ``_coerce_operation_kwarg``.
    """
    if isinstance(value, bool) or isinstance(default, bool):
        return value
    if isinstance(default, int) and isinstance(value, float):
        if math.isfinite(value) and float(value).is_integer():
            return int(value)
        return value
    if isinstance(default, float) and isinstance(value, int):
        return float(value)
    return value


def _cost_kwarg_defaults(func):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    return {name: p.default for name, p in sig.parameters.items() if p.default is not p.empty}


def check_cost_kwargs(raw_kwargs, func, cost_name, data_item_name=None):
    """Validate a data_item's ``cost_kwargs`` against the chosen cost func's signature.

    An unknown key is an error rather than being ignored: a stale or misspelled key would
    otherwise change nothing at all and quietly calibrate against the wrong cost, which is the
    failure mode that is hardest to notice from a converged-looking fit.

    Raises:
        ValueError: for a non-string key, a key circulatory_autogen supplies itself, or a key
            that is not a parameter of ``func``.
    """
    if not raw_kwargs:
        return
    where = f"data_item '{data_item_name}'" if data_item_name else "a data_item"
    label = cost_name if cost_name is not None else getattr(func, "__name__", "<unknown>")
    accepted, positional, accepts_any = get_cost_kwarg_spec(func)

    for key in raw_kwargs:
        if not isinstance(key, str):
            raise ValueError(
                f"Invalid 'cost_kwargs' in {where}: keys must be strings, got {key!r} "
                f"({type(key).__name__}). 'cost_kwargs' maps keyword-argument names of the "
                f"'cost_type' func to values.")
        if key in RESERVED_COST_KWARGS:
            raise ValueError(
                f"Invalid 'cost_kwargs' key '{key}' in {where}: '{key}' is supplied by "
                f"circulatory_autogen from the obs data and must not be set here. Set the "
                f"data_item's own '{key}' field instead.")
        if accepts_any:
            continue
        if key in positional:
            raise ValueError(
                f"Invalid 'cost_kwargs' key '{key}' in {where}: '{key}' is filled positionally "
                f"by cost func '{label}' (it receives the model output and the ground truth), "
                f"so it cannot also be given as a keyword.")
        if key not in accepted:
            raise ValueError(
                f"Invalid 'cost_kwargs' key '{key}' in {where}: cost func '{label}' has no "
                f"parameter '{key}'. Accepted: {sorted(set(accepted) - RESERVED_COST_KWARGS)}.")


def resolve_cost_kwargs(raw_kwargs, func):
    """Coerce a validated ``cost_kwargs`` mapping into the kwargs to splat into the call."""
    if not raw_kwargs:
        return {}
    defaults = _cost_kwarg_defaults(func)
    return {k: _coerce_cost_kwarg(v, defaults[k]) if k in defaults else v
            for k, v in raw_kwargs.items()}


def validate_cost_kwargs(obs_info, cost_funcs_dict, cost_types):
    """Check every data_item's ``cost_kwargs`` up front, before any simulation runs.

    Called at setup for the same reason ``validate_operation_kwargs`` is: a bad key should fail
    immediately, not after the first expensive forward solve.
    """
    raw_list = obs_info.get("cost_kwargs") if obs_info else None
    if not raw_list:
        return
    names = obs_item_names(obs_info) or obs_info.get("names", [])
    for i, raw in enumerate(raw_list):
        if not raw:
            continue
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid 'cost_kwargs' in obs index {i}: expected a dict of keyword arguments, "
                f"got {type(raw).__name__}.")
        cost_name = cost_types[i] if cost_types is not None and i < len(cost_types) else None
        func = cost_funcs_dict.get(cost_name) if cost_name is not None else None
        if func is None:
            continue
        check_cost_kwargs(raw, func, cost_name,
                          names[i] if i < len(names) else None)


def call_cost_func(func, *positional, std=None, weight=None, cost_kwargs=None):
    """Invoke a cost func with only the arguments it can accept.

    The single entry point the cost-assembly call sites use, so the rule about what a cost func
    receives lives in one place rather than being restated six times.
    """
    kwargs = framework_kwargs_for(func, std=std, weight=weight)
    if cost_kwargs:
        kwargs.update(resolve_cost_kwargs(cost_kwargs, func))
    return func(*positional, **kwargs)
