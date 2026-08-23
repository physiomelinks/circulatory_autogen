"""Built-in observable operations; dicts are built per mode via build_operation_funcs_dict.

This module also owns the ``operation_kwargs`` contract (issue #304): the helpers
``resolve_operation_kwargs`` / ``validate_operation_kwargs`` are the single place that decides
how a data_item's ``operation_kwargs`` mapping is checked and turned into the keyword arguments
of an operation func, so param-id, MCMC/UQ and sensitivity analysis all behave identically.
"""

import difflib
import inspect
import math
import os
import sys

from libcuflynx.param_id.differentiable import differentiable
from libcuflynx.param_id.math_backend import make_math_backend, bind_backend


def series_to_constant(func):
    func.series_to_constant = True
    return func


mb = make_math_backend("numpy")


@differentiable
@series_to_constant
def max(x, series_output=False):
    if series_output:
        return x
    return mb.max(x)


@differentiable
@series_to_constant
def min(x, series_output=False):
    if series_output:
        return x
    return mb.min(x)


@differentiable
@series_to_constant
def mean(x, series_output=False):
    if series_output:
        return x
    return mb.mean(x)


@differentiable
@series_to_constant
def max_minus_min(x, series_output=False):
    if series_output:
        return x
    return mb.max_minus_min(x)


@differentiable
def addition(x1, x2):
    return x1 + x2


@differentiable
def subtraction(x1, x2):
    return x1 - x2


@differentiable
def multiplication(x1, x2):
    return x1 * x2


@differentiable
def division(x1, x2):
    return x1 / x2


##
## Below here is the `operation_kwargs` contract (issue #304).
## These are public helpers, not observable operations, so they are excluded from registration.
##

# Keyword arguments that circulatory_autogen itself supplies to an operation func. A data_item's
# `operation_kwargs` must not set them, otherwise the call raises a bare
# "got multiple values for keyword argument".
RESERVED_OPERATION_KWARGS = frozenset({"series_output"})


def get_operation_kwarg_spec(func):
    """Introspect an operation func's signature for the ``operation_kwargs`` contract.

    Returns ``(accepted, from_operands, accepts_any)``:

    - ``accepted``: ordered list of parameter names that may be given in ``operation_kwargs``.
    - ``from_operands``: ordered list of parameters with no default, i.e. the ones
      circulatory_autogen fills positionally from the data_item's ``operands``.
    - ``accepts_any``: True when the func declares ``**kwargs``, in which case any key is
      accepted (e.g. ``calculate_two_observable_difference``).

    A func whose signature cannot be introspected is treated as ``accepts_any`` so that
    validation never blocks a legitimate call.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return [], [], True

    accepted = []
    from_operands = []
    accepts_any = False
    for name, param in sig.parameters.items():
        if param.kind == param.VAR_KEYWORD:
            accepts_any = True
        elif param.kind == param.VAR_POSITIONAL:
            # *args -> we cannot tell how many operands are consumed; don't second-guess.
            accepts_any = True
        elif param.kind == param.POSITIONAL_ONLY:
            from_operands.append(name)
        else:
            accepted.append(name)
            if param.default is param.empty:
                from_operands.append(name)
    return accepted, from_operands, accepts_any


def _operation_kwarg_defaults(func):
    """Map of parameter name -> default value for the keyword parameters of ``func``."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    return {name: p.default for name, p in sig.parameters.items() if p.default is not p.empty}


def _coerce_operation_kwarg(value, default):
    """Basic numeric coercion of a JSON-sourced kwarg towards the type of the func's default.

    JSON has a single number type, so a GUI (or a hand-edited obs_data.json) can easily write
    ``20.0`` where the operation func's default is the int ``20`` (used as a slice index) or
    ``1`` where the default is a float. Only these two integral<->float conversions are done;
    everything else (strings, bools, lists, dicts, None) is passed through untouched.
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


def check_operation_kwargs(raw_kwargs, func, operation_name, data_item_name=None, num_operands=0):
    """Validate a data_item's ``operation_kwargs`` against the chosen operation func's signature.

    An unknown key is an **error**, not silently ignored: a stale or misspelled key would
    otherwise change nothing and quietly calibrate against the wrong feature.

    Raises:
        ValueError: for a non-string key, a key reserved by circulatory_autogen, a key that
            duplicates an argument already filled from ``operands``, or a key that is not a
            parameter of ``func``.
    """
    if not raw_kwargs:
        return
    where = (f"data_item '{data_item_name}'" if data_item_name else "a data_item")
    op_label = operation_name if operation_name is not None else getattr(func, "__name__", "<unknown>")
    accepted, from_operands, accepts_any = get_operation_kwarg_spec(func)
    positional_filled = set(from_operands[:num_operands])

    for key in raw_kwargs:
        if not isinstance(key, str):
            raise ValueError(
                f"Invalid 'operation_kwargs' in {where}: keys must be strings, got {key!r} "
                f"({type(key).__name__}). 'operation_kwargs' maps keyword-argument names of the "
                f"operation func '{op_label}' to values."
            )
        if key in RESERVED_OPERATION_KWARGS:
            raise ValueError(
                f"Invalid 'operation_kwargs' key '{key}' in {where}: '{key}' is set by "
                f"circulatory_autogen when it calls the operation func '{op_label}' and must not "
                f"be given in obs_data.json. Remove it from 'operation_kwargs'."
            )
        if key in positional_filled:
            raise ValueError(
                f"Invalid 'operation_kwargs' key '{key}' in {where}: the operation func "
                f"'{op_label}' already receives '{key}' positionally from the data_item's "
                f"'operands' (operands fill {list(from_operands[:num_operands])}). Remove '{key}' "
                f"from 'operation_kwargs', or remove the corresponding entry from 'operands'."
            )
        if accepts_any or key in accepted:
            continue
        suggestions = difflib.get_close_matches(key, accepted, n=3, cutoff=0.6)
        hint = f" Did you mean {' or '.join(repr(s) for s in suggestions)}?" if suggestions else ""
        raise ValueError(
            f"Invalid 'operation_kwargs' key '{key}' in {where}: the operation func '{op_label}' "
            f"has no keyword argument '{key}'.{hint} Accepted keyword arguments are: "
            f"{sorted(set(accepted) - RESERVED_OPERATION_KWARGS)}. Fix the key in the data_item's "
            f"'operation_kwargs' in obs_data.json, or add '{key}' as a keyword argument of "
            f"'{op_label}'."
        )


def resolve_operation_kwargs(raw_kwargs, func, operation_name=None, data_item_name=None,
                             temp_results=None, num_operands=0, known_item_names=None):
    """Validate and resolve a data_item's ``operation_kwargs`` into call keyword arguments.

    Args:
        raw_kwargs: the data_item's ``operation_kwargs`` value. A missing field parses to ``{}``,
            but ``None``/``NaN``/any non-dict is tolerated and treated as "no kwargs".
        func: the operation func that will be called.
        operation_name: the data_item's ``operation`` (used in error messages).
        data_item_name: the data_item's ``data_item_name`` (used in error messages).
        temp_results: mapping of already-computed observable ``data_item_name`` -> value. Any
            **string** kwarg value that matches a key here is replaced by that observable's value,
            which is how an observable is built from earlier observables (see
            ``calculate_two_observable_difference``). A string that matches nothing is passed
            through unchanged, so plain string options still work.
        num_operands: number of operands passed positionally, used to detect a kwarg that
            duplicates a positional argument.
        known_item_names: every ``data_item_name`` in the study. A string that matches one of
            these but is not yet in ``temp_results`` is a *forward* reference -- the item it
            names has not been evaluated yet -- and raises instead of being passed through as a
            plain string, which used to surface as ``str - str`` or, worse, a plausible number.

    Returns:
        A new dict safe to splat into ``func(*operands, **kwargs)``.
    """
    if not isinstance(raw_kwargs, dict) or not raw_kwargs:
        # Covers the schema default {}, and a NaN/None left by a partially-filled obs_data file.
        return {}

    check_operation_kwargs(raw_kwargs, func, operation_name,
                           data_item_name=data_item_name, num_operands=num_operands)

    defaults = _operation_kwarg_defaults(func)
    kwargs = {}
    for key, value in raw_kwargs.items():
        if isinstance(value, str) and temp_results is not None and value in temp_results:
            # Reference to an earlier observable; substitute its value, never coerce it.
            kwargs[key] = temp_results[value]
        elif (isinstance(value, str) and known_item_names is not None
                and value in known_item_names):
            where = f"data_item {data_item_name!r}" if data_item_name else "a data_item"
            raise ValueError(
                f"{where}: 'operation_kwargs' key {key!r} references data_item {value!r}, "
                f"which has not been computed yet. References are resolved in order, so the "
                f"item referenced must come earlier in 'data_items' -- and, when it belongs to "
                f"another experiment or sub-experiment, that segment must be earlier too "
                f"(experiments in order, sub-experiments within each). Move {value!r} before "
                f"{data_item_name!r}.")
        elif key in defaults:
            kwargs[key] = _coerce_operation_kwarg(value, defaults[key])
        else:
            kwargs[key] = value
    return kwargs


def validate_operation_kwargs(obs_info, operation_funcs_dict):
    """Eagerly validate every data_item's ``operation_kwargs`` in ``obs_info``.

    Called once when the obs data is set so a stale obs_data.json fails immediately with a clear
    message instead of part-way through a calibration / sensitivity run. Operations that are not
    (yet) in ``operation_funcs_dict`` are skipped -- a user func may still be registered via
    ``add_user_operation_func``, and an genuinely unknown operation is reported by the evaluation
    path itself.
    """
    if not obs_info:
        return
    all_kwargs = obs_info.get("operation_kwargs") or []
    operations = obs_info.get("operations") or []
    operands = obs_info.get("operands") or []
    names = obs_info.get("data_item_names") or []
    for idx, raw_kwargs in enumerate(all_kwargs):
        if not isinstance(raw_kwargs, dict) or not raw_kwargs:
            continue
        operation_name = operations[idx] if idx < len(operations) else None
        func = operation_funcs_dict.get(operation_name) if operation_name is not None else None
        if func is None:
            continue
        item_operands = operands[idx] if idx < len(operands) else []
        num_operands = len(item_operands) if hasattr(item_operands, "__len__") else 0
        check_operation_kwargs(
            raw_kwargs, func, operation_name,
            data_item_name=names[idx] if idx < len(names) else None,
            num_operands=num_operands,
        )


##
## Below here are the organisational functions for building the operation functions dictionary
## They are not part of the public API
##

def register_core_operations(registry, backend):
    """
    Register every operation callable defined in this module, each bound to ``backend`` (so
    registries for different backends stay independent -- see math_backend.bind_backend, #315).

    Skips private names (``_`` prefix), ``series_to_constant``, and the dict builders.
    Imported callables are skipped via ``__module__`` checks.
    """
    global mb
    mb = backend
    g = globals()
    mod = __name__
    exclude = frozenset(
        {
            "series_to_constant",
            "register_core_operations",
            "build_operation_funcs_dict",
            "get_operation_funcs_dict_for_mode",
            # operation_kwargs contract helpers (#304) -- public API of this module, not operations
            "get_operation_kwarg_spec",
            "check_operation_kwargs",
            "resolve_operation_kwargs",
            "validate_operation_kwargs",
        }
    )
    for name, obj in g.items():
        if name.startswith("_") or name in exclude:
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != mod:
            continue
        registry[name] = bind_backend(obj, backend)


# Decorator/hook helper names an external operation-funcs file might define locally (mirroring
# operation_funcs_user.py); excluded so they are not registered as operations.
_EXTERNAL_OP_EXCLUDE = frozenset({"series_to_constant", "register_user_operations"})


def build_operation_funcs_dict(backend, external_path=None):
    """Build the observable-operation registry: built-in core ops, then the shipped ops in
    ``libcuflynx.funcs.operation_funcs_user``, then (if given) the ops in the external file
    ``external_path`` (issue #303). Later registrations win, so an external func may override a
    shipped/core one."""
    registry = {}
    register_core_operations(registry, backend)
    from libcuflynx.funcs import operation_funcs_user as ofu
    ofu.register_user_operations(registry, backend)
    if external_path:
        from libcuflynx.param_id.external_funcs import register_funcs_from_file
        register_funcs_from_file(external_path, registry, backend, exclude=_EXTERNAL_OP_EXCLUDE)
    return registry


def get_operation_funcs_dict_for_mode(mode="numpy", external_path=None):
    """Convenience for callers that only have a mode string."""
    return build_operation_funcs_dict(make_math_backend(mode), external_path=external_path)
