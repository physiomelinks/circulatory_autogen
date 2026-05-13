"""Built-in observable operations; dicts are built per mode via build_operation_funcs_dict."""

import os
import sys

_opp_dir = os.path.dirname(os.path.abspath(__file__))
if _opp_dir not in sys.path:
    sys.path.insert(0, _opp_dir)

from differentiable import differentiable
from math_backend import make_math_backend


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
## Below here are the organisational functions for building the operation functions dictionary
## They are not part of the public API
##

def register_core_operations(registry, backend):
    """
    Bind ``mb`` to ``backend`` and register every operation callable defined in this module.

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
        }
    )
    for name, obj in g.items():
        if name.startswith("_") or name in exclude:
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != mod:
            continue
        registry[name] = obj


def build_operation_funcs_dict(backend):
    registry = {}
    register_core_operations(registry, backend)
    try:
        import operation_funcs_user as ofu
    except ImportError:
        pass
    else:
        ofu.register_user_operations(registry, backend)
    return registry


def get_operation_funcs_dict_for_mode(mode="numpy"):
    """Convenience for callers that only have a mode string."""
    return build_operation_funcs_dict(make_math_backend(mode))
