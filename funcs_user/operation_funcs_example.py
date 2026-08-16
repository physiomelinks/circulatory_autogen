"""Template for your own observable operations (issue #303 / #433).

Copy this file, rename it, add your operations, and point the config at it::

    operation_funcs_external_path: funcs_user/my_ops.py

Every top-level callable defined here is registered as an ``operation`` for obs_data.json data
items, alongside the core operations in ``libcuflynx.param_id.operation_funcs`` and the ones
libcuflynx ships in ``libcuflynx.funcs.operation_funcs_user`` -- reusing a name overrides it.
Names starting with ``_`` are skipped, so keep helpers private. An operation's arguments are
filled from the data item's ``operands`` (and ``operation_kwargs``).

Decorators:
  ``@series_to_constant``  the operation reduces a series to a scalar; take a ``series_output``
                           keyword and return the series unreduced when it is True, so the
                           feature can be plotted on top of the trace
  ``@differentiable``      safe to execute symbolically under CasADi (AD gradients)

``mb`` is the math backend; libcuflynx rebinds it to numpy or casadi when it builds the
registry, so write against ``mb`` rather than ``np`` for anything that must differentiate.
"""
from libcuflynx.param_id.differentiable import differentiable
from libcuflynx.param_id.math_backend import make_math_backend
from libcuflynx.param_id.operation_funcs import series_to_constant  # noqa: F401

mb = make_math_backend("numpy")


# Delete this and write your own. It is registered as operation: peak_to_peak.
@differentiable
@series_to_constant
def peak_to_peak(x, series_output=False):
    """Amplitude of a trace: max minus min."""
    if series_output:
        return x
    return mb.max(x) - mb.min(x)
