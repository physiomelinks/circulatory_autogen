"""Template for your own modifier functions (issue #383 / #433).

Copy this file, rename it, add your modifiers, and point the config at it::

    modifier_funcs_external_path: funcs_user/my_modifiers.py

A modifier maps one calibrated value (theta) onto each model parameter a ``params_for_id``
entry names in ``modifies``::

    p_i = fn(theta, baseline_i, **inputs)

``baseline_i`` is the target's model-default value. Declare any extra ``inputs`` on the
decorator, typed ``'float'`` (the entry supplies one model qname; the function receives that
constant's default) or ``'list'`` (several qnames, received as a list of floats). Inputs are
resolved from the model defaults once at setup, so nothing compounds across iterations.

Only functions carrying ``@modifier_func`` are registered -- an undecorated helper is ignored
rather than half-registered without its input metadata. ``scale`` and ``remainder`` are built
in; reusing one of those names here deliberately overrides it.

**Every modifier must be affine in theta** (``a*theta + b`` for fixed inputs). This is probed
numerically at setup and a non-affine function is refused there: the analytic gradients apply
the constant chain-rule weight ``a = dp/dtheta``, and theta's starting value is obtained by
inverting the mapping at the baseline.

An entry then uses it by name::

    {"name": "R_offset", "modifies": ["par/R"], "modifier": "offset_from",
     "inputs": {"reference": "par/R_ref"}, "min": -1e6, "max": 1e6}
"""
from libcuflynx.funcs.modifier_funcs_user import modifier_func


# Delete this and write your own. It is registered as modifier: offset_from.
@modifier_func(inputs={'reference': 'float'},
               description='target = theta + reference')
def offset_from(theta, baseline, reference):
    return theta + reference
