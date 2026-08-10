"""User-defined modifier functions for params_for_id entries (issue #383).

A modifier function maps one calibrated value (theta) to each model parameter its entry
names in ``modifies``:

    p_i = fn(theta, baseline_i, **inputs)

``baseline_i`` is the target's model-default value. Extra ``inputs`` are declared on the
decorator with a type -- ``'float'`` (the params_for_id entry supplies one model qname, the
function receives that constant's default value) or ``'list'`` (a list of qnames, received as
a list of floats). Inputs are resolved from model defaults once at setup, never re-derived,
so nothing compounds across calibration iterations.

Every function must be **affine in theta** (a*theta + b for fixed inputs) -- verified at
setup -- because the analytic gradients use the constant chain-rule weight dp/dtheta and
theta's starting value is derived by inverting the mapping at the baseline.

Every top-level function carrying the ``@modifier_func`` decorator is registered
automatically; undecorated helpers are ignored. A params_for_id entry uses one by name:

    {"name": "q_tot", "modifies": ["heart/q_lv_init"], "operation": "remainder",
     "inputs": {"subtract": ["pvn/q_init", "par/q_init"]}, "min": 4e-3, "max": 6e-3}

(``scale`` and ``remainder`` are built in -- see src/param_id/modifier_funcs.py. Redefining a
built-in name here deliberately overrides it.)

Example (uncomment to use):

    @modifier_func(inputs={'reference': 'float'},
                   description='target = theta + reference')
    def offset_from(theta, baseline, reference):
        return theta + reference
"""
from param_id.modifier_funcs import modifier_func  # noqa: F401  (decorator for user funcs)
