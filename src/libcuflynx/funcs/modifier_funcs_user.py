"""Shipped modifier functions for params_for_id entries (issue #383), plus the decorator that
user-written modifier files import.

A modifier function maps one calibrated value (theta) to each model parameter its entry
names in ``modifies``. (An *operation* acts on an output and belongs in obs_data; a *modifier*
acts on a parameter and belongs in params_for_id.)

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

    {"name": "q_tot", "modifies": ["heart/q_lv_init"], "modifier": "remainder",
     "inputs": {"subtract": ["pvn/q_init", "par/q_init"]}, "min": 4e-3, "max": 6e-3}

``scale`` and ``remainder`` are built in -- see ``libcuflynx/param_id/modifier_funcs.py``.
This module currently ships no modifiers beyond those two; it exists as the third tier of the
registry (built-ins, this module, then the external file) and as the import site for the
decorator.

This module is *library* code (issue #433): do not add your own modifiers here, because an
upgrade of libcuflynx replaces the file. Put them in your own file and name it with the
``modifier_funcs_external_path`` config key -- redefining a built-in name there deliberately
overrides it. See ``funcs_user/modifier_funcs_example.py``::

    from libcuflynx.funcs.modifier_funcs_user import modifier_func

    @modifier_func(inputs={'reference': 'float'},
                   description='target = theta + reference')
    def offset_from(theta, baseline, reference):
        return theta + reference
"""
from libcuflynx.param_id.modifier_funcs import modifier_func  # noqa: F401  (decorator for user funcs)
