"""Template for your own cost functions (issue #303 / #433).

Copy this file, rename it, add your costs, and point the config at it::

    cost_funcs_external_path: funcs_user/my_costs.py

Every top-level callable defined here is registered as a ``cost_type`` alongside the costs
libcuflynx ships in ``libcuflynx.funcs.cost_funcs_user`` -- reusing a shipped name overrides
it. Names starting with ``_`` are skipped, so keep helpers private.

Decorators:
  ``@is_MLE``          the cost is a negative log-likelihood (required by the Bayesian method)
  ``@cost_combiner``   the cost combines the per-observable costs, rather than scoring one
  ``@differentiable``  safe to execute symbolically under CasADi (AD gradients)

``mb`` is the math backend; libcuflynx rebinds it to numpy or casadi when it builds the
registry, so write against ``mb`` rather than ``np`` for anything that must differentiate.
"""
from libcuflynx.funcs.cost_funcs_user import is_MLE, cost_combiner  # noqa: F401
from libcuflynx.param_id.differentiable import differentiable
from libcuflynx.param_id.math_backend import make_math_backend

mb = make_math_backend("numpy")


# Delete this and write your own. It is registered as cost_type: relative_MLE.
@differentiable
@is_MLE
def relative_MLE(output, desired_mean, std, weight):
    """Gaussian NLL on the *relative* error, for observables spanning several magnitudes."""
    per = mb.power((output - desired_mean) / (std * desired_mean), 2) * weight
    return 0.5 * mb.sum(per) / mb.numel(per)
