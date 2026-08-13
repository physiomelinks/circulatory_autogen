import numpy as np

from param_id.differentiable import differentiable
from param_id.math_backend import make_math_backend, bind_backend

"""
These functions can be used as cost functions. Specify a name of one of these functions as the "cost_type" in obs_data.json to
use it as the cost.

When making your own cost function make sure it works for scalars and vectors. Otherwise put an error message so that if it is used
for the wrong data type it gets called out and stopped.

IMPORTANT FOR BAYESIAN: For MLE estimators the functions below calculate the cost which equals the negative log likelihood.

Backend-dependent costs use the module-level ``mb`` (set to numpy or casadi when the cost dict is built).

All top-level functions defined in this file are registered as costs except private names
(leading ``_``), the decorator helpers ``is_MLE`` / ``cost_combiner``, and the registration
entrypoints. Put non-cost helpers in another module, or prefix them with ``_``.

# Decorators:
# "differentiable" decorator for functions that are differentiable
# "is_MLE" decorator for functions that are the MLE cost function
# "cost_combiner" decorator for functions that combine multiple cost functions

"""


def is_MLE(func):
    func.is_MLE = True
    return func


def cost_combiner(func):
    func.cost_combiner = True
    return func


mb = make_math_backend("numpy")


@differentiable
@is_MLE
def gaussian_MLE(output, desired_mean, std, weight):
    """Gaussian negative log-likelihood contribution (up to constants), averaged over elements.

    Always uses the ``0.5 * mean`` form so scalar outputs match the same NLL scaling as series
    (avoids a 2x Hessian / 0.5x covariance mismatch under ``ln L = -cost`` in paramID).
    """
    per = mb.power((output - desired_mean) / std, 2) * weight
    return 0.5 * mb.sum(per) / mb.numel(per)


@differentiable
def MSE(*args, **kwargs):
    return 2.0*gaussian_MLE(*args, **kwargs) # because the MLE cost function is the negative log likelihood, so we need to multiply by 2 to get the MSE


@is_MLE
def multimodal_gaussian(output, prob_dist_params, weight):
    if hasattr(output, "__len__"):
        print("ERROR: multimodal_gaussian cost function is not implemented for series data")

    allowable_keys_list = ["means", "stds", "scales"]
    allowable_keys_list.sort()
    keys_list = [*prob_dist_params]
    keys_list.sort()
    if not isinstance(prob_dist_params, dict):
        print("!!!!!!!!!!!!")
        print("ERROR prob_dist_params in obs_data.json needs to be a dict! The entries should be:")
        print(allowable_keys_list)
        print("!!!!!!!!!!!!")
        exit()

    if keys_list != allowable_keys_list:
        print("!!!!!!!!!!!!")
        print("ERROR prob_dist_params in obs_data.json needs to be a dict with entries:")
        print(allowable_keys_list)
        print("!!!!!!!!!!!!")
        exit()

    if sum(prob_dist_params["scales"]) != 1:
        print("!!!!!!!!!!!!")
        print("ERROR scales in prob_dist_params for multimodal_gaussian in obs_data.json need to sum to 1")
        print("!!!!!!!!!!!!")
        exit()

    v_vec = np.zeros(len(prob_dist_params["means"]))
    for idx, (desired_mean, std, scale) in enumerate(
        zip(prob_dist_params["means"], prob_dist_params["stds"], prob_dist_params["scales"])
    ):
        v_vec[idx] = np.power((output - desired_mean) / std, 2) * scale

    v_max = np.max(v_vec)
    sum_inner_term = np.sum(np.exp(v_vec - v_max))

    cost = (v_max + np.log(sum_inner_term)) * weight

    return cost


@is_MLE
def kernel_density_estimation(output, prob_dist_params, weight, bandwidth="scott"):
    """Negative log-likelihood under a kernel density estimate of the observed samples.

    For a target that is known only as a set of measurements rather than as a named
    distribution: the samples are turned into a smooth density with
    ``scipy.stats.gaussian_kde`` and the model output is scored against it. Unlike
    ``multimodal_gaussian`` this needs no assumption about how many modes there are, or
    where they sit.

    ``prob_dist_params`` is the data_item's ``{"data_points": [...]}`` -- the ground truth, the
    distribution-shaped alternative to ``value``/``std``. ``bandwidth`` is a *knob*, so it comes
    from the data_item's ``cost_kwargs`` (issue #84) and can be swept without touching the data;
    it is passed straight to gaussian_kde's ``bw_method`` ('scott', 'silverman', a scalar or a
    callable), defaulting to Scott's rule.
    """
    if hasattr(output, "__len__") and np.size(output) > 1:
        raise ValueError(
            "kernel_density_estimation cost function is not implemented for series data")

    if not isinstance(prob_dist_params, dict) or "data_points" not in prob_dist_params:
        raise ValueError(
            "prob_dist_params for kernel_density_estimation in obs_data.json must be a dict "
            "with a 'data_points' entry. Set the smoothing width in the data_item's "
            '"cost_kwargs": {"bandwidth": ...} instead.')

    data_points = np.asarray(prob_dist_params["data_points"], dtype=float)
    if data_points.size == 0:
        raise ValueError(
            "data_points for kernel_density_estimation in obs_data.json cannot be empty")

    from scipy.stats import gaussian_kde

    kde = gaussian_kde(data_points, bw_method=bandwidth)
    # logpdf returns an array even for one point; the cost must be a plain scalar so the
    # weighted sum over observables stays 0-d.
    return float(-np.ravel(kde.logpdf(np.ravel(output)))[0] * weight)


@is_MLE
def poisson_MLE(output, prob_dist_params, weight):
    """Poisson negative log-likelihood contribution, for count data.

    The model output is the rate (lambda); ``prob_dist_params['k']`` is the observed count.
    The constant ``log(k!)`` is dropped -- it does not depend on the parameters, so it shifts
    every cost by the same amount and changes no optimum.

    NLL = lambda - k*log(lambda), so it is minimised at lambda == k. (#367 had the sign the
    other way round, i.e. the log-likelihood, which an optimiser would have driven *away*
    from the data.)
    """
    if not isinstance(prob_dist_params, dict) or "k" not in prob_dist_params:
        raise ValueError(
            "prob_dist_params for poisson_MLE in obs_data.json must be a dict with a 'k' "
            "entry (the observed count)")

    # A Poisson rate is positive; clip so a parameter set that drives it to zero costs a lot
    # rather than producing -inf/nan and poisoning the whole cost.
    lam = np.clip(output, 1e-12, None)
    k = prob_dist_params["k"]

    cost = (lam - k * np.log(lam)) * weight

    if hasattr(cost, "__len__"):
        cost = np.sum(cost) / np.size(cost)

    return float(cost)


@differentiable
def AE(output, desired_mean, std, weight):
    cost = mb.abs((output - desired_mean) / std) * weight
    if mb.numel(output) > 1:
        cost = mb.sum(cost) / mb.numel(cost)
    return cost


@differentiable
@is_MLE
@cost_combiner
def additive(costs):
    cost = sum(costs)
    return cost


@differentiable
@cost_combiner
def norm_additive(costs):
    cost = sum(costs) / len(costs)
    return cost

##
## Below here are the organisational functions for building the cost functions dictionary
## They are not part of the public API
##

def register_cost_funcs(registry, backend):
    """Register all cost callables defined in this module, each bound to ``backend`` (so
    registries for different backends stay independent -- see math_backend.bind_backend, #315)."""
    global mb
    mb = backend
    g = globals()
    mod = __name__
    exclude = frozenset(
        {
            "is_MLE",
            "cost_combiner",
            "register_cost_funcs",
            "build_cost_funcs_dict",
            "get_cost_funcs_dict_for_mode",
            "cost_func_metadata",
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


# Decorator/hook helper names an external cost-funcs file might define locally; excluded so they
# are not registered as costs.
_EXTERNAL_COST_EXCLUDE = frozenset({"is_MLE", "cost_combiner", "register_cost_funcs"})


def build_cost_funcs_dict(backend, external_path=None):
    """Build the cost registry: the built-in costs in this module, then (if given) the costs in the
    external file ``external_path`` (issue #303). A later external func may override a built-in."""
    registry = {}
    register_cost_funcs(registry, backend)
    if external_path:
        from param_id.external_funcs import register_funcs_from_file
        register_funcs_from_file(external_path, registry, backend, exclude=_EXTERNAL_COST_EXCLUDE)
    return registry


def get_cost_funcs_dict_for_mode(mode="numpy", external_path=None):
    return build_cost_funcs_dict(make_math_backend(mode), external_path=external_path)


def cost_func_metadata(mode="numpy", external_path=None):
    """Discoverable metadata for every registered cost function, so an obs-data editor (e.g.
    CUFLynx) can offer the valid ``cost_type`` values and their flags without introspecting the
    callables. Returns ``{name: {"is_MLE": bool, "is_combiner": bool, "differentiable": bool}}``,
    covering both the built-ins and any user-added costs in this module.

    - ``is_MLE``: cost equals a negative log-likelihood (required by the Bayesian method).
    - ``is_combiner``: combines the per-observable costs (e.g. additive), not a per-item cost.
    - ``differentiable``: safe for CasADi symbolic execution (AD gradients).
    """
    from param_id.differentiable import is_circulatory_differentiable
    meta = {}
    for name, func in get_cost_funcs_dict_for_mode(mode, external_path=external_path).items():
        meta[name] = {
            "is_MLE": bool(getattr(func, "is_MLE", False)),
            "is_combiner": bool(getattr(func, "cost_combiner", False)),
            "differentiable": is_circulatory_differentiable(func),
        }
    return meta
