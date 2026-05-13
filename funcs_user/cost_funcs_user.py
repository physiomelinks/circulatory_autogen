import numpy as np

from param_id.differentiable import differentiable
from param_id.math_backend import make_math_backend

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
def kernel_density_estimation(output, prob_dist_params, weight):
    """
    KDE-based negative log-likelihood cost.
    """

    if hasattr(output, '__len__'):
        raise ValueError("KDE cost not implemented for series data")

    # Validate input
    if not isinstance(prob_dist_params, dict) or "data_points" not in prob_dist_params:
        raise ValueError("prob_dist_params must be {'data_points': [...]}")

    bandwidth = prob_dist_params.get('bandwidth', 'scott')  # default to Scott's rule if not provided
    data_points = np.asarray(prob_dist_params["data_points"])

    if data_points.size == 0:
        raise ValueError("data_points cannot be empty")

    from scipy.stats import gaussian_kde
    
    kde = gaussian_kde(prob_dist_params["data_points"], bw_method=bandwidth)
    
    log_density = kde.logpdf(output)
    cost = -1 * log_density * weight

    return cost

@is_MLE
def poisson_MLE(output, prob_dist_params, weight):
    """
    output: model prediction (rate or lambda)
    desired_mean: observed count (k)
    weight: scaling factor (like your Gaussian case)
    """
    eps = 1e-12

    # lambda (expected count)
    lam = np.clip(output, eps, None)
    k = prob_dist_params['k']

    # log-likelihood (dropping log(k!) since it's constant)
    cost = (k * np.log(lam) - lam) * weight

    if hasattr(output, '__len__'):
        cost = np.sum(cost) / len(output)

    return cost

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
    """Bind ``mb`` to ``backend`` and register all cost callables defined in this module."""
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


def build_cost_funcs_dict(backend):
    registry = {}
    register_cost_funcs(registry, backend)
    return registry


def get_cost_funcs_dict_for_mode(mode="numpy"):
    return build_cost_funcs_dict(make_math_backend(mode))
