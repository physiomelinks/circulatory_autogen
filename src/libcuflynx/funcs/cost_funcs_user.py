import numpy as np
from scipy.special import gammaln

from libcuflynx.param_id.differentiable import differentiable
from libcuflynx.param_id.math_backend import make_math_backend, bind_backend

"""
The cost functions libcuflynx ships. Specify a name of one of these functions as the "cost_type" in obs_data.json to
use it as the cost.

This module is *library* code (issue #433): do not add your own costs here, because an upgrade
of libcuflynx replaces the file. Put them in your own file and name it with the
``cost_funcs_external_path`` config key -- they are merged into the same registry as the
built-ins below, decorators and all, and may override a built-in by reusing its name. See
``libcuflynx/funcs/__init__.py`` and ``funcs_user/cost_funcs_example.py``.

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

#: log(sqrt(2*pi)), the Gaussian normalising constant. gaussian_MLE drops it -- it is the same
#: for every evaluation, so it changes no optimum. A *mixture* cannot drop it: the weight the
#: mixture gives each component depends on the ratio of their densities, so both have to be
#: normalised. See gaussian_MLE_robust.
_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


@differentiable
@is_MLE
def gaussian_MLE(output, desired_mean, std, weight):
    """Gaussian negative log-likelihood contribution (up to constants), averaged over elements.

    Always uses the ``0.5 * mean`` form so scalar outputs match the same NLL scaling as series
    (avoids a 2x Hessian / 0.5x covariance mismatch under ``ln L = -cost`` in paramID).
    """
    per = mb.power((output - desired_mean) / std, 2) * weight
    return 0.5 * mb.sum(per) / mb.numel(per)


#: How many terms of the Conway-Maxwell-Poisson series to sum. The distribution is
#: under-dispersed for nu > 1, so its tail is shorter than a Poisson of the same mean; 256
#: terms is far past the point where a term contributes at double precision for any count
#: this pipeline sees (spike counts in the tens).
COM_POISSON_TERMS = 256


#: ``j`` and ``log(j!)`` for the series, built once. Recomputing ``gammaln`` on every call
#: cost 1.1 ms per evaluation, which at eighteen count observables and sixty-four walkers is
#: 3.5 hours added to a 10000-step chain.
_COM_POISSON_J = np.arange(COM_POISSON_TERMS, dtype=float)
_COM_POISSON_LOG_FACT = gammaln(_COM_POISSON_J + 1.0)


def _com_poisson_log_terms(log_lam, nu, n_terms=COM_POISSON_TERMS):
    """log of the unnormalised weight of each count j: j*log(lambda) - nu*log(j!)."""
    if n_terms == COM_POISSON_TERMS:
        j, log_fact = _COM_POISSON_J, _COM_POISSON_LOG_FACT
    else:
        j = np.arange(n_terms, dtype=float)
        log_fact = gammaln(j + 1.0)
    return j * log_lam - nu * log_fact, j


def _com_poisson_log_Z(log_lam, nu, n_terms=COM_POISSON_TERMS):
    """log of the normalising constant Z(lambda, nu), by log-sum-exp."""
    terms, _ = _com_poisson_log_terms(log_lam, nu, n_terms)
    top = terms.max()
    return top + np.log(np.exp(terms - top).sum())


def _com_poisson_mean(log_lam, nu, n_terms=COM_POISSON_TERMS):
    """E[Y] under COM-Poisson. Monotone increasing in log_lam, which is what inverts it."""
    terms, j = _com_poisson_log_terms(log_lam, nu, n_terms)
    top = terms.max()
    w = np.exp(terms - top)
    return float((w * j).sum() / w.sum())


def _com_poisson_log_lam_for_mean(target, nu, n_terms=COM_POISSON_TERMS):
    """The log(lambda) whose COM-Poisson mean is ``target``.

    lambda is *not* the mean unless nu == 1, so a model that predicts an expected count has
    to be mapped onto it. The mean is strictly increasing in lambda, so a bisection on
    log(lambda) is exact to machine tolerance in ~60 iterations and cannot diverge -- which
    a Newton step on a distribution this flat-tailed can. The asymptotic
    ``lambda ~= (m + (nu-1)/(2nu))**nu`` is not used: it is poor for exactly the small means
    this pipeline has (spike counts of 0-3).
    """
    if target <= 0.0:
        return -np.inf
    # bracket around the Poisson answer (log lambda = log(mean) at nu == 1); the true value
    # moves monotonically with nu, so a wide but finite bracket around it is safe.
    guess = np.log(target)
    lo, hi = guess - 40.0, guess + 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _com_poisson_mean(mid, nu, n_terms) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


#: Cached (log lambda, log mean, log Z) grids, one per nu. Built on first use.
_COM_POISSON_TABLES = {}

#: Grid resolution for those tables. 20000 points over log lambda in [-50, 60] puts the
#: interpolation error below 1e-9 in the cost, against a 60-step bisection plus a 256-term
#: log-sum-exp *per call* -- which added two hours to a 10000-step chain.
_COM_POISSON_GRID = 20000


def _com_poisson_table(nu, n_terms=COM_POISSON_TERMS):
    """``(log_lam, log_mean, log_Z)`` on a grid, so a cost evaluation is two interpolations."""
    key = (float(nu), int(n_terms))
    cached = _COM_POISSON_TABLES.get(key)
    if cached is not None:
        return cached
    log_lam = np.linspace(-50.0, 60.0, _COM_POISSON_GRID)
    j = _COM_POISSON_J if n_terms == COM_POISSON_TERMS else np.arange(n_terms, dtype=float)
    log_fact = (_COM_POISSON_LOG_FACT if n_terms == COM_POISSON_TERMS
                else gammaln(j + 1.0))
    terms = log_lam[:, None] * j[None, :] - nu * log_fact[None, :]
    top = terms.max(axis=1, keepdims=True)
    w = np.exp(terms - top)
    total = w.sum(axis=1)
    log_Z = top[:, 0] + np.log(total)
    mean = (w * j[None, :]).sum(axis=1) / total
    keep = mean > 0                      # the smallest lambdas underflow to a zero mean
    log_lam, log_Z, log_mean = log_lam[keep], log_Z[keep], np.log(mean[keep])
    # Interpolate the residual from the analytic trend rather than the quantity itself.
    # mean ~ lambda**(1/nu), so log_lam ~ nu*log_mean and the leftover is small and flat --
    # which drops the linear-interpolation error from 7e-5 to below 1e-9 on the same grid.
    cached = (log_lam, log_mean, log_Z, log_lam - nu * log_mean)
    _COM_POISSON_TABLES[key] = cached
    return cached


@is_MLE
def com_poisson_MLE(output, prob_dist_params, weight, nu=1.0, background_rate=0.0):
    """Conway-Maxwell-Poisson NLL for a count whose spread is narrower than Poisson's.

    A Poisson cannot be told how tight it is: its variance equals its mean, so at an observed
    count of 0 or 1 it barely distinguishes a model that fires once from one that fires three
    times. COM-Poisson adds the one parameter that fixes this::

        P(Y=y) = lambda**y / ((y!)**nu * Z(lambda, nu)),   Z = sum_j lambda**j / (j!)**nu

    ``nu`` is the dispersion: 1 is exactly Poisson, ``nu > 1`` is under-dispersed (tighter),
    ``nu < 1`` over-dispersed. Unlike raising a data_item's ``weight``, this stays a proper
    probability model -- weighting tempers the likelihood by a power and is not a distribution
    over counts at all, so a posterior built from it is not a posterior for any model.

    **``output`` is the expected count, not lambda.** lambda is only the mean when nu == 1, so
    the model's prediction is mapped through :func:`_com_poisson_log_lam_for_mean` first. That
    keeps the optimum at ``E[Y] == k`` for every nu, which is what makes nu a pure
    tightness knob rather than something that also moves the fit.

    ``background_rate`` matches :func:`poisson_MLE`: a rate the model does not claim to
    explain, added to its prediction before scoring.

    The ``log(k!)`` term is kept rather than dropped (it carries ``nu``, so unlike the Poisson
    case it is not a constant across candidate nu), which means costs are comparable between
    nu values but not with ``poisson_MLE``'s.
    """
    if not isinstance(prob_dist_params, dict) or "k" not in prob_dist_params:
        raise ValueError(
            "prob_dist_params for com_poisson_MLE in obs_data.json must be a dict with a 'k' "
            "entry (the observed count)")
    if nu <= 0.0:
        raise ValueError(f"nu for com_poisson_MLE must be > 0, got {nu}")
    if background_rate < 0.0:
        raise ValueError(
            f"background_rate for com_poisson_MLE must be >= 0, got {background_rate}")

    k = float(prob_dist_params["k"])
    mean = float(np.clip(np.clip(output, 0.0, None) + background_rate, 1e-12, None))

    # The expensive half was inverting the mean -- sixty bisection steps, each a full
    # log-sum-exp. That is what the table replaces. log Z is then a single 256-term
    # log-sum-exp at the answer, so it stays exact rather than interpolated.
    _, grid_log_mean, _, grid_resid = _com_poisson_table(nu)
    log_mean = np.log(mean)
    log_lam = float(nu * log_mean + np.interp(log_mean, grid_log_mean, grid_resid))
    log_Z = float(_com_poisson_log_Z(log_lam, nu))
    cost = (nu * gammaln(k + 1.0) + log_Z - k * log_lam) * weight
    return float(cost)


@differentiable
@is_MLE
def gaussian_MLE_robust(output, desired_mean, std, weight,
                        p_outlier=0.0, outlier_width=170.0):
    """Gaussian NLL with an outlier component, for an observable the model can miss outright.

    ``gaussian_MLE`` assumes the model is always trying to hit the observation and is only ever
    off by measurement noise. That is false for an observable with a discontinuity in it. The
    maximum membrane voltage of a spiking cell is the spike peak when the cell fires and the
    subthreshold maximum when it does not, so the model produces one of two values ~110 mV apart
    and nothing in between. Against a measurement sigma of a few mV the gap is 25-30 sigma, and
    landing on the wrong side of it costs hundreds of nats -- more than every other observable in
    the study put together. That number is arithmetic on a category error: the branch mismatch is
    a *model discrepancy* event and sigma is a *measurement* scale.

    So score it as a two-component mixture::

        p(y | theta) = (1 - eps) * N(y ; m(theta), std**2)  +  eps * q(y)

    ``eps`` (``p_outlier``) is the probability the model's prediction is wrong for reasons outside
    theta -- for a jump observable, the probability it takes the wrong branch. ``q`` is what is
    believed about ``y`` when that happens: uniform over ``outlier_width``, the physiological span
    of the observable, so ``q(y) = 1/outlier_width``.

    Because ``y`` is data, ``q(y)`` is a constant, and the whole thing closes to::

        NLL = weight * [ log(W/eps) - log(1 + K * exp(-z**2 / 2)) ],
              z = (output - desired_mean)/std,   K = (1-eps)*W / (eps*std*sqrt(2*pi))

    ``K`` is constant, so ``K*exp(-z**2/2)`` is bounded above by ``K`` -- no overflow, no log of
    zero, and no log-sum-exp needed. It is smooth in theta, hence ``@differentiable``.

    What the two knobs do is worth keeping straight, because only one of them is a real dial:

    * ``p_outlier`` and ``outlier_width`` set the *height* of the cap, ``log(W/eps)``: the most
      this observable can ever cost, however wrong the model is.
    * ``std`` sets *where* the cap starts, at ``z_c = sqrt(2*ln((1-eps)*W/(eps*std*sqrt(2*pi))))``.
      Because eps and W enter through a logarithm, ``z_c`` is ~3.2-3.4 for any sensible pair --
      halving eps moves it by a few hundredths. ``std`` moves the crossover linearly, so it is
      ``std``, not eps, that decides which observables still inform the fit and which are written
      off as outliers.

    Set ``std`` accordingly: it has to cover the discrepancy you expect to *fit* (a systematically
    over-large spike amplitude, say) while leaving the discrepancy you want *capped* (the branch
    flip) outside. That makes it measurement error and model discrepancy in quadrature, not an
    instrument specification.

    ``p_outlier=0`` reduces to the ordinary Gaussian NLL **including** the normalising constant,
    so it differs from ``gaussian_MLE`` by ``log(std*sqrt(2*pi))``. Per-item constants cancel in a
    Metropolis ratio and move no optimum, but they do shift the reported cost, so numbers from a
    study using this cost are not comparable with numbers from one using ``gaussian_MLE``.
    """
    if not 0.0 <= p_outlier < 1.0:
        raise ValueError(
            f"p_outlier for gaussian_MLE_robust must be in [0, 1), got {p_outlier}. It is the "
            f"probability that the model's prediction is wrong for reasons outside the "
            f"parameters; 1 would mean the model is never informative.")
    if outlier_width <= 0.0:
        raise ValueError(
            f"outlier_width for gaussian_MLE_robust must be positive, got {outlier_width}. It is "
            f"the span of values this observable can plausibly take when the model is wrong, in "
            f"the observable's own units.")

    z = (output - desired_mean) / std
    if p_outlier == 0.0:
        # The eps -> 0 limit, written out rather than reached by dividing by zero.
        per = 0.5 * mb.power(z, 2) + mb.log(std) + _LOG_SQRT_2PI
    else:
        log_cap = float(np.log(outlier_width / p_outlier))
        gain = ((1.0 - p_outlier) / p_outlier) * outlier_width / (std * _SQRT_2PI)
        per = log_cap - mb.log(1.0 + gain * mb.exp(-0.5 * mb.power(z, 2)))
    per = per * weight
    return mb.sum(per) / mb.numel(per)


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
def poisson_MLE(output, prob_dist_params, weight, background_rate=0.0):
    """Poisson negative log-likelihood contribution, for count data.

    The model output is the rate (lambda); ``prob_dist_params['k']`` is the observed count.
    The constant ``log(k!)`` is dropped -- it does not depend on the parameters, so it shifts
    every cost by the same amount and changes no optimum.

    NLL = lambda - k*log(lambda), so it is minimised at lambda == k. (#367 had the sign the
    other way round, i.e. the log-likelihood, which an optimiser would have driven *away*
    from the data.)

    ``background_rate`` is a rate the model does not claim to explain -- spontaneous activity, or
    a detection artefact -- added to the model's own rate before scoring. It defaults to 0, which
    reproduces the behaviour below exactly, and exists because the clip is otherwise doing work
    it was never meant to do. A model that is silent where the recording fired k times pays
    ``-k*log(lambda)``, and with lambda pinned at the 1e-12 clip that is ``k * 27.6`` -- a number
    set by a numerical guard rather than by any belief about the cell. State it instead: at
    ``background_rate=0.01`` the same miss costs ``k * 4.6``.

    It is a strong dial, not a formality. Anywhere a jump observable has been capped (see
    ``gaussian_MLE_robust``) the counts are the only thing left pushing the model towards firing,
    so this value sets almost the whole of that push. It also moves the optimum to
    ``output == k - background_rate``, which is negligible at 0.01 but is a bias, not a rounding.
    """
    if not isinstance(prob_dist_params, dict) or "k" not in prob_dist_params:
        raise ValueError(
            "prob_dist_params for poisson_MLE in obs_data.json must be a dict with a 'k' "
            "entry (the observed count)")

    if background_rate < 0.0:
        raise ValueError(
            f"background_rate for poisson_MLE must be >= 0, got {background_rate}")
    # A Poisson rate is positive. background_rate is the stated floor; the 1e-12 clip stays
    # underneath it as the last-resort guard against -inf/nan when no floor was given.
    lam = np.clip(np.clip(output, 0.0, None) + background_rate, 1e-12, None)
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
        from libcuflynx.param_id.external_funcs import register_funcs_from_file
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
    from libcuflynx.param_id.differentiable import is_circulatory_differentiable
    meta = {}
    for name, func in get_cost_funcs_dict_for_mode(mode, external_path=external_path).items():
        meta[name] = {
            "is_MLE": bool(getattr(func, "is_MLE", False)),
            "is_combiner": bool(getattr(func, "cost_combiner", False)),
            "differentiable": is_circulatory_differentiable(func),
        }
    return meta
