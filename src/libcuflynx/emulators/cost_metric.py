"""A tuning metric denominated in the cost, not in the features' own spread.

An emulator here exists to drive ``cost(theta)``. Nothing else reads it: a GA walks
it and a sampler walks it, and both only ever see the cost. But the metric that
selects it is R2 -- autoemulate's default, which CA has never overridden -- and R2 is
the wrong currency twice over.

**It is scale-free.** R2 divides each feature's error by that feature's variance
*across the training design*, a quantity with no relationship to the cost. The cost
divides by ``std``, the measurement sigma the obs_data declares. A feature whose
design spread is wide and whose sigma is tight can keep 99 % of its variance and
still be tens of sigma out; measured on a real bundle, worst-feature R2 0.9943 came
with the cost understated 4.1x at the posterior median.

**It is averaged over features.** autoemulate's R2 is one number over the whole
multi-output target, so a feature contributing a third of the cost counts exactly as
much as one contributing nothing. The cost is a *sum*: an error in a tight-sigma
feature costs many times an identical error in a loose one.

:class:`CostWeightedError` fixes both by measuring, per feature, the error in units of
that feature's own sigma and weighting it by the same ``weight`` the cost uses. It is
the expected cost contribution of the emulator's error, up to a constant -- which is
the thing worth minimising.

**Why not std(dcost) or Spearman, which is what we judge a finished bundle on.**
Neither can be a tuning metric, for one structural reason each:

* ``std(dcost)`` needs the *total* cost, so it needs every feature at once. A
  multi-phase emulator is fitted in pieces -- a base regressor, a classifier per
  count, a pair of regressors per jump group -- and a hyperparameter search inside one
  piece cannot see the others. This metric is a sum over features, so it restricts to
  whatever subset a piece is fitted on and stays a genuine part of the same total.
* Spearman is rank-based: flat across most of a hyperparameter space, noisy on a
  single fold, and invariant to exactly the multiplicative cost distortion that
  reshapes a posterior. It is a guard against an inverted landscape, not an objective.

**What it does not capture.** The saturation of ``gaussian_MLE_robust`` (error past
the cap costs nothing more) and the asymmetry of the count likelihoods are both real
and both ignored here: this is a quadratic in ``z = error / sigma``, which is the
exact cost only for a plain ``gaussian_MLE`` term. It is deliberately the cheap,
decomposable approximation -- it needs one vector of numbers from the obs_data and no
cost machinery inside the cross-validation loop. Closing the remaining gap means
scoring real cost terms per fold, which is a bigger change than this one.
"""
import numpy as np

#: Features whose sigma the obs_data does not give -- a count scored by a
#: distribution declares ``prob_dist_params`` and no ``std`` -- are weighted by
#: their spread across the design instead, which is what R2 would have used. That
#: keeps them in the objective at a sane magnitude rather than dropping them or
#: dividing by zero, and how many fell back is reported when the metric is built.
FALLBACK_LABEL = 'design spread'


def sigma_weights(stds, weights, spans, y_train=None):
    """Per-feature multipliers that turn a scaled error into a cost contribution.

    autoemulate sees ``y_scaled = (y - shift) / span``, so an error of ``e`` there is
    ``e * span`` in real units and ``e * span / sigma`` in sigma units. The cost
    weights each term by the obs_data's own ``weight``, and a zero weight drops the
    item from the cost entirely, so it is dropped here too.

    Returns ``(weights, notes)`` where ``weights`` is one non-negative float per
    feature and ``notes`` lists the features whose sigma had to be substituted.
    """
    stds = np.asarray(stds, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    spans = np.asarray(spans, dtype=float).reshape(-1)
    if not (stds.size == weights.size == spans.size):
        raise ValueError(
            f'sigma_weights needs one std, weight and span per feature; got '
            f'{stds.size}, {weights.size} and {spans.size}')

    usable = np.isfinite(stds) & (stds > 0)
    effective = np.where(usable, stds, np.nan)

    notes = []
    if not usable.all():
        # The design's own spread, which is the scale R2 uses, so a feature with no
        # declared sigma is neither dropped nor allowed to dominate.
        if y_train is not None:
            spread = np.std(np.asarray(y_train, dtype=float), axis=0) * spans
        else:
            spread = np.full(stds.shape, np.nan)
        spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
        effective = np.where(usable, effective, spread)
        notes = [int(i) for i in np.flatnonzero(~usable)]

    spans = np.where(np.isfinite(spans) & (spans != 0), spans, 1.0)
    out = np.clip(weights, 0.0, None) * np.abs(spans) / effective
    # A feature the cost does not read contributes nothing, and a non-finite
    # multiplier would poison the whole metric.
    out = np.where(np.isfinite(out), out, 0.0)
    return out, notes


def build_metric(pid, y_scale, y_train=None, name='cost_weighted_error'):
    """A :class:`CostWeightedError` for this run, or None when it cannot be built.

    ``None`` rather than an exception: a study whose obs_info does not offer per-
    constant sigmas should fall back to R2 with a warning, not fail to train.
    """
    obs = getattr(pid, 'obs_info', None) or {}
    stds = obs.get('std_const_vec')
    weights = obs.get('weight_const_vec')
    if stds is None or weights is None:
        return None, ['obs_info has no per-constant std/weight vectors']

    spans = np.asarray((y_scale or {}).get('span', 1.0), dtype=float).reshape(-1)
    if spans.size == 1:
        spans = np.full(np.asarray(stds, dtype=float).reshape(-1).shape, spans[0])
    try:
        weights_vec, fell_back = sigma_weights(stds, weights, spans, y_train=y_train)
    except ValueError as error:
        return None, [str(error)]
    if not np.any(weights_vec > 0):
        return None, ['every feature has zero weight or no usable sigma']
    return CostWeightedError(weights_vec, name=name), fell_back


def _metric_base():
    """autoemulate's ``Metric`` when it is installed, else ``object``.

    It has to be the real base class: ``autoemulate.core.metrics.get_metric``
    dispatches on ``isinstance(metric, Metric)`` and rejects anything else outright, so
    a duck-typed metric is refused with "Unsupported metric type" the moment
    ``AutoEmulate`` is constructed. (Found by running a real fit -- an earlier version of
    this module was duck-typed on the assumption that only ``name``, ``maximize`` and
    ``__call__`` were read.)

    Resolved at import, falling back to ``object``, so this module stays importable in
    an environment with no autoemulate -- which is where its unit tests run, and where
    the rest of CA still has to work.
    """
    try:
        from autoemulate.core.metrics import Metric  # noqa: PLC0415

        return Metric
    except Exception:  # noqa: BLE001 - absent or broken autoemulate is the same here
        return object


class CostWeightedError(_metric_base()):
    """Root mean weighted squared error, minimised. An autoemulate ``Metric``."""

    def __init__(self, weights, name='cost_weighted_error'):
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self.name = str(name)
        #: Lower is better. autoemulate reads this to decide argmin vs argmax.
        self.maximize = False

    def __repr__(self):
        return f'CostWeightedError(name={self.name!r}, n_features={self.weights.size})'

    def __str__(self):
        return self.name

    # autoemulate stores metrics in dicts and sorts them by name.
    def __eq__(self, other):
        return getattr(other, 'name', None) == self.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < getattr(other, 'name', '')

    def __call__(self, y_pred, y_true, metric_params=None):
        """``sqrt(mean(w_i * (y_pred - y_true)_i ** 2))`` over features and points.

        A distribution prediction is reduced to its mean, the way autoemulate's own
        ``TorchMetrics`` does -- this metric scores the point estimate the cost would
        be evaluated at, not the spread around it.
        """
        import torch  # noqa: PLC0415 - only needed when autoemulate is in play

        if hasattr(y_pred, 'mean') and not isinstance(y_pred, torch.Tensor):
            try:
                y_pred = y_pred.mean
            except Exception:  # noqa: BLE001 - fall back the way TorchMetrics does
                n = getattr(metric_params, 'n_samples', 1000) or 1000
                y_pred = y_pred.rsample(torch.Size([n])).mean(dim=0)

        pred = torch.as_tensor(y_pred, dtype=torch.float64)
        true = torch.as_tensor(y_true, dtype=torch.float64).to(pred.device)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if true.ndim == 1:
            true = true.reshape(-1, 1)

        weights = torch.as_tensor(self.weights, dtype=torch.float64, device=pred.device)
        if weights.numel() != pred.shape[-1]:
            # A piece of a multi-phase fit can be handed a column subset. Refusing is
            # better than silently scoring with mismatched weights, which would look
            # like a working metric and rank on nonsense.
            raise ValueError(
                f'{self.name} was built for {weights.numel()} feature(s) but scored '
                f'{pred.shape[-1]}')

        squared = (pred - true) ** 2 * (weights ** 2)
        reduction = getattr(metric_params, 'reduction', 'mean') or 'mean'
        if reduction == 'none':
            return torch.sqrt(squared.mean(dim=0))
        return torch.sqrt(squared.mean())
