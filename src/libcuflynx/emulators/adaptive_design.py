"""A second sampling stage that spends its points where the model changes fastest.

A Sobol design spreads points evenly over the parameter box, which is the right thing
to do when nothing is known about the response. It is the wrong thing to do once
something *is*: an evenly-spread design spends as many simulations describing a region
where the output barely moves as it does resolving the place where it jumps.

Neuron models make that expensive. Below rheobase a spike count is exactly zero, above
it the cell fires, and the whole of what an emulator gets wrong lives in the thin shell
between the two. Points on the flat side are nearly free to predict and nearly useless
to have.

So: run a Sobol design, look at what came back, and draw the rest of the budget near
the pairs of neighbouring points whose outputs disagree most. The gradient estimate is
literally between points -- for neighbours i and j,

    g = ||y_i - y_j|| / ||x_i - x_j||

with x on the unit cube and each output scaled by its own range, so a current near 1e-9
and a firing rate near 10 count the same. New points are drawn *along* the segment
joining a chosen pair, which is what puts them inside the shell rather than merely
near it: a cliff detected between two points is somewhere between those two points.

Two ways to ask where to look next, and they are different questions:

``gradient_weighted_design`` asks where the *model* changes fastest, from the samples
themselves. ``error_weighted_design`` asks where a cheap surrogate of those samples is
*wrong*, which is not the same thing -- a response can be steep and easy to fit, or
flat and hard -- and because it interpolates that error across the whole box rather
than reading it between pairs, it can propose points nowhere near an existing sample.

This is adaptive sampling, and two cautions apply.

The first: a later stage can only refine structure an earlier one found. A feature
whose active region no Sobol point ever landed in has no pair to draw a gradient from
and no measured error to interpolate, and no amount of stage two will discover it.
Stage one has to be big enough to see the phenomenon; stage two makes it sharp.

The second is sharper, and is asserted in the tests rather than left as advice:
clustering points across a discontinuity helps an emulator that can represent one --
a forest, or the classifier half of a two-phase emulator -- and *hurts* a smooth
global fit, which is then forced through steeply-disagreeing neighbours and rings
around the jump. Measured on a step response, a thin-plate spline gets worse when the
second stage is added while a random forest gets better. So an adaptive stage is not
free improvement: it has to be matched to the emulator being fitted.
"""
import numpy as np

#: Neighbours per point used to estimate a local gradient. Small enough to stay local
#: -- the estimate is meaningless across the box -- and large enough that a point does
#: not hang the whole estimate on its single nearest neighbour.
DEFAULT_NEIGHBOURS = 8

#: Exponent on the pair gradients before they become sampling weights. 1.0 makes the
#: draw proportional to the gradient; higher concentrates harder on the steepest pairs
#: at the cost of covering fewer of them.
DEFAULT_POWER = 1.0

#: Candidate points drawn per requested sample before weighting, when a stage picks
#: from a pool rather than from pairs. Large enough that the weighting has something to
#: prefer, small enough to stay a rounding error next to running the model.
CANDIDATES_PER_SAMPLE = 32

#: Folds used to estimate how wrong a cheap surrogate is at each existing sample.
DEFAULT_ERROR_FOLDS = 5

#: New points land on the segment between the chosen pair, plus a nudge of this much of
#: the pair's separation. Without it every point for a given pair falls on one line,
#: which in more than two dimensions is a set of measure zero and a poor basis to fit
#: from; the nudge gives the cliff some thickness in the other directions.
DEFAULT_JITTER = 0.1


def error_weighted_design(x, y, n_samples, mins, maxs, seed=0, log_scale=False,
                          weight=1.0, folds=DEFAULT_ERROR_FOLDS):
    """``n_samples`` new points, concentrated where a surrogate is currently wrong.

    Where ``gradient_weighted_design`` asks where the *model* changes fastest, this asks
    where the *emulator* is failing -- which is not the same question. A response can be
    steep and easy to fit, and flat and hard: what an emulator needs is points where its
    own prediction is poor, wherever those happen to be.

    The error at the existing samples is measured by cross-validating a cheap radial
    basis surrogate over them, so it costs no simulations. Those errors are then
    interpolated by a second radial basis fit to give an error surface over the whole
    box, and candidate points are drawn against it. The surface is the reason this can
    propose points nowhere near an existing sample: a gradient is only defined between
    points that exist, but an interpolated error is defined everywhere.

    Falls back to a uniform draw whenever the surrogate cannot be fitted or says nothing
    -- too few points for the dimension, a degenerate response, a singular system.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    n_samples = int(n_samples)
    rng = np.random.default_rng(seed)

    if n_samples <= 0:
        return np.empty((0, mins.size), dtype=float)
    if x.ndim != 2 or len(x) < 2 or len(y) != len(x):
        return _uniform(rng, n_samples, mins, maxs, log_scale)

    unit = _to_unit(x, mins, maxs, log_scale)
    scaled = _scale_features(y)
    if scaled.shape[1] == 0:
        return _uniform(rng, n_samples, mins, maxs, log_scale)

    errors = _cross_validated_error(unit, scaled, folds)
    if errors is None or not np.any(errors > 0):
        return _uniform(rng, n_samples, mins, maxs, log_scale)

    candidates = rng.random((n_samples * CANDIDATES_PER_SAMPLE, unit.shape[1]))
    predicted = _interpolate(unit, errors, candidates)
    if predicted is None:
        return _uniform(rng, n_samples, mins, maxs, log_scale)
    # An interpolant is free to go negative between samples; a negative error is not a
    # reason to avoid a region, it is the interpolant saying "about zero here".
    predicted = np.clip(predicted, 0.0, None)

    chosen = _weighted_choice(rng, predicted, n_samples, weight)
    return _from_unit(candidates[chosen], mins, maxs, log_scale)


def _cross_validated_error(unit, scaled, folds):
    """Held-out error of a cheap surrogate at every existing sample.

    Cross-validated rather than measured in place: a surrogate that interpolates its
    own training points reports zero error everywhere, which would say the design is
    finished no matter how bad it is.
    """
    n_points = len(unit)
    folds = int(min(max(int(folds), 2), n_points))
    if n_points <= unit.shape[1] + 1:
        return None

    # Deterministic folds: this runs inside a seeded design, and a training run has to
    # be repeatable.
    assignment = np.arange(n_points) % folds
    errors = np.full(n_points, np.nan, dtype=float)
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        if train.sum() <= unit.shape[1] + 1 or not test.any():
            continue
        predicted = _interpolate(unit[train], scaled[train], unit[test])
        if predicted is None:
            return None
        residual = predicted.reshape(test.sum(), -1) - scaled[test]
        errors[test] = np.sqrt(np.mean(np.square(residual), axis=1))
    if not np.any(np.isfinite(errors)):
        return None
    # A fold that could not be fitted leaves gaps; they carry no claim either way, so
    # they get the average rather than a zero that would steer sampling away from them.
    return np.nan_to_num(errors, nan=float(np.nanmean(errors)))


def _interpolate(points, values, at):
    """Radial basis interpolation, or ``None`` if the system cannot be solved."""
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:  # pragma: no cover - scipy ships with the solver stack
        return None
    values = np.asarray(values, dtype=float)
    flat = values.ndim == 1
    values = values.reshape(len(values), -1)
    try:
        # `neighbors` keeps this local and keeps the solve small; a global fit over
        # thousands of points in a dozen dimensions is both slower and less stable.
        neighbours = int(min(len(points), max(points.shape[1] + 2, 50)))
        interpolator = RBFInterpolator(points, values, neighbors=neighbours,
                                       kernel='thin_plate_spline', smoothing=1e-8)
        predicted = np.asarray(interpolator(at), dtype=float)
    except Exception:  # noqa: BLE001 - singular, degenerate or unsupported: caller falls back
        return None
    if not np.all(np.isfinite(predicted)):
        return None
    return predicted.reshape(len(at)) if flat else predicted


def _weighted_choice(rng, weights, n_samples, weight, replace=False):
    """Indices drawn against ``weights``, mixed with a uniform draw by ``weight``.

    ``weight`` is a single dial over the whole range from ignoring the weights to
    following them closely:

    * ``0``   -- uniform. The weights are computed and then not used, which is the
      honest way to turn an adaptive stage into a plain random top-up.
    * ``0.5`` -- half the probability mass spread uniformly, half distributed by weight.
    * ``1``   -- drawn by weight.
    * ``>1``  -- drawn by weight raised to that power, concentrating harder on the
      worst regions and covering fewer of them.

    Mixing in probability space rather than drawing two separate batches keeps every
    point a single draw from one distribution, so the stage has no seam in it.
    """
    weight = float(weight)
    n_candidates = len(weights)
    # Without replacement a pool cannot hand back more points than it holds; with it,
    # the same pair can be drawn from repeatedly, which is how a handful of steep pairs
    # can absorb a whole stage.
    take = int(n_samples if replace else min(n_samples, n_candidates))

    if weight <= 0:
        return rng.choice(n_candidates, size=take, replace=replace)

    weights = np.asarray(weights, dtype=float)
    largest = weights.max()
    if not np.isfinite(largest) or largest <= 0:
        return rng.choice(n_candidates, size=take, replace=replace)

    # Normalised before the power so the exponent means the same thing whatever units
    # the weights arrived in.
    shaped = np.power(weights / largest, max(weight, 1.0))
    total = shaped.sum()
    if not np.isfinite(total) or total <= 0:
        return rng.choice(n_candidates, size=take, replace=replace)

    mix = min(weight, 1.0)
    probability = (1.0 - mix) / n_candidates + mix * (shaped / total)
    probability = probability / probability.sum()
    return rng.choice(n_candidates, size=take, replace=replace, p=probability)


def gradient_weighted_design(x, y, n_samples, mins, maxs, seed=0,
                             log_scale=False, weight=DEFAULT_POWER,
                             n_neighbours=DEFAULT_NEIGHBOURS, jitter=DEFAULT_JITTER):
    """``n_samples`` new points, concentrated where neighbouring outputs disagree.

    ``x``/``y`` are the first stage's design and its simulated features. Returns points
    in the same units as ``x``, inside ``[mins, maxs]``. ``weight`` is the same dial as
    everywhere else: 0 ignores the gradients, 0.5 half-follows them, 1 follows them, and
    above 1 concentrates harder on the steepest pairs (see :func:`_weighted_choice`).

    Falls back to a uniform draw over the box whenever there is nothing to be adaptive
    about -- fewer than two simulated points, or a response with no variation anywhere.
    Falling back is the honest answer there: a weight vector of zeros carries no
    information about where to look, and inventing one would concentrate the budget on
    whatever numerical dust happened to be largest.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    n_samples = int(n_samples)
    rng = np.random.default_rng(seed)

    if n_samples <= 0:
        return np.empty((0, mins.size), dtype=float)
    if x.ndim != 2 or len(x) < 2 or len(y) != len(x):
        return _uniform(rng, n_samples, mins, maxs, log_scale)

    unit = _to_unit(x, mins, maxs, log_scale)
    scaled = _scale_features(y)
    if scaled.shape[1] == 0:
        # Every feature came back constant, so no pair disagrees with any other and
        # there is no cliff to refine. A plain space-filling top-up is what is left.
        return _uniform(rng, n_samples, mins, maxs, log_scale)

    left, right, weights = _pair_gradients(unit, scaled, n_neighbours)
    if left.size == 0 or not np.any(weights > 0):
        return _uniform(rng, n_samples, mins, maxs, log_scale)

    chosen = _weighted_choice(rng, weights, n_samples, weight, replace=True)
    a = unit[left[chosen]]
    b = unit[right[chosen]]
    # Uniform along the segment: the crossing is somewhere between the two points and
    # nothing in the pair says where, so no position on it is preferred.
    t = rng.random((n_samples, 1))
    points = a + t * (b - a)
    if jitter > 0:
        separation = np.linalg.norm(b - a, axis=1, keepdims=True)
        points = points + rng.normal(scale=float(jitter), size=points.shape) * separation
    np.clip(points, 0.0, 1.0, out=points)
    return _from_unit(points, mins, maxs, log_scale)


def _pair_gradients(unit, scaled, n_neighbours):
    """``(i, j, gradient)`` over each point's nearest neighbours, deduplicated.

    A pair reached from both ends is one pair; keeping both would double the weight of
    exactly the mutual-nearest-neighbour pairs, which are the ones most likely to
    straddle a cliff.
    """
    n_points = len(unit)
    k = int(min(max(n_neighbours, 1), n_points - 1))
    neighbours = _nearest(unit, k)

    left = np.repeat(np.arange(n_points), k)
    right = neighbours.reshape(-1)
    keep = left != right
    left, right = left[keep], right[keep]

    low = np.minimum(left, right)
    high = np.maximum(left, right)
    _, unique = np.unique(np.stack([low, high], axis=1), axis=0, return_index=True)
    left, right = low[unique], high[unique]

    dx = np.linalg.norm(unit[left] - unit[right], axis=1)
    dy = np.linalg.norm(scaled[left] - scaled[right], axis=1)
    # Coincident points carry no direction to step in; their gradient is undefined
    # rather than infinite, so they are dropped instead of dominating the draw.
    usable = dx > 0
    return left[usable], right[usable], dy[usable] / dx[usable]


def _nearest(unit, k):
    """Indices of each point's ``k`` nearest neighbours, itself excluded."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy ships with the solver stack
        distances = np.linalg.norm(unit[:, None, :] - unit[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        return np.argsort(distances, axis=1)[:, :k]
    # k+1 because the query returns the point itself first.
    _, indices = cKDTree(unit).query(unit, k=k + 1)
    indices = np.atleast_2d(indices)
    return indices[:, 1:k + 1]


def _scale_features(y):
    """Each feature divided by its own range; features with no range dropped.

    Without this the norm is whichever feature happens to carry the largest numbers.
    A membrane current near 1e-9 and a firing rate near 10 have to count the same,
    because a jump is a jump in whichever of them it happens.
    """
    finite = np.isfinite(y)
    y = np.where(finite, y, np.nan)
    lows = np.nanmin(y, axis=0)
    highs = np.nanmax(y, axis=0)
    spans = highs - lows
    keep = np.isfinite(spans) & (spans > 0)
    if not keep.any():
        return np.zeros((len(y), 0), dtype=float)
    scaled = (y[:, keep] - lows[keep]) / spans[keep]
    # A feature that failed on one sample must not delete that sample's whole row;
    # it contributes nothing to the distance instead.
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def _uniform(rng, n_samples, mins, maxs, log_scale):
    return _from_unit(rng.random((n_samples, mins.size)), mins, maxs, log_scale)


def _to_unit(x, mins, maxs, log_scale):
    if log_scale:
        return (np.log(x) - np.log(mins)) / (np.log(maxs) - np.log(mins))
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    return (x - mins) / spans


def _from_unit(unit, mins, maxs, log_scale):
    if log_scale:
        return np.exp(np.log(mins) + unit * (np.log(maxs) - np.log(mins)))
    return mins + unit * (maxs - mins)
