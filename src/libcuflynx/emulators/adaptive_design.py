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

This is adaptive sampling, and the usual caution applies -- the second stage can only
refine structure the first stage found. A feature whose active region no Sobol point
ever landed in has no pair to draw a gradient from, and no amount of stage two will
discover it. Stage one has to be big enough to see the phenomenon; stage two makes it
sharp.
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

#: New points land on the segment between the chosen pair, plus a nudge of this much of
#: the pair's separation. Without it every point for a given pair falls on one line,
#: which in more than two dimensions is a set of measure zero and a poor basis to fit
#: from; the nudge gives the cliff some thickness in the other directions.
DEFAULT_JITTER = 0.1


def gradient_weighted_design(x, y, n_samples, mins, maxs, seed=0,
                             log_scale=False, n_neighbours=DEFAULT_NEIGHBOURS,
                             power=DEFAULT_POWER, jitter=DEFAULT_JITTER):
    """``n_samples`` new points, concentrated where neighbouring outputs disagree.

    ``x``/``y`` are the first stage's design and its simulated features. Returns points
    in the same units as ``x``, inside ``[mins, maxs]``.

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

    weights = np.power(weights, float(power))
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return _uniform(rng, n_samples, mins, maxs, log_scale)

    chosen = rng.choice(len(weights), size=n_samples, replace=True, p=weights / total)
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
