"""The two likelihood costs added for uncertainty quantification (from #367).

Both are ``@is_MLE``, so paramID reads them as ``ln L = -cost``: the value they return must be a
*negative* log-likelihood, positive and minimised at the best fit, matching ``gaussian_MLE``.
That convention is what these tests pin -- getting the sign wrong does not fail loudly, it just
makes the optimiser walk away from the data.
"""
import numpy as np
import pytest

from libcuflynx.funcs.cost_funcs_user import (
    cost_func_metadata, gaussian_MLE, kernel_density_estimation, poisson_MLE)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_new_costs_are_discoverable_as_MLE_and_not_differentiable():
    """The obs-data editor builds its cost_type menu from this metadata. Both are MLE (so they
    are selectable for MCMC / Laplace) and neither is differentiable -- a KDE's logpdf and the
    Poisson NLL are numpy-only, so they must not be offered as an AD cost."""
    meta = cost_func_metadata()
    for name in ('kernel_density_estimation', 'poisson_MLE'):
        assert name in meta, name + ' is not registered as a cost function'
        assert meta[name]['is_MLE'] is True
        assert meta[name]['differentiable'] is False
        assert meta[name]['is_combiner'] is False


# ---------------------------------------------------------------------------
# poisson_MLE
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_poisson_cost_is_the_negative_log_likelihood():
    """NLL = lambda - k*log(lambda), dropping the parameter-independent log(k!)."""
    k, lam = 4, 3.0
    expected = lam - k * np.log(lam)
    assert poisson_MLE(lam, {'k': k}, 1.0) == pytest.approx(expected)


@pytest.mark.unit
def test_poisson_cost_is_minimised_when_the_rate_matches_the_count():
    """The sign check that matters. #367 returned k*log(lambda) - lambda -- the log-likelihood --
    which is *maximised* at lambda == k, so minimising it would have driven the rate away from
    the observed count in both directions."""
    k = 4
    at_truth = poisson_MLE(float(k), {'k': k}, 1.0)
    for wrong_rate in (0.5, 1.0, 2.0, 8.0, 40.0):
        assert poisson_MLE(wrong_rate, {'k': k}, 1.0) > at_truth, wrong_rate

    # and it really is the minimum, not merely better than a few samples
    rates = np.linspace(0.1, 20.0, 400)
    costs = [poisson_MLE(r, {'k': k}, 1.0) for r in rates]
    assert rates[int(np.argmin(costs))] == pytest.approx(k, abs=0.1)


@pytest.mark.unit
def test_poisson_cost_scales_with_weight_and_returns_a_scalar():
    cost = poisson_MLE(3.0, {'k': 4}, 1.0)
    assert poisson_MLE(3.0, {'k': 4}, 2.5) == pytest.approx(2.5 * cost)
    assert isinstance(cost, float)


@pytest.mark.unit
def test_poisson_cost_survives_a_rate_driven_to_zero():
    """An unclipped log(0) would return -inf and poison the summed cost for every other
    observable, turning one bad parameter set into an uninformative total."""
    cost = poisson_MLE(0.0, {'k': 4}, 1.0)
    assert np.isfinite(cost)
    assert cost > poisson_MLE(4.0, {'k': 4}, 1.0)


@pytest.mark.unit
def test_poisson_cost_requires_the_observed_count():
    with pytest.raises(ValueError, match="'k'"):
        poisson_MLE(3.0, {'lambda': 4}, 1.0)


# ---------------------------------------------------------------------------
# kernel_density_estimation
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_kde_cost_matches_scipy_gaussian_kde():
    gaussian_kde = pytest.importorskip('scipy.stats').gaussian_kde
    samples = [1.0, 1.2, 0.9, 1.1, 1.05, 0.95]

    expected = -float(gaussian_kde(np.asarray(samples)).logpdf([1.0])[0])
    assert kernel_density_estimation(1.0, {'data_points': samples}, 1.0) == pytest.approx(expected)


@pytest.mark.unit
def test_kde_cost_is_lowest_where_the_samples_are_dense():
    """The whole point of a KDE likelihood: the cost follows the observed sample density rather
    than any assumed shape, so a bimodal sample set scores both modes better than the gap
    between them -- which a single gaussian_MLE cannot do."""
    pytest.importorskip('scipy.stats')
    samples = list(np.concatenate([np.linspace(-1.2, -0.8, 40), np.linspace(0.8, 1.2, 40)]))
    kwargs = {'data_points': samples}

    at_left_mode = kernel_density_estimation(-1.0, kwargs, 1.0)
    at_right_mode = kernel_density_estimation(1.0, kwargs, 1.0)
    in_the_gap = kernel_density_estimation(0.0, kwargs, 1.0)

    assert in_the_gap > at_left_mode and in_the_gap > at_right_mode
    assert at_left_mode == pytest.approx(at_right_mode, rel=0.05), 'the modes are symmetric'


@pytest.mark.unit
def test_kde_cost_honours_the_bandwidth_and_the_weight():
    pytest.importorskip('scipy.stats')
    samples = [1.0, 1.2, 0.9, 1.1, 1.05, 0.95]

    # bandwidth is a cost_kwarg, not part of the ground truth: it tunes the comparison rather
    # than stating the measurements, so it can be swept without editing them (issue #421).
    narrow = kernel_density_estimation(2.0, {'data_points': samples}, 1.0, bandwidth=0.05)
    wide = kernel_density_estimation(2.0, {'data_points': samples}, 1.0, bandwidth=1.0)
    assert narrow > wide, 'a narrower kernel must penalise a far-out point more'

    cost = kernel_density_estimation(1.0, {'data_points': samples}, 1.0)
    assert kernel_density_estimation(1.0, {'data_points': samples}, 3.0) == pytest.approx(3 * cost)
    assert isinstance(cost, float)


@pytest.mark.unit
def test_kde_cost_rejects_unusable_inputs():
    pytest.importorskip('scipy.stats')
    with pytest.raises(ValueError, match='data_points'):
        kernel_density_estimation(1.0, {'samples': [1.0, 2.0]}, 1.0)
    with pytest.raises(ValueError, match='empty'):
        kernel_density_estimation(1.0, {'data_points': []}, 1.0)
    with pytest.raises(ValueError, match='series'):
        kernel_density_estimation(np.array([1.0, 2.0]), {'data_points': [1.0, 2.0]}, 1.0)


# ---------------------------------------------------------------------------
# consistency with the existing MLE cost
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_both_costs_share_gaussian_MLEs_sign_convention():
    """All three are @is_MLE and are summed together into one cost, so they have to agree on
    which direction is 'better' -- otherwise mixing them in one obs_data cancels them out."""
    pytest.importorskip('scipy.stats')
    samples = [1.0, 1.2, 0.9, 1.1, 1.05, 0.95]

    def better_than(cost_at_truth, cost_when_wrong):
        assert cost_when_wrong > cost_at_truth

    better_than(gaussian_MLE(1.0, 1.0, 0.1, 1.0), gaussian_MLE(2.0, 1.0, 0.1, 1.0))
    better_than(kernel_density_estimation(1.05, {'data_points': samples}, 1.0),
                kernel_density_estimation(5.0, {'data_points': samples}, 1.0))
    better_than(poisson_MLE(4.0, {'k': 4}, 1.0), poisson_MLE(20.0, {'k': 4}, 1.0))
