"""``gaussian_MLE_robust`` and ``poisson_MLE``'s background rate.

Both exist for observables with a discontinuity in them: a maximum membrane voltage is the
spike peak when the cell fires and the subthreshold maximum when it does not, so the model
produces one of two values ~110 mV apart and nothing between. Under a plain Gaussian that gap
is 25-30 sigma and one wrong branch costs more than every other observable put together --
which is arithmetic on a category error, because a branch flip is model discrepancy and sigma
is measurement noise.

The properties asserted here are the ones that make the mixture usable rather than merely
smaller: a *stated* ceiling, a crossover in a known place, exact reduction to the Gaussian at
eps = 0, and no overflow anywhere in between.
"""
import math

import numpy as np
import pytest

from libcuflynx.funcs.cost_funcs_user import (
    cost_func_metadata,
    gaussian_MLE,
    gaussian_MLE_robust,
    get_cost_funcs_dict_for_mode,
    poisson_MLE,
)
from libcuflynx.param_id.cost_kwargs import call_cost_func, check_cost_kwargs

EPS = 0.04
WIDTH = 170.0
STD = 6.0
TRUTH = 20.0


def _cost(z, **over):
    kwargs = dict(p_outlier=EPS, outlier_width=WIDTH)
    kwargs.update(over)
    std = kwargs.pop("std", STD)
    return gaussian_MLE_robust(TRUTH + z * std, TRUTH, std, 1.0, **kwargs)


def _crossover_z(eps=EPS, width=WIDTH, std=STD):
    """Where the outlier component overtakes the Gaussian one."""
    return math.sqrt(2.0 * math.log((1.0 - eps) * width / (eps * std * math.sqrt(2.0 * math.pi))))


@pytest.mark.unit
def test_the_cost_is_capped_at_log_width_over_epsilon():
    """However wrong the model is, one observable can never cost more than log(W/eps).

    This is the whole point: the ceiling is a number the modeller chose, not a number the
    scale of the discontinuity chose for them.
    """
    cap = math.log(WIDTH / EPS)
    assert _cost(6.0) == pytest.approx(cap, rel=1e-6)
    assert _cost(50.0) == pytest.approx(cap, rel=1e-12)
    assert _cost(1e4) == pytest.approx(cap, rel=1e-12)
    assert _cost(-1e4) == pytest.approx(cap, rel=1e-12)
    assert _cost(0.0) < cap


@pytest.mark.unit
def test_at_the_crossover_the_cost_is_exactly_the_cap_minus_log_two():
    """The two components are equal there, so their sum is twice either one.

    Pinning the crossover matters because it, not the cap, decides which observables still
    inform the fit: anything whose *best achievable* error is beyond it is written off.
    """
    for eps, width, std in ((0.04, 170.0, 6.0), (0.05, 170.0, 4.0), (0.01, 40.0, 1.5)):
        z_c = _crossover_z(eps, width, std)
        cap = math.log(width / eps)
        got = _cost(z_c, p_outlier=eps, outlier_width=width, std=std)
        assert got == pytest.approx(cap - math.log(2.0), rel=1e-9)


@pytest.mark.unit
def test_the_crossover_barely_moves_with_epsilon_but_scales_with_std():
    """eps and the width enter through a logarithm; std multiplies.

    So std is the dial that decides what is fitted and what is capped, and eps only sets how
    much a capped observable costs. Getting this backwards is the easy mistake.
    """
    base = _crossover_z(0.05, WIDTH, STD)
    # a 25% change in eps -- the size of decision a modeller actually makes here
    assert _crossover_z(0.04, WIDTH, STD) == pytest.approx(base, abs=0.1)
    # and even a tenfold change moves it by less than one sigma, in either direction
    assert abs(_crossover_z(0.005, WIDTH, STD) - base) < 1.0
    assert abs(_crossover_z(0.5, WIDTH, STD) - base) < 1.6
    # in the observable's own units, though, it tracks std almost exactly
    assert _crossover_z(EPS, WIDTH, 12.0) * 12.0 > 1.8 * _crossover_z(EPS, WIDTH, 6.0) * 6.0


@pytest.mark.unit
def test_zero_p_outlier_is_the_normalised_gaussian():
    """The eps -> 0 limit, and the documented offset from gaussian_MLE.

    A mixture cannot drop the normalising constant -- the weight each component gets depends on
    the ratio of their densities -- so this cost keeps it and gaussian_MLE does not.
    """
    offset = math.log(STD * math.sqrt(2.0 * math.pi))
    for z in (0.0, 0.5, 2.0, 7.5):
        plain = gaussian_MLE(TRUTH + z * STD, TRUTH, STD, 1.0)
        assert _cost(z, p_outlier=0.0) == pytest.approx(plain + offset, rel=1e-12)


@pytest.mark.unit
def test_it_is_monotone_in_the_error_and_finite_everywhere():
    zs = np.concatenate([np.linspace(0.0, 12.0, 400), [1e3, 1e6]])
    costs = np.array([_cost(z) for z in zs])
    assert np.all(np.isfinite(costs))
    assert np.all(np.diff(costs) >= -1e-12)
    assert costs[0] == pytest.approx(_cost(-0.0))
    # symmetric in the sign of the error
    assert _cost(-2.7) == pytest.approx(_cost(2.7), rel=1e-12)


@pytest.mark.unit
def test_a_tiny_p_outlier_does_not_overflow():
    """gain = (1-eps)W/(eps*std*sqrt(2pi)) grows as eps shrinks, but only ever multiplies
    exp(-z^2/2) <= 1, so nothing exponentiates a large number."""
    for eps in (1e-3, 1e-6, 1e-12):
        assert np.isfinite(_cost(0.0, p_outlier=eps))
        assert _cost(40.0, p_outlier=eps) == pytest.approx(math.log(WIDTH / eps), rel=1e-9)


@pytest.mark.unit
def test_weight_and_series_behave_like_gaussian_MLE():
    assert _cost(1.3, ) * 3.0 == pytest.approx(
        gaussian_MLE_robust(TRUTH + 1.3 * STD, TRUTH, STD, 3.0,
                            p_outlier=EPS, outlier_width=WIDTH))
    series = np.array([TRUTH, TRUTH + STD, TRUTH + 40.0 * STD])
    got = gaussian_MLE_robust(series, TRUTH, STD, 1.0, p_outlier=EPS, outlier_width=WIDTH)
    expected = np.mean([_cost(0.0), _cost(1.0), math.log(WIDTH / EPS)])
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
def test_an_impossible_p_outlier_is_refused(bad):
    with pytest.raises(ValueError, match="p_outlier"):
        _cost(0.0, p_outlier=bad)


@pytest.mark.unit
def test_a_non_positive_outlier_width_is_refused():
    with pytest.raises(ValueError, match="outlier_width"):
        _cost(0.0, outlier_width=0.0)


@pytest.mark.unit
def test_it_is_registered_and_reaches_the_cost_call_path():
    """Registration is automatic, and cost_kwargs is the route the knobs travel by."""
    registry = get_cost_funcs_dict_for_mode("numpy")
    assert "gaussian_MLE_robust" in registry
    meta = cost_func_metadata()["gaussian_MLE_robust"]
    assert meta["is_MLE"] and meta["differentiable"] and not meta["is_combiner"]

    got = call_cost_func(registry["gaussian_MLE_robust"], TRUTH + 200.0, TRUTH,
                         std=STD, weight=1.0,
                         cost_kwargs={"p_outlier": EPS, "outlier_width": WIDTH})
    assert got == pytest.approx(math.log(WIDTH / EPS), rel=1e-6)

    # a mistyped knob is caught at setup, not after the first forward solve
    with pytest.raises(ValueError):
        check_cost_kwargs({"p_outlyer": EPS}, registry["gaussian_MLE_robust"],
                          "gaussian_MLE_robust", "V_max")


@pytest.mark.unit
def test_poisson_background_rate_defaults_to_the_previous_behaviour():
    for output, k in ((0.0, 4), (3.0, 3), (7.5, 2), (-1.0, 1)):
        lam = np.clip(output, 1e-12, None)
        assert poisson_MLE(output, {"k": k}, 1.0) == pytest.approx(lam - k * np.log(lam))


@pytest.mark.unit
def test_poisson_background_rate_bounds_the_silent_model():
    """A model that never fires is what the clip was silently pricing; now the price is stated."""
    k = 4
    clipped = poisson_MLE(0.0, {"k": k}, 1.0)
    assert clipped == pytest.approx(k * math.log(1e12), rel=1e-6)
    for lam0 in (0.001, 0.01, 0.1):
        got = poisson_MLE(0.0, {"k": k}, 1.0, background_rate=lam0)
        assert got == pytest.approx(lam0 - k * math.log(lam0), rel=1e-12)
        assert got < clipped
    # and the optimum moves to k - background_rate, which is a bias and not a rounding
    lam0 = 0.01
    at_shifted = poisson_MLE(k - lam0, {"k": k}, 1.0, background_rate=lam0)
    assert at_shifted <= poisson_MLE(float(k), {"k": k}, 1.0, background_rate=lam0)


@pytest.mark.unit
def test_a_negative_background_rate_is_refused():
    with pytest.raises(ValueError, match="background_rate"):
        poisson_MLE(1.0, {"k": 1}, 1.0, background_rate=-0.1)


class _Inner:
    """The surface ``ensure_mle_cost_type_for_bayesian_inner`` touches."""

    def __init__(self, cost_types):
        self.obs_info = {"num_obs": len(cost_types), "cost_type": list(cost_types)}
        self.cost_type = self.obs_info["cost_type"]
        self.cost_funcs_dict = get_cost_funcs_dict_for_mode("numpy")


@pytest.mark.unit
def test_bayesian_setup_keeps_an_observable_that_already_names_an_mle_cost():
    """The MCMC path must not undo the obs_data's per-item choices.

    It used to replace the whole vector with one name, which turned a ``poisson_MLE`` count --
    scored against ``prob_dist_params`` and deliberately carrying no ``value`` -- into a
    ``gaussian_MLE`` against a ground truth of nan, and stripped the outlier component off
    every ``gaussian_MLE_robust`` item. Both are MLE costs and neither needed touching.
    """
    from libcuflynx.param_id.paramID import ensure_mle_cost_type_for_bayesian_inner

    inner = _Inner(["gaussian_MLE", "poisson_MLE", "gaussian_MLE_robust", "poisson_MLE"])
    ensure_mle_cost_type_for_bayesian_inner(
        inner, {"UQ_options": {"cost_type": "gaussian_MLE"}})
    assert inner.cost_type == ["gaussian_MLE", "poisson_MLE",
                               "gaussian_MLE_robust", "poisson_MLE"]
    assert inner.obs_info["cost_type"] is inner.cost_type


@pytest.mark.unit
def test_bayesian_setup_still_replaces_a_cost_that_is_not_an_mle():
    """The rule it exists to enforce is unchanged: ln L = -cost needs an MLE everywhere."""
    from libcuflynx.param_id.paramID import ensure_mle_cost_type_for_bayesian_inner

    inner = _Inner(["MSE", "AE", "poisson_MLE"])
    ensure_mle_cost_type_for_bayesian_inner(
        inner, {"UQ_options": {"cost_type": "gaussian_MLE"}})
    assert inner.cost_type == ["gaussian_MLE", "gaussian_MLE", "poisson_MLE"]

    # and with no usable option it falls back to gaussian_MLE, as before
    inner = _Inner(["MSE", None])
    ensure_mle_cost_type_for_bayesian_inner(inner, {})
    assert inner.cost_type == ["gaussian_MLE", "gaussian_MLE"]


# ------------------------------------------------- Conway-Maxwell-Poisson (under-dispersed)

class TestComPoisson:
    """``com_poisson_MLE``: a count likelihood whose tightness is a parameter.

    A Poisson's variance is its mean, so at an observed count of 0 or 1 it barely separates a
    model that fires once from one that fires three times. COM-Poisson adds ``nu``: 1 is
    Poisson, above 1 is under-dispersed. This is the proper-model alternative to raising a
    data_item's ``weight``, which tempers the likelihood by a power and is not a distribution.
    """

    def test_nu_one_is_exactly_poisson(self):
        from libcuflynx.funcs.cost_funcs_user import com_poisson_MLE, poisson_MLE
        from scipy.special import gammaln
        for k in (0, 1, 4, 8):
            for mean in (0.5, 1.0, 3.0, 8.0):
                com = com_poisson_MLE(mean, {'k': k}, 1.0, nu=1.0)
                # poisson_MLE drops log(k!); com_poisson_MLE keeps it because it carries nu
                assert com == pytest.approx(poisson_MLE(mean, {'k': k}, 1.0) + gammaln(k + 1),
                                            abs=1e-8)

    def test_output_is_the_expected_count_for_every_nu(self):
        """The optimum stays at E[Y] == k, so nu is tightness alone and does not move the fit."""
        from libcuflynx.funcs.cost_funcs_user import com_poisson_MLE
        for nu in (1.0, 2.0, 3.0, 5.0):
            for k in (1, 4, 8):
                at_k = com_poisson_MLE(float(k), {'k': k}, 1.0, nu=nu)
                for off in (0.5, -0.5, 2.0, -0.9):
                    if k + off <= 0:
                        continue
                    assert com_poisson_MLE(k + off, {'k': k}, 1.0, nu=nu) > at_k

    def test_larger_nu_is_under_dispersed(self):
        from libcuflynx.funcs.cost_funcs_user import (
            _com_poisson_log_lam_for_mean, _com_poisson_log_terms)
        import numpy as np

        def variance(mean, nu):
            terms, j = _com_poisson_log_terms(_com_poisson_log_lam_for_mean(mean, nu), nu)
            w = np.exp(terms - terms.max()); w /= w.sum()
            m = (w * j).sum()
            return float((w * (j - m) ** 2).sum())

        for mean in (1.0, 3.0, 8.0):
            assert variance(mean, 1.0) == pytest.approx(mean, rel=1e-6)   # Poisson: var == mean
            assert variance(mean, 2.0) < variance(mean, 1.0)
            assert variance(mean, 5.0) < variance(mean, 2.0)

    def test_larger_nu_punishes_over_firing_harder(self):
        """The whole point: at k=1 a model firing three times should cost more as nu rises."""
        from libcuflynx.funcs.cost_funcs_user import com_poisson_MLE
        excess = [com_poisson_MLE(3.0, {'k': 1}, 1.0, nu=nu)
                  - com_poisson_MLE(1.0, {'k': 1}, 1.0, nu=nu) for nu in (1.0, 2.0, 3.0, 5.0)]
        assert excess == sorted(excess)
        assert excess[2] > 2 * excess[0]        # nu=3 more than doubles the Poisson penalty

    def test_the_silence_penalty_is_left_alone(self):
        """nu must sharpen over-firing without inflating 'the cell fired and the model did not'.

        After ``gaussian_MLE_robust`` caps the jump observables the counts are the only thing
        pushing the model towards firing, so that direction must not move much with nu.
        """
        from libcuflynx.funcs.cost_funcs_user import com_poisson_MLE
        silent = [com_poisson_MLE(0.0, {'k': 1}, 1.0, nu=nu, background_rate=0.01)
                  - com_poisson_MLE(1.0, {'k': 1}, 1.0, nu=nu) for nu in (1.0, 3.0)]
        assert silent[1] / silent[0] < 1.2

    def test_it_is_registered_and_rejects_bad_arguments(self):
        from libcuflynx.funcs.cost_funcs_user import com_poisson_MLE, cost_func_metadata
        meta = cost_func_metadata()
        assert meta['com_poisson_MLE']['is_MLE'] is True
        with pytest.raises(ValueError):
            com_poisson_MLE(1.0, {'k': 1}, 1.0, nu=0.0)
        with pytest.raises(ValueError):
            com_poisson_MLE(1.0, {'k': 1}, 1.0, background_rate=-1.0)
        with pytest.raises(ValueError):
            com_poisson_MLE(1.0, {'no_k': 1}, 1.0)
