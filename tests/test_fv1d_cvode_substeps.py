"""The FV 1D solver's weighting of the 0D CVODE step attempts within one coupling step
(libcuflynx.solver1d.auxCouplingFunctions.CvodeSubstepWeights).

The generated 0D model reports each CVODE step attempt once, as (end time, step size). An attempt
CVODE rejects is retried from the same start with a smaller step, and the retry must replace it:
counted twice, the weights of cvs_model_with_arm_hybrid's first coupling step summed to 1.29 and
the 1D solver stopped ("Sum of integration weights > 1").
"""
import numpy as np
import pytest

from libcuflynx.solver1d.auxCouplingFunctions import CvodeSubstepWeights

TOL = 1e-11


def _feed(dt, reports, flux_of):
    """Run ``reports`` ((end time, dtLoc) per right-hand-side call) through the weights, adding
    weight * flux_of(call index, end time) to one accumulating boundary flux, as main1D does."""
    w = CvodeSubstepWeights(dt, TOL)
    acc = np.zeros(2)
    for i, (t_end, h) in enumerate(reports):
        weight = w.add(t_end, h, [acc])
        acc[:] = acc + weight * flux_of(i, t_end)
    return w, acc


@pytest.mark.unit
def test_accepted_steps_tile_the_coupling_step():
    dt = 2.0e-4
    steps = [0.1, 0.25, 0.4, 0.25]
    reports, t = [], 0.0
    for f in steps:
        t += f * dt
        reports += [(t, f * dt), (t, 0.0), (t, 0.0)]  # first call of the attempt, then Newton calls
    w, acc = _feed(dt, reports, lambda i, t: np.array([1.0, 2.0]))
    assert w.total() == pytest.approx(1.0)
    assert not w.overshoots()
    np.testing.assert_allclose(acc, [1.0, 2.0])


@pytest.mark.unit
def test_a_failed_attempt_is_replaced_by_its_retry():
    """The sequence of cvs_model_with_arm_hybrid's first coupling step: three accepted steps, an
    attempt to the end of the step (0.889) that fails its error test, its retry, and a last step."""
    dt = 2.0912e-4
    h = [0.0013588, 0.0135880, 0.0960475]
    t = np.cumsum(h) * dt
    start = t[-1]
    retry = 0.5958
    reports = [(t[0], h[0] * dt), (t[1], h[1] * dt), (t[2], h[2] * dt), (t[2], 0.0),
               (start + 0.8890057 * dt, 0.8890057 * dt), (start + 0.8890057 * dt, 0.0),  # rejected
               (start + retry * dt, retry * dt), (start + retry * dt, 0.0),              # its retry
               (dt, dt - (start + retry * dt))]
    # the rejected attempt's flux (call 4) must not reach the sum
    flux_of = lambda i, te: np.array([1.0, -1.0]) * (100.0 if i == 4 else 1.0)
    w, acc = _feed(dt, reports, flux_of)
    assert w.total() == pytest.approx(1.0, abs=1e-12)
    assert not w.overshoots()
    np.testing.assert_allclose(acc, [1.0, -1.0])
    assert w.weights[4] == 0.0  # the rejected attempt


@pytest.mark.unit
def test_repeated_retries_restore_the_flux_before_the_first():
    dt = 1.0
    # a step from 0.5 rejected twice, then accepted with 0.125, then the rest
    reports = [(0.5, 0.5), (1.0, 0.5), (0.75, 0.25), (0.625, 0.125), (1.0, 0.375)]
    w, acc = _feed(dt, reports, lambda i, t: np.array([t, 1.0]))
    assert w.total() == pytest.approx(1.0)
    np.testing.assert_allclose(acc, [0.5 * 0.5 + 0.125 * 0.625 + 0.375 * 1.0, 1.0])


@pytest.mark.unit
def test_an_overshoot_beyond_the_time_tolerance_is_reported():
    dt = 2.0e-4
    w = CvodeSubstepWeights(dt, TOL)
    acc = np.zeros(1)
    w.add(0.6 * dt, 0.6 * dt, [acc])
    w.add(1.0 * dt + 0.5 * TOL, 0.4 * dt + 0.5 * TOL, [acc])  # within the 0D model's time rounding
    assert not w.overshoots()
    w.add(1.3 * dt, 0.3 * dt, [acc])
    assert w.overshoots()
