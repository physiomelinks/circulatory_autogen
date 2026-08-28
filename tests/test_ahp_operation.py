"""``min_between_first_two_spikes``: the afterhyperpolarisation trough as an observable.

The point of locating the spikes in each trace, rather than taking a window fixed from the
recording, is that the model spikes at its own times. On one SN_full experiment the model had
a spike inside the recording's window in only 10 of 40 posterior draws, so the "trough" was
often a rising subthreshold ramp -- a spike-timing measurement wearing a trough's name.
"""
import numpy as np
import pytest

from libcuflynx.funcs.operation_funcs_user import min_between_first_two_spikes as ahp

pytestmark = pytest.mark.unit


def trace(spike_times, trough=-70.0, rest=-60.0, n=2001):
    t = np.linspace(0, 1, n)
    V = np.full_like(t, rest)
    for a, b in zip(spike_times[:-1], spike_times[1:]):
        V[(t > a + .02) & (t < b - .02)] = trough
    for s in spike_times:
        V[(t > s - .005) & (t < s + .005)] = 30.0
    return t, V


def test_it_returns_the_trough_between_the_first_two_spikes():
    t, V = trace([0.2, 0.6])
    assert ahp(t, V, spike_min_thresh=-10) == pytest.approx(-70.0)


def test_a_deeper_trough_later_is_ignored():
    """Only the *first* interspike interval counts, however the trace evolves after it."""
    t, V = trace([0.2, 0.5, 0.8])
    V[(t > 0.55) & (t < 0.75)] = -95.0          # much deeper, but after the second spike
    assert ahp(t, V, spike_min_thresh=-10) == pytest.approx(-70.0)


def test_timing_does_not_change_the_answer():
    """The whole reason for locating spikes per-trace: shift them and the depth is unchanged."""
    a = ahp(*trace([0.10, 0.30]), spike_min_thresh=-10)
    b = ahp(*trace([0.55, 0.85]), spike_min_thresh=-10)
    assert a == pytest.approx(b)


@pytest.mark.parametrize('spikes', [[], [0.3]])
def test_fewer_than_two_spikes_returns_the_maximum(spikes):
    """A sentinel far from any trough, so gaussian_MLE_robust caps it rather than the
    subthreshold minimum being scored against a real AHP."""
    t, V = trace(spikes) if spikes else (np.linspace(0, 1, 501), np.full(501, -60.0))
    got = ahp(t, V, spike_min_thresh=-10)
    assert got == pytest.approx(np.max(V))
    assert got > -60.0 or not spikes


def test_series_output_passes_the_trace_through():
    t, V = trace([0.2, 0.6])
    assert np.array_equal(ahp(t, V, series_output=True, spike_min_thresh=-10), V)


def test_it_is_registered_as_an_operation():
    import libcuflynx.funcs.operation_funcs_user as ofu
    assert callable(getattr(ofu, 'min_between_first_two_spikes'))


# ------------------------------------------- the difference form, AHP minus steady state

from libcuflynx.funcs.operation_funcs_user import AHP_minus_steady_state_min as ahp_diff


def two_trough_trace(first_trough, late_trough, spikes=(0.10, 0.30), n=2001):
    """Spikes early, a first trough between them, then a different steady-state trough."""
    t = np.linspace(0, 1, n)
    V = np.full_like(t, -60.0)
    if len(spikes) >= 2:
        V[(t > spikes[0] + .02) & (t < spikes[1] - .02)] = first_trough
    V[t > 0.55] = late_trough
    for s in spikes:
        V[(t > s - .005) & (t < s + .005)] = 30.0
    return t, V


def test_it_measures_accommodation_of_the_trough():
    t, V = two_trough_trace(first_trough=-70.0, late_trough=-65.0)
    assert ahp_diff(t, V, spike_min_thresh=-10) == pytest.approx(-5.0)


def test_no_accommodation_gives_zero():
    t, V = two_trough_trace(first_trough=-65.0, late_trough=-65.0)
    assert ahp_diff(t, V, spike_min_thresh=-10) == pytest.approx(0.0)


def test_a_common_offset_cancels():
    """The point of a difference: anything shared by both minima drops out."""
    t, V = two_trough_trace(-70.0, -65.0)
    a = ahp_diff(t, V, spike_min_thresh=-10)
    b = ahp_diff(t, V - 12.0, spike_min_thresh=-10)      # whole trace shifted
    assert a == pytest.approx(b)


def test_a_silent_trace_returns_zero_not_a_far_sentinel():
    """A silent model must not be charged for its silence a third time.

    ``min_between_first_two_spikes`` deliberately returns the trace maximum when there is no
    interspike interval, because an absolute trough needs an obviously-wrong sentinel. A
    difference wants the opposite: the counts and the jump observables already assert that the
    model did not fire, so this one should stay quiet. Measured on ox1, a far sentinel would
    add ~24 raw nats to the silence penalty against ~0.6 with zero.
    """
    t, V = two_trough_trace(-70.0, -65.0, spikes=())
    assert ahp_diff(t, V, spike_min_thresh=-10) == pytest.approx(0.0)
    t, V = two_trough_trace(-70.0, -65.0, spikes=(0.10,))       # a single spike
    assert ahp_diff(t, V, spike_min_thresh=-10) == pytest.approx(0.0)


def test_the_silent_sentinel_costs_far_less_than_a_far_one():
    # Through the registry, not the bare module function: the funcs dispatch on a module-level
    # ``mb`` that each register_* hook rebinds, so a bare call inherits whichever backend the
    # last-built registry left behind (casadi's ``numel`` needs a casadi type, #315).
    from libcuflynx.funcs.cost_funcs_user import get_cost_funcs_dict_for_mode
    gaussian_MLE_robust = get_cost_funcs_dict_for_mode("numpy")["gaussian_MLE_robust"]
    kw = dict(p_outlier=0.04, outlier_width=60.0)
    observed, sigma = -1.6, 1.5
    best = gaussian_MLE_robust(observed, observed, sigma, 1.0, **kw)
    quiet = gaussian_MLE_robust(0.0, observed, sigma, 1.0, **kw)
    far = gaussian_MLE_robust(5.0, observed, sigma, 1.0, **kw)
    assert quiet - best < 0.15 * (far - best)


def test_series_output_passes_through():
    t, V = two_trough_trace(-70.0, -65.0)
    assert np.array_equal(ahp_diff(t, V, series_output=True, spike_min_thresh=-10), V)


# ------------------------------------------------------ V_plateau, the interspike plateau

from libcuflynx.funcs.operation_funcs_user import V_plateau, steady_state_min


def firing_trace(plateau=-55.0, ahp=-75.0, spikes=(0.2, 0.5, 0.8), n=4001):
    """A trace with a distinct AHP trough and a distinct interspike plateau."""
    t = np.linspace(0, 1, n)
    V = np.full_like(t, plateau)
    for s in spikes:
        up = (t > s - .004) & (t < s + .004)
        V[up] = np.linspace(plateau, 30.0, up.sum())
        dn = (t > s + .004) & (t < s + .02)
        V[dn] = np.linspace(30.0, ahp, dn.sum())
        rec = (t > s + .02) & (t < s + .06)
        V[rec] = np.linspace(ahp, plateau, rec.sum())
    return t, V


def test_it_finds_the_plateau_not_the_trough():
    """The whole point: steady_state_min returns the AHP on a firing trace, this does not."""
    t, V = firing_trace(plateau=-55.0, ahp=-75.0)
    assert V_plateau(t, V, spike_min_thresh=-10) == pytest.approx(-55.0, abs=1.5)
    assert steady_state_min(V) == pytest.approx(-75.0, abs=0.5)


def test_it_tracks_the_plateau_when_the_trough_is_unchanged():
    """Move the plateau, hold the AHP: the plateau observable must follow the plateau."""
    a = V_plateau(*firing_trace(plateau=-55.0, ahp=-75.0), spike_min_thresh=-10)
    b = V_plateau(*firing_trace(plateau=-45.0, ahp=-75.0), spike_min_thresh=-10)
    assert b - a == pytest.approx(10.0, abs=1.5)


def test_it_ignores_the_trough_depth():
    """Move the AHP, hold the plateau: the plateau observable must barely move."""
    a = V_plateau(*firing_trace(plateau=-55.0, ahp=-75.0), spike_min_thresh=-10)
    b = V_plateau(*firing_trace(plateau=-55.0, ahp=-90.0), spike_min_thresh=-10)
    assert abs(b - a) < 2.0


def test_a_silent_trace_falls_back_to_the_second_half_mean():
    t = np.linspace(0, 1, 1001)
    V = np.full_like(t, -62.0)
    assert V_plateau(t, V, spike_min_thresh=-10) == pytest.approx(-62.0)


def test_a_single_spike_still_gives_a_plateau():
    """One AP has no next threshold, so the window runs to the end of the trace."""
    t, V = firing_trace(plateau=-55.0, ahp=-75.0, spikes=(0.2,))
    assert V_plateau(t, V, spike_min_thresh=-10) == pytest.approx(-55.0, abs=1.5)


def test_series_output_passes_through():
    t, V = firing_trace()
    assert np.array_equal(V_plateau(t, V, series_output=True, spike_min_thresh=-10), V)
