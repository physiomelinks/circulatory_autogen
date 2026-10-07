"""``mean_AP_threshold``: the voltage at which an action potential's upstroke takes off.

The synthetic upstroke is an exponential, V = Vb + a exp(t/tau), whose slope is (V - Vb)/tau,
so the slope reaches the criterion at exactly V = Vb + criterion * tau -- a threshold known in
closed form, which the operation must recover.
"""
import numpy as np
import pytest

from libcuflynx.funcs.operation_funcs_user import mean_AP_threshold as threshold
from libcuflynx.funcs.cost_funcs_user import gaussian_MLE_robust

pytestmark = pytest.mark.unit

DT = 1e-5            # s
REST = -65.0         # mV
CRITERION = 10e3     # mV/s, i.e. 10 mV/ms


def _spike(vb, tau):
    """An exponential upstroke from just above rest to +30 mV, then a linear fall back to rest."""
    t = np.arange(0, 20 * tau, DT)
    up = vb + 0.5 * np.exp(t / tau)
    up = up[up < 30.0]
    down = np.linspace(30.0, REST, int(2e-3 / DT))
    return np.concatenate([np.linspace(REST, up[0], int(5e-3 / DT)), up, down])


def trace(spikes, total=0.4):
    """``spikes``: (start time s, vb mV, tau s). Flat at REST elsewhere."""
    V = np.full(int(total / DT), REST)
    for start, vb, tau in spikes:
        s = _spike(vb, tau)
        i = int(start / DT)
        V[i:i + s.size] = s[:V.size - i]
    return np.arange(V.size) * DT, V


def expected(vb, tau):
    return vb + CRITERION * tau


def test_it_recovers_a_known_threshold():
    t, V = trace([(0.1, -60.0, 1e-3)])
    assert threshold(t, V, spike_min_thresh=-10) == pytest.approx(expected(-60.0, 1e-3), abs=0.2)


def test_it_averages_over_the_spikes():
    t, V = trace([(0.05, -60.0, 1e-3), (0.2, -56.0, 1e-3)])
    mean = (expected(-60.0, 1e-3) + expected(-56.0, 1e-3)) / 2
    assert threshold(t, V, spike_min_thresh=-10) == pytest.approx(mean, abs=0.2)


def test_the_criterion_moves_the_threshold_as_the_slope_says():
    t, V = trace([(0.1, -60.0, 1e-3)])
    assert threshold(t, V, spike_min_thresh=-10, dV_dt_thresh=7.5e3) == \
        pytest.approx(-60.0 + 7.5e3 * 1e-3, abs=0.2)


def test_a_window_keeps_only_its_own_spikes():
    t, V = trace([(0.05, -60.0, 1e-3), (0.3, -50.0, 1e-3)])
    assert threshold(t, V, spike_min_thresh=-10, start_frac=0.5) == \
        pytest.approx(expected(-50.0, 1e-3), abs=0.2)
    assert threshold(t, V, spike_min_thresh=-10, end_frac=0.5) == \
        pytest.approx(expected(-60.0, 1e-3), abs=0.2)


def test_the_default_window_is_the_whole_trace():
    t, V = trace([(0.05, -60.0, 1e-3), (0.3, -50.0, 1e-3)])
    assert threshold(t, V, spike_min_thresh=-10) == \
        threshold(t, V, spike_min_thresh=-10, start_frac=0.0, end_frac=1.0)


def test_a_silent_window_returns_its_mean_voltage():
    t, V = trace([(0.3, -60.0, 1e-3)])
    got = threshold(t, V, spike_min_thresh=-10, end_frac=0.5)
    assert got == pytest.approx(REST)


def test_peaks_without_an_upstroke_return_the_mean_not_a_sentinel():
    """A slow hump above the peak criterion has no threshold (depolarisation block looks like
    this). It used to return 9999, which swamped every other term of a cost."""
    t = np.arange(0, 0.4, DT)
    V = REST + 70.0 * np.exp(-((t - 0.2) / 0.05) ** 2)      # peaks at +5 mV, slope < 3 mV/ms
    got = threshold(t, V, spike_min_thresh=-10)
    assert got == pytest.approx(np.mean(V))
    assert got < 0


def test_no_spike_value_replaces_the_mean_when_silent():
    t, V = trace([(0.3, -60.0, 1e-3)])
    assert threshold(t, V, spike_min_thresh=-10, end_frac=0.5, no_spike_value=100.0) == 100.0


def test_no_spike_value_leaves_a_firing_trace_alone():
    t, V = trace([(0.1, -60.0, 1e-3)])
    assert threshold(t, V, spike_min_thresh=-10, no_spike_value=100.0) == \
        threshold(t, V, spike_min_thresh=-10)


def test_a_silent_model_costs_the_robust_cap_and_firing_never_costs_more():
    """The point of ``no_spike_value``: under the mixture a silent model sits on the cap,
    log(W/eps), and a firing one with any threshold error costs at most that."""
    std, eps, width, target = 6.0, 0.04, 170.0, -50.0
    cap = np.log(width / eps)
    silent = gaussian_MLE_robust(np.array([100.0]), np.array([target]), np.array([std]),
                                 np.array([1.0]), p_outlier=eps, outlier_width=width)
    assert float(np.sum(silent)) == pytest.approx(cap)
    for error in (0.0, 5.0, 15.0, 30.0, 60.0):
        firing = gaussian_MLE_robust(np.array([target + error]), np.array([target]),
                                     np.array([std]), np.array([1.0]),
                                     p_outlier=eps, outlier_width=width)
        assert float(np.sum(firing)) <= cap + 1e-12


def test_series_output_returns_the_trace():
    t, V = trace([(0.1, -60.0, 1e-3)])
    assert threshold(t, V, series_output=True) is V
