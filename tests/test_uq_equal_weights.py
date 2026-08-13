"""Issue #193: every feature entering the likelihood carries equal weight.

Calibration weights say which features the optimiser should care about most -- a modelling
choice. A posterior is not a modelling choice. Under ``ln L = -cost``, a weight w on a feature
raises its likelihood term to the power w, i.e. claims w independent observations of it, so a
weighted likelihood reports credible intervals the data does not support.

These tests pin the flattening at the seam both classes share, so they cannot drift apart.
"""
import numpy as np
import pytest

from param_id.paramID import OpencorMCMC, OpencorParamID


def _engine(cls, weights):
    """An engine with only the protocol_info the weight lookup reads."""
    obj = cls.__new__(cls)
    obj.protocol_info = {
        'scaled_weight_const_from_exp_sub': [[np.asarray(weights['const'], dtype=float)]],
        'scaled_weight_series_from_exp_sub': [[np.asarray(weights['series'], dtype=float)]],
        'scaled_weight_amp_from_exp_sub': [[np.asarray(weights['amp'], dtype=float)]],
        'scaled_weight_phase_from_exp_sub': [[np.asarray(weights['phase'], dtype=float)]],
        'scaled_weight_prob_dist_from_exp_sub': [[np.asarray(weights['prob_dist'], dtype=float)]],
    }
    return obj


_MIXED = {
    'const': [1.0, 10.0, 0.0, 0.5],
    'series': [2.0, 0.0],
    'amp': [3.0],
    'phase': [0.0],
    'prob_dist': [7.5, 1.0],
}


@pytest.mark.unit
def test_calibration_keeps_the_weights_the_user_set():
    """The base engine must be untouched -- weighting is exactly what a calibration is for."""
    engine = _engine(OpencorParamID, _MIXED)
    const, series, amp, phase, prob_dist = engine._cost_weight_vectors(0, 0)

    assert list(const) == [1.0, 10.0, 0.0, 0.5]
    assert list(series) == [2.0, 0.0]
    assert list(amp) == [3.0]
    assert list(prob_dist) == [7.5, 1.0]


@pytest.mark.unit
def test_uq_flattens_every_non_zero_weight_to_one():
    engine = _engine(OpencorMCMC, _MIXED)
    for vec in engine._cost_weight_vectors(0, 0):
        assert set(np.unique(vec)) <= {0.0, 1.0}, 'a UQ weight may only be 0 or 1'


@pytest.mark.unit
def test_uq_preserves_zero_weights_because_they_exclude_an_observable():
    """A zero does not mean 'unimportant', it means the observable is not part of this
    sub-experiment. Reinstating it would add a feature the user excluded -- and would also
    desynchronise the cached _num_weighted_obs_by_exp_sub denominator, which counts non-zeros."""
    engine = _engine(OpencorMCMC, _MIXED)
    const, series, amp, phase, prob_dist = engine._cost_weight_vectors(0, 0)

    assert list(const) == [1.0, 1.0, 0.0, 1.0]
    assert list(series) == [1.0, 0.0]
    assert list(phase) == [0.0]

    # the non-zero count is what the cost denominator is built from, so it must not move
    base = _engine(OpencorParamID, _MIXED)._cost_weight_vectors(0, 0)
    flat = engine._cost_weight_vectors(0, 0)
    for original, flattened in zip(base, flat):
        assert np.count_nonzero(original) == np.count_nonzero(flattened)


@pytest.mark.unit
def test_uq_warns_once_when_the_weights_actually_changed(capsys):
    """A user who tuned weights for a calibration and then ran UQ on the same obs_data has to
    find out that they no longer apply -- but not once per cost evaluation."""
    engine = _engine(OpencorMCMC, _MIXED)

    engine._cost_weight_vectors(0, 0)
    first = capsys.readouterr().out
    assert '#193' in first and 'weighted 1' in first

    for _ in range(5):
        engine._cost_weight_vectors(0, 0)
    assert capsys.readouterr().out == '', 'the warning must not repeat per evaluation'


@pytest.mark.unit
def test_uq_is_silent_when_the_weights_were_already_uniform(capsys):
    """The common case -- nothing changed, so there is nothing to say."""
    engine = _engine(OpencorMCMC, {
        'const': [1.0, 1.0, 0.0], 'series': [1.0], 'amp': [0.0],
        'phase': [0.0], 'prob_dist': [1.0],
    })
    engine._cost_weight_vectors(0, 0)
    assert capsys.readouterr().out == ''


@pytest.mark.unit
def test_the_flattening_is_the_only_difference_from_the_calibration_path():
    """OpencorMCMC must not reimplement cost_calc: it inherits it and overrides only the weight
    lookup, so a change to the cost machinery cannot apply to calibration and miss UQ."""
    assert 'cost_calc' not in vars(OpencorMCMC), \
        'OpencorMCMC should inherit cost_calc, not override it'
    assert '_cost_weight_vectors' in vars(OpencorMCMC)
    assert OpencorMCMC.cost_calc is OpencorParamID.cost_calc


# ---------------------------------------------------------------------------
# the flattening must not escape into reported costs
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_calibration_weighting_restores_the_users_weights():
    """best_cost is a calibration artifact: plot_param_id and simulate_once re-derive it and
    compare against the saved value, and the calibration optimised the *weighted* cost. Writing
    it on the flat scale makes it a different quantity under the same name -- measured on
    3compartment, 0.0377 saved against 0.1058 recomputed, which tripped simulate_once's
    consistency check and failed CI."""
    engine = _engine(OpencorMCMC, _MIXED)
    engine._flatten_weights = True

    with engine.calibration_weighting():
        const, series, amp, phase, prob_dist = engine._cost_weight_vectors(0, 0)
        assert list(const) == [1.0, 10.0, 0.0, 0.5], 'the user weights must be back'
        assert list(series) == [2.0, 0.0]

    # ...and the flattening resumes afterwards, for the likelihood being sampled
    assert list(engine._cost_weight_vectors(0, 0)[0]) == [1.0, 1.0, 0.0, 1.0]


@pytest.mark.unit
def test_calibration_weighting_restores_the_flag_even_on_an_error():
    engine = _engine(OpencorMCMC, _MIXED)
    engine._flatten_weights = True

    with pytest.raises(RuntimeError):
        with engine.calibration_weighting():
            raise RuntimeError('boom')

    assert engine._flatten_weights is True, 'the flag must not be left off by an exception'


@pytest.mark.unit
def test_the_costs_reported_by_a_uq_run_use_calibration_weighting():
    """Pins the call site, not just the helper. Since #413 a UQ run no longer overwrites
    best_cost; it reports the cost at the posterior median and mean into mcmc_statistics.json
    alongside calibration_best_cost. Those three are read against each other, so they have to be
    the same quantity."""
    import inspect

    source = inspect.getsource(OpencorMCMC.save_mcmc_statistics)
    assert 'with self.calibration_weighting():' in source, (
        'the costs reported alongside calibration_best_cost must be the same quantity as it, '
        'not the flat likelihood weighting')


@pytest.mark.unit
def test_sampling_still_uses_flat_weights_by_default():
    """The default has to stay flattened -- that is #193."""
    engine = _engine(OpencorMCMC, _MIXED)
    engine._flatten_weights = True
    for vec in engine._cost_weight_vectors(0, 0):
        assert set(np.unique(vec)) <= {0.0, 1.0}
