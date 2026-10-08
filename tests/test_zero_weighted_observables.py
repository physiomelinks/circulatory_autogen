"""A weight-0 observable the emulator cannot predict, and does not have to.

``EmulatorTrainer`` refuses non-scalar data_items up front, because the emulator
predicts scalar features and a ``series`` cost would otherwise fail a hundred
simulations later with a shape mismatch. That check was on ``data_type`` alone,
which made it stricter than the thing it protects: a zero-weighted item is
dropped from the cost entirely, so the emulator is never asked for it --
``emulated_feature_labels`` is built from ``const_idx_to_obs_idx`` and excludes
non-constants already.

The case that matters is carrying a recorded trace at weight 0 so it can be drawn
behind the simulated one. Without it an output-vs-time plot has nothing to
compare against but a few horizontal lines reduced from the same recording.
Before this, that forced a second obs_data file whose only purpose was to keep
the emulator from refusing.
"""
import numpy as np
import pytest

from libcuflynx.emulators.emulator_trainer import EmulatorTrainer


def _trainer(data_types, series_weights=(), series_idx=()):
    """Just enough of a trainer to run the check in isolation.

    Built by hand rather than through init_from_dict: constructing a real one
    needs a model, a solver and a parsed obs_data, none of which this is about.
    """
    obs_info = {
        'data_types': list(data_types),
        'weight_series_vec': np.array(series_weights, dtype=float),
        'series_idx_to_obs_idx': list(series_idx),
    }
    pid = type('FakePID', (), {'obs_info': obs_info})()
    return type('FakeTrainer', (), {'pid': pid})()


def _check(trainer):
    EmulatorTrainer._check_observables_are_scalar(trainer)


def test_all_scalar_observables_pass():
    _check(_trainer(['constant'] * 3))


def test_a_zero_weighted_series_does_not_block_training():
    """The point of the change: a trace carried only to be plotted."""
    _check(_trainer(['constant', 'constant', 'series'],
                    series_weights=[0.0], series_idx=[2]))


def test_many_zero_weighted_series_do_not_block_training():
    """One per experiment is what a study emits, so the count grows with it."""
    types = ['constant'] * 4 + ['series'] * 8
    _check(_trainer(types, series_weights=[0.0] * 8, series_idx=list(range(4, 12))))


def test_a_weighted_series_is_still_refused():
    """The check still has a job: this one really is in the cost."""
    with pytest.raises(ValueError, match='scalar data_item features only'):
        _check(_trainer(['constant', 'series'], series_weights=[1.0], series_idx=[1]))


def test_only_the_weighted_one_is_named():
    """A message listing the zero-weighted items too would send the reader to
    remove the very things that are fine."""
    with pytest.raises(ValueError) as excinfo:
        _check(_trainer(['constant', 'series', 'series'],
                        series_weights=[0.0, 1.0], series_idx=[1, 2]))
    message = str(excinfo.value)
    assert 'index(es) [2]' in message


def test_the_message_offers_weight_zero_as_the_fix():
    with pytest.raises(ValueError, match='weight 0'):
        _check(_trainer(['series'], series_weights=[1.0], series_idx=[0]))


def test_a_vector_weight_of_all_zeros_still_counts_as_off():
    """weight may be per-sample; CA treats an all-zero vector as switched off."""
    obs_info = {
        'data_types': ['constant', 'series'],
        'weight_series_vec': np.array([[0.0, 0.0, 0.0]]),
        'series_idx_to_obs_idx': [1],
    }
    pid = type('FakePID', (), {'obs_info': obs_info})()
    _check(type('FakeTrainer', (), {'pid': pid})())


# ── the same rule, at both places that enforce it ──────────────────────────
"""There are two checks, not one: the trainer refuses before simulating, and
paramID refuses at use time when ``use_emulator`` is set. They had the rule
written out separately and drifted -- the trainer learned about zero weights and
the use-time check did not, so a recorded trace carried for plotting trained an
emulator successfully and then failed every run that tried to use it.
"""

from libcuflynx.emulators.emulator_bundle import weighted_non_scalar_obs


def _obs_info(data_types, series_weights=(), series_idx=()):
    return {
        'data_types': list(data_types),
        'weight_series_vec': np.array(series_weights, dtype=float),
        'series_idx_to_obs_idx': list(series_idx),
    }


def test_the_shared_rule_exempts_a_zero_weighted_series():
    assert weighted_non_scalar_obs(
        _obs_info(['constant', 'series'], [0.0], [1])) == {}


def test_the_shared_rule_still_names_a_weighted_series():
    assert weighted_non_scalar_obs(
        _obs_info(['constant', 'series'], [1.0], [1])) == {1: 'series'}


def test_the_use_time_check_agrees_with_the_trainer():
    """The property that matters: whatever one refuses, so does the other."""
    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer

    for weights, idx, types in (
            ([0.0], [1], ['constant', 'series']),
            ([1.0], [1], ['constant', 'series']),
            ([0.0, 1.0], [1, 2], ['constant', 'series', 'series']),
            ([], [], ['constant', 'constant']),
    ):
        info = _obs_info(types, weights, idx)
        shared = weighted_non_scalar_obs(info)

        trainer = type('T', (), {'pid': type('P', (), {'obs_info': info})()})()
        try:
            EmulatorTrainer._check_observables_are_scalar(trainer)
            trainer_refused = False
        except ValueError:
            trainer_refused = True

        assert trainer_refused == bool(shared), (types, weights)


def _sobol_with(info):
    """A SobolSA with only what _configure_emulator reads, and a helper whose bundle accepts."""
    from libcuflynx.sensitivity_analysis.sobolSA import sobol_SA as SobolSA

    calls = []
    bundle = type('B', (), {'check_matches': lambda self, fp: calls.append('matches'),
                            'check_quality': lambda self, r2: calls.append('quality')})()
    helper = type('H', (), {'bundle': bundle,
                            'set_obs_map': lambda self, *a, **k: calls.append('map')})()
    info = dict(info, const_idx_to_obs_idx=[0], operations=[None] * len(info['data_types']))
    # A real instance without __init__, so every method _configure_emulator calls exists.
    sa = SobolSA.__new__(SobolSA)
    sa.__dict__.update({'obs_info': info, 'param_id_info': None, 'protocol_info': None,
                        'model_path': None, 'emulator_settings': {},
                        'include_prediction_items': False})
    return SobolSA, sa, helper, calls


def test_sobol_on_an_emulator_accepts_a_zero_weighted_series(monkeypatch):
    """Sobol had its own copy of the scalar-only rule, without the weight exemption, so an
    obs_data that trained an emulator could not then be analysed on it."""
    import libcuflynx.emulators.emulator_bundle as eb
    monkeypatch.setattr(eb, 'fingerprint', lambda *a: None)
    SobolSA, sa, helper, calls = _sobol_with(_obs_info(['constant', 'series'], [0.0], [1]))
    SobolSA._configure_emulator(sa, helper)
    assert calls == ['matches', 'quality', 'map']


def test_sobol_on_an_emulator_still_refuses_a_weighted_series(monkeypatch):
    import libcuflynx.emulators.emulator_bundle as eb
    monkeypatch.setattr(eb, 'fingerprint', lambda *a: None)
    SobolSA, sa, helper, _ = _sobol_with(_obs_info(['constant', 'series'], [1.0], [1]))
    with pytest.raises(ValueError, match=r'\[1\]'):
        SobolSA._configure_emulator(sa, helper)
