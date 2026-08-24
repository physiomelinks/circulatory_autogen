"""Two-phase emulation: classify which side of the cliff, then regress.

A regressor is the wrong shape for an observable pinned at one value over most of
its range. A spike count is exactly zero below rheobase and jumps to one spike per
window above it, with nothing in between -- so a smooth fit splits the difference,
predicting fractions of a spike across the floor and undershooting above it.

These use a synthetic cliff rather than a real study: ``y = 0`` below a threshold
in the first input and rising above it, which is the shape without the cost.
"""
import numpy as np
import pytest

from libcuflynx.emulators import internal_emulators as ie


# --- naming ---------------------------------------------------------------------

def test_a_two_phase_name_is_recognised_and_unwrapped():
    assert ie.is_two_phase('two_phase_MLP')
    assert not ie.is_two_phase('MLP')
    assert ie.base_emulator_name('two_phase_MLP') == 'MLP'
    assert ie.base_emulator_name('MLP') == 'MLP'


def test_wrapping_is_idempotent():
    """So a caller that already has the prefix does not get it twice."""
    assert ie.two_phase_name('MLP') == 'two_phase_MLP'
    assert ie.two_phase_name('two_phase_MLP') == 'two_phase_MLP'


def test_a_variant_is_offered_for_every_base_emulator():
    assert ie.two_phase_model_names(['MLP', 'RandomForest']) == [
        'two_phase_MLP', 'two_phase_RandomForest']


# --- finding the floor ----------------------------------------------------------

def test_a_feature_pinned_at_one_value_is_found():
    y = np.column_stack([np.r_[np.zeros(80), np.linspace(4, 20, 20)],
                         np.linspace(-70, 30, 100)])
    has_floor, floor_value = ie.floor_mask(y)

    assert list(has_floor) == [True, False]
    assert floor_value[0] == 0.0


def test_the_floor_need_not_be_zero():
    """Nothing about spiking is hardcoded -- it is whichever value repeats."""
    y = np.column_stack([np.r_[np.full(60, -85.0), np.linspace(-60, 10, 40)]])
    has_floor, floor_value = ie.floor_mask(y)

    assert has_floor[0]
    assert floor_value[0] == -85.0


def test_a_continuous_feature_has_no_floor():
    y = np.linspace(0, 1, 200).reshape(-1, 1)
    assert not ie.floor_mask(y)[0][0]


def test_the_share_required_is_adjustable():
    y = np.r_[np.zeros(30), np.linspace(1, 9, 70)].reshape(-1, 1)
    assert ie.floor_mask(y, floor_share=0.25)[0][0]
    assert not ie.floor_mask(y, floor_share=0.5)[0][0]


# --- predicting -----------------------------------------------------------------

class _Ramp:
    """Stands in for a fitted regressor: predicts the second column of x."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return np.column_stack([x[:, 0] * self.scale, x[:, 0] * 0 + 5.0])


class _Gate:
    """Stands in for the classifier: active where the first input exceeds 0.5."""

    def predict(self, x):
        return (np.asarray(x, dtype=float)[:, 0] > 0.5).reshape(-1, 1)


def test_the_floor_is_returned_exactly_below_the_boundary():
    """Not "close to zero" -- a spike count of 0.3 is not a thing the model does."""
    model = ie.TwoPhaseEmulator(_Ramp(), _Ramp(scale=10.0), _Gate(),
                                has_floor=[True, False], floor_value=[0.0, 0.0],
                                base_name='MLP')
    out = model.predict(np.array([[0.1, 0.0], [0.2, 0.0]]))

    assert np.all(out[:, 0] == 0.0)


def test_the_active_regressor_supplies_the_value_above_the_boundary():
    model = ie.TwoPhaseEmulator(_Ramp(), _Ramp(scale=10.0), _Gate(),
                                has_floor=[True, False], floor_value=[0.0, 0.0],
                                base_name='MLP')
    out = model.predict(np.array([[0.8, 0.0]]))

    assert out[0, 0] == pytest.approx(8.0)   # the active model, not the base


def test_a_feature_without_a_floor_comes_from_the_base_regressor():
    """The gate applies only to the features that have a cliff."""
    model = ie.TwoPhaseEmulator(_Ramp(), _Ramp(scale=10.0), _Gate(),
                                has_floor=[True, False], floor_value=[0.0, 0.0],
                                base_name='MLP')
    out = model.predict(np.array([[0.1, 0.0], [0.8, 0.0]]))

    assert np.all(out[:, 1] == 5.0)


def test_no_classifier_means_the_base_model_answers_everything():
    """What a design with no floored feature produces -- recorded as two-phase so
    the bundle says what was asked for, rather than silently degrading."""
    model = ie.TwoPhaseEmulator(_Ramp(), None, None, has_floor=[False, False],
                                floor_value=[0.0, 0.0], base_name='MLP')
    out = model.predict(np.array([[0.9, 0.0]]))

    assert out[0, 0] == pytest.approx(0.9)


def test_it_reports_the_two_phase_name():
    model = ie.TwoPhaseEmulator(_Ramp(), None, None, [False], [0.0], 'RandomForest')
    assert model.model_name == 'two_phase_RandomForest'


def test_a_single_row_is_accepted():
    model = ie.TwoPhaseEmulator(_Ramp(), None, None, [False, False], [0.0, 0.0], 'MLP')
    assert model.predict(np.array([0.4, 0.0])).shape == (1, 2)


# --- the classifier -------------------------------------------------------------

def test_a_column_that_never_changes_gets_a_constant_not_a_fit():
    """A single-class column has no boundary to learn, and sklearn raises on one."""
    x = np.random.default_rng(0).uniform(size=(30, 2))
    classifier = ie._fit_classifier(x, np.ones((30, 1), dtype=bool))

    assert np.all(classifier.predict(x))


def test_the_boundary_is_learned_when_there_is_one():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(200, 2))
    labels = (x[:, 0] > 0.5).reshape(-1, 1)
    classifier = ie._fit_classifier(x, labels)

    predicted = classifier.predict(np.array([[0.1, 0.5], [0.9, 0.5]]))
    assert not predicted[0, 0] and predicted[1, 0]


# --- what a settings form is offered, and what 'all' sweeps up -------------------

def _autoemulate_or_skip():
    from libcuflynx.emulators.emulator_trainer import autoemulate_available
    if not autoemulate_available():
        pytest.skip('autoemulate is not installed')


def test_a_settings_form_is_offered_both_kinds():
    """CUFLynx reads this list at runtime rather than hardcoding one, so the
    two-phase variants reach its emulator settings without a change on its side."""
    _autoemulate_or_skip()
    from libcuflynx.emulators.emulator_trainer import (
        base_emulator_model_names, emulator_model_names)

    offered = emulator_model_names()
    base = base_emulator_model_names()

    assert len(offered) == 2 * len(base)
    for name in base:
        assert ie.two_phase_name(name) in offered


def test_all_does_not_sweep_up_the_two_phase_variants():
    """Opt-in by name. Two stages cost more to fit than one, and only pay off when
    the features really do have a floor -- sweeping them into 'all' would double
    every comparison for studies that need none of it."""
    _autoemulate_or_skip()
    from libcuflynx.emulators.emulator_trainer import _parse_models

    assert not any(ie.is_two_phase(name) for name in _parse_models('all'))


def test_asking_for_one_by_name_reaches_the_trainer():
    from libcuflynx.emulators.emulator_trainer import _parse_models

    assert _parse_models('two_phase_MLP') == ['two_phase_MLP']


def test_a_helper_exists_for_every_offered_emulator():
    _autoemulate_or_skip()
    from libcuflynx.emulators.emulator_trainer import base_emulator_model_names

    for name in base_emulator_model_names():
        helper = getattr(ie, ie.two_phase_name(name))
        assert helper() == ie.two_phase_name(name)


def test_a_two_phase_name_cannot_be_mixed_with_others():
    """It is fitted on its own -- there is no comparison to win, and silently
    dropping the others would be worse than saying so."""
    _autoemulate_or_skip()
    import numpy as np
    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer

    trainer = EmulatorTrainer.__new__(EmulatorTrainer)
    trainer.settings = {'models': 'two_phase_MLP,MLP', 'random_seed': 0}
    trainer.pid = None
    with pytest.raises(ValueError, match='fitted on its own'):
        trainer.fit(np.zeros((10, 2)), np.zeros((10, 2)))
