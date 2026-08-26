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


def test_a_settings_form_is_offered_every_kind():
    """CUFLynx reads this list at runtime rather than hardcoding one, so a new
    variant reaches its emulator settings without a change on its side."""
    _autoemulate_or_skip()
    from libcuflynx.emulators.emulator_trainer import (
        base_emulator_model_names, emulator_model_names)

    offered = emulator_model_names()
    base = base_emulator_model_names()

    # plain, two-phase and multi-phase
    assert len(offered) == 3 * len(base)
    for name in base:
        assert ie.two_phase_name(name) in offered
        assert ie.multi_phase_name(name) in offered


def test_all_does_not_sweep_up_the_two_phase_variants():
    """Opt-in by name. Two stages cost more to fit than one, and only pay off when
    the features really do have a floor -- sweeping them into 'all' would double
    every comparison for studies that need none of it."""
    _autoemulate_or_skip()
    from libcuflynx.emulators.emulator_trainer import _parse_models

    assert not any(ie.is_two_phase(name) for name in _parse_models('all'))
    assert not any(ie.is_multi_phase(name) for name in _parse_models('all'))


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


# ---------------------------------------------------------------------------
# multi-phase: counts, jumps and smooth observables in one emulator
# ---------------------------------------------------------------------------
def _multi_phase_design(seed=0, n=400):
    """One column of each kind, with a quiet branch that genuinely varies.

    The quiet branch varying is the point: a constant one is representable by the
    two-phase emulator, so it would not distinguish the two.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 3))
    count = np.where(x[:, 0] < 0.4, 0, np.round(1 + 4 * (x[:, 0] - 0.4) / 0.6)).astype(float)
    jump = np.where(x[:, 1] < 0.5, -60 + 18 * x[:, 2], 20 + 25 * x[:, 2])
    smooth = 3.0 + 2.0 * x[:, 0] - 1.5 * x[:, 1] ** 2
    return x, np.column_stack([count, jump, smooth])


def test_the_three_kinds_are_found_from_the_data_not_from_a_name():
    from libcuflynx.emulators.internal_emulators import classify_features

    _, y = _multi_phase_design()
    assert list(classify_features(y)) == ["count", "jump", "smooth"]


def test_a_count_is_only_a_count_when_it_is_a_small_set_of_non_negative_integers():
    import numpy as np

    from libcuflynx.emulators.internal_emulators import is_count_column

    assert is_count_column(np.array([0.0, 1, 2, 3, 1, 0, 2]))
    # negative -> not a count
    assert not is_count_column(np.array([0.0, -1, 2, 3]))
    # non-integer -> not a count
    assert not is_count_column(np.array([0.0, 1.5, 2, 3]))
    # a single value is a constant, not a count
    assert not is_count_column(np.zeros(20))
    # too many levels is a continuous quantity for every practical purpose
    assert not is_count_column(np.arange(200.0))


def test_a_jump_is_two_populations_not_a_repeated_value():
    """``floor_mask`` cannot see this shape, which is why it needed its own test.

    A peak voltage that either spikes or does not has two *branches*, each varying;
    no single value repeats, so the floor test finds nothing.
    """
    import numpy as np

    from libcuflynx.emulators.internal_emulators import floor_mask, jump_threshold

    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 400)
    column = np.where(x < 0.5, -60 + 18 * x, 20 + 25 * x)

    assert jump_threshold(column) is not None
    has_floor, _ = floor_mask(column.reshape(-1, 1))
    assert not has_floor.any()


def test_a_smooth_column_has_no_jump():
    import numpy as np

    from libcuflynx.emulators.internal_emulators import jump_threshold

    rng = np.random.default_rng(4)
    assert jump_threshold(np.sort(rng.uniform(0, 1, 400))) is None


@pytest.mark.parametrize("family", ["RadialBasisFunctions", "MLP"])
def test_multi_phase_predicts_each_kind_the_way_its_shape_asks(family):
    """Counts stay non-negative, and every kind is recovered."""
    import numpy as np

    from libcuflynx.emulators.emulator_trainer import _load_autoemulate
    from libcuflynx.emulators.internal_emulators import classify_features, fit_multi_phase

    pytest.importorskip("autoemulate")
    x, y = _multi_phase_design()
    kinds = classify_features(y)
    shift, span = y.min(0), np.ptp(y, 0)
    scaled = (y - shift) / span
    train, test = slice(0, 320), slice(320, None)

    model, _ = fit_multi_phase(
        x[train], scaled[train], x[test], scaled[test], family,
        _load_autoemulate(), dict(n_iter=3, n_splits=3, random_seed=0), kinds)

    assert model.model_name == f"multi_phase_{family}"
    predicted = model.predict(x[test]) * span + shift

    # A count is never negative -- a Poisson cost clips a negative lambda to 1e-12 and
    # charges k*log(1e12) for it, with no gradient to climb back out.
    assert predicted[:, 0].min() >= -1e-9

    for column in range(3):
        truth = y[test][:, column]
        residual = ((truth - predicted[:, column]) ** 2).sum()
        total = ((truth - truth.mean()) ** 2).sum()
        assert 1 - residual / total > 0.9, f"column {column} fitted poorly"


def test_both_sides_of_a_jump_get_their_own_regressor():
    """The difference from two-phase, stated as the number it changes.

    Two-phase substitutes one constant on the inactive side, so a quiet branch that
    varies is not merely approximated badly -- it cannot be represented at all.
    """
    import numpy as np

    from libcuflynx.emulators.emulator_trainer import _load_autoemulate
    from libcuflynx.emulators.internal_emulators import (
        classify_features, fit_multi_phase, fit_two_phase)

    pytest.importorskip("autoemulate")
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(500, 3))
    y = np.where(x[:, 1] < 0.5, -60 + 18 * x[:, 2], 20 + 25 * x[:, 2]).reshape(-1, 1)
    shift, span = y.min(0), np.ptp(y, 0)
    scaled = (y - shift) / span
    train, test = slice(0, 400), slice(400, None)
    autoemulate, kwargs = _load_autoemulate(), dict(n_iter=3, n_splits=3, random_seed=0)

    multi, _ = fit_multi_phase(x[train], scaled[train], x[test], scaled[test],
                               "RadialBasisFunctions", autoemulate, kwargs,
                               classify_features(y))
    two, _ = fit_two_phase(x[train], scaled[train], x[test], scaled[test],
                           "RadialBasisFunctions", autoemulate, kwargs)

    quiet = x[test][:, 1] < 0.5
    truth = y[test].ravel()[quiet]

    def quiet_r2(model):
        predicted = (model.predict(x[test]) * span + shift).ravel()[quiet]
        return 1 - ((truth - predicted) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()

    assert quiet_r2(multi) > 0.9
    assert quiet_r2(multi) > quiet_r2(two) + 0.5
    assert len(multi.jump_groups) == 1
    assert multi.jump_groups[0]["low"] is not None
    assert multi.jump_groups[0]["high"] is not None


def test_columns_that_jump_together_share_one_pair_of_regressors():
    """Otherwise every observable of one trace pays for its own pair."""
    import numpy as np

    from libcuflynx.emulators.emulator_trainer import _load_autoemulate
    from libcuflynx.emulators.internal_emulators import classify_features, fit_multi_phase

    pytest.importorskip("autoemulate")
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, size=(400, 3))
    spiking = x[:, 1] < 0.5
    y = np.column_stack([
        np.where(spiking, -60 + 18 * x[:, 2], 20 + 25 * x[:, 2]),
        np.where(spiking, -40 + 9 * x[:, 0], 35 + 12 * x[:, 0]),
    ])
    shift, span = y.min(0), np.ptp(y, 0)
    scaled = (y - shift) / span
    model, _ = fit_multi_phase(x[:320], scaled[:320], x[320:], scaled[320:],
                               "RadialBasisFunctions", _load_autoemulate(),
                               dict(n_iter=3, n_splits=3, random_seed=0),
                               classify_features(y))
    assert len(model.jump_groups) == 1
    assert sorted(model.jump_groups[0]["columns"]) == [0, 1]


def test_a_multi_phase_name_is_refused_alongside_another_model():
    from libcuflynx.emulators.internal_emulators import (
        base_emulator_name, is_multi_phase, multi_phase_name)

    assert is_multi_phase("multi_phase_MLP")
    assert not is_multi_phase("two_phase_MLP")
    assert base_emulator_name("multi_phase_MLP") == "MLP"
    assert base_emulator_name("two_phase_MLP") == "MLP"
    assert base_emulator_name("MLP") == "MLP"
    assert multi_phase_name("MLP") == "multi_phase_MLP"
    assert multi_phase_name("multi_phase_MLP") == "multi_phase_MLP"


def test_a_jump_side_too_thin_to_fold_falls_back_rather_than_raising():
    """A side needs enough rows to *fold*, not merely to fit.

    autoemulate cross-validates with n_splits folds and, when every candidate errors,
    keeps no results at all -- so ``best_result()`` raises and one branch of one
    observable would take the whole training run with it.
    """
    import numpy as np

    from libcuflynx.emulators.internal_emulators import _fit_side

    sentinel = object()

    def explode(*_args, **_kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("a side this thin should not have been fitted")

    rows = np.zeros(200, dtype=bool)
    rows[:9] = True  # over MIN_SIDE_ROWS, under 2 * n_splits
    got = _fit_side(np.zeros((200, 2)), np.zeros((200, 1)), None, None, rows,
                    "RadialBasisFunctions", explode, {"n_splits": 5}, sentinel)
    assert got is sentinel


def test_a_jump_side_that_fails_to_fit_falls_back_rather_than_raising():
    import numpy as np

    from libcuflynx.emulators.internal_emulators import _fit_side

    sentinel = object()

    def fails(*_args, **_kwargs):
        raise ValueError("No results available. Please run AutoEmulate.compare() first.")

    rows = np.ones(200, dtype=bool)
    got = _fit_side(np.zeros((200, 2)), np.zeros((200, 1)), None, None, rows,
                    "RadialBasisFunctions", fails, {"n_splits": 5}, sentinel)
    assert got is sentinel


# ----------------------------------------------------- counts never come back negative (#498)

def test_the_class_cap_is_left_where_it_performs_best():
    """The cap stays at 24, and that is a measured choice rather than an oversight.

    Raising it so that counts with 26-53 levels became classifiers removed the negative
    predictions of #498, but roughly doubled the error on those very features (RMSE +107%
    on ox1, +81% on cpvt) and made the emulator worse overall -- std(dcost) 2.675 -> 3.097
    and 3.385 -> 4.343 on the two 12000-sample studies. Negatives are clamped in predict
    instead, which costs nothing. If this number is ever raised, re-measure both.
    """
    from libcuflynx.emulators.internal_emulators import is_count_column, COUNT_MAX_CLASSES
    import numpy as np
    rng = np.random.default_rng(0)
    assert COUNT_MAX_CLASSES == 24
    assert is_count_column(rng.integers(0, 20, size=4000).astype(float))
    # beyond the cap a count is handled by the regressor, and the clamp below keeps it sane
    assert not is_count_column(rng.integers(0, 40, size=4000).astype(float))
    assert not is_count_column(rng.normal(size=500))
    assert not is_count_column(rng.integers(0, 5, size=500).astype(float) - 3.0)


def test_multi_phase_never_predicts_a_negative_count():
    """Whatever produced the value, a count column is clamped at zero on the way out."""
    from libcuflynx.emulators.internal_emulators import MultiPhaseEmulator

    class Negative:
        """Stands in for the plain regressor a demoted count would fall through to."""
        def predict(self, x):
            return np.full((len(x), 3), -2.5)

    kinds = np.array(['count', 'smooth', 'count'], dtype=object)
    # count_models deliberately empty: this is the fall-through path that produced the bug
    model = MultiPhaseEmulator(Negative(), kinds, {}, [], 'RadialBasisFunctions',
                               count_columns=[0, 2])
    out = model.predict(np.zeros((7, 4)))
    assert (out[:, [0, 2]] >= 0).all(), 'a count came back negative'
    assert (out[:, 1] == -2.5).all(), 'a smooth column must not be clamped'


def test_an_older_pickle_without_count_columns_still_loads():
    """count_columns defaults to the modelled counts, so an existing bundle keeps working."""
    from libcuflynx.emulators.internal_emulators import MultiPhaseEmulator

    class Zero:
        def predict(self, x):
            return np.zeros((len(x), 2))

    model = MultiPhaseEmulator(Zero(), np.array(['count', 'smooth'], dtype=object),
                               {0: None}, [], 'MLP')
    assert model.count_columns == [0]


# ------------------------------------ floored counts get the floor treatment (#498)

def test_a_count_past_the_class_cap_is_a_floored_count_not_smooth():
    """The demoted counts of #498 are recognised rather than dropped to the regressor.

    They are integer, non-negative, and spend most of the design on zero -- 31-92% on the
    SN_full bundles -- so they have a floor to put exactly where it belongs.
    """
    from libcuflynx.emulators.internal_emulators import classify_features
    rng = np.random.default_rng(0)
    n = 600
    few = rng.integers(0, 10, size=n).astype(float)             # under the cap
    many = np.where(rng.random(n) < 0.7, 0.0,
                    rng.integers(1, 45, size=n).astype(float))  # over the cap, floored at 0
    smooth = rng.normal(size=n)
    kinds = list(classify_features(np.column_stack([few, many, smooth])))
    assert kinds[0] == 'count'
    assert kinds[1] == 'floored_count'
    assert kinds[2] == 'smooth'


def test_a_floored_count_never_predicts_below_its_floor():
    """The floor is assigned, not smoothed through -- so no clamping is needed for it."""
    from libcuflynx.emulators.internal_emulators import MultiPhaseEmulator

    class Rippling:
        """A regressor that dips below zero the way a smooth fit does at a floor."""
        def predict(self, x):
            return np.full((len(x), 2), -0.4)

    class SaysFloor:
        def predict(self, x):
            return np.zeros(len(x), dtype=bool)      # every row is on the floor

    model = MultiPhaseEmulator(
        Rippling(), np.array(['floored_count', 'smooth'], dtype=object), {}, [],
        'RadialBasisFunctions', count_columns=[0],
        floor_groups=[{'columns': [0], 'classifier': SaysFloor(),
                       'floor_value': {0: 0.0}, 'magnitude': None}])
    out = model.predict(np.zeros((5, 3)))
    assert (out[:, 0] == 0.0).all(), 'the floor should be assigned exactly'
    assert (out[:, 1] == -0.4).all(), 'a smooth column must be left alone'


def test_the_magnitude_regressor_is_used_off_the_floor():
    from libcuflynx.emulators.internal_emulators import MultiPhaseEmulator

    class Base:
        def predict(self, x):
            return np.full((len(x), 1), -99.0)

    class Magnitude:
        def predict(self, x):
            return np.full((len(x), 1), 7.0)

    class Alternating:
        def predict(self, x):
            return np.arange(len(x)) % 2 == 1

    model = MultiPhaseEmulator(
        Base(), np.array(['floored_count'], dtype=object), {}, [], 'MLP',
        count_columns=[0],
        floor_groups=[{'columns': [0], 'classifier': Alternating(),
                       'floor_value': {0: 0.0}, 'magnitude': Magnitude()}])
    out = model.predict(np.zeros((4, 2)))[:, 0]
    assert list(out) == [0.0, 7.0, 0.0, 7.0]


def test_an_older_bundle_without_floor_groups_still_predicts():
    from libcuflynx.emulators.internal_emulators import MultiPhaseEmulator

    class Zero:
        def predict(self, x):
            return np.zeros((len(x), 1))

    model = MultiPhaseEmulator(Zero(), np.array(['smooth'], dtype=object), {}, [], 'MLP')
    assert model.floor_groups == []
    assert model.predict(np.zeros((3, 2))).shape == (3, 1)
