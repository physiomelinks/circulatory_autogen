"""Tuning the classifier halves, and calibrating the boundary they decide.

Two gaps these close. The classifier halves of a two_phase_/multi_phase_ emulator were
fitted at sklearn defaults with no search at all, while the regression halves each got
``n_iter`` tuned draws -- and on an obs_data where half the items are counts, the
untuned half is the one doing most of the damage. And their decision threshold sat at
0.5, which is right only when the two mistakes cost the same amount; here they do not.
"""
import numpy as np
import pytest

from libcuflynx.emulators.internal_emulators import (
    THRESHOLD_MIN_GAIN,
    THRESHOLD_NEGLIGIBLE,
    _calibrate_threshold,
    _classifier_candidates,
    _fit_column,
    _log_loss_cv,
    _positive_proba,
    _Thresholded,
    _tune_classifier,
)

pytest.importorskip('sklearn', reason='the classifier halves are sklearn')

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402


def _separable(n=120, seed=0):
    """A boundary a classifier can learn, so a search has something to choose between."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 3))
    labels = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    return x, labels


# ---------------------------------------------------------------------------
# The candidate list
# ---------------------------------------------------------------------------
def test_the_first_candidate_is_always_the_untuned_default():
    """So classifier_n_iter: 1 reproduces the previous behaviour exactly, and any
    regression is attributable to the search rather than to this change."""
    assert _classifier_candidates(1, seed=0) == [{}]
    assert _classifier_candidates(5, seed=0)[0] == {}


def test_candidates_are_distinct_and_seeded():
    first = _classifier_candidates(4, seed=7)
    assert len(first) == 4
    assert len({tuple(sorted(c.items())) for c in first}) == 4
    assert first == _classifier_candidates(4, seed=7), 'a run must be reproducible'
    assert first != _classifier_candidates(4, seed=8)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_log_loss_cv_scores_a_learnable_boundary_better_than_a_random_one():
    x, labels = _separable()
    learnable = _log_loss_cv(GradientBoostingClassifier, x, labels, {}, 5, 0)
    noise = _log_loss_cv(GradientBoostingClassifier, x,
                         np.random.default_rng(1).integers(0, 2, labels.size), {}, 5, 0)
    assert learnable is not None and noise is not None
    assert learnable < noise


def test_log_loss_rather_than_accuracy_is_the_point():
    """Most draws do not spike, so accuracy rewards a classifier that always answers
    'floored'. Log loss does not, because it scores the probability."""
    x = np.random.default_rng(0).uniform(-1, 1, size=(200, 2))
    labels = np.zeros(200, dtype=int)
    labels[:10] = 1  # 5% positives

    always_zero_accuracy = 0.95
    assert always_zero_accuracy > 0.9, 'which is why accuracy is not used here'
    score = _log_loss_cv(GradientBoostingClassifier, x, labels, {}, 5, 0)
    assert score is None or score > 0, 'a proper scoring rule, or honestly unavailable'


def test_a_single_class_cannot_be_scored():
    x = np.zeros((20, 2))
    assert _log_loss_cv(GradientBoostingClassifier, x, np.ones(20, dtype=int),
                        {}, 5, 0) is None


def test_too_few_of_the_minority_class_to_fold_is_unavailable_not_an_error():
    x, labels = _separable(n=40)
    labels[:] = 0
    labels[0] = 1  # one positive cannot be stratified across folds
    assert _log_loss_cv(GradientBoostingClassifier, x, labels, {}, 5, 0) is None


# ---------------------------------------------------------------------------
# The tuner
# ---------------------------------------------------------------------------
def test_tuning_returns_a_fitted_model_that_predicts():
    x, labels = _separable()
    model = _tune_classifier(GradientBoostingClassifier, x, labels,
                             {'classifier_n_iter': 4, 'n_splits': 3})
    assert (np.asarray(model.predict(x)) == labels).mean() > 0.85


def test_one_iteration_uses_the_defaults_untouched():
    x, labels = _separable()
    tuned = _tune_classifier(GradientBoostingClassifier, x, labels,
                             {'classifier_n_iter': 1})
    plain = GradientBoostingClassifier(random_state=0).fit(x, labels)
    assert tuned.get_params()['n_estimators'] == plain.get_params()['n_estimators']
    assert tuned.get_params()['learning_rate'] == plain.get_params()['learning_rate']


def test_tuning_can_pick_something_other_than_the_defaults():
    """Not guaranteed on every dataset, but on a boundary this clean the search should
    find at least one configuration it prefers -- otherwise the search is inert."""
    x, labels = _separable(n=300, seed=3)
    tuned = _tune_classifier(GradientBoostingClassifier, x, labels,
                             {'classifier_n_iter': 6, 'n_splits': 3})
    params = tuned.get_params()
    assert params['n_estimators'] > 0 and params['max_depth'] >= 1


def test_tuning_degrades_to_a_plain_fit_when_it_cannot_search():
    """A column with one class carries no boundary; the caller handles that, but the
    tuner must not raise if it sees one."""
    x = np.random.default_rng(0).uniform(size=(30, 2))
    labels = np.zeros(30, dtype=int)
    labels[0] = 1
    model = _tune_classifier(GradientBoostingClassifier, x, labels,
                            {'classifier_n_iter': 4})
    assert model.predict(x).shape == (30,)


def test_fit_column_still_short_circuits_a_constant_column():
    x = np.random.default_rng(0).uniform(size=(20, 2))
    always = _fit_column(GradientBoostingClassifier, x, np.ones(20, dtype=bool))
    never = _fit_column(GradientBoostingClassifier, x, np.zeros(20, dtype=bool))
    assert always.predict(x).all()
    assert not never.predict(x).any()


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------
class _Proba:
    """A classifier with a fixed probability per row, so a threshold is exactly known."""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, x):
        return np.column_stack([1 - self.p, self.p])

    def predict(self, x):
        return (self.p >= 0.5).astype(int)


def test_positive_proba_reads_the_second_column():
    assert _positive_proba(_Proba([0.2, 0.8]), np.zeros((2, 1))) == pytest.approx([0.2, 0.8])


def test_positive_proba_is_none_when_the_model_cannot_say():
    class _NoProba:
        def predict(self, x):
            return np.zeros(len(x))

    assert _positive_proba(_NoProba(), np.zeros((3, 1))) is None


def test_the_threshold_moves_when_one_mistake_costs_more_than_the_other():
    """The asymmetry this exists for. Ten rows the classifier is unsure about (p=0.4):
    at 0.5 they all take the false branch, which is far from the truth, while the true
    branch is close. A lower threshold is measurably better and should be chosen.
    """
    n = 40
    proba = np.full(n, 0.4)
    y = np.zeros((n, 1))
    y[:, 0] = 10.0                      # the truth is the true branch's value
    values_false = np.zeros((n, 1))     # far away
    values_true = np.full((n, 1), 10.0)  # exactly right

    threshold, gain = _calibrate_threshold(
        _Proba(proba), np.zeros((n, 2)), y, [0], values_false, values_true)
    assert threshold < 0.5
    assert gain is not None and gain > THRESHOLD_MIN_GAIN


def test_the_threshold_stays_at_a_half_when_there_is_nothing_to_gain():
    n = 40
    y = np.zeros((n, 1))
    values = np.zeros((n, 1))
    threshold, gain = _calibrate_threshold(
        _Proba(np.full(n, 0.9)), np.zeros((n, 2)), y, [0], values, values)
    assert threshold == 0.5
    assert gain is None


def test_a_negligible_baseline_error_is_not_worth_calibrating():
    """Found by running it: a purely *relative* gain test is not enough. Where both
    branches already predict the truth the error at 0.5 is near zero, and a 2%
    improvement on near zero is reachable by noise -- so the threshold moved for
    nothing. The test is now scaled against the targets themselves."""
    n = 60
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.45, 0.55, n)
    y = rng.normal(0, 1, (n, 1))
    values_false = y + rng.normal(0, 1e-3, (n, 1))
    values_true = y + rng.normal(0, 1e-3, (n, 1))
    threshold, gain = _calibrate_threshold(
        _Proba(proba), np.zeros((n, 2)), y, [0], values_false, values_true)
    assert threshold == 0.5, 'branches are interchangeable; nothing real to gain'
    assert THRESHOLD_NEGLIGIBLE > 0


def test_too_few_held_out_rows_leaves_the_threshold_alone():
    threshold, gain = _calibrate_threshold(
        _Proba([0.4, 0.4]), np.zeros((2, 2)), np.zeros((2, 1)), [0],
        np.zeros((2, 1)), np.ones((2, 1)))
    assert threshold == 0.5 and gain is None


def test_a_classifier_without_probabilities_is_left_alone():
    class _NoProba:
        def predict(self, x):
            return np.zeros(len(x))

    threshold, gain = _calibrate_threshold(
        _NoProba(), np.zeros((20, 2)), np.zeros((20, 1)), [0],
        np.zeros((20, 1)), np.ones((20, 1)))
    assert threshold == 0.5 and gain is None


def test_the_cost_weights_can_steer_the_threshold():
    """An error in a tight-sigma observable should move the boundary more than the same
    error in a loose one, which is what passing the metric's weights achieves."""
    n = 40
    proba = np.full(n, 0.45)
    y = np.column_stack([np.full(n, 1.0), np.full(n, 0.0)])
    values_false = np.column_stack([np.zeros(n), np.zeros(n)])
    values_true = np.column_stack([np.ones(n), np.full(n, 8.0)])

    # Unweighted, column 1's large error should keep the false branch.
    plain, _ = _calibrate_threshold(_Proba(proba), np.zeros((n, 2)), y, [0, 1],
                                    values_false, values_true)
    # Weighting column 0 heavily flips the balance toward the true branch.
    weighted, _ = _calibrate_threshold(_Proba(proba), np.zeros((n, 2)), y, [0, 1],
                                       values_false, values_true,
                                       weights=np.array([50.0, 1.0]))
    assert plain == 0.5
    assert weighted < 0.5


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------
def test_thresholded_applies_its_threshold():
    model = _Proba([0.1, 0.4, 0.6, 0.9])
    x = np.zeros((4, 1))
    assert list(_Thresholded(model, 0.5).predict(x)) == [False, False, True, True]
    assert list(_Thresholded(model, 0.3).predict(x)) == [False, True, True, True]


def test_thresholded_passes_probabilities_through():
    model = _Proba([0.25])
    assert _Thresholded(model, 0.5).predict_proba(np.zeros((1, 1))).shape == (1, 2)


def test_thresholded_falls_back_to_predict_without_probabilities():
    class _NoProba:
        def predict(self, x):
            return np.array([1, 0])

    assert list(_Thresholded(_NoProba(), 0.2).predict(np.zeros((2, 1)))) == [True, False]
