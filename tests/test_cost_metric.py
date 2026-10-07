"""The tuning metric denominated in the cost rather than in the features' own spread.

R2 divides each feature's error by that feature's variance across the design, a
quantity with no relationship to the cost. These tests pin the difference that makes:
an error in a tight-sigma observable must count for more than the same error in a
loose one, which is exactly what R2 cannot express.

No autoemulate needed -- the metric is duck-typed against its ``Metric`` protocol
precisely so that it can be tested (and imported) without it.
"""
import numpy as np
import pytest

from libcuflynx.emulators.cost_metric import (
    CostWeightedError,
    build_metric,
    sigma_weights,
)

torch = pytest.importorskip('torch', reason='the metric scores torch tensors')


class _Pid:
    """Just enough of a ParamID for build_metric: it reads obs_info and nothing else."""

    def __init__(self, stds, weights):
        self.obs_info = {'std_const_vec': stds, 'weight_const_vec': weights}


def _scale(spans):
    return {'span': list(spans), 'shift': [0.0] * len(spans)}


# ---------------------------------------------------------------------------
# The weights
# ---------------------------------------------------------------------------
def test_a_tight_sigma_weighs_more_than_a_loose_one():
    """The whole point. Two features, identical design spread, sigmas 10x apart: the
    tight one must dominate, because the cost divides by sigma."""
    weights, notes = sigma_weights(stds=[1.0, 10.0], weights=[1.0, 1.0],
                                   spans=[1.0, 1.0])
    assert weights[0] == pytest.approx(10 * weights[1])
    assert notes == []


def test_the_span_undoes_autoemulates_scaling():
    """autoemulate sees (y - shift) / span, so an error of e there is e * span in real
    units. Without the span the metric would weigh a feature by how wide its design
    happened to be -- which is the mistake R2 makes."""
    weights, _ = sigma_weights(stds=[2.0, 2.0], weights=[1.0, 1.0], spans=[1.0, 100.0])
    assert weights[1] == pytest.approx(100 * weights[0])


def test_an_unweighted_item_contributes_nothing():
    """A zero weight drops an item from the cost entirely, so it must drop out here
    too -- otherwise the emulator is tuned partly on observables nobody is fitting."""
    weights, _ = sigma_weights(stds=[1.0, 1.0], weights=[1.0, 0.0], spans=[1.0, 1.0])
    assert weights[1] == 0.0
    assert weights[0] > 0


@pytest.mark.parametrize('bad', [np.nan, 0.0, -1.0, np.inf])
def test_a_feature_with_no_usable_sigma_falls_back_to_its_design_spread(bad):
    """Counts scored by a distribution declare prob_dist_params and no std. Dropping
    them would tune on a subset of the cost; dividing by zero would poison the whole
    metric. They get the scale R2 would have used, and are named in the notes."""
    y_train = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)])
    weights, notes = sigma_weights(stds=[1.0, bad], weights=[1.0, 1.0],
                                   spans=[1.0, 1.0], y_train=y_train)
    assert notes == [1]
    assert np.isfinite(weights[1]) and weights[1] > 0


def test_a_mismatched_vector_length_is_refused():
    with pytest.raises(ValueError, match='one std, weight and span per feature'):
        sigma_weights(stds=[1.0, 2.0], weights=[1.0], spans=[1.0, 1.0])


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------
def _score(metric, pred, true):
    return float(metric(torch.tensor(pred, dtype=torch.float64),
                        torch.tensor(true, dtype=torch.float64)))


def test_a_perfect_prediction_scores_zero():
    metric = CostWeightedError([1.0, 1.0])
    assert _score(metric, [[1.0, 2.0]], [[1.0, 2.0]]) == pytest.approx(0.0)


def test_lower_is_better():
    """autoemulate reads `maximize` to pick argmin or argmax; getting it backwards
    would select the worst emulator of every family."""
    assert CostWeightedError([1.0]).maximize is False


def test_the_same_raw_error_costs_more_in_a_tight_sigma_feature():
    metric = CostWeightedError([10.0, 1.0])
    tight = _score(metric, [[1.0, 0.0]], [[0.0, 0.0]])
    loose = _score(metric, [[0.0, 1.0]], [[0.0, 0.0]])
    assert tight == pytest.approx(10 * loose)


def test_r2_would_have_called_those_two_errors_equal():
    """Stated as a test because it is the reason this module exists. Two features with
    the same design variance and sigmas 10x apart: R2 sees one number, the cost sees
    two."""
    design = np.column_stack([np.linspace(-1, 1, 200), np.linspace(-1, 1, 200)])
    r2_scale = design.std(axis=0)
    assert r2_scale[0] == pytest.approx(r2_scale[1]), 'identical to R2'

    weights, _ = sigma_weights(stds=[0.1, 1.0], weights=[1.0, 1.0], spans=[1.0, 1.0])
    assert weights[0] == pytest.approx(10 * weights[1]), 'not identical to the cost'


def test_it_reduces_over_points_and_features():
    metric = CostWeightedError([1.0, 1.0])
    # errors of 1 in every cell: rms is 1 whatever the shape.
    assert _score(metric, [[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]) \
        == pytest.approx(1.0)


def test_per_output_reduction_is_available_for_reporting():
    from autoemulate.core.metrics import MetricParams  # noqa: PLC0415

    metric = CostWeightedError([1.0, 3.0])
    out = metric(torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0, 0.0]]),
                 MetricParams(reduction='none'))
    assert out.numel() == 2
    assert float(out[1]) == pytest.approx(3 * float(out[0]))


def test_a_column_count_mismatch_is_refused_rather_than_scored():
    """A piece of a multi-phase fit could be handed a subset. Scoring it with
    mismatched weights would look like a working metric and rank on nonsense."""
    metric = CostWeightedError([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match='built for 3 feature'):
        _score(metric, [[1.0, 1.0]], [[0.0, 0.0]])


def test_a_one_dimensional_target_is_accepted():
    metric = CostWeightedError([2.0])
    assert _score(metric, [1.0, 1.0], [0.0, 0.0]) == pytest.approx(2.0)


def test_a_distribution_prediction_is_reduced_to_its_mean():
    """The cost is evaluated at a point estimate, so that is what is scored."""
    class _Dist:
        mean = torch.tensor([[1.0]])

    metric = CostWeightedError([1.0])
    assert float(metric(_Dist(), torch.tensor([[0.0]]))) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Building it from a study
# ---------------------------------------------------------------------------
def test_build_metric_from_obs_info():
    metric, notes = build_metric(_Pid([1.0, 4.0], [1.0, 1.0]), _scale([1.0, 1.0]))
    assert isinstance(metric, CostWeightedError)
    assert notes == []
    assert metric.weights[0] == pytest.approx(4 * metric.weights[1])


def test_build_metric_degrades_rather_than_raising_without_sigmas():
    """A study whose obs_info cannot offer them should fall back to R2 with a reason,
    not fail to train."""
    metric, notes = build_metric(_Pid(None, None), _scale([1.0]))
    assert metric is None
    assert notes and 'std/weight' in notes[0]


def test_build_metric_degrades_when_nothing_is_weighted():
    metric, notes = build_metric(_Pid([1.0, 1.0], [0.0, 0.0]), _scale([1.0, 1.0]))
    assert metric is None
    assert notes and 'zero weight' in notes[0]


def test_a_scalar_span_is_broadcast():
    metric, _ = build_metric(_Pid([1.0, 2.0], [1.0, 1.0]), {'span': 3.0})
    assert metric is not None
    assert metric.weights.size == 2


def test_the_metric_names_itself_for_autoemulates_result_tables():
    metric = CostWeightedError([1.0], name='cost_weighted_error')
    assert str(metric) == 'cost_weighted_error'
    assert metric == CostWeightedError([2.0], name='cost_weighted_error')
    assert hash(metric) == hash(CostWeightedError([9.0], name='cost_weighted_error'))


# ---------------------------------------------------------------------------
# Reaching the fit
# ---------------------------------------------------------------------------
class _Trainer:
    """The two methods under test, lifted off EmulatorTrainer without a ParamID."""

    def __init__(self, settings, pid):
        from libcuflynx.emulators.emulator_trainer import EmulatorTrainer

        self._settings = settings
        self.pid = pid
        self.rank = 1  # silences the rank-0 prints
        self._tuning_metric = EmulatorTrainer._tuning_metric.__get__(self)
        self._phase_settings = EmulatorTrainer._phase_settings.__get__(self)

    def _setting(self, name, default=None):
        return self._settings.get(name, default)

    @property
    def feature_labels(self):
        return ['a', 'b']


def test_r2_is_the_default_so_no_existing_run_changes():
    trainer = _Trainer({}, _Pid([1.0, 1.0], [1.0, 1.0]))
    assert trainer._tuning_metric(_scale([1.0, 1.0]), None) is None, (
        'None means "leave autoemulate on its own default"')


def test_asking_for_cost_builds_the_metric():
    trainer = _Trainer({'tuning_metric': 'cost'}, _Pid([1.0, 4.0], [1.0, 1.0]))
    metric = trainer._tuning_metric(_scale([1.0, 1.0]), None)
    assert isinstance(metric, CostWeightedError)
    assert metric.maximize is False


def test_an_unbuildable_cost_metric_degrades_to_r2_rather_than_failing_the_run():
    trainer = _Trainer({'tuning_metric': 'cost'}, _Pid(None, None))
    assert trainer._tuning_metric(_scale([1.0]), None) is None


def test_an_unknown_tuning_metric_is_refused_with_the_choices():
    trainer = _Trainer({'tuning_metric': 'rmse'}, _Pid([1.0], [1.0]))
    with pytest.raises(ValueError, match="expected 'r2' or 'cost'"):
        trainer._tuning_metric(_scale([1.0]), None)


def test_the_classifier_settings_do_not_leak_into_autoemulates_kwargs():
    """They travel by a separate argument on purpose: kwargs is splatted into
    AutoEmulate(...), where an unknown keyword is a TypeError, not a warning."""
    trainer = _Trainer({'classifier_n_iter': 6, 'n_splits': 3},
                       _Pid([1.0, 1.0], [1.0, 1.0]))
    settings = trainer._phase_settings(seed=11)
    assert settings['classifier_n_iter'] == 6
    assert settings['n_splits'] == 3
    assert settings['random_seed'] == 11
    assert settings['feature_weights'] is None


def test_the_cost_weights_are_handed_to_the_threshold_calibration():
    trainer = _Trainer({'tuning_metric': 'cost'}, _Pid([1.0, 4.0], [1.0, 1.0]))
    metric = trainer._tuning_metric(_scale([1.0, 1.0]), None)
    settings = trainer._phase_settings(seed=0, tuning=metric)
    assert settings['feature_weights'] is not None
    assert np.allclose(settings['feature_weights'], metric.weights)
