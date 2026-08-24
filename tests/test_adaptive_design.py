"""The second sampling stage has to spend its points on the cliff, or it is pure cost.

``gradient_weighted_design`` is worth testing directly rather than only through a
training run: the claim it makes is geometric -- that its points land near a threshold
an even design straddles -- and that claim can be checked exactly, against a response
whose threshold is known, without simulating anything.

The fallbacks matter as much as the concentration. This runs after a first stage that
may have produced a flat, degenerate or nearly-empty set of results, and in every one
of those cases it has to return a usable design rather than raise or return points
piled on top of each other.
"""
import numpy as np
import pytest

from libcuflynx.emulators.adaptive_design import gradient_weighted_design
from libcuflynx.emulators.emulator_trainer import EmulatorTrainer

MINS = np.array([0.0, 0.0])
MAXS = np.array([1.0, 1.0])
THRESHOLD = 0.5


def _step_response(x):
    """A cliff at ``x0 = 0.5`` and nothing else: the shape a rheobase makes."""
    return np.column_stack([(x[:, 0] > THRESHOLD).astype(float)])


def _first_stage(n=200, seed=0):
    x = np.random.default_rng(seed).random((n, 2))
    return x, _step_response(x)


def test_the_second_stage_lands_on_the_cliff():
    """The whole point: closer to the threshold than the design it was drawn from.

    Compared against the first stage rather than an absolute tolerance -- the useful
    property is *concentration relative to spreading points evenly*, which is the
    alternative use of the same simulations.
    """
    x, y = _first_stage()
    follow_up = gradient_weighted_design(x, y, 400, MINS, MAXS, seed=1)

    before = np.median(np.abs(x[:, 0] - THRESHOLD))
    after = np.median(np.abs(follow_up[:, 0] - THRESHOLD))
    assert after < before / 3, (
        f'stage 2 sits {after:.3f} from the cliff and stage 1 sat {before:.3f}; it is not '
        f'concentrating, so the extra simulations buy nothing an even design would not')


def test_it_does_not_concentrate_in_the_direction_that_carries_no_signal():
    """``x1`` does nothing to the response, so the points must stay spread across it.

    A method that narrowed here would be finding structure that is not there, and would
    leave the emulator blind along a parameter it still has to cover.
    """
    x, y = _first_stage()
    follow_up = gradient_weighted_design(x, y, 400, MINS, MAXS, seed=1)
    assert follow_up[:, 1].std() > 0.2, (
        'stage 2 collapsed along the parameter the response does not depend on')


def test_every_point_stays_inside_the_box():
    """Jitter pushes points off the segment, and the bounds are not negotiable."""
    x, y = _first_stage()
    follow_up = gradient_weighted_design(x, y, 500, MINS, MAXS, seed=3)
    assert follow_up.shape == (500, 2)
    assert (follow_up >= MINS).all() and (follow_up <= MAXS).all()


def test_the_same_seed_gives_the_same_design():
    """A training run has to be repeatable, and this stage is part of the design."""
    x, y = _first_stage()
    first = gradient_weighted_design(x, y, 50, MINS, MAXS, seed=7)
    second = gradient_weighted_design(x, y, 50, MINS, MAXS, seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, gradient_weighted_design(x, y, 50, MINS, MAXS, seed=8))


def test_a_flat_response_falls_back_to_covering_the_box():
    """No gradient anywhere means no information about where to look.

    The honest answer is a space-filling top-up. Concentrating on whatever floating-point
    dust happened to be largest would be worse than useless -- it would look adaptive.
    """
    x = np.random.default_rng(0).random((100, 2))
    y = np.ones((100, 3))
    follow_up = gradient_weighted_design(x, y, 300, MINS, MAXS, seed=1)
    assert follow_up.shape == (300, 2)
    assert follow_up[:, 0].std() > 0.2 and follow_up[:, 1].std() > 0.2


def test_features_are_weighed_by_their_own_range():
    """A tiny-magnitude feature that jumps must not be drowned by a large smooth one.

    CA parameters and observables routinely span decades -- a current near 1e-9 beside a
    firing rate near 10 -- so an unscaled norm would be decided entirely by the feature
    with the biggest numbers, and the cliff would never be seen.
    """
    x = np.random.default_rng(0).random((200, 2))
    y = np.column_stack([
        1e-9 * (x[:, 0] > THRESHOLD).astype(float),   # the cliff, in tiny units
        1e3 * x[:, 1],                                # a large, smooth ramp
    ])
    follow_up = gradient_weighted_design(x, y, 400, MINS, MAXS, seed=1)
    assert np.median(np.abs(follow_up[:, 0] - THRESHOLD)) < 0.1, (
        'the 1e-9 cliff was ignored in favour of the 1e3 ramp, so features are being '
        'compared in their own units instead of by range')


@pytest.mark.parametrize('n_points', [0, 1])
def test_too_few_points_to_estimate_anything(n_points):
    """One point has no neighbour, so there is no between-points gradient to weight by."""
    x = np.random.default_rng(0).random((n_points, 2))
    follow_up = gradient_weighted_design(x, _step_response(x) if n_points else
                                         np.empty((0, 1)), 20, MINS, MAXS, seed=1)
    assert follow_up.shape == (20, 2)
    assert (follow_up >= MINS).all() and (follow_up <= MAXS).all()


def test_asking_for_nothing_returns_nothing():
    """fraction_2nd_stage rounding to zero points must not produce a stray sample."""
    x, y = _first_stage()
    assert gradient_weighted_design(x, y, 0, MINS, MAXS).shape == (0, 2)


def test_log_scaled_parameters_come_back_inside_their_bounds():
    """A design spanning decades is placed in log space and has to return in real units."""
    mins, maxs = np.array([1e-9, 1e-3]), np.array([1e-6, 1e0])
    unit = np.random.default_rng(0).random((100, 2))
    x = np.exp(np.log(mins) + unit * (np.log(maxs) - np.log(mins)))
    y = np.column_stack([(x[:, 0] > 1e-7).astype(float)])
    follow_up = gradient_weighted_design(x, y, 200, mins, maxs, seed=1, log_scale=True)
    assert (follow_up >= mins).all() and (follow_up <= maxs).all()


# ---------------------------------------------------------------------------------
# How num_stages / frac_per_stage / method_per_stage turn into a plan.
#
# Exercised against the property rather than a training run: the arithmetic and the
# refusals are where a multi-stage design goes wrong silently -- a budget that does not
# add up, or a stage order that cannot work -- and none of that needs a simulation.
# ---------------------------------------------------------------------------------

class _Stages:
    """Just enough trainer to ask for a sampling plan, with no model behind it."""

    def __init__(self, **settings):
        self.settings = settings

    def _setting(self, name, default):
        value = self.settings.get(name, default)
        return default if value is None else value

    @property
    def num_train_samples(self):
        return int(self.settings.get('num_train_samples', 1000))


def _plan(**settings):
    trainer = _Stages(**settings)
    return EmulatorTrainer.sampling_stages.fget(trainer)


def test_one_stage_is_the_design_ca_has_always_built():
    """The default has to be untouched: every existing study depends on it."""
    assert _plan(num_train_samples=128) == [
        {'method': 'sobol', 'num_samples': 128, 'weight': 1.0}]
    assert _plan(num_train_samples=128, sample_type='latin_hypercube') == [
        {'method': 'latin_hypercube', 'num_samples': 128, 'weight': 1.0}]


def test_the_stages_spend_exactly_the_budget():
    """Rounding must not lose or invent a simulation; the last stage takes the remainder."""
    stages = _plan(num_train_samples=1000, num_stages=3, frac_per_stage=[1/3, 1/3, 1/3])
    assert sum(stage['num_samples'] for stage in stages) == 1000


def test_defaults_put_the_space_filling_stage_first():
    """"Two stages" without further instruction means sobol, then adapt."""
    stages = _plan(num_train_samples=800, num_stages=2)
    assert [stage['method'] for stage in stages] == ['sobol', 'gradient_weighted']
    assert [stage['num_samples'] for stage in stages] == [400, 400]


def test_a_comma_separated_string_reads_the_same_as_a_list():
    """A form-driven tool has only strings to offer; a yaml naturally has lists."""
    as_list = _plan(num_train_samples=1000, num_stages=2, frac_per_stage=[0.6, 0.4],
                    method_per_stage=['sobol', 'gradient_weighted'])
    as_string = _plan(num_train_samples=1000, num_stages=2, frac_per_stage='0.6,0.4',
                      method_per_stage='sobol,gradient_weighted')
    assert as_list == as_string == [
        {'method': 'sobol', 'num_samples': 600, 'weight': 1.0},
        {'method': 'gradient_weighted', 'num_samples': 400, 'weight': 1.0}]


def test_an_adaptive_first_stage_is_refused():
    """It places points using features from earlier stages, and there are none.

    Refused up front rather than at run time: the alternative is discovering it after
    the model has been generated and the first simulations have been paid for.
    """
    with pytest.raises(ValueError, match='there are none'):
        _plan(num_stages=2, method_per_stage=['gradient_weighted', 'sobol'])


def test_fractions_that_do_not_add_up_are_refused():
    """Silently normalising would spend a budget the user did not ask to spend."""
    with pytest.raises(ValueError, match='sums to'):
        _plan(num_stages=2, frac_per_stage=[0.5, 0.2])


@pytest.mark.parametrize('settings, match', [
    ({'num_stages': 3, 'frac_per_stage': [0.5, 0.5]}, 'one entry per stage'),
    ({'num_stages': 2, 'method_per_stage': ['sobol']}, 'one entry per stage'),
    ({'num_stages': 2, 'frac_per_stage': [1.0, 0.0]}, 'positive share'),
    ({'num_stages': 2, 'method_per_stage': ['sobol', 'nonesuch']}, 'unknown method_per_stage'),
    ({'num_stages': 0}, 'at least one stage'),
])
def test_a_plan_that_cannot_run_is_refused_with_a_reason(settings, match):
    with pytest.raises(ValueError, match=match):
        _plan(**settings)


# ---------------------------------------------------------------------------------
# Does adaptive sampling actually produce a better emulator?
#
# Everything above checks that the stages place points where they claim to. That is not
# the same claim, and it is the weaker one: points can land exactly where intended and
# still buy nothing. These measure the thing the feature exists for -- fit a surrogate
# on a single-stage design of N points, fit another on a two-stage design of the same
# N, and compare both against the same held-out truth.
#
# The budget is equal by construction. An adaptive design that wins by simulating more
# points has not won.
# ---------------------------------------------------------------------------------

from libcuflynx.emulators.adaptive_design import error_weighted_design  # noqa: E402


def _surrogate_error(x_train, y_train, x_test, y_test):
    """RMSE of an RBF surrogate fitted on the design, over a fixed held-out set."""
    from scipy.interpolate import RBFInterpolator

    model = RBFInterpolator(x_train, y_train, kernel='thin_plate_spline', smoothing=1e-8,
                            neighbors=min(len(x_train), 50))
    return float(np.sqrt(np.mean(np.square(model(x_test).reshape(y_test.shape) - y_test))))


def _sobol(n, seed, dim=2):
    from scipy.stats import qmc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return qmc.Sobol(d=dim, scramble=True, seed=seed).random(n)


def _held_out(response, seed=99, n=500, dim=2):
    """A fixed, evenly spread test set. Sobol, deliberately -- see the note below."""
    x = _sobol(n, seed, dim)
    return x, response(x)


def _peaked(x):
    """Smooth almost everywhere, with one narrow feature an even design under-resolves."""
    return np.column_stack([np.exp(-60.0 * ((x[:, 0] - 0.35) ** 2 + (x[:, 1] - 0.6) ** 2))])


def test_error_weighting_beats_an_even_design_of_the_same_size():
    """The generic case: no cliff, just a response that is harder in some places.

    Both designs simulate 160 points. The two-stage one spends the second half where a
    surrogate cross-validated on the first half is worst, and has to come out ahead on a
    held-out set neither of them has seen.
    """
    budget, first = 160, 80
    x_test, y_test = _held_out(_peaked)

    even = _sobol(budget, seed=0)
    even_error = _surrogate_error(even, _peaked(even), x_test, y_test)

    stage1 = _sobol(first, seed=0)
    stage2 = error_weighted_design(stage1, _peaked(stage1), budget - first, MINS, MAXS, seed=1)
    staged = np.vstack([stage1, stage2])
    staged_error = _surrogate_error(staged, _peaked(staged), x_test, y_test)

    assert len(staged) == len(even), 'the comparison is only fair at an equal budget'
    assert staged_error < even_error, (
        f'error-weighted sampling scored {staged_error:.4g} against {even_error:.4g} for a '
        f'plain design of the same size, so the second stage paid for nothing')


def _cliff(x):
    """A threshold in x0, as a neuron's rheobase is, plus a mild ramp in x1."""
    return np.column_stack([np.where(x[:, 0] > THRESHOLD, 1.0, 0.0) + 0.1 * x[:, 1]])


def _forest_error(x_train, y_train, x_test, y_test):
    """RMSE of a random forest -- a model class that can represent a step.

    Deliberately not the RBF surrogate the smooth cases use. Concentrating points across
    a discontinuity is actively bad for a smooth global interpolator: it is forced
    through steeply-disagreeing neighbours and rings around the jump. Measuring an
    adaptive design that way conflates "were the points placed well" with "can this
    model represent a cliff at all", and the second question is answered no before the
    first is asked. See the caveat test below, which pins that behaviour down.
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(x_train, y_train.ravel())
    return float(np.sqrt(np.mean(np.square(model.predict(x_test) - y_test.ravel()))))


def test_gradient_weighting_beats_an_even_design_on_a_model_with_a_cliff():
    """The case the feature was written for: a threshold, as a neuron's rheobase is.

    Equal budgets: 160 simulations either way. The staged design spends half of them
    between the pairs that straddle the step, and has to come out ahead on a held-out
    set neither design has seen.
    """
    budget, first = 160, 80
    x_test, y_test = _held_out(_cliff)

    even = _sobol(budget, seed=0)
    even_error = _forest_error(even, _cliff(even), x_test, y_test)

    stage1 = _sobol(first, seed=0)
    stage2 = gradient_weighted_design(stage1, _cliff(stage1), budget - first, MINS, MAXS,
                                      seed=1)
    staged = np.vstack([stage1, stage2])
    staged_error = _forest_error(staged, _cliff(staged), x_test, y_test)

    assert len(staged) == len(even), 'the comparison is only fair at an equal budget'
    assert staged_error < even_error, (
        f'gradient-weighted sampling scored {staged_error:.4g} against {even_error:.4g} for '
        f'an even design of the same size, so it is not resolving the cliff')


def test_clustering_on_a_cliff_hurts_a_smooth_interpolator():
    """The caveat, pinned down rather than left for a user to discover.

    A thin-plate spline forced through tightly-spaced points on either side of a jump
    oscillates around it, and the oscillation costs more than the sharper boundary
    gains. So an adaptive stage is not free: it pays off for a model that can represent
    the discontinuity -- a forest, or the classifier half of a two-phase emulator -- and
    costs accuracy for one that cannot.

    Asserted, not just documented, because it decides whether a study should turn this
    on. If a change ever makes clustering harmless to a smooth interpolator, this test
    fails and the guidance in the settings description needs rewriting.
    """
    budget, first = 160, 80
    x_test, y_test = _held_out(_cliff)

    even = _sobol(budget, seed=0)
    stage1 = _sobol(first, seed=0)
    staged = np.vstack([stage1, gradient_weighted_design(
        stage1, _cliff(stage1), budget - first, MINS, MAXS, seed=1)])

    assert (_surrogate_error(staged, _cliff(staged), x_test, y_test)
            > _surrogate_error(even, _cliff(even), x_test, y_test))


def test_three_stages_sobol_then_gradient_then_error_run_together():
    """The stages compose: each one sees everything simulated before it.

    Worth its own test because the third stage is the first to be handed a design that
    is *not* space-filling -- half of what it reads was placed by the stage before it,
    clustered on the cliff. It has to cope with that input and still return points in
    the box.
    """
    cliff = _cliff
    stage1 = _sobol(80, seed=0)
    y1 = cliff(stage1)

    stage2 = gradient_weighted_design(stage1, y1, 40, MINS, MAXS, seed=1)
    so_far = np.vstack([stage1, stage2])
    y2 = cliff(so_far)

    stage3 = error_weighted_design(so_far, y2, 40, MINS, MAXS, seed=2)
    everything = np.vstack([so_far, stage3])

    assert len(everything) == 160
    assert (everything >= MINS).all() and (everything <= MAXS).all()
    # Each adaptive stage found something to aim at rather than falling back to uniform.
    assert np.median(np.abs(stage2[:, 0] - THRESHOLD)) < 0.15
    assert np.isfinite(stage3).all()

    x_test, y_test = _held_out(cliff)
    staged = _forest_error(everything, cliff(everything), x_test, y_test)
    even = _sobol(160, seed=0)
    assert staged < _forest_error(even, cliff(even), x_test, y_test)


@pytest.mark.parametrize('weight, description', [
    (0.0, 'ignores the scores entirely'),
    (0.5, 'half uniform, half by score'),
    (1.0, 'by score'),
    (3.0, 'concentrated on the worst'),
])
def test_the_weight_dial_runs_from_uniform_to_concentrated(weight, description):
    """0 must be genuinely uniform, and raising it must genuinely concentrate."""
    x, y = _first_stage(n=120)
    drawn = gradient_weighted_design(x, y, 300, MINS, MAXS, seed=1, weight=weight)
    assert drawn.shape == (300, 2), description
    assert (drawn >= MINS).all() and (drawn <= MAXS).all()


def test_weight_zero_is_a_plain_random_top_up():
    """The documented meaning of 0: the scores are computed and then not used."""
    x, y = _first_stage()
    drawn = gradient_weighted_design(x, y, 400, MINS, MAXS, seed=1, weight=0.0)
    # Uniform over the box, so nowhere near as tight to the cliff as weight 1 gets.
    assert np.median(np.abs(drawn[:, 0] - THRESHOLD)) > 0.15


def test_a_higher_weight_concentrates_harder_than_a_lower_one():
    """Monotone in the dial, which is what makes it a dial rather than a switch."""
    x, y = _first_stage()
    loose = gradient_weighted_design(x, y, 400, MINS, MAXS, seed=1, weight=0.5)
    tight = gradient_weighted_design(x, y, 400, MINS, MAXS, seed=1, weight=4.0)
    assert (np.median(np.abs(tight[:, 0] - THRESHOLD))
            < np.median(np.abs(loose[:, 0] - THRESHOLD)))


def test_error_weighting_falls_back_when_there_is_nothing_to_learn_from():
    """A flat response, and too few points for the dimension, both fall back cleanly."""
    x = np.random.default_rng(0).random((60, 2))
    flat = error_weighted_design(x, np.ones((60, 2)), 100, MINS, MAXS, seed=1)
    assert flat.shape == (100, 2) and flat[:, 0].std() > 0.2

    tiny = np.random.default_rng(0).random((2, 2))
    assert error_weighted_design(tiny, _step_response(tiny), 20, MINS, MAXS,
                                 seed=1).shape == (20, 2)


# ---------------------------------------------------------------------------------
# What the reported R2 is measured on.
# ---------------------------------------------------------------------------------

from libcuflynx.emulators.emulator_trainer import _train_test_split  # noqa: E402


def test_only_space_filling_samples_are_held_out():
    """The validation set is drawn from the sobol stage, never from an adaptive one.

    An adaptive stage puts its points exactly where the model is hardest. Holding those
    out would measure the emulator on the worst corner of the box and report it as the
    error everywhere -- and report it inconsistently, since how many such points fell in
    the test set is chance. It would also waste them: they were simulated to teach the
    emulator the difficult region, and a held-out point teaches it nothing.
    """
    n_sobol, n_adaptive = 80, 40
    x = np.arange(n_sobol + n_adaptive, dtype=float).reshape(-1, 1)
    y = x.copy()
    space_filling = np.array([True] * n_sobol + [False] * n_adaptive)

    _, _, x_test, _ = _train_test_split(x, y, 0.2, seed=0, test_candidates=space_filling)

    held_out = x_test.ravel().astype(int)
    assert len(held_out) == 24, 'the test set should still be test_fraction of the design'
    assert (held_out < n_sobol).all(), (
        f'{(held_out >= n_sobol).sum()} adaptive samples were held out; validation has to '
        f'come from the evenly-spread stage')


def test_every_adaptive_sample_reaches_the_training_set():
    """The other half of the same claim: nothing hard-won is spent on validation."""
    n_sobol, n_adaptive = 60, 40
    x = np.arange(n_sobol + n_adaptive, dtype=float).reshape(-1, 1)
    space_filling = np.array([True] * n_sobol + [False] * n_adaptive)

    x_train, _, _, _ = _train_test_split(x, x.copy(), 0.2, seed=1,
                                         test_candidates=space_filling)
    trained_on = set(x_train.ravel().astype(int))
    assert set(range(n_sobol, n_sobol + n_adaptive)) <= trained_on


def test_a_single_stage_design_splits_exactly_as_it_always_did():
    """No mask, and an all-true mask, must both take the original path.

    Every existing emulator was validated by the unmasked split; a study that retrains
    without changing anything has to get the same split back, or its R2 moves for a
    reason that has nothing to do with the emulator.
    """
    x = np.arange(50, dtype=float).reshape(-1, 1)
    plain = _train_test_split(x, x.copy(), 0.2, seed=0)
    masked = _train_test_split(x, x.copy(), 0.2, seed=0,
                               test_candidates=np.ones(50, dtype=bool))
    for left, right in zip(plain, masked):
        assert np.array_equal(left, right)


def test_the_fit_is_never_starved_of_training_points():
    """A design that is almost all adaptive still leaves something to hold out."""
    space_filling = np.array([True] * 3 + [False] * 97)
    x = np.arange(100, dtype=float).reshape(-1, 1)
    x_train, _, x_test, _ = _train_test_split(x, x.copy(), 0.2, seed=0,
                                              test_candidates=space_filling)
    assert 1 <= len(x_test) <= 3
    assert len(x_train) == 100 - len(x_test)
