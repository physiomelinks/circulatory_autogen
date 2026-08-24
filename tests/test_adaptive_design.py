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
    assert _plan(num_train_samples=128) == [{'method': 'sobol', 'num_samples': 128}]
    assert _plan(num_train_samples=128, sample_type='latin_hypercube') == [
        {'method': 'latin_hypercube', 'num_samples': 128}]


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
    assert as_list == as_string == [{'method': 'sobol', 'num_samples': 600},
                                    {'method': 'gradient_weighted', 'num_samples': 400}]


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
