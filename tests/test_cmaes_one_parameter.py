"""CMA-ES on one calibrated parameter finds the minimum of a smooth cost in its box.

In one dimension nevergrad's ``CMA`` selects its OnePlusOne search, which on a bounded variable
can keep proposing points on the bounds: with the box, start and budget below it ended more than
1% of the box from the minimum in about a tenth of the seeds (up to 60 of 80 candidates on a
bound). ``CMAESOptimiser`` gives nevergrad a second, unused variable so that it runs CMA.
"""
import numpy as np
import pytest

pytest.importorskip('nevergrad', reason='CMA-ES needs nevergrad')

from libcuflynx.param_id.optimisers import CMAESOptimiser  # noqa: E402

LO, HI = 5.0e-5, 1.15e-4
WIDTH = HI - LO
TRUE = LO + 0.71 * WIDTH
BUDGET = 80


class _OneParameterCost:
    """A smooth cost with one minimum, steeper above it than below, and the start value."""

    def __init__(self, start):
        self.param_init = np.array([start])
        self.points = []

    def get_cost_from_params(self, params):
        (x,) = np.ravel(params)
        self.points.append(float(x))
        width = (0.083 if x < TRUE else 0.055) * WIDTH
        return float(((x - TRUE) / width) ** 2)


def _calibrate(start, seed):
    cost = _OneParameterCost(start)
    info = {'param_names': [['a/k']], 'param_mins': [LO], 'param_maxs': [HI]}
    optimiser = CMAESOptimiser(cost, info, None, 1, None, optimiser_options={
        'num_calls_to_function': BUDGET, 'max_patience': BUDGET, 'cost_convergence': 0.0,
        'seed': seed})
    optimiser.run()
    return optimiser, cost


@pytest.mark.unit
@pytest.mark.parametrize('start', [LO + 0.31 * WIDTH, LO], ids=['start mid-box', 'start on a bound'])
def test_cma_es_finds_a_one_parameter_minimum_from_every_seed(start):
    errors, on_a_bound = [], []
    for seed in range(40):
        optimiser, cost = _calibrate(start, seed)
        assert np.shape(optimiser.best_param_vals) == (1,)
        errors.append(abs(float(optimiser.best_param_vals[0]) - TRUE) / WIDTH)
        on_a_bound.append(sum(x in (LO, HI) for x in cost.points[1:]))
        assert all(LO <= x <= HI for x in cost.points)
    assert max(errors) < 0.01, f'worst seed ends {max(errors):.1%} of the box from the minimum'
    assert max(on_a_bound) <= 5, f'{max(on_a_bound)} of {BUDGET} candidates on a bound'


@pytest.mark.unit
def test_only_a_one_parameter_search_gets_the_unused_variable():
    info = {'param_names': [['a/k'], ['b/k']], 'param_mins': [LO, 1.0], 'param_maxs': [HI, 2.0]}
    two = CMAESOptimiser(_OneParameterCost(LO), info, None, 2, None,
                         optimiser_options={'num_calls_to_function': 4})
    init, lower, upper = two._search_box(np.array([LO, 1.5]))
    assert list(init) == [LO, 1.5] and list(lower) == [LO, 1.0] and list(upper) == [HI, 2.0]

    info = {'param_names': [['a/k']], 'param_mins': [LO], 'param_maxs': [HI]}
    one = CMAESOptimiser(_OneParameterCost(LO), info, None, 1, None,
                         optimiser_options={'num_calls_to_function': 4})
    init, lower, upper = one._search_box(np.array([LO]))
    assert list(init) == [LO, 0.5] and list(lower) == [LO, 0.0] and list(upper) == [HI, 1.0]
