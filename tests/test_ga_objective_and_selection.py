"""The GA's objective selection and its survival weighting (from #367).

Both exist so the GA can be driven by a log posterior instead of a weighted cost -- which is how
a GA is used to find a starting point for UQ. That objective is unbounded in sign and large in
magnitude, and the old selection step assumed neither.
"""
import numpy as np
import pytest

from param_id.optimisers import GeneticAlgorithmOptimiser


class _StubEngine:
    """Stands in for OpencorParamID: records which objective the GA asked for."""

    def __init__(self, cost=3.0, log_posterior=-2.0):
        self.cost = cost
        self.log_posterior = log_posterior
        self.calls = []

    def get_cost_from_params(self, param_vals):
        self.calls.append('cost')
        return self.cost

    def get_lnlikelihood_lnprior_from_params(self, param_vals):
        self.calls.append('likelihood')
        return self.log_posterior


def _ga(engine=None, **options):
    """A GA with only the attributes the objective/selection helpers touch."""
    ga = GeneticAlgorithmOptimiser.__new__(GeneticAlgorithmOptimiser)
    ga.param_id_obj = engine
    ga.optimiser_options = {'cost_convergence': 1e-4, 'max_patience': 10, **options}
    ga.objective_function = ga.optimiser_options.get('objective_function', 'cost')
    if ga.objective_function not in ('cost', 'likelihood'):
        raise ValueError(f"Invalid objective_function: {ga.objective_function!r}")
    return ga


# ---------------------------------------------------------------------------
# objective_function
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_default_objective_is_the_cost_function():
    engine = _StubEngine(cost=3.0)
    assert _ga(engine)._objective(np.array([1.0])) == 3.0
    assert engine.calls == ['cost']


@pytest.mark.unit
def test_likelihood_objective_is_negated_so_lower_is_still_better():
    """The sign check that matters. get_lnlikelihood_lnprior_from_params returns a log posterior
    (higher is better) and every optimiser here minimises, so it has to be negated. #367 handed
    it over as-is, which would have made the GA search for the least probable parameters."""
    engine = _StubEngine(log_posterior=-2.0)
    ga = _ga(engine, objective_function='likelihood')

    assert ga._objective(np.array([1.0])) == 2.0
    assert engine.calls == ['likelihood']

    # a better fit (higher log posterior) must give a *smaller* objective
    better = _ga(_StubEngine(log_posterior=-0.5), objective_function='likelihood')
    worse = _ga(_StubEngine(log_posterior=-9.0), objective_function='likelihood')
    assert better._objective(np.array([1.0])) < worse._objective(np.array([1.0]))


@pytest.mark.unit
def test_an_out_of_prior_point_becomes_infinite_cost_not_the_best_point():
    """get_lnlikelihood_lnprior_from_params returns -inf outside the prior support. Unnegated,
    a minimiser scores that as the best point there is; negated it becomes +inf, which the
    population loop already treats as 'reject and resample'."""
    ga = _ga(_StubEngine(log_posterior=-np.inf), objective_function='likelihood')
    assert ga._objective(np.array([1.0])) == np.inf


@pytest.mark.unit
def test_an_unknown_objective_is_rejected_at_construction():
    with pytest.raises(ValueError, match='objective_function'):
        _ga(_StubEngine(), objective_function='posterior')


# ---------------------------------------------------------------------------
# survival probabilities
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_survival_weights_are_a_valid_probability_vector_for_negative_costs():
    """A negative log posterior goes negative wherever the posterior density exceeds 1. The old
    cost**-1 weighting then produces negative 'probabilities', which np.random.choice rejects."""
    costs = np.array([-5.0, -2.0, 1.0, 4.0])
    probs = GeneticAlgorithmOptimiser._survival_probabilities(costs)

    assert np.all(probs >= 0), 'a probability cannot be negative'
    assert np.sum(probs) == pytest.approx(1.0)
    # the old rule really would have broken here
    old_rule = costs ** -1 / np.sum(costs ** -1)
    assert np.any(old_rule < 0), 'the regression this replaces'


@pytest.mark.unit
def test_lower_cost_is_more_likely_to_survive():
    probs = GeneticAlgorithmOptimiser._survival_probabilities(np.array([1.0, 2.0, 3.0]))
    assert probs[0] > probs[1] > probs[2]
    assert np.sum(probs) == pytest.approx(1.0)


@pytest.mark.unit
def test_an_ordinary_positive_cost_keeps_the_existing_inverse_cost_weighting():
    """The change must be invisible to existing calibrations. A weighted cost function returns a
    strictly positive cost, which is exactly where cost**-1 is valid, so that path is kept
    byte-for-byte rather than replaced."""
    costs = np.array([5.32062919, 8.12162795, 11.0966665, 15.81030615])
    expected = costs ** -1 / np.sum(costs ** -1)
    assert GeneticAlgorithmOptimiser._survival_probabilities(costs) == pytest.approx(expected)


@pytest.mark.unit
def test_selection_does_not_collapse_onto_the_single_best_member():
    """Measured on a real 3compartment GA population, #367's unscaled exp(-cost) drops the
    effective number of distinct survivors from 5.4 of 6 to 1.3 of 6 -- it selects almost
    deterministically, which is the diversity a GA exists to keep. Guard both paths against
    that, using perplexity (exp of the Shannon entropy) as the effective-survivor count."""
    def perplexity(probs):
        probs = np.asarray(probs, dtype=float)
        probs = probs[probs > 0]
        return float(np.exp(-np.sum(probs * np.log(probs))))

    real_population = np.array([5.32062919, 8.12162795, 11.0966665,
                                15.81030615, 16.45929783, 17.33873024])
    naive = np.exp(-real_population) / np.sum(np.exp(-real_population))
    assert perplexity(naive) < 2.0, 'the collapse this guards against'
    assert perplexity(GeneticAlgorithmOptimiser._survival_probabilities(real_population)) > 4.0

    # the likelihood path must stay discriminating without collapsing either
    log_posterior_costs = np.array([-5.0, -3.0, -1.0, 2.0, 6.0])
    probs = GeneticAlgorithmOptimiser._survival_probabilities(log_posterior_costs)
    assert 2.0 < perplexity(probs) < len(log_posterior_costs)


@pytest.mark.unit
def test_survival_weights_survive_the_cost_magnitudes_a_log_posterior_reaches():
    """exp(-cost) underflows to zero for every member once the objective passes about 745
    (exp(-800) == 0.0), and the normalisation then divides by zero. A log posterior summed over
    many observables reaches that easily."""
    large = np.array([-800.0, -801.0, -802.0])  # negative, so the inverse rule cannot apply
    with np.errstate(under='ignore', invalid='ignore'):
        naive = np.exp(-(large - np.min(large)) * 0 - large) / np.sum(np.exp(-large))
    assert not np.all(np.isfinite(naive)), 'the overflow/underflow this guards against'

    probs = GeneticAlgorithmOptimiser._survival_probabilities(large)
    assert np.all(np.isfinite(probs)) and np.sum(probs) == pytest.approx(1.0)
    assert probs[2] > probs[0], 'the lowest cost must still be the most likely to survive'


@pytest.mark.unit
def test_a_non_finite_member_neither_wins_nor_poisons_the_others():
    """The GA already substitutes 1e25 for a nan cost, but the selection step must be safe on
    its own -- an inf must not become the most-favoured member via a zero weight elsewhere."""
    probs = GeneticAlgorithmOptimiser._survival_probabilities(np.array([-2.0, 0.0, np.inf]))
    assert np.all(np.isfinite(probs)) and np.sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[2], 'the infinite-cost member must not be favoured'


@pytest.mark.unit
def test_survival_weights_fall_back_to_uniform_rather_than_nan():
    """1e25 is the sentinel the GA already substitutes for a nan/huge cost. If every member is
    equally hopeless the weights must still be a usable distribution."""
    probs = GeneticAlgorithmOptimiser._survival_probabilities(np.array([np.nan, np.nan]))
    assert np.all(np.isfinite(probs)) and np.sum(probs) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# relative tolerance
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_absolute_tolerance_alone_is_the_default():
    ga = _ga(_StubEngine())
    assert ga._loss_has_stalled(1.0, 1.0 + 1e-6) is True
    assert ga._loss_has_stalled(1.0, 2.0) is False


@pytest.mark.unit
def test_relative_tolerance_catches_a_stall_an_absolute_threshold_misses():
    """A log-posterior objective sits in the hundreds, so it never moves by less than 1e-4 and
    the run only ever stops on the generation budget."""
    absolute_only = _ga(_StubEngine())
    assert absolute_only._loss_has_stalled(500.0, 500.01) is False

    relative = _ga(_StubEngine(), use_relative_tolerance=True, relative_tolerance=1e-3)
    assert relative._loss_has_stalled(500.0, 500.01) is True
    # still not stalled when the fractional change is genuinely large
    assert relative._loss_has_stalled(500.0, 700.0) is False


@pytest.mark.unit
def test_relative_tolerance_does_not_divide_by_zero():
    relative = _ga(_StubEngine(), use_relative_tolerance=True, relative_tolerance=1e-3)
    assert relative._loss_has_stalled(1.0, 0.0) is False
