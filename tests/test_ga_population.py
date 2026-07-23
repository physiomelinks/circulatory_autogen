"""Unit tests for the user-configurable genetic-algorithm population sizing
(num_elite / num_survivors / num_mutations_per_survivor / num_cross_breed)."""
import pytest

from param_id.optimisers import GeneticAlgorithmOptimiser


def _make_ga(options, debug):
    # Only optimiser_options and DEBUG are consulted by _population_sizes; the rest can be dummies.
    return GeneticAlgorithmOptimiser(
        param_id_obj=None, param_id_info=None, param_norm_obj=None,
        num_params=3, output_dir=None, optimiser_options=dict(options), DEBUG=debug)


@pytest.mark.unit
def test_ga_population_defaults_match_debug_flag():
    """With nothing set, the sizes are the historical DEBUG-dependent defaults (unchanged
    behaviour): the small 'quick' sizes under DEBUG, the full production sizes otherwise."""
    assert _make_ga({}, False)._population_sizes() == {
        'num_elite': 12, 'num_survivors': 48, 'num_mutations_per_survivor': 12,
        'num_cross_breed': 120}
    assert _make_ga({}, True)._population_sizes() == {
        'num_elite': 4, 'num_survivors': 6, 'num_mutations_per_survivor': 2,
        'num_cross_breed': 10}


@pytest.mark.unit
def test_ga_population_user_overrides_and_none_falls_back():
    ga = _make_ga({'num_survivors': 10, 'num_cross_breed': 20, 'num_mutations_per_survivor': 3,
                   'num_elite': None}, debug=False)
    sizes = ga._population_sizes()
    # explicit values win
    assert (sizes['num_survivors'], sizes['num_cross_breed'], sizes['num_mutations_per_survivor']) \
        == (10, 20, 3)
    # an explicit None (or an omitted key) falls back to the DEBUG-dependent default
    assert sizes['num_elite'] == 12
    # population = num_survivors + num_survivors*num_mutations_per_survivor + num_cross_breed
    num_pop = (sizes['num_survivors'] + sizes['num_survivors'] * sizes['num_mutations_per_survivor']
               + sizes['num_cross_breed'])
    assert num_pop == 10 + 10 * 3 + 20
