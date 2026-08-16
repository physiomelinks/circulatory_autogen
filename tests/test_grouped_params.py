"""Grouped params_for_id rows: one calibrated value driving several vessels (issue #355).

The feature had no tests at all, and was broken end to end: `zip(names, vals)` with N names and
one shared value stops at the shorter sequence, so only the first vessel was ever set. A grouped
row produced a cost curve bit-identical to the ungrouped one -- nothing downstream could tell it
was calibrating a single vessel.
"""
import numpy as np
import pytest

from libcuflynx.solver_wrappers.param_grouping import (
    as_name_list, as_value_list, pair_names_with_values)


@pytest.mark.unit
def test_one_shared_value_reaches_every_member_of_a_group():
    """The whole point of a grouped row. zip() silently dropped everything after the first."""
    pairs = pair_names_with_values(['aortic_root/C', 'par/C'], 1.5e-8)
    assert pairs == [('aortic_root/C', 1.5e-8), ('par/C', 1.5e-8)]


@pytest.mark.unit
def test_a_group_of_three_broadcasts_to_all_three():
    pairs = pair_names_with_values(['a/R', 'b/R', 'c/R'], 2.0)
    assert [n for n, _ in pairs] == ['a/R', 'b/R', 'c/R']
    assert {v for _, v in pairs} == {2.0}


@pytest.mark.unit
def test_an_ungrouped_entry_is_unchanged():
    assert pair_names_with_values('aortic_root/C', 1.5e-8) == [('aortic_root/C', 1.5e-8)]


@pytest.mark.unit
def test_per_name_values_are_paired_positionally():
    """N names with N values keeps the existing meaning -- broadcasting must not override it."""
    pairs = pair_names_with_values(['a/R', 'b/R'], [1.0, 2.0])
    assert pairs == [('a/R', 1.0), ('b/R', 2.0)]


@pytest.mark.unit
def test_a_length_mismatch_raises_instead_of_truncating():
    """The failure this whole issue was: zip() drops the tail without a word."""
    with pytest.raises(ValueError, match='3 name'):
        pair_names_with_values(['a/R', 'b/R', 'c/R'], [1.0, 2.0])


@pytest.mark.unit
def test_a_protocol_trace_key_is_one_value_not_a_string_of_characters():
    """A string value is a protocol_traces key. Splitting it per character would pair 'p','a','c'
    with the group's names and set three nonsense values."""
    pairs = pair_names_with_values(['a/u', 'b/u'], 'pace')
    assert pairs == [('a/u', 'pace'), ('b/u', 'pace')]


@pytest.mark.unit
def test_numpy_arrays_are_accepted_as_value_lists():
    pairs = pair_names_with_values(['a/R', 'b/R'], np.array([1.0, 2.0]))
    assert [v for _, v in pairs] == [1.0, 2.0]


@pytest.mark.unit
def test_a_zero_dimensional_numpy_scalar_broadcasts():
    pairs = pair_names_with_values(['a/R', 'b/R'], np.float64(3.0))
    assert [v for _, v in pairs] == [3.0, 3.0]


@pytest.mark.unit
def test_normalisers_agree_with_the_pairing():
    assert as_name_list('x') == ['x'] and as_name_list(['x', 'y']) == ['x', 'y']
    assert as_value_list('trace') == ['trace']
    assert as_value_list(1.0) == [1.0]


@pytest.mark.unit
def test_sobol_keeps_groups_for_setting_and_flattens_only_for_labels():
    """The SA fix: a group is one variable to the sampler but N parameters to set.

    Collapsing to the first name for both is what made a grouped SA vary one vessel while
    calibration varied all of them -- the two silently answered different questions.
    """
    from libcuflynx.sensitivity_analysis.sobolSA import sobol_SA

    sa = sobol_SA.__new__(sobol_SA)
    sa.param_id_info = {
        'param_names': [['aortic_root/C', 'par/C'], 'heart/R'],
        'param_mins': [1e-9, 1.0],
        'param_maxs': [5e-8, 2.0],
    }
    info = sa._create_SA_info('sobol', 8)

    # one variable per row for the sampler...
    assert info['param_labels'] == ['aortic_root/C+par/C', 'heart/R']
    assert sa.num_params == 2
    # ...but the full group survives for set_param_vals
    assert info['param_names'] == [['aortic_root/C', 'par/C'], 'heart/R']


@pytest.mark.unit
def test_the_casadi_backend_now_supports_groups_by_folding_per_member_derivatives():
    """CasADi used to refuse a grouped row, because its symbolic subset took one symbol per
    *row* and would have differentiated w.r.t. the first member alone.

    It now takes one symbol per *member* and folds the per-member derivatives back with each
    entry's chain-rule weights -- 1.0 for a shared-value group, the probed affine coefficient
    for a modifier -- which is the same combination the Myokit/FSA arm applies. The guard
    that remains is against an *unflattened* name reaching the helper, which would silently
    reintroduce the first-member-only bug.
    """
    from libcuflynx.param_id.casadi_backend import flatten_entries, fold_entry_rows

    info = {'param_names': [['a/C', 'b/C'], ['heart/R']]}
    flat, entry_map = flatten_entries(info)
    assert flat == ['a/C', 'b/C', 'heart/R']
    # a shared-value group carries unit weights: d/dtheta = sum_i d/dp_i
    assert entry_map == [[(0, 1.0), (1, 1.0)], [(2, 1.0)]]
    folded = fold_entry_rows(np.array([3.0, 7.0, 11.0]), entry_map, 2)
    assert list(folded) == [10.0, 11.0]
    # ... and the first-member-only answer is distinguishable, or this proves nothing
    assert folded[0] != 3.0
