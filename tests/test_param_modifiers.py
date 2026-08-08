"""Modifier parameters: one calibrated value that scales several model parameters.

A modifier occupies one slot in the optimiser's vector but names N model parameters, applying
`theta * baseline_i` to each. To every consumer it looks exactly like a grouped row -- one
variable to the sampler, N parameters to set -- which is the distinction #376 already drew.
"""
import numpy as np
import pytest

from parsers.PrimitiveParsers import (
    DEFAULT_PARAM_MODIFIER_OPERATION, ObsAndParamDataParser, PARAM_MODIFIER_OPERATIONS,
    expand_modifier_param_vals, param_modifier_operations, resolve_modifier_baselines)


def _parser():
    return ObsAndParamDataParser()


def _doc(**overrides):
    entry = {'name': 'compliance_scale', 'modifies': ['aortic_root/C', 'par/C'],
             'operation': 'scale', 'min': 0.5, 'max': 2.0}
    entry.update(overrides)
    return {'version': 1, 'params': [entry]}


def _info(doc):
    parser = _parser()
    return parser._build_param_id_info_from_entries(parser.resolve_params_for_id_doc(doc))


class _FakeHelper:
    """Returns model defaults regardless of what has been 'written' -- the property a baseline
    needs, and the one CasADi's get_init_param_vals does not have."""

    def __init__(self, defaults):
        self.defaults = defaults

    def get_default_param_vals(self, param_names):
        return [[self.defaults[q] for q in group] if isinstance(group, list)
                else self.defaults[group] for group in param_names]


# ------------------------------------------------------------------ vocabulary


@pytest.mark.unit
def test_the_operation_vocabulary_is_exported_as_data():
    """A front-end builds its menu by reading this rather than hardcoding CA's vocabulary."""
    ops = param_modifier_operations()
    assert 'scale' in ops
    assert DEFAULT_PARAM_MODIFIER_OPERATION in ops
    meta = ops['scale']
    for key in ('description', 'applies_to', 'dimensionless', 'default_min', 'default_max'):
        assert key in meta, key
    assert meta['dimensionless'] is True
    assert meta['default_min'] < meta['default_max']


@pytest.mark.unit
def test_the_exported_vocabulary_is_a_copy():
    """A consumer mutating what it introspected must not change CA's own table."""
    param_modifier_operations()['scale']['default_min'] = 999
    assert PARAM_MODIFIER_OPERATIONS['scale']['default_min'] != 999


# ------------------------------------------------------------------ parsing


@pytest.mark.unit
def test_a_modifier_looks_like_a_grouped_row_to_param_names():
    info = _info(_doc())
    assert info['param_names'] == [['aortic_root/C', 'par/C']]


@pytest.mark.unit
def test_a_modifier_is_labelled_by_its_own_name_not_by_its_targets():
    """theta is its own quantity -- dimensionless for scale -- so labelling it with a target's
    name would misreport what was calibrated."""
    info = _info(_doc())
    assert info['param_labels'] == ['compliance_scale']
    assert '+' not in info['param_labels'][0]


@pytest.mark.unit
def test_a_modifier_uses_name_for_plotting_when_given():
    info = _info(_doc(name_for_plotting=r'\theta_{C}'))
    assert info['param_labels'] == [r'\theta_{C}']


@pytest.mark.unit
def test_the_modifiers_record_carries_what_a_downstream_tool_needs():
    info = _info(_doc())
    assert len(info['modifiers']) == 1
    mod = info['modifiers'][0]
    assert mod['index'] == 0
    assert mod['name'] == 'compliance_scale'
    assert mod['operation'] == 'scale'
    assert mod['targets'] == ['aortic_root/C', 'par/C']
    # unresolved is None rather than absent, so a consumer can tell it apart from "no baseline"
    assert mod['baselines'] is None


@pytest.mark.unit
def test_operation_defaults_to_scale():
    doc = _doc()
    del doc['params'][0]['operation']
    assert _info(doc)['modifiers'][0]['operation'] == DEFAULT_PARAM_MODIFIER_OPERATION


@pytest.mark.unit
def test_the_index_is_computed_after_filtering():
    """idxs_to_ignore shifts positions, and a stale index would point a slider at the wrong
    parameter."""
    parser = _parser()
    doc = {'version': 1, 'params': [
        {'name': 'free', 'targets': ['x/C'], 'min': 1, 'max': 2},
        {'name': 'scale_it', 'modifies': ['y/C'], 'min': 0.5, 'max': 2.0},
    ]}
    entries = parser.resolve_params_for_id_doc(doc)
    info = parser._build_param_id_info_from_entries(entries, idxs_to_ignore=[0])
    assert info['param_labels'] == ['scale_it']
    assert info['modifiers'][0]['index'] == 0


# ------------------------------------------------------------------ hard errors


@pytest.mark.unit
def test_a_modified_parameter_may_not_also_be_free():
    """(theta, p) and (theta*k, p/k) give an identical cost, so the optimiser wanders a flat
    ridge and both reported values are meaningless."""
    parser = _parser()
    with pytest.raises(ValueError, match='structurally unidentifiable'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'free_C', 'targets': ['aortic_root/C'], 'min': 1e-9, 'max': 5e-8},
            {'name': 'scale_C', 'modifies': ['aortic_root/C'], 'min': 0.5, 'max': 2.0},
        ]})


@pytest.mark.unit
def test_the_error_names_both_entries():
    parser = _parser()
    with pytest.raises(ValueError, match='free_C'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'free_C', 'targets': ['aortic_root/C'], 'min': 1e-9, 'max': 5e-8},
            {'name': 'scale_C', 'modifies': ['aortic_root/C'], 'min': 0.5, 'max': 2.0},
        ]})


@pytest.mark.unit
def test_a_modifier_may_not_modify_another_modifier():
    parser = _parser()
    # `modifies` holds qnames, and a modifier's name defaults to its first modified qname, so a
    # chain is reachable whenever a modifier's name is qname-shaped.
    with pytest.raises(ValueError, match='itself a modifier'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'x/C', 'modifies': ['a/C'], 'min': 0.5, 'max': 2.0},
            {'name': 'outer', 'modifies': ['x/C'], 'min': 0.5, 'max': 2.0},
        ]})


@pytest.mark.unit
def test_two_modifiers_may_not_act_on_the_same_parameter():
    """p = theta_1 * theta_2 * baseline: only the product is identifiable, so each factor alone
    is meaningless -- the same flat ridge as a modified parameter that is also free."""
    parser = _parser()
    with pytest.raises(ValueError, match='modified by both'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'first', 'modifies': ['a/C'], 'min': 0.5, 'max': 2.0},
            {'name': 'second', 'modifies': ['a/C'], 'min': 0.5, 'max': 2.0},
        ]})


@pytest.mark.unit
def test_an_entry_may_not_set_both_targets_and_modifies():
    parser = _parser()
    with pytest.raises(ValueError, match='both "targets" and "modifies"'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'both', 'targets': ['a/C'], 'modifies': ['b/C'], 'min': 1, 'max': 2}]})


@pytest.mark.unit
def test_operation_without_modifies_is_refused():
    parser = _parser()
    with pytest.raises(ValueError, match='no "modifies"'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'x', 'targets': ['a/C'], 'operation': 'scale', 'min': 1, 'max': 2}]})


@pytest.mark.unit
def test_an_unknown_operation_is_refused():
    parser = _parser()
    with pytest.raises(ValueError, match='unknown operation'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'x', 'modifies': ['a/C'], 'operation': 'offset', 'min': 1, 'max': 2}]})


@pytest.mark.unit
def test_a_scale_range_crossing_zero_warns():
    """A multiplier range straddling zero flips the sign of every target somewhere inside it."""
    parser = _parser()
    with pytest.warns(UserWarning, match='crossing zero'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'x', 'modifies': ['a/C'], 'min': -1.0, 'max': 2.0}]})


# ------------------------------------------------------------------ baselines and expansion


@pytest.mark.unit
def test_baselines_are_resolved_from_the_model_defaults():
    info = _info(_doc())
    helper = _FakeHelper({'aortic_root/C': 1.2028e-08, 'par/C': 3.09077e-10})
    resolve_modifier_baselines(info, helper)
    assert info['modifiers'][0]['baselines'] == [1.2028e-08, 3.09077e-10]


@pytest.mark.unit
def test_theta_expands_to_one_value_per_target():
    info = _info(_doc())
    resolve_modifier_baselines(info, _FakeHelper({'aortic_root/C': 2.0, 'par/C': 5.0}))
    assert expand_modifier_param_vals(info, [1.5]) == [[3.0, 7.5]]


@pytest.mark.unit
def test_free_parameters_pass_through_the_expansion_untouched():
    parser = _parser()
    doc = {'version': 1, 'params': [
        {'name': 'free', 'targets': ['x/C'], 'min': 1, 'max': 2},
        {'name': 'scale_it', 'modifies': ['y/C', 'z/C'], 'min': 0.5, 'max': 2.0},
    ]}
    info = parser._build_param_id_info_from_entries(parser.resolve_params_for_id_doc(doc))
    resolve_modifier_baselines(info, _FakeHelper({'y/C': 10.0, 'z/C': 100.0}))
    assert expand_modifier_param_vals(info, [7.0, 2.0]) == [7.0, [20.0, 200.0]]


@pytest.mark.unit
def test_scaling_does_not_compound_across_iterations():
    """The highest-value test here: theta must always multiply the *model default*, never the
    value the previous iteration wrote.

    Re-deriving baselines from the live parameter array would give theta=1.2 twice as 1.44x,
    silently corrupting a long optimisation. This is not hypothetical -- CasADi's
    get_init_param_vals reads the live array, which is why baselines are resolved once from the
    frozen snapshot and never again.
    """
    info = _info(_doc())
    helper = _FakeHelper({'aortic_root/C': 1.0, 'par/C': 2.0})
    resolve_modifier_baselines(info, helper)

    first = expand_modifier_param_vals(info, [1.2])
    # a second calibration step at the same theta must give the same absolute values
    second = expand_modifier_param_vals(info, [1.2])
    assert first == second == [[1.2, 2.4]]
    # and re-resolving must not move the baselines either
    resolve_modifier_baselines(info, helper)
    assert info['modifiers'][0]['baselines'] == [1.0, 2.0]
    assert expand_modifier_param_vals(info, [1.2]) == [[1.2, 2.4]]


@pytest.mark.unit
def test_expanding_before_baselines_are_resolved_raises():
    """Silently expanding with a default of 1.0 would calibrate the wrong quantity."""
    info = _info(_doc())
    with pytest.raises(ValueError, match='no resolved baselines'):
        expand_modifier_param_vals(info, [1.5])


@pytest.mark.unit
def test_a_backend_without_default_param_vals_raises_clearly():
    info = _info(_doc())

    class _NoDefaults:
        pass

    with pytest.raises(NotImplementedError, match='get_default_param_vals'):
        resolve_modifier_baselines(info, _NoDefaults())


@pytest.mark.unit
def test_expansion_is_a_noop_without_modifiers():
    parser = _parser()
    info = parser._build_param_id_info_from_entries(parser.resolve_params_for_id_doc(
        {'version': 1, 'params': [{'name': 'a', 'targets': ['x/C'], 'min': 1, 'max': 2}]}))
    assert info['modifiers'] == []
    assert expand_modifier_param_vals(info, [3.0]) == [3.0]


@pytest.mark.unit
def test_the_expanded_values_pair_positionally_with_the_target_names():
    """The seam with #376: N names against N values is paired positionally, which is why no
    backend needs to know modifiers exist."""
    from solver_wrappers.param_grouping import pair_names_with_values

    info = _info(_doc())
    resolve_modifier_baselines(info, _FakeHelper({'aortic_root/C': 2.0, 'par/C': 5.0}))
    names = info['param_names'][0]
    values = expand_modifier_param_vals(info, [3.0])[0]
    assert pair_names_with_values(names, values) == [('aortic_root/C', 6.0), ('par/C', 15.0)]
