"""Modifier functions: a calibrated theta computes its targets through a registered,
user-definable function (issue #383).

    p_i = fn(theta, baseline_i, **inputs)

The motivating case: calibrate a total volume q_tot and derive
q_lv_init = q_tot - sum(other volumes) -- the built-in ``remainder`` function. Inputs are
model constants named by qname, resolved to their defaults once at setup (never re-derived,
like the baselines, so nothing compounds). Every function must be affine in theta so the
analytic gradients keep a constant chain-rule weight and theta's x0 is derivable by inversion.
"""
import numpy as np
import pytest

from libcuflynx.param_id import fsa_backend
from libcuflynx.param_id.modifier_funcs import (
    BUILTIN_MODIFIER_FUNCS, get_modifier_funcs, modifier_func, probe_affine)
from libcuflynx.parsers.PrimitiveParsers import (
    ObsAndParamDataParser, apply_modifier_identity_nominals, expand_modifier_param_vals,
    modifier_weights_by_index, param_modifiers, resolve_modifier_baselines)


def _parser(external_path=None):
    return ObsAndParamDataParser(modifier_funcs_external_path=external_path)


def _doc(**overrides):
    entry = {'name': 'q_tot', 'modifies': ['heart/q_lv_init'], 'modifier': 'remainder',
             'inputs': {'subtract': ['pvn/q_init', 'par/q_init']}, 'min': 4e-3, 'max': 6e-3}
    entry.update(overrides)
    return {'version': 1, 'params': [entry]}


def _info(doc, external_path=None):
    parser = _parser(external_path)
    from libcuflynx.param_id.modifier_funcs import get_modifier_funcs as gmf
    entries = parser.resolve_params_for_id_doc(doc, modifier_funcs=gmf(external_path))
    return parser._build_param_id_info_from_entries(entries)


class _FakeHelper:
    """Returns model defaults regardless of what has been 'written' -- the property both the
    baselines and the resolved inputs need."""

    def __init__(self, defaults):
        self.defaults = defaults

    def get_default_param_vals(self, param_names):
        return [[self.defaults[q] for q in group] if isinstance(group, list)
                else self.defaults[group] for group in param_names]


_DEFAULTS = {'heart/q_lv_init': 2e-3, 'pvn/q_init': 1e-3, 'par/q_init': 5e-4}


def _resolved_info(doc=None, defaults=_DEFAULTS):
    info = _info(doc or _doc())
    resolve_modifier_baselines(info, _FakeHelper(defaults))
    return info


# ------------------------------------------------------------------ registry


@pytest.mark.unit
def test_builtins_are_registered():
    funcs = get_modifier_funcs()
    assert 'scale' in funcs and 'remainder' in funcs
    assert funcs['scale'] is BUILTIN_MODIFIER_FUNCS['scale']


@pytest.mark.unit
def test_the_decorator_records_inputs_and_description():
    @modifier_func(inputs={'ref': 'float'}, description='d')
    def f(theta, baseline, ref):
        return theta + ref
    assert f.is_modifier_func and f.modifier_inputs == {'ref': 'float'}
    assert f.modifier_description == 'd'


@pytest.mark.unit
def test_an_invalid_input_type_is_refused_at_declaration():
    with pytest.raises(ValueError, match="must be one of"):
        modifier_func(inputs={'x': 'int'})


@pytest.mark.unit
def test_an_external_file_adds_functions(tmp_path):
    ext = tmp_path / 'my_modifiers.py'
    ext.write_text(
        "from libcuflynx.param_id.modifier_funcs import modifier_func\n"
        "@modifier_func(inputs={'ref': 'float'}, description='target = theta + ref')\n"
        "def offset_from(theta, baseline, ref):\n"
        "    return theta + ref\n"
        "def _helper_ignored(x):\n"
        "    return x\n"
        "def undecorated_ignored(theta, baseline):\n"
        "    return theta\n")
    funcs = get_modifier_funcs(str(ext))
    assert 'offset_from' in funcs
    assert '_helper_ignored' not in funcs and 'undecorated_ignored' not in funcs
    with pytest.raises(FileNotFoundError):
        get_modifier_funcs(str(tmp_path / 'missing.py'))


# ------------------------------------------------------------------ vocabulary (CUFLynx)


@pytest.mark.unit
def test_the_vocabulary_exposes_inputs_and_user_defined():
    """A front-end renders the modifier form from this: which functions exist, what inputs
    each takes and of what type, and whether it is user code."""
    ops = param_modifiers()
    assert ops['remainder']['inputs'] == {'subtract': 'list'}
    assert ops['remainder']['user_defined'] is False
    assert ops['remainder']['applies_to'] == 'value'
    # scale keeps its static UI metadata alongside the new keys
    assert ops['scale']['inputs'] == {}
    assert ops['scale']['identity'] == 1.0
    assert ops['scale']['default_min'] < ops['scale']['default_max']


@pytest.mark.unit
def test_external_functions_are_marked_user_defined(tmp_path):
    ext = tmp_path / 'm.py'
    ext.write_text(
        "from libcuflynx.param_id.modifier_funcs import modifier_func\n"
        "@modifier_func(inputs={})\n"
        "def my_op(theta, baseline):\n"
        "    return theta\n")
    ops = param_modifiers(str(ext))
    assert ops['my_op']['user_defined'] is True


# ------------------------------------------------------------------ entry validation


@pytest.mark.unit
def test_a_remainder_entry_parses_and_records_inputs():
    info = _info(_doc())
    (mod,) = info['modifiers']
    assert mod['modifier'] == 'remainder'
    assert mod['inputs'] == {'subtract': ['pvn/q_init', 'par/q_init']}
    assert mod['resolved_inputs'] is None and mod['affine'] is None


@pytest.mark.unit
def test_missing_declared_inputs_are_refused():
    with pytest.raises(ValueError, match="requires input"):
        _info(_doc(inputs=None))


@pytest.mark.unit
def test_undeclared_inputs_are_refused():
    with pytest.raises(ValueError, match="does not take input"):
        _info(_doc(inputs={'subtract': ['pvn/q_init'], 'typo': 'x/y'}))


@pytest.mark.unit
def test_a_float_input_takes_one_qname_not_a_list(tmp_path):
    ext = tmp_path / 'm.py'
    ext.write_text(
        "from libcuflynx.param_id.modifier_funcs import modifier_func\n"
        "@modifier_func(inputs={'ref': 'float'})\n"
        "def offset_from(theta, baseline, ref):\n"
        "    return theta + ref\n")
    doc = _doc(modifier='offset_from', inputs={'ref': ['a/x', 'b/x']})
    with pytest.raises(ValueError, match='single component/param qname'):
        _info(doc, external_path=str(ext))


@pytest.mark.unit
def test_inputs_without_modifies_are_refused():
    doc = {'version': 1, 'params': [
        {'name': 'p', 'targets': ['a/x'], 'inputs': {'ref': 'b/y'},
         'min': 0.0, 'max': 1.0}]}
    with pytest.raises(ValueError, match='sets "inputs" but has no "modifies"'):
        _info(doc)


# ------------------------------------------------------------------ the q_tot case, end to end


@pytest.mark.unit
def test_resolve_fills_inputs_and_affine_from_model_defaults():
    info = _resolved_info()
    (mod,) = info['modifiers']
    assert mod['baselines'] == [pytest.approx(2e-3)]
    assert mod['resolved_inputs'] == {'subtract': [pytest.approx(1e-3), pytest.approx(5e-4)]}
    # remainder: p = theta - sum(subtract) -> a=1, b=-1.5e-3
    assert mod['affine']['a'] == [pytest.approx(1.0)]
    assert mod['affine']['b'] == [pytest.approx(-1.5e-3)]


@pytest.mark.unit
def test_expansion_computes_the_target_from_theta():
    """q_tot = 5e-3 with 1.5e-3 of other volume -> q_lv_init = 3.5e-3."""
    info = _resolved_info()
    out = expand_modifier_param_vals(info, [5e-3])
    assert out == [[pytest.approx(3.5e-3)]]


@pytest.mark.unit
def test_expansion_does_not_compound_across_iterations():
    """The inputs are resolved defaults, so calling twice with the same theta gives the same
    targets -- the guarantee that made baselines resolve-once (#378)."""
    info = _resolved_info()
    first = expand_modifier_param_vals(info, [5e-3])
    second = expand_modifier_param_vals(info, [5e-3])
    assert first == second


@pytest.mark.unit
def test_theta_x0_is_the_inversion_at_the_baseline():
    """theta0 = q_lv_default + sum(others) = 2e-3 + 1.5e-3, so the run starts with the target
    at its model default."""
    info = _resolved_info()
    nominal = np.array([0.0])
    apply_modifier_identity_nominals(info, nominal)
    assert nominal[0] == pytest.approx(3.5e-3)
    # and expanding at that theta reproduces the default exactly
    assert expand_modifier_param_vals(info, [nominal[0]]) == [[pytest.approx(2e-3)]]


@pytest.mark.unit
def test_scale_x0_is_still_one_via_the_same_inversion():
    doc = {'version': 1, 'params': [
        {'name': 'C_scale', 'modifies': ['aortic_root/C', 'par/C'], 'modifier': 'scale',
         'min': 0.5, 'max': 2.0}]}
    info = _info(doc)
    resolve_modifier_baselines(info, _FakeHelper({'aortic_root/C': 1.2e-8, 'par/C': 3e-10}))
    nominal = np.array([99.0])
    apply_modifier_identity_nominals(info, nominal)
    assert nominal[0] == pytest.approx(1.0)


# ------------------------------------------------------------------ gradients


@pytest.mark.unit
def test_remainder_weights_are_one_and_scale_weights_are_baselines():
    info = _resolved_info()
    assert modifier_weights_by_index(info) == {0: [pytest.approx(1.0)]}

    doc = {'version': 1, 'params': [
        {'name': 'C_scale', 'modifies': ['aortic_root/C', 'par/C'], 'modifier': 'scale',
         'min': 0.5, 'max': 2.0}]}
    scale_info = _info(doc)
    resolve_modifier_baselines(scale_info, _FakeHelper({'aortic_root/C': 2.0, 'par/C': 0.5}))
    assert modifier_weights_by_index(scale_info) == {0: [pytest.approx(2.0),
                                                         pytest.approx(0.5)]}


@pytest.mark.unit
def test_the_probed_weights_reach_the_fsa_chain_rule():
    """ensure_setup consumes modifier_weights_by_index, so a remainder entry's members carry
    weight a=1 into the combined sensitivity -- no fsa_backend change needed."""
    info = _resolved_info(_doc(modifies=['heart/q_lv_init', 'pvn/q_init'],
                               inputs={'subtract': ['par/q_init']}))

    class _Pid:
        param_id_info = info
        obs_info = {"operands": [['v']], "const_idx_to_obs_idx": [0]}

        class sim_helper:
            _fsa_sensitivities_history = []

            @staticmethod
            def enable_fsa(deps, indeps):
                return []
    pid = _Pid()
    fsa_backend.ensure_setup(pid)
    assert pid._fsa_entry_members == [[('heart/q_lv_init', pytest.approx(1.0)),
                                       ('pvn/q_init', pytest.approx(1.0))]]


@pytest.mark.unit
def test_a_non_affine_function_is_refused_at_resolve(tmp_path):
    ext = tmp_path / 'm.py'
    ext.write_text(
        "from libcuflynx.param_id.modifier_funcs import modifier_func\n"
        "@modifier_func(inputs={})\n"
        "def squared(theta, baseline):\n"
        "    return theta * theta * baseline\n")
    doc = _doc(modifier='squared', inputs=None)
    doc['params'][0].pop('inputs')
    info = _info(doc, external_path=str(ext))
    info['modifier_funcs_external_path'] = str(ext)
    with pytest.raises(NotImplementedError, match='not affine in theta'):
        resolve_modifier_baselines(info, _FakeHelper(_DEFAULTS))


@pytest.mark.unit
def test_probe_affine_accepts_affine_and_returns_coefficients():
    a, b = probe_affine(lambda t, base: 3.0 * t - 2.0, 0.0, {}, 'lin')
    assert (a, b) == (pytest.approx(3.0), pytest.approx(-2.0))


# ------------------------------------------------------------------ naming: modifier vs operation
#
# "Operations" act on *outputs* and are declared in obs_data; what acts on *parameters* is a
# modifier, declared in params_for_id. The #378 spelling of the key was 'operation', which
# collided with that vocabulary; it is still accepted, with a warning.


@pytest.mark.unit
def test_the_entry_key_is_modifier():
    info = _info(_doc())
    (mod,) = info['modifiers']
    assert mod['modifier'] == 'remainder'
    assert 'operation' not in mod


@pytest.mark.unit
def test_the_operation_key_is_accepted_as_a_deprecated_alias():
    doc = _doc()
    doc['params'][0]['operation'] = doc['params'][0].pop('modifier')
    with pytest.warns(UserWarning, match='"operation" is deprecated'):
        info = _info(doc)
    assert info['modifiers'][0]['modifier'] == 'remainder'


@pytest.mark.unit
def test_setting_both_modifier_and_operation_is_refused():
    doc = _doc()
    doc['params'][0]['operation'] = 'scale'
    with pytest.raises(ValueError, match='sets both "modifier" and "operation"'):
        _info(doc)


@pytest.mark.unit
def test_a_legacy_modifiers_record_still_resolves():
    """A param_modifiers.json written by a #378-era run names the function under 'operation'."""
    info = _info(_doc())
    (mod,) = info['modifiers']
    mod['operation'] = mod.pop('modifier')
    resolve_modifier_baselines(info, _FakeHelper(_DEFAULTS))
    assert expand_modifier_param_vals(info, [5e-3]) == [[pytest.approx(3.5e-3)]]
