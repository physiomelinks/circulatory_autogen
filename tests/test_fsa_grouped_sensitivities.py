"""The FSA chain rule for grouped and modifier params_for_id entries.

Since #376 ``set_param_vals`` moves every member of a grouped row, so on the Myokit/FSA path
the cost is a function of all members. The gradient must follow:

    ungrouped            dJ/dp
    shared-value group   dJ/dtheta = sum_i  dJ/dp_i
    scale modifier       dJ/dtheta = sum_i (dJ/dp_i) * baseline_i

These tests discriminate **by construction**, not empirically: an FSA-vs-FD comparison on a
real model cannot catch a first-member-only gradient when the members' influences differ by
orders of magnitude (on 3compartment [aortic_root/C, par/C] the wrong and right answers agree
to 1e-5 relative). Here the member sensitivities are synthetic and comparable, so the plain
sum, the weighted sum, and the first member alone are three numbers far apart.
"""
import numpy as np
import pytest

from libcuflynx.param_id import fsa_backend
from libcuflynx.parsers.PrimitiveParsers import (
    apply_modifier_identity_nominals, modifier_weights_by_index, param_entry_labels)


# Two members with *comparable* influence -- see the module docstring for why this matters.
_S_A = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_S_B = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
_TRACE = np.array([0.5, 0.6, 0.7, 0.8, 0.9])


class _FakeSimHelper:
    def __init__(self, sens_map, eligible):
        self._sens_map = sens_map            # {dep_name: {member: trace}}
        self._fsa_eligible_param_names = list(eligible)
        self._fsa_chain_rule_map = {}
        self._fsa_sensitivities_history = [object()]   # one sub-experiment
        self.enable_fsa_received = None

    def enable_fsa(self, dependent_names, independent_param_names):
        self.enable_fsa_received = list(independent_param_names)
        return []

    def get_sensitivities(self, dependent_names, param_names, sensitivities=None):
        return {d: {p: self._sens_map[d][p].copy()
                    for p in param_names if p in self._sens_map.get(d, {})}
                for d in dependent_names}


class _FakePid:
    """The surface fsa_backend calls back into, over one observable ('v') in one
    sub-experiment, with cost J = sum(v) and feature mean(v) -- both linear in the trace, so
    the central directional difference is exact and the expected numbers are integers."""

    def __init__(self, param_id_info, eligible=('a/C', 'b/C')):
        self.param_id_info = param_id_info
        self.obs_info = {"operands": [['v']], "const_idx_to_obs_idx": [0]}
        self.protocol_info = {"num_experiments": 1, "num_sub_per_exp": [1],
                              "sim_times": [[1.0]]}
        self.sim_helper = _FakeSimHelper({'v': {'a/C': _S_A, 'b/C': _S_B}}, eligible)

    def get_cost_obs_and_pred_from_params(self, param_vals, reset=True, only_one_exp=0):
        return 0.0, [[[_TRACE.copy()]]], []

    def get_cost_from_operands(self, operands, exp_idx=0, sub_idx=0):
        return float(np.sum(operands[0][0]))

    def get_cost_from_params(self, param_vals):
        raise AssertionError("the analytic path must not fall back to full-cost FD here")

    def _total_weighted_obs_denominator(self):
        return 1.0

    def _observable_label(self, obs_idx):
        return f"obs{obs_idx}"

    def get_obs_output_dict(self, operands):
        return {'const': [float(np.mean(operands[0][0]))]}


def _grouped_info():
    return {"param_names": [['a/C', 'b/C']], "param_labels": ['a/C+b/C']}


def _modifier_info(baselines=(2.0, 0.5)):
    return {"param_names": [['a/C', 'b/C']], "param_labels": ['theta_C'],
            "modifiers": [{"index": 0, "name": "theta_C", "operation": 'scale',
                           "targets": ['a/C', 'b/C'], "baselines": list(baselines)}]}


# dJ/dp_i for J = sum(v) is sum(S_i).
_DJ_DA = float(np.sum(_S_A))    # 15
_DJ_DB = float(np.sum(_S_B))    # 150


# ------------------------------------------------------------------ setup / flattening


@pytest.mark.unit
def test_every_member_reaches_enable_fsa():
    """The bug was structural: only n[0] of each group was handed to CVODES, so no column ever
    existed for the other members. All members must be independents."""
    info = {"param_names": [['a/C', 'b/C'], ['heart/R']],
            "param_labels": ['a/C+b/C', 'heart/R']}
    pid = _FakePid(info)
    fsa_backend.ensure_setup(pid)
    assert pid.sim_helper.enable_fsa_received == ['a/C', 'b/C', 'heart/R']
    assert pid._fsa_param_names_flat == ['a/C', 'b/C', 'heart/R']
    assert pid._fsa_entry_members == [[('a/C', 1.0), ('b/C', 1.0)], [('heart/R', 1.0)]]


@pytest.mark.unit
def test_modifier_members_carry_their_baselines_as_weights():
    pid = _FakePid(_modifier_info(baselines=(2.0, 0.5)))
    fsa_backend.ensure_setup(pid)
    assert pid._fsa_entry_members == [[('a/C', 2.0), ('b/C', 0.5)]]


@pytest.mark.unit
def test_unresolved_baselines_refuse_setup():
    """A weight guessed at is a gradient silently wrong, the exact failure this work removes."""
    info = _modifier_info()
    info["modifiers"][0]["baselines"] = None
    with pytest.raises(ValueError, match="baselines"):
        fsa_backend.ensure_setup(_FakePid(info))


@pytest.mark.unit
def test_modifier_weights_by_index_ignores_plain_entries():
    assert modifier_weights_by_index(_grouped_info()) == {}
    assert modifier_weights_by_index(_modifier_info()) == {0: [2.0, 0.5]}


# ------------------------------------------------------------------ the chain rule itself


@pytest.mark.unit
def test_a_shared_value_group_gradient_is_the_plain_sum():
    pid = _FakePid(_grouped_info())
    grad = fsa_backend.get_jac_cost(pid, np.array([1.0]))
    assert grad == pytest.approx([_DJ_DA + _DJ_DB])          # 165, not 15


@pytest.mark.unit
def test_a_modifier_gradient_is_the_baseline_weighted_sum():
    pid = _FakePid(_modifier_info(baselines=(2.0, 0.5)))
    grad = fsa_backend.get_jac_cost(pid, np.array([1.0]))
    assert grad == pytest.approx([2.0 * _DJ_DA + 0.5 * _DJ_DB])   # 105, not 15 or 165


@pytest.mark.unit
def test_the_first_member_alone_is_not_the_answer():
    """The pre-fix behaviour, asserted unreachable. Kept explicit because the natural
    empirical test (FSA vs FD on a real model) could not tell these apart."""
    for info in (_grouped_info(), _modifier_info()):
        grad = fsa_backend.get_jac_cost(_FakePid(info), np.array([1.0]))
        assert grad[0] != pytest.approx(_DJ_DA)


@pytest.mark.unit
def test_an_ungrouped_entry_is_unchanged():
    info = {"param_names": [['a/C']], "param_labels": ['a/C']}
    grad = fsa_backend.get_jac_cost(_FakePid(info), np.array([1.0]))
    assert grad == pytest.approx([_DJ_DA])


@pytest.mark.unit
def test_a_partially_ineligible_entry_falls_back_to_fd_as_a_whole():
    """With one member's column missing, sum_i w_i * S_i would silently drop that member's
    contribution -- the entry must leave the analytic path entirely."""
    pid = _FakePid(_grouped_info(), eligible=('a/C',))   # b/C has no CVODES column

    fd_calls = []

    def fd_cost(param_vals):
        fd_calls.append(np.array(param_vals, dtype=float))
        return 7.0 * float(param_vals[0])
    pid.get_cost_from_params = fd_cost

    grad = fsa_backend.get_jac_cost(pid, np.array([1.0]))
    assert grad == pytest.approx([7.0])
    assert fd_calls, "expected the full-cost FD fallback to run"


# ------------------------------------------------------------------ observable sensitivities


@pytest.mark.unit
def test_observable_sensitivities_combine_members_and_key_by_label():
    pid = _FakePid(_modifier_info(baselines=(2.0, 0.5)))
    sens = fsa_backend.observable_feature_sensitivities(pid, np.array([1.0]))
    # feature = mean(v), so d(feature)/dtheta = mean(2*S_A + 0.5*S_B) = 2*3 + 0.5*30 = 21.
    assert set(sens.keys()) == {"obs0"}
    assert sens["obs0"] == {"theta_C": pytest.approx(21.0)}


@pytest.mark.unit
def test_grouped_observable_sensitivities_key_by_the_joined_label():
    pid = _FakePid(_grouped_info())
    sens = fsa_backend.observable_feature_sensitivities(pid, np.array([1.0]))
    assert sens["obs0"] == {"a/C+b/C": pytest.approx(np.mean(_S_A) + np.mean(_S_B))}


# ------------------------------------------------------------------ labels and nominals


@pytest.mark.unit
def test_param_entry_labels_reads_param_labels_and_falls_back():
    assert param_entry_labels(_modifier_info()) == ['theta_C']
    # A param_id_info from before #355 has no param_labels: derive, joining groups.
    assert param_entry_labels({"param_names": [['a/C', 'b/C'], ['heart/R']]}) \
        == ['a/C+b/C', 'heart/R']


@pytest.mark.unit
def test_the_casadi_arm_no_longer_refuses_modifiers(monkeypatch):
    """The CasADi arm used to refuse modifier entries outright. It now folds its per-member
    jacobian with the same affine weights the FSA arm uses, so the dispatch must reach the
    backend rather than raise. The numbers are pinned end to end, against the Myokit arm and
    a closed form, in tests/test_modifier_backend_equivalence.py."""
    from libcuflynx.param_id import paramID

    class _Stub(paramID.OpencorParamID):
        def __init__(self):
            pass

    pid = _Stub()
    pid.model_type = 'casadi_python'
    pid.param_id_info = _modifier_info()
    called = []
    monkeypatch.setattr(paramID.casadi_backend, 'get_observable_sensitivities',
                        lambda p, v: called.append(p) or {})
    pid.get_observable_sensitivities(np.array([1.0]))
    assert called == [pid]


@pytest.mark.unit
def test_the_fsa_arm_no_longer_refuses_modifiers(monkeypatch):
    from libcuflynx.param_id import paramID

    class _Stub(paramID.OpencorParamID):
        def __init__(self):
            pass

    pid = _Stub()
    pid.model_type = 'cellml_only'
    pid.do_ad = True
    pid.param_id_info = _modifier_info()
    pid.sim_helper = _FakeSimHelper({'v': {'a/C': _S_A, 'b/C': _S_B}}, ['a/C', 'b/C'])

    called = []
    monkeypatch.setattr(paramID.fsa_backend, 'observable_feature_sensitivities',
                        lambda p, v: called.append(p) or {})
    pid.get_observable_sensitivities(np.array([1.0]))
    assert called == [pid]


@pytest.mark.unit
def test_a_modifier_nominal_is_the_operation_identity_not_a_model_value():
    """get_init_param_vals over first members would put the first target's default in theta's
    slot -- and theta = 1.2e-8 scales every target by 1.2e-8. Identity theta (scale -> 1.0)
    leaves every target at its baseline, which is what 'nominal' means."""
    info = {"param_names": [['heart/R'], ['a/C', 'b/C']], "param_labels": ['heart/R', 'th'],
            "modifiers": [{"index": 1, "name": "th", "operation": 'scale',
                           "targets": ['a/C', 'b/C'], "baselines": [1.0, 2.0]}]}
    nominal = np.array([3.5, 1.2e-8])
    apply_modifier_identity_nominals(info, nominal)
    assert nominal == pytest.approx([3.5, 1.0])
