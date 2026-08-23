"""FSA accuracy against closed-form truth, and modifier-vs-native equivalence (issue #387).

CA's other FSA/modifier tests check *internal consistency*: that the gradient matches a finite
difference of CA's own cost, or that two CA code paths agree. Those cannot catch an error
shared by both sides. This file pins CA's numbers against an **analytically solvable model**,
and against the same system written two equivalent ways:

* ``affine_native.cellml``   -- one constant ``theta`` scales two rate constants *in the
  model's own math* (``k1 = theta*c1``, ``k2 = theta*c2``), so the solver differentiates
  through the scaling directly and no modifier machinery is involved.
* ``affine_modifier.cellml`` -- ``k1``/``k2`` are independent constants defaulting to exactly
  ``c1``/``c2``, driven by a params_for_id ``scale`` modifier over both.

At ``theta = 1`` these are the same point of the same system, so every cost, feature and
gradient must agree -- which is the multi-target modifier chain rule (#380/#383) checked
against a model that computes the same thing without it.

Both are solvable in closed form:

    x(T) = x0*exp(-theta*c1*T)   =>   d ln x/d ln theta = -c1*T   exactly
    y(T) = y0*exp(-theta*c2*T)   =>   d ln y/d ln theta = -c2*T   exactly

independent of x0, y0 and of the solver, so the assertions are the true answer rather than
"the two paths agree with each other".

Provenance: the fixture pair and the closed-form assertions come from CUFLynx #208, moved here
because the behaviour they pin is CA's arithmetic, not CUFLynx's seams -- and because CUFLynx
CI has no Myokit, so a numerical claim there never actually runs.
"""
import json
import os

import numpy as np
import pytest

from libcuflynx.param_id.paramID import CVS0DParamID

# The fixture pair lives in tests/test_inputs/ -- it is test scaffolding (an analytically
# solvable toy), not a resource a user would run, which is what resources/ is for.
TEST_INPUTS = os.path.join(os.path.dirname(__file__), 'test_inputs')

_C1 = 0.7314          # affine/k1 default, and the native model's c1
_C2 = 0.1129          # affine/k2 default, and the native model's c2
_X0, _Y0 = 2.0, 3.0
_T_END = 4.0
_DT = 0.01

# Ground truths/stds for the two `min` observables (x and y decay, so min is the end value).
_GT_X, _STD_X = 0.15, 0.05
_GT_Y, _STD_Y = 1.8, 0.2


def _obs_doc():
    def item(var, operand, gt, std):
        # cost_type pinned rather than defaulted: the chain-rule test below differentiates
        # the cost by hand, so it must know which cost it is differentiating. Leaving it to
        # CA's default made the test silently wrong when that default changed from MSE to
        # gaussian_MLE (which is exactly 0.5x MSE -- see libcuflynx/funcs/cost_funcs_user.py).
        return {"data_item_name": var, "trace_name_for_plotting": var, "data_type": "constant",
                "operation": "min", "operands": [operand], "unit": "dimensionless",
                "weight": 1.0, "value": gt, "std": std, "cost_type": "gaussian_MLE",
                "experiment_idx": 0, "subexperiment_idx": 0, "plot_type": "horizontal"}
    return {"protocol_info": {"pre_times": [0.0], "sim_times": [[_T_END]],
                              "params_to_change": {}},
            "prediction_items": [],
            "data_items": [item("x_end", "affine/x", _GT_X, _STD_X),
                           item("y_end", "affine/y", _GT_Y, _STD_Y)]}


_NATIVE_PARAMS = {'version': 1, 'params': [
    {'name': 'theta', 'targets': ['affine/theta'], 'min': 0.2, 'max': 5.0,
     'name_for_plotting': 'theta'}]}

# The modifier arm: ONE calibrated theta scaling BOTH rate constants. Its baselines are the
# model defaults c1 and c2, so theta = 1 reproduces the native model exactly.
_MODIFIER_PARAMS = {'version': 1, 'params': [
    {'name': 'theta', 'modifies': ['affine/k1', 'affine/k2'], 'modifier': 'scale',
     'min': 0.2, 'max': 5.0, 'name_for_plotting': 'theta'}]}


def _write(tmp_path, name, doc):
    path = os.path.join(str(tmp_path), name)
    with open(path, 'w') as f:
        json.dump(doc, f)
    return path


def _engine(tmp_path, model, params_doc, solver_info=None, dt=_DT):
    """A param-id engine on one of the affine fixtures, FSA gradients enabled."""
    out_dir = os.path.join(str(tmp_path), f'out_{model}')
    os.makedirs(out_dir, exist_ok=True)
    outer = CVS0DParamID(
        model_path=os.path.join(TEST_INPUTS, f'affine_{model}.cellml'),
        model_type='cellml', param_id_method='sp_minimize', file_name_prefix='affine',
        params_for_id_path=_write(tmp_path, f'{model}_params.json', params_doc),
        param_id_obs_path=_write(tmp_path, f'{model}_obs.json', _obs_doc()),
        sim_time=_T_END, pre_time=0.0, dt=dt,
        solver_info=solver_info or {'solver': 'CVODE_myokit'},
        do_ad=True, DEBUG=True, param_id_output_dir=out_dir)
    engine = outer.param_id
    engine.output_dir = out_dir
    return engine


@pytest.fixture
def native(tmp_path):
    return _engine(tmp_path, 'native', _NATIVE_PARAMS)


@pytest.fixture
def modifier(tmp_path):
    return _engine(tmp_path, 'modifier', _MODIFIER_PARAMS)


def _features(engine, theta):
    """The scalar observables at theta, in const order (x_end, y_end)."""
    _, operands_list, _ = engine.get_cost_obs_and_pred_from_params(
        np.array([float(theta)]), reset=True, only_one_exp=0)
    return np.asarray(engine.get_obs_output_dict(operands_list[0])['const'], dtype=float)


def _cost_fd(engine, theta, h=1e-6):
    """Central difference of CA's own cost -- the operational truth a calibration descends."""
    plus = float(engine.get_cost_from_params(np.array([theta + h])))
    minus = float(engine.get_cost_from_params(np.array([theta - h])))
    return (plus - minus) / (2.0 * h)


# --------------------------------------------------------------- fixture validation


@pytest.mark.integration
def test_the_two_models_are_the_same_system_at_theta_one(native, modifier):
    """Fixture validation: if the pair ever stops describing one system, every equivalence
    assertion below becomes vacuous, so check it directly."""
    assert _features(native, 1.0) == pytest.approx(_features(modifier, 1.0), rel=1e-9)
    assert float(native.get_cost_from_params(np.array([1.0]))) == pytest.approx(
        float(modifier.get_cost_from_params(np.array([1.0]))), rel=1e-9)


@pytest.mark.integration
def test_the_features_match_the_closed_form(native):
    """x(T) = x0*exp(-theta*c1*T); min over a decaying trace is the end value."""
    x_end, y_end = _features(native, 1.0)
    # rel 1e-4: this is the *state* accuracy at CA's default CVODE_myokit tolerances
    # (rel 1e-6 / abs 1e-8 since #379), not a claim about the sensitivities.
    assert x_end == pytest.approx(_X0 * np.exp(-_C1 * _T_END), rel=1e-4)
    assert y_end == pytest.approx(_Y0 * np.exp(-_C2 * _T_END), rel=1e-4)


# --------------------------------------------------------------- closed-form sensitivities


@pytest.mark.integration
@pytest.mark.parametrize('arm', ['native', 'modifier'])
def test_output_sensitivities_match_the_closed_form(arm, tmp_path):
    """d ln(Y)/d ln(theta) = -c*T exactly, for both the in-model scaling and the modifier.

    The elasticity is the sharp form: it removes x0/y0 and any solver scaling, so a wrong
    chain-rule weight (the #380 failure mode) shows up immediately.
    """
    params = _NATIVE_PARAMS if arm == 'native' else _MODIFIER_PARAMS
    engine = _engine(tmp_path, arm, params)
    theta = 1.0
    sens = engine.get_observable_sensitivities(np.array([theta]), gradient_method='FSA')
    features = _features(engine, theta)

    # obs_info order is the data_items order: x_end, y_end.
    labels = [engine._observable_label(i)
              for i in engine.obs_info['const_idx_to_obs_idx']]
    key = 'theta' if arm == 'modifier' else 'affine/theta'
    for label, feature, c in zip(labels, features, (_C1, _C2)):
        d_abs = sens[label][key]
        elasticity = d_abs * theta / feature
        assert elasticity == pytest.approx(-c * _T_END, rel=1e-5), (
            f'{arm} {label}: d ln/d ln theta = {elasticity}, expected {-c * _T_END}')


# --------------------------------------------------------------- the cost gradient


@pytest.mark.integration
@pytest.mark.parametrize('arm', ['native', 'modifier'])
def test_the_fsa_cost_gradient_matches_a_finite_difference_of_the_cost(
        arm, tmp_path):
    """The gradient a do_ad calibration actually descends, against differencing CA's own cost.

    Issue #387 reported this as wrong by 1e-3..1e-1; at CA's default FSA tolerances it is
    accurate to ~1e-6. See test_tightening_tolerances_past_the_sensitivity_floor_is_refused
    for the configuration that does break it, and why.
    """
    params = _NATIVE_PARAMS if arm == 'native' else _MODIFIER_PARAMS
    engine = _engine(tmp_path, arm, params)
    theta = 1.0
    _, grad = engine.get_cost_and_jac_fsa(np.array([theta]))
    assert float(grad[0]) == pytest.approx(_cost_fd(engine, theta), rel=1e-4)


@pytest.mark.integration
@pytest.mark.parametrize('arm', ['native', 'modifier'])
def test_the_cost_gradient_is_the_chain_rule_over_its_own_sensitivities(
        arm, tmp_path):
    """dJ/dtheta must equal sum_k dJ/d(feature_k) * d(feature_k)/dtheta, built from CA's *own*
    output sensitivities.

    This is issue #387's decisive check: the cost arm and the feature arm share the same
    sensitivity traces, so any disagreement is in the cost reconstruction alone (and any
    agreement localises a residual error to the traces, which the closed-form tests above
    then pin).
    """
    params = _NATIVE_PARAMS if arm == 'native' else _MODIFIER_PARAMS
    engine = _engine(tmp_path, arm, params)
    theta = 1.0
    key = 'theta' if arm == 'modifier' else 'affine/theta'

    _, grad = engine.get_cost_and_jac_fsa(np.array([theta]))
    sens = engine.get_observable_sensitivities(np.array([theta]), gradient_method='FSA')
    features = _features(engine, theta)
    labels = [engine._observable_label(i)
              for i in engine.obs_info['const_idx_to_obs_idx']]

    # gaussian_MLE is 0.5 * sum_k w_k * ((feature_k - gt_k)/std_k)^2 (the negative
    # log-likelihood up to constants), divided by the weighted denominator -- so
    # d(cost)/d(feature_k) is (feature_k - gt_k)/std_k^2, not twice that. MSE is exactly
    # 2x gaussian_MLE, which is why getting this factor wrong shows up as a clean 2x.
    denom = float(engine._total_weighted_obs_denominator())
    by_hand = 0.0
    for label, feature, gt, std in zip(labels, features,
                                       (_GT_X, _GT_Y), (_STD_X, _STD_Y)):
        by_hand += (feature - gt) / (std ** 2) * sens[label][key]
    by_hand /= denom

    assert float(grad[0]) == pytest.approx(by_hand, rel=1e-6)


@pytest.mark.integration
def test_the_modifier_reproduces_the_native_gradient(native, modifier):
    """One theta scaling two model constants must give the same cost gradient whether the
    scaling is written into the CellML math or applied by CA's scale modifier.

    This is the multi-target modifier chain rule (sum_i baseline_i * dJ/dp_i, #380/#383)
    checked against a model that computes the same derivative without any modifier.
    """
    theta = 1.0
    _, grad_native = native.get_cost_and_jac_fsa(np.array([theta]))
    _, grad_modifier = modifier.get_cost_and_jac_fsa(np.array([theta]))
    assert float(grad_modifier[0]) == pytest.approx(float(grad_native[0]), rel=1e-5)
    # ... and both against the operational truth, so agreeing on a wrong value cannot pass.
    assert float(grad_modifier[0]) == pytest.approx(_cost_fd(modifier, theta), rel=1e-4)


@pytest.mark.integration
def test_the_modifier_gradient_sums_both_targets(tmp_path):
    """The modifier's derivative is sum_i baseline_i * dJ/dp_i -- dropping either target (the
    pre-#380 first-member-only bug) leaves a materially different number, so assert the sum
    against per-target gradients obtained by calibrating k1 and k2 as free parameters."""
    free = {'version': 1, 'params': [
        {'name': 'k1', 'targets': ['affine/k1'], 'min': 0.05, 'max': 5.0},
        {'name': 'k2', 'targets': ['affine/k2'], 'min': 0.01, 'max': 2.0}]}
    free_engine = _engine(tmp_path, 'modifier', free)
    _, grad_free = free_engine.get_cost_and_jac_fsa(np.array([_C1, _C2]))

    mod_engine = _engine(tmp_path, 'modifier', _MODIFIER_PARAMS)
    _, grad_mod = mod_engine.get_cost_and_jac_fsa(np.array([1.0]))

    expected = _C1 * float(grad_free[0]) + _C2 * float(grad_free[1])
    assert float(grad_mod[0]) == pytest.approx(expected, rel=1e-5)
    # the first-member-only value must be clearly distinguishable, or this test proves nothing
    first_only = _C1 * float(grad_free[0])
    assert abs(first_only - expected) > 0.01 * abs(expected)


# --------------------------------------------------------------- the tolerance floor (#387)


@pytest.mark.integration
def test_tightening_rtol_past_the_floor_degrades_the_gradient_and_warns(
        tmp_path):
    """The configuration issue #387 was actually measured under, pinned as known behaviour.

    Myokit excludes the CVODES sensitivity variables from the local error test and uses a
    finite-difference sensitivity RHS sized by sqrt(rtol), so rtol=1e-12 makes the *gradient*
    worse while the *states* stay exact. CA warns; this asserts both the warning and the
    degradation it is warning about, so the day Myokit fixes its side, this test fails and
    the guard can be dropped.
    """
    from libcuflynx.solver_wrappers.myokit_helper import FSA_MIN_SAFE_REL_TOL

    tight = {'solver': 'CVODE_myokit', 'rtol': 1e-12, 'atol': 1e-12}
    engine = _engine(tmp_path, 'native', _NATIVE_PARAMS, solver_info=tight)
    # The warning belongs to enabling FSA, which happens on the first gradient call (the
    # simulation is rebuilt with sensitivities then), not to constructing the helper.
    with pytest.warns(UserWarning, match='less. accurate as rtol'):
        _, grad = engine.get_cost_and_jac_fsa(np.array([1.0]))
    rel_err = abs(float(grad[0]) - _cost_fd(engine, 1.0)) / abs(_cost_fd(engine, 1.0))
    assert rel_err > 1e-4, (
        f'rtol=1e-12 no longer degrades the FSA gradient (rel err {rel_err:.2e}) -- if Myokit '
        f'now error-controls its sensitivities, drop the rtol < {FSA_MIN_SAFE_REL_TOL:g} '
        f'warning and this test')


@pytest.mark.integration
def test_the_default_tolerances_do_not_warn_and_are_accurate(tmp_path):
    """The other half: at CA's defaults there is nothing to warn about, and the gradient is
    accurate -- so the warning cannot become background noise on ordinary runs."""
    import warnings as _warnings
    engine = _engine(tmp_path, 'native', _NATIVE_PARAMS)
    with _warnings.catch_warnings():
        _warnings.simplefilter('error', UserWarning)
        _, grad = engine.get_cost_and_jac_fsa(np.array([1.0]))
    assert float(grad[0]) == pytest.approx(_cost_fd(engine, 1.0), rel=1e-4)
