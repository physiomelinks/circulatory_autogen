"""One theta scaling FIVE model constants: modifier vs native, Myokit vs CasADi.

The `affine` pair (tests/test_fsa_analytic_accuracy.py) pins two members on the Myokit/FSA
arm. This file widens both axes at once:

* **five members**, spread over an order of magnitude (c = 0.11 .. 0.89), so the chain-rule
  sum is over a number where dropping *any* single member is unambiguous -- with two members
  a bug can hide behind a small one.
* **both analytic backends**: `cellml_only` + `CVODE_myokit` (CVODES forward sensitivities)
  and `casadi_python` + `casadi_integrator` (symbolic AD). CasADi gained modifier/grouped
  support by folding its per-member jacobian with the same affine weights the FSA arm uses;
  this is what proves the two backends answer the same question.

Four arms, one system:

    scaling_native.cellml     k_i = theta * c_i written into the model's math
    scaling_modifier.cellml   k_i independent constants, one params_for_id scale modifier
      x each of {Myokit, CasADi}

At theta = 1 all four are the same point of the same system, and it is solvable in closed
form (``x_i(T) = x_i0*exp(-theta*c_i*T)``, so ``d ln x_i/d ln theta = -c_i*T`` exactly), so
every observable sensitivity and every cost gradient is checked against the truth as well as
against the other arms.
"""
import json
import math
import os

import numpy as np
import pytest

from param_id.paramID import CVS0DParamID

TEST_INPUTS = os.path.join(os.path.dirname(__file__), 'test_inputs')

# Must match tests/test_inputs/scaling_*.cellml.
_C = [0.7314, 0.1129, 0.4207, 0.2561, 0.8893]
_X0 = [2.0, 3.0, 1.5, 4.0, 2.5]
_T_END = 4.0
_DT = 0.01
_N = len(_C)

_OBS_PATH = os.path.join(TEST_INPUTS, 'scaling_obs_data.json')

_NATIVE_PARAMS = {'version': 1, 'params': [
    {'name': 'theta', 'targets': ['scaling/theta'], 'min': 0.2, 'max': 5.0}]}

_MODIFIER_PARAMS = {'version': 1, 'params': [
    {'name': 'theta', 'modifies': [f'scaling/k{i}' for i in range(1, _N + 1)],
     'modifier': 'scale', 'min': 0.2, 'max': 5.0}]}

_THETA_KEY = {'native': 'scaling/theta', 'modifier': 'theta'}


@pytest.fixture(scope='module')
def casadi_models(tmp_path_factory):
    """The two CellML fixtures generated once as casadi_python modules.

    PythonGenerator takes a plain CellML file, so the same fixture pair drives both backends
    -- there is no second, hand-maintained copy of the model that could drift from the first.
    """
    pytest.importorskip('casadi')
    from generators.PythonGenerator import PythonGenerator
    out = str(tmp_path_factory.mktemp('scaling_casadi'))
    return {arm: PythonGenerator(os.path.join(TEST_INPUTS, f'scaling_{arm}.cellml'),
                                 output_dir=out, casadi_compat=True).generate()
            for arm in ('native', 'modifier')}


def _engine(tmp_path, arm, backend, casadi_models=None):
    params_doc = _NATIVE_PARAMS if arm == 'native' else _MODIFIER_PARAMS
    params_path = os.path.join(str(tmp_path), f'{arm}_{backend}_params.json')
    with open(params_path, 'w') as f:
        json.dump(params_doc, f)
    out_dir = os.path.join(str(tmp_path), f'out_{arm}_{backend}')
    os.makedirs(out_dir, exist_ok=True)

    if backend == 'myokit':
        model_path = os.path.join(TEST_INPUTS, f'scaling_{arm}.cellml')
        model_type, solver_info = 'cellml_only', {'solver': 'CVODE_myokit'}
    else:
        model_path = casadi_models[arm]
        model_type = 'casadi_python'
        # 'bdf' rather than cvodes/idas: the adjoint integrators are excluded from
        # ad_suitable_methods, so the symbolic gradient needs the symbolic BDF.
        solver_info = {'solver': 'casadi_integrator', 'method': 'bdf',
                       'max_step': 0.01, 'max_num_steps': 100000}

    outer = CVS0DParamID(
        model_path=model_path, model_type=model_type, param_id_method='sp_minimize',
        file_name_prefix='scaling', params_for_id_path=params_path,
        param_id_obs_path=_OBS_PATH, sim_time=_T_END, pre_time=0.0, dt=_DT,
        solver_info=solver_info, do_ad=True, DEBUG=True, param_id_output_dir=out_dir)
    engine = outer.param_id
    engine.output_dir = out_dir
    return engine


def _labels(engine):
    return [engine._observable_label(i) for i in engine.obs_info['const_idx_to_obs_idx']]


def _sensitivities(engine, arm, theta=1.0):
    """{observable label: d(feature)/d(theta)} for the one calibrated variable."""
    sens = engine.get_observable_sensitivities(np.array([theta]))
    return {label: sens[label][_THETA_KEY[arm]] for label in _labels(engine)}


# The exact feature values at theta = 1 (min of a decaying trace is its end value). Taken
# from the closed form rather than from the engine: on the CasADi arm the operands are
# symbolic once AD is armed, and a *closed-form* denominator is the stronger choice anyway --
# the elasticity is then checked against truth on both sides of the ratio.
_FEATURES_EXACT = [x0 * math.exp(-c * _T_END) for c, x0 in zip(_C, _X0)]


def _cost_fd(tmp_path, arm, backend, casadi_models=None, theta=1.0, h=1e-6):
    """Central difference of the *numeric* cost -- the operational truth.

    Deliberately on a fresh engine that never arms AD, and through
    ``get_cost_from_params`` (the re-simulating numpy path) rather than ``get_cost``. On
    casadi_python ``get_cost`` evaluates the symbolic cost, which for a computed-constant
    parameter is flat in theta for the same reason its gradient is zero (issue #389) -- so
    comparing an AD gradient against an AD cost would agree at 0 == 0 and prove nothing.
    """
    engine = _engine(tmp_path, arm, backend, casadi_models)
    plus = float(engine.get_cost_from_params(np.array([theta + h])))
    minus = float(engine.get_cost_from_params(np.array([theta - h])))
    return (plus - minus) / (2.0 * h)


# native-casadi is expected to fail: on casadi_python a parameter that acts only through a
# *computed constant* (which is exactly what the native model's theta does) gets an
# identically-zero AD gradient, because compute_computed_constants is applied numerically and
# never enters the symbolic graph. Pre-existing and unrelated to modifiers -- confirmed on
# master @ 5798334 -- and filed as issue #389. strict=True, so the day it is fixed this flips
# to XPASS and the marks come off rather than quietly staying dead.
_CASADI_COMPUTED_CONSTANT_XFAIL = pytest.mark.xfail(
    strict=True,
    reason='issue #389: casadi_python gives a zero AD gradient for a parameter acting only '
           'through a computed constant')

_ARMS_OK = [('native', 'myokit'), ('modifier', 'myokit'), ('modifier', 'casadi')]
_ARMS = _ARMS_OK + [pytest.param('native', 'casadi',
                                 marks=_CASADI_COMPUTED_CONSTANT_XFAIL)]
_IDS = ['native-myokit', 'modifier-myokit', 'modifier-casadi', 'native-casadi']


# --------------------------------------------------------------- closed form, every arm


@pytest.mark.integration
@pytest.mark.parametrize('arm,backend', _ARMS, ids=_IDS)
def test_observable_sensitivities_match_the_closed_form(arm, backend, tmp_path,
                                                        casadi_models):
    """d ln(x_i)/d ln(theta) = -c_i*T exactly, for all five members, on both backends and
    whether the scaling is in the model's math or applied by the modifier."""
    engine = _engine(tmp_path, arm, backend, casadi_models)
    sens = _sensitivities(engine, arm)
    for label, feature, c in zip(_labels(engine), _FEATURES_EXACT, _C):
        elasticity = sens[label] * 1.0 / feature
        assert elasticity == pytest.approx(-c * _T_END, rel=2e-4), (
            f'{arm}/{backend} {label}: {elasticity} != {-c * _T_END}')


@pytest.mark.integration
@pytest.mark.parametrize('arm,backend', _ARMS, ids=_IDS)
def test_the_cost_gradient_matches_a_finite_difference_of_the_cost(arm, backend, tmp_path,
                                                                   casadi_models):
    engine = _engine(tmp_path, arm, backend, casadi_models)
    grad = np.asarray(engine.get_gradient(np.array([1.0]))).ravel()
    truth = _cost_fd(tmp_path, arm, backend, casadi_models)
    assert abs(truth) > 1.0, f'a flat cost would make this vacuous (got {truth})'
    assert float(grad[0]) == pytest.approx(truth, rel=2e-3)


# --------------------------------------------------------------- cross-arm equivalence


@pytest.mark.integration
def test_all_four_arms_agree_on_the_observable_sensitivities(tmp_path, casadi_models):
    """The headline claim: modifier == native and Myokit == CasADi, per observable.

    Over the three arms that work today; native-casadi is issue #389 (see the dedicated
    test below), so including it here would only restate that one failure three times.
    """
    got = {f'{arm}-{backend}': _sensitivities(
               _engine(tmp_path, arm, backend, casadi_models), arm)
           for arm, backend in _ARMS_OK}
    reference = got['native-myokit']
    assert len(reference) == _N
    for name, sens in got.items():
        assert set(sens) == set(reference), name
        for label, value in reference.items():
            assert sens[label] == pytest.approx(value, rel=2e-3), f'{name} {label}'


@pytest.mark.integration
def test_all_four_arms_agree_on_the_cost_gradient(tmp_path, casadi_models):
    grads = {f'{arm}-{backend}': float(np.asarray(
                 _engine(tmp_path, arm, backend, casadi_models)
                 .get_gradient(np.array([1.0]))).ravel()[0])
             for arm, backend in _ARMS_OK}
    reference = grads['native-myokit']
    assert abs(reference) > 1e-6, 'a zero gradient would make this vacuous'
    for name, value in grads.items():
        assert value == pytest.approx(reference, rel=3e-3), f'{name}: {grads}'


# --------------------------------------------------------------- the sum is over all five


@pytest.mark.integration
@pytest.mark.parametrize('backend', ['myokit', 'casadi'])
def test_the_modifier_sums_all_five_members(backend, tmp_path, casadi_models):
    """dJ/dtheta = sum_i baseline_i * dJ/dk_i over every member.

    Asserted against the five free-parameter gradients from the same model, and -- the part
    that makes it a real test -- against the value each *partial* sum would give, so dropping
    any single member is detectable.
    """
    free = {'version': 1, 'params': [
        {'name': f'k{i}', 'targets': [f'scaling/k{i}'], 'min': 0.01, 'max': 5.0}
        for i in range(1, _N + 1)]}
    params_path = os.path.join(str(tmp_path), f'free_{backend}.json')
    with open(params_path, 'w') as f:
        json.dump(free, f)

    out_dir = os.path.join(str(tmp_path), f'out_free_{backend}')
    os.makedirs(out_dir, exist_ok=True)
    if backend == 'myokit':
        model_path = os.path.join(TEST_INPUTS, 'scaling_modifier.cellml')
        model_type, solver_info = 'cellml_only', {'solver': 'CVODE_myokit'}
    else:
        model_path = casadi_models['modifier']
        model_type = 'casadi_python'
        solver_info = {'solver': 'casadi_integrator', 'method': 'bdf',
                       'max_step': 0.01, 'max_num_steps': 100000}
    free_engine = CVS0DParamID(
        model_path=model_path, model_type=model_type, param_id_method='sp_minimize',
        file_name_prefix='scaling', params_for_id_path=params_path,
        param_id_obs_path=_OBS_PATH, sim_time=_T_END, pre_time=0.0, dt=_DT,
        solver_info=solver_info, do_ad=True, DEBUG=True,
        param_id_output_dir=out_dir).param_id
    free_engine.output_dir = out_dir

    per_member = np.asarray(free_engine.get_gradient(np.array(_C, dtype=float))).ravel()
    assert per_member.shape[0] == _N

    mod_engine = _engine(tmp_path, 'modifier', backend, casadi_models)
    grad = float(np.asarray(mod_engine.get_gradient(np.array([1.0]))).ravel()[0])

    expected = float(np.dot(np.asarray(_C), per_member))
    assert grad == pytest.approx(expected, rel=3e-3)

    # Every leave-one-out sum must be clearly distinguishable, or the assertion above could
    # pass while a member was being dropped.
    for i in range(_N):
        partial = expected - _C[i] * per_member[i]
        assert abs(partial - expected) > 1e-2 * abs(expected), (
            f'member {i} contributes too little to detect: this fixture no longer proves '
            f'the sum covers all {_N} members')


@pytest.mark.integration
def test_casadi_refuses_a_nested_param_name_rather_than_flattening_silently(tmp_path,
                                                                           casadi_models):
    """The helper's own guard: reaching it with an unflattened grouped row used to mean
    differentiating w.r.t. the first member alone (the pre-#380 bug), so it must raise."""
    engine = _engine(tmp_path, 'modifier', 'casadi', casadi_models)
    with pytest.raises(ValueError, match='expects flat parameter names'):
        engine.sim_helper._create_param_subset([['scaling/k1', 'scaling/k2']], None)


# --------------------------------------------------------------- issue #389


@pytest.mark.integration
def test_a_computed_constant_parameter_has_a_zero_casadi_gradient(tmp_path, casadi_models):
    """Pin the #389 behaviour so the fix is detected.

    The native model's theta reaches the dynamics only through `k_i = theta*c_i`, which
    libCellML classes as a COMPUTED_CONSTANT. The CasADi helper gives each of those its own
    independent SX symbol and applies the defining relation numerically, so the symbolic
    graph has no path from theta to the rates and the gradient is *structurally* zero -- not
    small, exactly 0.0. The Myokit arm of the same model gives a real number, so this is a
    backend defect and not a property of the fixture.

    When #389 is fixed this test fails, and it should be replaced by the native-casadi arm
    of the equivalence tests above (drop the xfail marks at the same time).
    """
    casadi_engine = _engine(tmp_path, 'native', 'casadi', casadi_models)
    casadi_grad = float(np.asarray(
        casadi_engine.get_gradient(np.array([1.0]))).ravel()[0])
    assert casadi_grad == 0.0, (
        f'issue #389 appears fixed (casadi gradient {casadi_grad}) -- drop the xfail marks '
        f'on the native-casadi arms and delete this test')

    myokit_engine = _engine(tmp_path, 'native', 'myokit')
    myokit_grad = float(np.asarray(
        myokit_engine.get_gradient(np.array([1.0]))).ravel()[0])
    assert abs(myokit_grad) > 1.0, (
        'the fixture must have a materially non-zero gradient, or the zero above proves '
        f'nothing (got {myokit_grad})')
