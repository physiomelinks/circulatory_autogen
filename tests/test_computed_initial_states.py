"""States that CellML initialises from computed constants (SN_simple: ``m = m_init`` with
``m_init = f(V_rest_approx, ...)``) in the generated Python and CasADi models.

libCellML sets such a state inside ``compute_computed_constants``; the generated function takes it as
an optional ``states`` argument, and the solver helpers read it from there.
"""
import importlib.util
import os

import numpy as np
import pytest

from libcuflynx.generators.PythonGenerator import PythonGenerator
from libcuflynx.solver_wrappers import get_simulation_helper

pytestmark = [pytest.mark.integration, pytest.mark.solver]

_DT, _T = 0.001, 0.05
# bounded to 50 ms of SN_simple: about 10 s in all
_TOLERANCE = 1e-5


@pytest.fixture
def sn_models(generated_cellml_model_factory, tmp_path):
    cellml = generated_cellml_model_factory("SN_simple", "SN_simple_parameters.json")
    return {
        "cellml": cellml,
        "python": PythonGenerator(cellml, output_dir=str(tmp_path), module_name="sn_python").generate(),
        "casadi": PythonGenerator(cellml, output_dir=str(tmp_path), module_name="sn_casadi",
                                  casadi_compat=True).generate(),
        "python_flat": PythonGenerator(cellml, output_dir=str(tmp_path), module_name="sn_flat",
                                       human_readable=False).generate(),
    }


def _trace(helper, name):
    return np.asarray(helper.get_results([name], flatten=True)[0], dtype=float).ravel()


def _myokit(cellml):
    helper = get_simulation_helper(
        model_path=cellml, model_type="cellml", solver="CVODE_myokit", dt=_DT, sim_time=_T,
        pre_time=0.0, solver_info={"MaximumStep": 0.001, "rtol": 1e-9, "atol": 1e-9})
    assert helper.run()
    return helper


def _casadi(path):
    return get_simulation_helper(
        model_path=path, model_type="casadi_python", solver="casadi_integrator", dt=_DT,
        sim_time=_T, pre_time=0.0,
        solver_info={"method": "cvodes", "max_step_size": 0.001, "max_num_steps": 100000,
                     "abstol": 1e-10, "reltol": 1e-10, "jit": False})


def _worst_relative_error(reference, other):
    worst = 0.0
    for name in other.state_name_to_idx:
        ref, got = _trace(reference, name), _trace(other, name)
        n = min(len(ref), len(got))
        assert n > 1
        worst = max(worst, np.max(np.abs(got[:n] - ref[:n])) / max(np.max(np.abs(ref)), 1e-12))
    return worst


def test_python_trajectory_matches_myokit(sn_models):
    reference = _myokit(sn_models["cellml"])
    helper = get_simulation_helper(
        model_path=sn_models["python"], model_type="python", solver="solve_ivp", dt=_DT,
        sim_time=_T, pre_time=0.0,
        solver_info={"method": "BDF", "rtol": 1e-9, "atol": 1e-9, "max_step": 0.001})
    assert helper.run()
    assert _worst_relative_error(reference, helper) < _TOLERANCE


def test_casadi_cvodes_trajectory_matches_myokit(sn_models):
    reference = _myokit(sn_models["cellml"])
    helper = _casadi(sn_models["casadi"])
    assert helper.run()
    assert _worst_relative_error(reference, helper) < _TOLERANCE


def test_casadi_gradient_through_an_initial_state_matches_finite_differences(sn_models):
    """Cai_init sets the initial m_SK (and z_BK) through computed constants, so the gradient of the m_SK
    trace only matches if the initial state stays symbolic in it."""
    import casadi as ca
    parameter, state = "soma_SN/Cai_init", "soma_SN/m_SK"

    def mean_m_sk(value):
        helper = _casadi(sn_models["casadi"])
        helper.set_param_vals([[parameter]], [[value]])
        assert helper.run()
        return float(np.mean(_trace(helper, state)))

    helper = _casadi(sn_models["casadi"])
    p0 = float(helper.get_init_param_vals([[parameter]])[0])
    helper._create_param_subset([[parameter]], [p0])
    assert helper.run()
    observable = helper.get_results([[state]])[0][0]
    jacobian = ca.Function("j", [helper.states_symb, helper.variables_symb],
                           [ca.jacobian(ca.sum2(observable) / observable.numel(),
                                        helper.variables_symb_subset)])
    ad = float(np.array(jacobian(ca.DM(helper._x0_numeric()), ca.DM(helper.variables))).ravel()[0])
    step = abs(p0) * 1e-5
    fd = (mean_m_sk(p0 + step) - mean_m_sk(p0 - step)) / (2 * step)
    assert abs(fd) > 1.0, f"a flat feature would make this vacuous ({fd})"
    assert ad == pytest.approx(fd, rel=1e-4)


def test_set_param_vals_moves_the_states_computed_from_it(sn_models):
    for kind, model_type, solver in (("python", "python", "solve_ivp"),
                                      ("casadi", "casadi_python", "casadi_integrator")):
        helper = get_simulation_helper(model_path=sn_models[kind], model_type=model_type,
                                       solver=solver, dt=_DT, sim_time=_T, pre_time=0.0)
        idx = helper.state_name_to_idx["soma_SN/m_SK"]
        before = float(helper.states[idx])
        helper.set_param_vals([["soma_SN/Cai_init"]], [[2e-4]])
        assert float(helper.states[idx]) != pytest.approx(before, rel=1e-3), kind


def _load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["python", "python_flat", "casadi"])
def test_compute_computed_constants_fills_the_optional_states(sn_models, kind):
    module = _load(sn_models[kind])
    states, rates, variables = (module.create_states_array(), module.create_states_array(),
                                module.create_variables_array())
    module.initialise_variables(states, rates, variables)
    module.compute_computed_constants(variables)  # variables alone is still a valid call
    module.compute_computed_constants(variables, states)
    assert np.isfinite([float(s) for s in states]).all()


def test_models_without_computed_states_keep_the_variables_only_signature(
        generated_cellml_model_factory, tmp_path):
    cellml = generated_cellml_model_factory("3compartment", "3compartment_parameters.json")
    for flat in (True, False):
        path = PythonGenerator(cellml, output_dir=str(tmp_path), module_name=f"threec_{flat}",
                               human_readable=not flat).generate()
        with open(path) as fh:
            assert "def compute_computed_constants(variables):" in fh.read()
