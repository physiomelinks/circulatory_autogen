"""casadi_integrator/cvodes solves the stiff `physiological` model with an LU linear solver.

CasADi's default CVODES linear solver is sparse QR. On this model the Newton matrix I - gamma*J
has a condition number ~4e20 (Jacobian entries from 5e-140 to 1e20, from 1/(I*C) in tiny cerebral
vessels), and QR solves it wrong: the run takes five times the steps and the cerebral states end
up wrong (worst relative error ~1.8 against a converged CVODE_myokit). Sparse LU (``csparse``)
gives ~1e-5. The default is ``csparse``; ``options: {linear_solver: ...}`` overrides it.
"""
import os
import importlib.util

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SIM, _DT = 1.0, 0.01


@pytest.fixture(scope="module")
def physiological(tmp_path_factory):
    pytest.importorskip("casadi")
    pytest.importorskip("myokit")
    from libcuflynx.generators.PythonGenerator import PythonGenerator
    spec = importlib.util.spec_from_file_location(
        "casadi_coverage", os.path.join(REPO, "benchmarks", "casadi_coverage.py"))
    cov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cov)
    work = str(tmp_path_factory.mktemp("physiological"))
    cellml = cov._generate_cellml("physiological", work)
    module = PythonGenerator(cellml, output_dir=work, module_name="physiological_casadi",
                             casadi_compat=True).generate()
    return {"cellml": cellml, "casadi": module, "cov": cov}


def _worst(cov, myokit, casadi):
    names = list(casadi.state_name_to_idx)
    return cov._worst_error(cov._states(myokit, names), cov._states(casadi, names))


@pytest.fixture(scope="module")
def reference(physiological):
    from libcuflynx.solver_wrappers import get_simulation_helper
    h = get_simulation_helper(
        model_path=physiological["cellml"], model_type="cellml", solver="CVODE_myokit",
        dt=_DT, sim_time=_SIM, pre_time=0.0,
        solver_info={"MaximumStep": 0.001, "rtol": 1e-8, "atol": 1e-10})
    assert h.run()
    return h


def _casadi(physiological, options=None):
    from libcuflynx.solver_wrappers import get_simulation_helper
    info = {"method": "cvodes", "max_step_size": 0.001, "max_num_steps": 100000,
            "reltol": 1e-8, "abstol": 1e-10, "jit": False}
    if options is not None:
        info["options"] = options
    h = get_simulation_helper(
        model_path=physiological["casadi"], model_type="casadi_python",
        solver="casadi_integrator", dt=_DT, sim_time=_SIM, pre_time=0.0, solver_info=info)
    assert h.run()
    return h


@pytest.mark.integration
@pytest.mark.solver
def test_default_linear_solver_is_csparse_and_matches_myokit(physiological, reference):
    h = _casadi(physiological)
    assert h._build_integrator_opts()["linear_solver"] == "csparse"
    err, name = _worst(physiological["cov"], reference, h)
    assert err < 1e-3, (err, name)


@pytest.mark.integration
@pytest.mark.solver
def test_user_option_overrides_linear_solver(physiological, reference):
    h = _casadi(physiological, {"linear_solver": "qr"})
    assert h._build_integrator_opts()["linear_solver"] == "qr"
    err, _ = _worst(physiological["cov"], reference, h)
    assert err > 1e-2  # QR mis-solves this model's Newton matrix
