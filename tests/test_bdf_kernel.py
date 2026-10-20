"""
Tests for the bdf_kernel gradient strategy via the pip-installable aadc-ode package
(solver_info method 'semi_implicit_signed', gradient_strategy 'kernel').

Verifies on the stiff 3compartment model (27 states):
  1. cost_and_grad on the kernel path runs and returns a finite cost + gradient
  2. the AD gradient matches a central finite difference of the SAME estimator

No custom AADC build is needed: `pip install aadc aadc-ode` (both on PyPI). If either is missing the
tests skip. Supersedes the old proprietary bdf_record_and_evaluate path (#381/#382).
"""
import os
import types

import numpy as np
import pytest

_PNAME = "q_lv_us"      # a sensitive constant parameter
_SNAME = "q_lv"         # a state observable


@pytest.fixture(scope="module")
def aadc_3compartment(tmp_path_factory):
    """Generate the aadc_python 3compartment model once for the module."""
    pytest.importorskip("aadc")
    pytest.importorskip("aadc_ode")
    from libcuflynx.utilities.utility_funcs import get_default_inp_data_dict
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture

    resources = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
    d = str(tmp_path_factory.mktemp("bdf_kernel"))
    inp = get_default_inp_data_dict("3compartment", "3compartment_parameters.csv", resources)
    inp.update({"model_type": "aadc_python", "generated_models_dir": d,
                "solver": "aadc_semi_implicit", "solver_info": {"method": "semi_implicit_signed"}})
    assert generate_with_new_architecture(False, inp), "3compartment aadc model generation failed"
    return os.path.join(d, "3compartment", "3compartment.py")


def _kernel_pid(model_path, sim_time, max_step=0.001):
    """Build a minimal param-id object that routes cost_and_grad to the aadc-ode kernel path."""
    from libcuflynx.solver_wrappers import get_simulation_helper
    sim = get_simulation_helper(
        model_path=model_path, solver="aadc_semi_implicit", model_type="aadc_python",
        dt=0.01, sim_time=sim_time, pre_time=0.0,
        solver_info={"method": "semi_implicit_signed", "gradient_strategy": "kernel",
                     "max_step": max_step, "jac_lag": 10})
    sim.run()
    pid = types.SimpleNamespace(
        sim_helper=sim, param_id_info={"param_names": [_PNAME]},
        obs_info={"ground_truth_const": [0.0], "std_const_vec": [1.0],
                  "operations": ["mean"], "operands": [_SNAME], "data_types": ["constant"]},
        protocol_info=None, cost_type=["gaussian_MLE"])
    p0 = float(sim._numeric_variables_all[sim._resolve_name(_PNAME)[1]])
    return pid, p0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("sim_time", [1.0, 5.0])
def test_bdf_kernel_gradient_vs_fd(aadc_3compartment, sim_time):
    """AD gradient from the aadc-ode kernel path matches a central FD of the same cost."""
    pytest.importorskip("aadc")
    pytest.importorskip("aadc_ode")
    from libcuflynx.param_id import aadc_backend

    pid, p0 = _kernel_pid(aadc_3compartment, sim_time)
    cost, grad = aadc_backend.cost_and_grad(pid, [p0])
    assert np.isfinite(cost) and np.all(np.isfinite(grad)), "kernel cost/grad not finite"

    h = abs(p0) * 1e-5 if p0 != 0 else 1e-6
    cp, _ = aadc_backend.cost_and_grad(pid, [p0 + h])
    cm, _ = aadc_backend.cost_and_grad(pid, [p0 - h])
    fd = (cp - cm) / (2 * h)

    if abs(fd) > 1e-30:
        ratio = grad[0] / fd
        # frozen-Jacobian-diagonal preconditioner -> ~1% at dt=0.01, improving with finer stepping
        assert abs(ratio - 1.0) < 0.02, f"AD/FD = {ratio:.6f} (sim_time={sim_time}), expected ~1.0"
    else:
        assert abs(grad[0]) < 1e-9, f"FD≈0 but AD={grad[0]:.3e}"
