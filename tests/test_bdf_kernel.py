"""
Tests for the bdf_kernel gradient strategy via the pip-installable aadc-ode package
(solver_info method 'semi_implicit_signed', gradient_strategy 'kernel').

On the stiff 3compartment model (27 states), with the kernel path forced (the tape fallback
raises, so a silent fallback fails the test):
  - the AD gradient matches a central finite difference of the SAME estimator, with and without
    a pre_time phase (pre_time > 0 runs aadc_ode.bdf_solve);
  - max_minus_min and algebraic-variable operands are on the tape cost;
  - an observable the kernel cannot represent raises.
These need `pip install "libcuflynx[aadc]"` and skip without it.

Without aadc-ode, the kernel strategy warns and falls back to the tape strategy; that test and
the newton_iters one use a stand-in sim helper and run anywhere.
"""
import os
import sys
import types

import numpy as np
import pytest

_PNAME = "q_lv_us"      # a sensitive constant parameter
_SNAME = "q_lv"         # a state observable
_VNAME = "aortic_root/u"  # an algebraic-variable observable (a pressure)


@pytest.fixture(scope="module")
def aadc_3compartment(tmp_path_factory, aadc_licensed):
    """Generate the aadc_python 3compartment model once for the module."""
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


@pytest.fixture
def kernel_only(monkeypatch):
    """Make the tape fallback fail, and count aadc_ode.bdf_solve calls."""
    import aadc_ode
    from libcuflynx.param_id import aadc_backend

    def no_fallback(pid, param_vals):
        raise AssertionError("gradient_strategy 'kernel' fell back to the tape strategy")

    monkeypatch.setattr(aadc_backend, "_cost_and_grad_bdf_tape", no_fallback)
    calls = []
    real_bdf_solve = aadc_ode.bdf_solve

    def counting_bdf_solve(*args, **kwargs):
        calls.append(kwargs.get("n_steps"))
        return real_bdf_solve(*args, **kwargs)

    monkeypatch.setattr(aadc_ode, "bdf_solve", counting_bdf_solve)
    return calls


def _kernel_pid(model_path, sim_time, pre_time=0.0, operation="mean", operand=_SNAME,
                max_step=0.001):
    """Build a minimal param-id object that routes cost_and_grad to the aadc-ode kernel path."""
    from libcuflynx.solver_wrappers import get_simulation_helper
    sim = get_simulation_helper(
        model_path=model_path, solver="aadc_semi_implicit", model_type="aadc_python",
        dt=0.01, sim_time=sim_time, pre_time=pre_time,
        solver_info={"method": "semi_implicit_signed", "gradient_strategy": "kernel",
                     "max_step": max_step, "jac_lag": 10})
    sim.run()
    pid = types.SimpleNamespace(
        sim_helper=sim, param_id_info={"param_names": [_PNAME]},
        obs_info={"ground_truth_const": [0.0], "std_const_vec": [1.0],
                  "operations": [operation], "operands": [operand], "data_types": ["constant"]},
        protocol_info=None, cost_type=["gaussian_MLE"])
    p0 = float(sim._numeric_variables_all[sim._resolve_name(_PNAME)[1]])
    return pid, p0


def _kernel_cost_and_grad_vs_fd(pid, p0, tol=0.02):
    """Run the kernel path at p0 and check its gradient against a central FD of the same cost."""
    from libcuflynx.param_id import aadc_backend

    cost, grad = aadc_backend.cost_and_grad(pid, [p0])
    assert hasattr(pid, "_aadc_ode_kernel"), "the aadc-ode kernel was not recorded"
    assert np.isfinite(cost) and np.all(np.isfinite(grad)), "kernel cost/grad not finite"

    h = abs(p0) * 1e-5 if p0 != 0 else 1e-6
    cp, _ = aadc_backend.cost_and_grad(pid, [p0 + h])
    cm, _ = aadc_backend.cost_and_grad(pid, [p0 - h])
    fd = (cp - cm) / (2 * h)

    if abs(fd) > 1e-30:
        ratio = grad[0] / fd
        # frozen-Jacobian-diagonal preconditioner -> ~1% at dt=0.01, improving with finer stepping
        assert abs(ratio - 1.0) < tol, f"AD/FD = {ratio:.6f}, expected ~1.0"
    else:
        assert abs(grad[0]) < 1e-9, f"FD≈0 but AD={grad[0]:.3e}"
    return cost, grad


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("sim_time", [1.0, 5.0])
def test_bdf_kernel_gradient_vs_fd(aadc_3compartment, kernel_only, sim_time):
    """AD gradient from the aadc-ode kernel path matches a central FD of the same cost."""
    pid, p0 = _kernel_pid(aadc_3compartment, sim_time)
    _kernel_cost_and_grad_vs_fd(pid, p0)
    assert kernel_only == [], "pre_time=0 has no pre-steps, so bdf_solve should not run"


@pytest.mark.integration
@pytest.mark.slow
def test_bdf_kernel_with_pre_time_runs_bdf_solve(aadc_3compartment, kernel_only):
    """A pre_time phase goes through aadc_ode.bdf_solve, and the gradient through it is right."""
    pid, p0 = _kernel_pid(aadc_3compartment, sim_time=1.0, pre_time=1.0)
    _kernel_cost_and_grad_vs_fd(pid, p0)
    # three evaluations (p0 and p0 +/- h), each 1.0 s / 0.001 s of pre-steps
    assert kernel_only == [1000, 1000, 1000]


@pytest.mark.integration
@pytest.mark.slow
def test_bdf_kernel_max_minus_min(aadc_3compartment, kernel_only):
    """max_minus_min is the running max less the running min, with a correct gradient."""
    from libcuflynx.param_id import aadc_backend

    costs = {}
    for op in ("max", "min"):
        pid, p0 = _kernel_pid(aadc_3compartment, sim_time=1.0, operation=op)
        costs[op], _ = aadc_backend.cost_and_grad(pid, [p0])
    pid, p0 = _kernel_pid(aadc_3compartment, sim_time=1.0, operation="max_minus_min")
    cost, _ = _kernel_cost_and_grad_vs_fd(pid, p0)

    # ground truth 0, std 1, gaussian_MLE: cost = 0.5 * value**2, and q_lv stays positive
    q_max, q_min = np.sqrt(2 * costs["max"]), np.sqrt(2 * costs["min"])
    assert q_max > q_min > 0
    assert np.sqrt(2 * cost) == pytest.approx(q_max - q_min, rel=1e-9)


@pytest.mark.integration
@pytest.mark.slow
def test_bdf_kernel_algebraic_operand(aadc_3compartment, kernel_only):
    """An algebraic-variable operand is recomputed on the tape, not dropped from the cost."""
    pid, p0 = _kernel_pid(aadc_3compartment, sim_time=1.0, operand=_VNAME)
    # _kernel_pid has run the forward solve at p0
    forward_mean = float(np.mean(pid.sim_helper.get_results([_VNAME], flatten=True)[0]))
    cost, grad = _kernel_cost_and_grad_vs_fd(pid, p0)
    assert grad[0] != 0.0

    # The tape's mean matches the forward solve's mean of the same variable (same scheme,
    # sampled on the sub-step grid rather than the output grid).
    assert np.sqrt(2 * cost) == pytest.approx(abs(forward_mean), rel=0.02)


@pytest.mark.integration
def test_bdf_kernel_refuses_an_observable_it_cannot_tape(aadc_3compartment, kernel_only):
    """An operation the kernel does not re-implement raises rather than leaving the cost."""
    from libcuflynx.param_id import aadc_backend

    pid, p0 = _kernel_pid(aadc_3compartment, sim_time=0.1, operation="std")
    with pytest.raises(NotImplementedError, match="cannot represent"):
        aadc_backend.cost_and_grad(pid, [p0])


def _stub_pid(**solver_info):
    """A param-id stand-in that reaches the kernel strategy without a model or AADC."""
    sim_helper = types.SimpleNamespace(
        solver_info={"method": "semi_implicit_signed", "gradient_strategy": "kernel",
                     **solver_info},
        set_param_vals=lambda names, vals: None,
        _resolve_name=lambda name: ("var", 0))
    return types.SimpleNamespace(sim_helper=sim_helper, param_id_info={"param_names": ["p"]})


@pytest.fixture
def no_aadc_ode(monkeypatch):
    """aadc_ode is not importable, and the tape strategy returns a recognisable result."""
    from libcuflynx.param_id import aadc_backend

    monkeypatch.setitem(sys.modules, "aadc_ode", None)
    tape_result = (1.25, np.array([0.5]))
    monkeypatch.setattr(aadc_backend, "_cost_and_grad_bdf_tape",
                        lambda pid, param_vals: tape_result)
    return tape_result


@pytest.mark.unit
def test_kernel_without_aadc_ode_warns_and_falls_back_to_tape(no_aadc_ode):
    from libcuflynx.param_id import aadc_backend

    with pytest.warns(UserWarning, match=r'falling back to gradient_strategy .tape.*'
                                         r'pip install "libcuflynx\[aadc\]"'):
        result = aadc_backend.cost_and_grad(_stub_pid(), [1.0])
    assert result is no_aadc_ode


@pytest.mark.unit
def test_kernel_says_newton_iters_is_ignored(no_aadc_ode):
    from libcuflynx.param_id import aadc_backend

    with pytest.warns(UserWarning, match="newton_iters=2 is ignored"):
        aadc_backend.cost_and_grad(_stub_pid(newton_iters=2), [1.0])
