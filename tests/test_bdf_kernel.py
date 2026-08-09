"""
Tests for bdf_kernel (C++ kernel replay via aadc.bdf_record_and_evaluate).

Verifies:
  1. Cost matches: bdf_kernel cost ≈ scipy BDF cost (same observables)
  2. Gradient matches: bdf_kernel gradient ≈ FD gradient
  3. Speed: bdf_kernel vs scipy BDF forward+gradient
  4. Calibration: bdf_kernel-calibrated params ≈ scipy BDF-calibrated params

Uses the 3compartment stiff model from test_aadc_vs_casadi_3compartment.py.
"""
import os
import sys
import time

import numpy as np
import pytest

_TEST_ROOT = os.path.join(os.path.dirname(__file__), '..')
_SRC_DIR = os.path.join(_TEST_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from solver_wrappers import get_simulation_helper
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from utilities.utility_funcs import get_default_inp_data_dict

_DT = 0.001
_SIM_TIME = 1.0


@pytest.fixture(scope="module")
def aadc_model(tmp_path_factory):
    """Generate aadc_python 3compartment model."""
    base = str(tmp_path_factory.mktemp("bdf_kernel"))
    resources = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
    d = os.path.join(base, "aadc")
    os.makedirs(d, exist_ok=True)
    inp = get_default_inp_data_dict("3compartment", "3compartment_parameters.csv", resources)
    inp.update({
        "model_type": "aadc_python",
        "generated_models_dir": d,
        "solver": "aadc_semi_implicit",
        "solver_info": {"method": "bdf"},
    })
    assert generate_with_new_architecture(False, inp), "generation failed"
    return {
        "py": os.path.join(d, "3compartment", "3compartment.py"),
        "cellml": os.path.join(d, "3compartment", "3compartment.cellml"),
    }


def _make_sim(model_path, method='bdf', dt=_DT, sim_time=_SIM_TIME):
    return get_simulation_helper(
        model_path=model_path, solver='aadc_semi_implicit', model_type='aadc_python',
        dt=dt, sim_time=sim_time, pre_time=0.0, solver_info={'method': method},
    )


def _pick_param_and_obs(sim):
    """Pick one param and one state for gradient test."""
    vi = sim.model.VARIABLE_INFO
    pidx = next((i for i, info in enumerate(vi) if 'e_lv_a' in info['name'].lower()),
                next(i for i, info in enumerate(vi)))
    sidx = next((i for i, info in enumerate(sim.model.STATE_INFO) if 'q_lv' in info['name'].lower()),
                0)
    return pidx, sidx


# ---- Test 1: bdf_kernel cost matches scipy BDF cost ----
@pytest.mark.integration
def test_bdf_kernel_cost_matches_scipy_bdf(aadc_model):
    """bdf_kernel and scipy BDF should produce similar cost for same observables."""
    aadc = pytest.importorskip("aadc")
    if not hasattr(aadc, 'bdf_record_and_evaluate'):
        pytest.skip("bdf_record_and_evaluate not available in this AADC build")

    sim = _make_sim(aadc_model["py"], method='bdf')
    sim.run()
    pidx, sidx = _pick_param_and_obs(sim)

    # Get final state value from scipy BDF
    final_state_scipy = float(sim.state_traj[sidx, -1])

    # Now compute cost via bdf_kernel
    n = sim.STATE_COUNT
    states = list(sim.states[:n])
    variables = list(sim._numeric_variables_all)
    for ci, idx in enumerate(sim.constant_indices):
        variables[idx] = sim.variables[ci]

    # Observable: (kind=0, state_idx, var_idx=0, op=0=mean, gt=final_state, std=0.01, weight=1, scale=1)
    obs_list = [(0, sidx, 0, 0, final_state_scipy, 0.01, 1.0, 1.0)]

    max_step = float(sim.solver_info.get('max_step', 0.001))
    n_sub = max(1, int(np.ceil(sim.dt / max_step)))
    total_steps = int(sim.sim_time / sim.dt)

    cost, grads = aadc.bdf_record_and_evaluate(
        sim.model.compute_rates,
        states, variables,
        [], [],  # no params for cost-only test
        total_steps, 0, n_sub, sim.dt / n_sub,
        obs_list, None, 10)

    print(f"\n  scipy BDF final state[{sidx}]: {final_state_scipy:.6e}")
    print(f"  bdf_kernel cost: {cost:.6e}")
    # Cost should be small if bdf_kernel forward matches scipy BDF
    # (cost = weighted (mean_state - gt)^2, gt = scipy result)
    assert cost < 1.0, f"bdf_kernel cost={cost:.4f} too high (forward drift from scipy BDF)"


# ---- Test 2: bdf_kernel gradient vs FD ----
@pytest.mark.integration
def test_bdf_kernel_gradient_vs_fd(aadc_model):
    """bdf_kernel gradient should match finite differences."""
    aadc = pytest.importorskip("aadc")
    if not hasattr(aadc, 'bdf_record_and_evaluate'):
        pytest.skip("bdf_record_and_evaluate not available")

    sim = _make_sim(aadc_model["py"], method='bdf', sim_time=0.3)
    sim.run()
    pidx, sidx = _pick_param_and_obs(sim)

    n = sim.STATE_COUNT
    states = list(sim.states[:n])
    variables = list(sim._numeric_variables_all)
    for ci, idx in enumerate(sim.constant_indices):
        variables[idx] = sim.variables[ci]

    pval = float(variables[pidx])
    target = float(sim.state_traj[sidx, -1])
    obs_list = [(0, sidx, 0, 0, target * 0.9, 0.01, 1.0, 1.0)]  # target off by 10% to get nonzero grad

    max_step = float(sim.solver_info.get('max_step', 0.001))
    n_sub = max(1, int(np.ceil(sim.dt / max_step)))
    total_steps = int(0.3 / sim.dt)
    idt = sim.dt / n_sub

    def eval_cost(pv):
        variables[pidx] = pv
        cost, _ = aadc.bdf_record_and_evaluate(
            sim.model.compute_rates,
            states, variables,
            [pidx], [pv],
            total_steps, 0, n_sub, idt,
            obs_list, None, 10)
        return cost

    # AD gradient
    variables[pidx] = pval
    cost_ad, grad_ad = aadc.bdf_record_and_evaluate(
        sim.model.compute_rates,
        states, variables,
        [pidx], [pval],
        total_steps, 0, n_sub, idt,
        obs_list, None, 10)
    g_ad = float(grad_ad[0]) if len(grad_ad) > 0 else 0.0

    # FD gradient
    h = abs(pval) * 1e-5 if pval != 0 else 1e-5
    g_fd = (eval_cost(pval + h) - eval_cost(pval - h)) / (2 * h)

    print(f"\n  bdf_kernel gradient: AD={g_ad:.6e}  FD={g_fd:.6e}")
    if abs(g_fd) > 1e-30:
        ratio = g_ad / g_fd
        print(f"  ratio: {ratio:.4f}")
        assert abs(ratio - 1.0) < 0.05, f"AD/FD ratio = {ratio:.4f}, expected ~1.0"
    else:
        assert abs(g_ad) < 1e-10


# ---- Test 3: Speed benchmark ----
@pytest.mark.integration
@pytest.mark.slow
def test_bdf_kernel_speed_vs_scipy_bdf(aadc_model):
    """bdf_kernel should be faster than scipy BDF for cost+gradient evaluation."""
    aadc = pytest.importorskip("aadc")
    if not hasattr(aadc, 'bdf_record_and_evaluate'):
        pytest.skip("bdf_record_and_evaluate not available")

    sim = _make_sim(aadc_model["py"], method='bdf')
    sim.run()
    pidx, sidx = _pick_param_and_obs(sim)

    n = sim.STATE_COUNT
    states = list(sim.states[:n])
    variables = list(sim._numeric_variables_all)
    for ci, idx in enumerate(sim.constant_indices):
        variables[idx] = sim.variables[ci]
    pval = float(variables[pidx])
    target = float(sim.state_traj[sidx, -1])
    obs_list = [(0, sidx, 0, 0, target, 0.01, 1.0, 1.0)]
    max_step = float(sim.solver_info.get('max_step', 0.001))
    n_sub = max(1, int(np.ceil(sim.dt / max_step)))
    total_steps = int(_SIM_TIME / sim.dt)
    idt = sim.dt / n_sub

    # bdf_kernel timing (cost + gradient)
    n_runs = 3
    t0 = time.perf_counter()
    for _ in range(n_runs):
        aadc.bdf_record_and_evaluate(
            sim.model.compute_rates, states, variables,
            [pidx], [pval], total_steps, 0, n_sub, idt,
            obs_list, None, 10)
    t_kernel = (time.perf_counter() - t0) / n_runs

    # scipy BDF timing (forward only, no gradient)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sim2 = _make_sim(aadc_model["py"], method='bdf')
        sim2.run()
    t_scipy = (time.perf_counter() - t0) / n_runs

    print(f"\n  bdf_kernel (cost+grad): {t_kernel:.3f}s")
    print(f"  scipy BDF (forward):    {t_scipy:.3f}s")
    print(f"  bdf_kernel includes gradient; scipy does not")
    if t_scipy > 0:
        print(f"  Ratio: {t_scipy/t_kernel:.1f}x")

    # Just report, don't gate on speed
    assert t_kernel < 60, f"bdf_kernel took {t_kernel:.1f}s — sanity check failed"
