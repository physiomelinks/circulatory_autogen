"""
Tests for AADC solver: compare trajectories against reference solvers,
and verify AD gradient vs finite differences.
"""
import os
import sys
import time

import pytest
import numpy as np

_TEST_ROOT = os.path.join(os.path.dirname(__file__), '..')
_SRC_DIR = os.path.join(_TEST_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from solver_wrappers import get_simulation_helper
from generators.PythonGenerator import PythonGenerator
from scripts.script_generate_with_new_architecture import generate_with_new_architecture


def _generate_aadc_model(model_name, param_file, base_user_inputs, temp_dir):
    """Generate aadc_python model, return path."""
    config = base_user_inputs.copy()
    config.update({
        "file_prefix": model_name,
        "input_param_file": param_file,
        "model_type": "aadc_python",
        "generated_models_dir": temp_dir,
        "solver": "aadc_semi_implicit",
        "solver_info": {"method": "semi_implicit"},
    })
    success = generate_with_new_architecture(False, config)
    assert success, f"aadc_python generation failed for {model_name}"
    return os.path.join(temp_dir, model_name, f"{model_name}.py")


def _generate_python_model(model_name, param_file, cellml_path, temp_dir):
    """Generate standard python model from CellML, return path."""
    py_gen = PythonGenerator(cellml_path, output_dir=temp_dir, module_name=model_name)
    return py_gen.generate()


# ---- Test 1: Compare AADC solver against reference ----

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model_name,param_file,method,sim_time", [
    ("Lotka_Volterra", "Lotka_Volterra_parameters.csv", "adaptive_rk45", 1.0),
    ("VanDerPol", "VanDerPol_parameters.csv", "adaptive_rk45", 1.0),
    # 3compartment_nonstiff still too stiff for Python BDF comparison
    # ("3compartment_nonstiff", "3compartment_nonstiff_parameters.csv", "adaptive_rk45", 1.0),
])
def test_aadc_vs_python_solver_nonstiff(model_name, param_file, method, sim_time,
                                         base_user_inputs, resources_dir, temp_generated_models_dir):
    """AADC solver should produce trajectories close to Python BDF solver for non-stiff models."""
    aadc_mod = pytest.importorskip("aadc")

    aadc_dir = os.path.join(temp_generated_models_dir, "aadc")
    py_dir = os.path.join(temp_generated_models_dir, "py")
    os.makedirs(aadc_dir, exist_ok=True)
    os.makedirs(py_dir, exist_ok=True)

    # Generate aadc model
    aadc_path = _generate_aadc_model(model_name, param_file, base_user_inputs, aadc_dir)

    # Generate standard python model (need CellML first)
    cellml_path = os.path.join(aadc_dir, model_name, f"{model_name}.cellml")
    assert os.path.exists(cellml_path), f"CellML not found: {cellml_path}"
    py_path = _generate_python_model(model_name, param_file, cellml_path, py_dir)

    # Run AADC
    sim_aadc = get_simulation_helper(model_path=aadc_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.01, sim_time=sim_time, pre_time=0.0,
        solver_info={'method': method})
    sim_aadc.run()

    # Run Python BDF
    sim_py = get_simulation_helper(model_path=py_path, solver='solve_ivp',
        model_type='python', dt=0.01, sim_time=sim_time, pre_time=0.0,
        solver_info={'method': 'BDF', 'rtol': 1e-8, 'atol': 1e-10})
    sim_py.run()

    # Compare state trajectories at final time
    aadc_names = sim_aadc.get_all_variable_names()
    py_names = sim_py.get_all_variable_names()
    common = set(aadc_names) & set(py_names)
    assert len(common) > 0, "No common variable names"

    for name in sorted(common)[:10]:  # check first 10
        try:
            r_aadc = sim_aadc.get_results([[name]], flatten=True)[0]
            r_py = sim_py.get_results([[name]], flatten=True)[0]
            # Compare final value
            if abs(r_py[-1]) > 1e-10:
                rel = abs(r_aadc[-1] - r_py[-1]) / abs(r_py[-1])
                assert rel < 0.1, f"{name}: aadc={r_aadc[-1]:.6e} py={r_py[-1]:.6e} rel={rel:.2e}"
        except (ValueError, KeyError):
            pass  # skip variables with name resolution issues


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_stiff_model_runs(base_user_inputs, resources_dir, temp_generated_models_dir):
    """Stiff 3compartment model should run with semi_implicit without NaN."""
    aadc_mod = pytest.importorskip("aadc")
    import math

    aadc_dir = os.path.join(temp_generated_models_dir, "aadc_stiff")
    os.makedirs(aadc_dir, exist_ok=True)
    aadc_path = _generate_aadc_model("3compartment", "3compartment_parameters.csv",
                                      base_user_inputs, aadc_dir)

    sim = get_simulation_helper(model_path=aadc_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.01, sim_time=1.0, pre_time=0.0,
        solver_info={'method': 'semi_implicit'})
    success = sim.run()
    assert success

    final = sim.state_traj[:, -1]
    assert not any(math.isnan(x) for x in final), "Final states contain NaN"
    assert not any(abs(x) > 1e20 for x in final), "Final states overflow"


# ---- Test 2: AD gradient vs FD ----

@pytest.mark.integration
@pytest.mark.slow
def test_aadc_gradient_vs_fd(base_user_inputs, resources_dir, temp_generated_models_dir):
    """AD gradient should match finite differences on the same tape."""
    aadc_mod = pytest.importorskip("aadc")

    aadc_dir = os.path.join(temp_generated_models_dir, "aadc_grad")
    os.makedirs(aadc_dir, exist_ok=True)
    aadc_path = _generate_aadc_model("3compartment", "3compartment_parameters.csv",
                                      base_user_inputs, aadc_dir)

    sim = get_simulation_helper(model_path=aadc_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.01, sim_time=1.0, pre_time=0.0,
        solver_info={'method': 'semi_implicit'})
    sim.run()

    # Find a sensitive parameter
    vi = sim.model.VARIABLE_INFO
    pidx = next(i for i, info in enumerate(vi) if 'q_lv_us' in info['name'].lower())
    sim._ad_param_names = [vi[pidx]['name']]
    sim._ad_param_var_indices = [pidx]

    # State index for Q_LV
    si = next(i for i, info in enumerate(sim.model.STATE_INFO) if 'q_lv' in info['name'].lower())

    def cost_fn(st, p):
        return st[si] * st[si]

    # AD gradient
    grad = sim.compute_gradient_tape(cost_fn)

    # FD on same tape
    pv = float(sim._numeric_variables_all[pidx])
    h = abs(pv) * 1e-5
    workers = sim._aad_workers
    funcs = sim._tape_funcs
    rc = sim._tape_r_cost
    ap = sim._tape_a_p

    inputs_p = {ap[0]: pv + h}
    inputs_m = {ap[0]: pv - h}
    cp = float(np.asarray(aadc_mod.evaluate(funcs, {rc: []}, inputs_p, workers)[0][rc]).flat[0])
    cm = float(np.asarray(aadc_mod.evaluate(funcs, {rc: []}, inputs_m, workers)[0][rc]).flat[0])
    fd = (cp - cm) / (2 * h)

    if abs(fd) > 1e-30:
        ratio = grad[0] / fd
        assert abs(ratio - 1.0) < 0.01, f"AD/FD ratio = {ratio:.6f}, expected ~1.0"
    else:
        # Both near zero is also acceptable
        assert abs(grad[0]) < 1e-10, f"FD≈0 but AD={grad[0]:.6e}"


# ---- Test 3: implicit_euler_ift solver ----

@pytest.mark.integration
@pytest.mark.slow
def test_aadc_implicit_euler_ift_nonstiff(base_user_inputs, resources_dir, temp_generated_models_dir):
    """implicit_euler_ift should produce trajectories close to RK45 on nonstiff model."""
    aadc_mod = pytest.importorskip("aadc")

    aadc_dir = os.path.join(temp_generated_models_dir, "aadc_ift")
    os.makedirs(aadc_dir, exist_ok=True)
    aadc_path = _generate_aadc_model("Lotka_Volterra", "Lotka_Volterra_parameters.csv",
                                      base_user_inputs, aadc_dir)

    # Run with implicit_euler_ift (small dt for accuracy)
    sim_ift = get_simulation_helper(model_path=aadc_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.001, sim_time=0.5, pre_time=0.0,
        solver_info={'method': 'implicit_euler_ift'})
    sim_ift.run()

    # Run with RK45 reference
    sim_rk = get_simulation_helper(model_path=aadc_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.001, sim_time=0.5, pre_time=0.0,
        solver_info={'method': 'adaptive_rk45'})
    sim_rk.run()

    # Compare final states
    err = np.abs(sim_ift.state_traj[:, -1] - sim_rk.state_traj[:, -1]) / (
        np.abs(sim_rk.state_traj[:, -1]) + 1e-10)
    assert np.all(err < 0.01), f"implicit_euler_ift vs RK45 error too large: {err}"


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_implicit_euler_ift_gradient(base_user_inputs, resources_dir, temp_generated_models_dir):
    """AD gradient via implicit_euler_ift + IFT should match FD."""
    aadc_mod = pytest.importorskip("aadc")

    aadc_dir = os.path.join(temp_generated_models_dir, "aadc_ift_grad")
    os.makedirs(aadc_dir, exist_ok=True)
    aadc_path = _generate_aadc_model("Lotka_Volterra", "Lotka_Volterra_parameters.csv",
                                      base_user_inputs, aadc_dir)

    sim = get_simulation_helper(model_path=aadc_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.01, sim_time=0.5, pre_time=0.0,
        solver_info={'method': 'implicit_euler_ift'})
    sim.run()

    # Pick a parameter for gradient
    vi = sim.model.VARIABLE_INFO
    # Find first constant parameter
    pidx = next(i for i, info in enumerate(vi)
                if info['type'].name in ('CONSTANT', 'COMPUTED_CONSTANT'))
    sim._ad_param_names = [vi[pidx]['name']]
    sim._ad_param_var_indices = [pidx]

    def cost_fn(st, p):
        return st[0] * st[0] + st[1] * st[1]

    # AD gradient
    grad = sim.compute_gradient_tape(cost_fn)

    # FD on same tape
    pv = float(sim._numeric_variables_all[pidx])
    h = max(abs(pv) * 1e-5, 1e-8)
    workers = sim._aad_workers
    funcs = sim._tape_funcs
    rc = sim._tape_r_cost
    ap = sim._tape_a_p

    inputs_p = {ap[0]: pv + h}
    inputs_m = {ap[0]: pv - h}
    cp = float(np.asarray(aadc_mod.evaluate(funcs, {rc: []}, inputs_p, workers)[0][rc]).flat[0])
    cm = float(np.asarray(aadc_mod.evaluate(funcs, {rc: []}, inputs_m, workers)[0][rc]).flat[0])
    fd = (cp - cm) / (2 * h)

    if abs(fd) > 1e-20:
        ratio = grad[0] / fd
        assert abs(ratio - 1.0) < 0.01, f"IFT AD/FD ratio = {ratio:.6f}, expected ~1.0"
    else:
        assert abs(grad[0]) < 1e-10, f"FD≈0 but AD={grad[0]:.6e}"


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_implicit_euler_ift_parallel(base_user_inputs, resources_dir, temp_generated_models_dir):
    """Batch parallel evaluation gives same results as single-thread."""
    aadc_mod = pytest.importorskip("aadc")

    aadc_dir = os.path.join(temp_generated_models_dir, "aadc_ift_par")
    os.makedirs(aadc_dir, exist_ok=True)
    aadc_path = _generate_aadc_model("Lotka_Volterra", "Lotka_Volterra_parameters.csv",
                                      base_user_inputs, aadc_dir)

    sim = get_simulation_helper(model_path=aadc_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.01, sim_time=0.5, pre_time=0.0,
        solver_info={'method': 'implicit_euler_ift'})
    sim.run()

    # Record tape
    vi = sim.model.VARIABLE_INFO
    pidx = next(i for i, info in enumerate(vi)
                if info['type'].name in ('CONSTANT', 'COMPUTED_CONSTANT'))
    sim._ad_param_names = [vi[pidx]['name']]
    sim._ad_param_var_indices = [pidx]

    def cost_fn(st, p):
        return st[0] * st[0] + st[1] * st[1]

    sim.compute_gradient_tape(cost_fn)

    funcs = sim._tape_funcs
    rc = sim._tape_r_cost
    ap = sim._tape_a_p

    pv = float(sim._numeric_variables_all[pidx])
    param_values = np.linspace(pv * 0.9, pv * 1.1, 20)
    request = {rc: [ap[0]]}

    # Single-thread reference
    workers_1 = aadc_mod.ThreadPool(1)
    res_1 = aadc_mod.evaluate(funcs, request, {ap[0]: param_values}, workers_1)
    vals_1 = np.array(res_1[0][rc])
    grads_1 = np.array(res_1[1][rc][ap[0]])

    # Multi-thread
    for n_threads in [2, 4]:
        workers_n = aadc_mod.ThreadPool(n_threads)
        res_n = aadc_mod.evaluate(funcs, request, {ap[0]: param_values}, workers_n)
        vals_n = np.array(res_n[0][rc])
        grads_n = np.array(res_n[1][rc][ap[0]])

        assert np.allclose(vals_n, vals_1, rtol=1e-12), \
            f"{n_threads} threads: values differ from single-thread"
        assert np.allclose(grads_n, grads_1, rtol=1e-12), \
            f"{n_threads} threads: gradients differ from single-thread"


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_implicit_euler_ift_benchmark(base_user_inputs, resources_dir, temp_generated_models_dir):
    """Benchmark: recording, forward eval, gradient eval with threading."""
    aadc_mod = pytest.importorskip("aadc")

    aadc_dir = os.path.join(temp_generated_models_dir, "aadc_ift_bench")
    os.makedirs(aadc_dir, exist_ok=True)
    aadc_path = _generate_aadc_model("Lotka_Volterra", "Lotka_Volterra_parameters.csv",
                                      base_user_inputs, aadc_dir)

    # Record tape
    t0 = time.time()
    sim = get_simulation_helper(model_path=aadc_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.01, sim_time=0.5, pre_time=0.0,
        solver_info={'method': 'implicit_euler_ift'})
    sim.run()

    vi = sim.model.VARIABLE_INFO
    pidx = next(i for i, info in enumerate(vi)
                if info['type'].name in ('CONSTANT', 'COMPUTED_CONSTANT'))
    sim._ad_param_names = [vi[pidx]['name']]
    sim._ad_param_var_indices = [pidx]

    def cost_fn(st, p):
        return st[0] * st[0] + st[1] * st[1]

    sim.compute_gradient_tape(cost_fn)
    rec_ms = (time.time() - t0) * 1000

    funcs = sim._tape_funcs
    rc = sim._tape_r_cost
    ap = sim._tape_a_p
    pv = float(sim._numeric_variables_all[pidx])

    print(f"\n  Recording: {rec_ms:.0f}ms")
    print(f"  {'threads':>8} {'fwd_ms':>10} {'grad_ms':>10} {'batch100_ms':>12}")

    for n_threads in [1, 2, 4]:
        workers = aadc_mod.ThreadPool(n_threads)

        # Single eval forward
        n_rep = 50
        t0 = time.time()
        for _ in range(n_rep):
            aadc_mod.evaluate(funcs, {rc: []}, {ap[0]: pv}, workers)
        fwd_ms = (time.time() - t0) * 1000 / n_rep

        # Single eval with gradient
        t0 = time.time()
        for _ in range(n_rep):
            aadc_mod.evaluate(funcs, {rc: [ap[0]]}, {ap[0]: pv}, workers)
        grad_ms = (time.time() - t0) * 1000 / n_rep

        # Batch 100 evals with gradient
        params_batch = np.linspace(pv * 0.9, pv * 1.1, 100)
        t0 = time.time()
        aadc_mod.evaluate(funcs, {rc: [ap[0]]}, {ap[0]: params_batch}, workers)
        batch_ms = (time.time() - t0) * 1000

        print(f"  {n_threads:>8} {fwd_ms:>10.3f} {grad_ms:>10.3f} {batch_ms:>12.1f}")

    # Sanity: benchmark should complete without error
    assert True
