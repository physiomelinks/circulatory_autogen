"""AADC vs CasADi on a coupled pipeline of two generated CellML models.

Two models are chained onto a single AADC tape:

    Model A: Lotka-Volterra   (250 RK4 steps, 2 states)
      -> max(prey) * 0.1 -> mu ->
    Model B: Van der Pol      (250 RK4 steps, 2 states)
      -> cost = (u_final - 0.5)^2 + v_final^2

so the gradient has to flow back through 500 integration steps and the coupling point in one
reverse pass. This is the case AADC's tape is meant to be good at, and the test asserts the
gradient is *exact* (agrees with a finite difference over a full numpy re-solve) both at the
true parameters and at perturbed ones -- checking only at the true parameters would not catch
a gradient that goes wrong once an optimiser starts moving.

Was tests/benchmark_aadc_coupled_pipeline.py, which pytest never collected (it doesn't match
python_files = test_*.py) and which scavenged models other tests happened to leave behind in
tests/test_outputs. It now generates its own models and runs under the aadc_licensed fixture,
so it executes wherever a Matlogica licence exists and skips cleanly everywhere else.
"""
import os
import time

import numpy as np
import pytest

from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from solver_wrappers import get_simulation_helper

# The coupled pipeline. Kept as module constants so the AADC tape, the CasADi graph and the
# numpy reference below all integrate exactly the same system.
DT = 0.02
N_STEPS_LV = 250
N_STEPS_VDP = 250
COUPLING_GAIN = 0.1
VDP_TARGET_U = 0.5
FD_STEP = 1e-7
# The tape gradient must agree with a finite difference over a full re-solve to this
# relative tolerance. It is tight on purpose: an "approximately right" gradient is what
# breaks the L-BFGS-B line search.
GRADIENT_RTOL = 0.01


def _generate_aadc_model(model_name, param_file, base_user_inputs, temp_dir):
    config = base_user_inputs.copy()
    config.update({
        "file_prefix": model_name,
        "input_param_file": param_file,
        "model_type": "aadc_python",
        "generated_models_dir": temp_dir,
        "solver": "aadc_semi_implicit",
        "solver_info": {"method": "semi_implicit"},
    })
    assert generate_with_new_architecture(False, config), \
        f"aadc_python generation failed for {model_name}"
    return os.path.join(temp_dir, model_name, f"{model_name}.py")


def _load_sim(model_path):
    return get_simulation_helper(
        model_path=model_path, solver='aadc_semi_implicit', model_type='aadc_python',
        dt=DT, sim_time=5.0, pre_time=0.0, solver_info={'method': 'adaptive_rk45'})


def _resolved_variables(sim):
    """The model's variable vector with the calibratable constants substituted in."""
    variables = list(sim._numeric_variables_all)
    for const_pos, const_idx in enumerate(sim.constant_indices):
        variables[const_idx] = sim.variables[const_pos]
    return variables


def _variable_by_name(sim, needle):
    variables = _resolved_variables(sim)
    idx = next(i for i, info in enumerate(sim.model.VARIABLE_INFO)
               if needle in info['name'].lower())
    return float(variables[idx])


def _make_numpy_reference(delta, gamma, u0, v0):
    """The same coupled pipeline in plain numpy, used as the finite-difference reference.

    Deliberately independent of both AD backends: a finite difference taken *on the tape*
    would still agree with the tape gradient if the tape itself were wrong.
    """
    def coupled(params):
        alpha, beta = params
        x, y = 20.0, 10.0
        max_prey = x
        for _ in range(N_STEPS_LV):
            kx1 = alpha * x - beta * x * y
            ky1 = delta * x * y - gamma * y
            xm, ym = x + .5 * DT * kx1, y + .5 * DT * ky1
            kx2 = alpha * xm - beta * xm * ym
            ky2 = delta * xm * ym - gamma * ym
            xm, ym = x + .5 * DT * kx2, y + .5 * DT * ky2
            kx3 = alpha * xm - beta * xm * ym
            ky3 = delta * xm * ym - gamma * ym
            xm, ym = x + DT * kx3, y + DT * ky3
            kx4 = alpha * xm - beta * xm * ym
            ky4 = delta * xm * ym - gamma * ym
            x += DT / 6 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
            y += DT / 6 * (ky1 + 2 * ky2 + 2 * ky3 + ky4)
            max_prey = max(max_prey, x)

        mu = max_prey * COUPLING_GAIN
        u, v = u0, v0
        for _ in range(N_STEPS_VDP):
            du = v
            dv = mu * (1 - u * u) * v - u
            u += DT * du
            v += DT * dv
        return (u - VDP_TARGET_U) ** 2 + v ** 2

    return coupled


def _central_difference(func, params):
    grad = np.zeros(len(params))
    for j in range(len(params)):
        up, down = np.array(params, dtype=float), np.array(params, dtype=float)
        up[j] += FD_STEP
        down[j] -= FD_STEP
        grad[j] = (func(up) - func(down)) / (2 * FD_STEP)
    return grad


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_coupled_pipeline_gradient_is_exact_and_faster_than_casadi(
        aadc_licensed, base_user_inputs, resources_dir, temp_generated_models_dir):
    import aadc
    from aadc.recording_ctx import record_kernel
    from aadc.evaluate_wrappers import evaluate_kernel

    lv_path = _generate_aadc_model('Lotka_Volterra', 'Lotka_Volterra_parameters.csv',
                                   base_user_inputs, temp_generated_models_dir)
    vdp_path = _generate_aadc_model('VanDerPol', 'VanDerPol_parameters.csv',
                                    base_user_inputs, temp_generated_models_dir)

    sim_lv = _load_sim(lv_path)
    sim_vdp = _load_sim(vdp_path)
    sim_lv.run()
    sim_vdp.run()

    alpha = _variable_by_name(sim_lv, 'alpha')
    beta = _variable_by_name(sim_lv, 'beta')
    delta = _variable_by_name(sim_lv, 'delta')
    gamma = _variable_by_name(sim_lv, 'gamma')
    u0 = float(sim_vdp.states[0])
    v0 = float(sim_vdp.states[1])

    p_true = np.array([alpha, beta])
    coupled_numpy = _make_numpy_reference(delta, gamma, u0, v0)

    # ---- record both models onto one AADC tape ----
    with record_kernel() as kernel:
        a = aadc.idouble(alpha)
        a_arg = a.mark_as_input()
        b = aadc.idouble(beta)
        b_arg = b.mark_as_input()

        d_, g_ = aadc.idouble(delta), aadc.idouble(gamma)
        x, y = aadc.idouble(20.0), aadc.idouble(10.0)
        max_prey = x

        for _ in range(N_STEPS_LV):
            kx1 = a * x - b * x * y
            ky1 = d_ * x * y - g_ * y
            xm = x + aadc.idouble(.5 * DT) * kx1
            ym = y + aadc.idouble(.5 * DT) * ky1
            kx2 = a * xm - b * xm * ym
            ky2 = d_ * xm * ym - g_ * ym
            xm = x + aadc.idouble(.5 * DT) * kx2
            ym = y + aadc.idouble(.5 * DT) * ky2
            kx3 = a * xm - b * xm * ym
            ky3 = d_ * xm * ym - g_ * ym
            xm = x + aadc.idouble(DT) * kx3
            ym = y + aadc.idouble(DT) * ky3
            kx4 = a * xm - b * xm * ym
            ky4 = d_ * xm * ym - g_ * ym
            two = aadc.idouble(2)
            x = x + aadc.idouble(DT / 6) * (kx1 + two * kx2 + two * kx3 + kx4)
            y = y + aadc.idouble(DT / 6) * (ky1 + two * ky2 + two * ky3 + ky4)
            max_prey = aadc.iif(x > max_prey, x, max_prey)

        mu = max_prey * aadc.idouble(COUPLING_GAIN)  # the coupling point

        u, v = aadc.idouble(u0), aadc.idouble(v0)
        for _ in range(N_STEPS_VDP):
            du = v
            dv = mu * (aadc.idouble(1.0) - u * u) * v - u
            u = u + aadc.idouble(DT) * du
            v = v + aadc.idouble(DT) * dv

        cost = (u - aadc.idouble(VDP_TARGET_U)) ** 2 + v ** 2
        cost_out = cost.mark_as_output()

    args = [a_arg, b_arg]
    inputs = {a_arg: alpha, b_arg: beta}

    result = evaluate_kernel(kernel, {cost_out: args}, inputs, num_threads=1)
    aadc_cost = result.values[cost_out].item()
    aadc_grad = np.array([result.derivs[cost_out][arg].item() for arg in args])

    assert np.isfinite(aadc_cost)
    assert np.all(np.isfinite(aadc_grad))

    # ---- the gradient must be exact, at the true params and away from them ----
    fd_true = _central_difference(coupled_numpy, p_true)
    np.testing.assert_allclose(
        aadc_grad, fd_true, rtol=GRADIENT_RTOL,
        err_msg='AADC tape gradient disagrees with a finite difference over a full re-solve '
                'at the true parameters')

    # An optimiser does not sit at the true parameters, so check where it would actually go.
    p_perturbed = p_true * np.array([1.2, 0.8])
    perturbed = evaluate_kernel(
        kernel, {cost_out: args}, {a_arg: p_perturbed[0], b_arg: p_perturbed[1]}, num_threads=1)
    grad_perturbed = np.array([perturbed.derivs[cost_out][arg].item() for arg in args])
    fd_perturbed = _central_difference(coupled_numpy, p_perturbed)
    np.testing.assert_allclose(
        grad_perturbed, fd_perturbed, rtol=GRADIENT_RTOL,
        err_msg='AADC tape gradient disagrees with finite differences at perturbed parameters, '
                'so it cannot be trusted to drive a gradient-based optimiser')

    # ---- timing: AADC tape vs CasADi symbolic graph vs finite differences ----
    def time_aadc():
        t0 = time.time()
        for _ in range(200):
            evaluate_kernel(kernel, {cost_out: args}, inputs, num_threads=1)
        return (time.time() - t0) / 200

    t_aadc = time_aadc()

    t_casadi = None
    try:
        import casadi as ca
    except ImportError:
        ca = None

    if ca is not None:
        a2, b2 = ca.SX.sym('a'), ca.SX.sym('b')
        params_sym = ca.vertcat(a2, b2)
        x2, y2 = ca.SX(20.0), ca.SX(10.0)
        max_prey2 = x2
        for _ in range(N_STEPS_LV):
            kx1 = a2 * x2 - b2 * x2 * y2
            ky1 = delta * x2 * y2 - gamma * y2
            xm, ym = x2 + .5 * DT * kx1, y2 + .5 * DT * ky1
            kx2 = a2 * xm - b2 * xm * ym
            ky2 = delta * xm * ym - gamma * ym
            xm, ym = x2 + .5 * DT * kx2, y2 + .5 * DT * ky2
            kx3 = a2 * xm - b2 * xm * ym
            ky3 = delta * xm * ym - gamma * ym
            xm, ym = x2 + DT * kx3, y2 + DT * ky3
            kx4 = a2 * xm - b2 * xm * ym
            ky4 = delta * xm * ym - gamma * ym
            x2 = x2 + DT / 6 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
            y2 = y2 + DT / 6 * (ky1 + 2 * ky2 + 2 * ky3 + ky4)
            max_prey2 = ca.if_else(x2 > max_prey2, x2, max_prey2)

        mu2 = max_prey2 * COUPLING_GAIN
        u2, v2 = ca.SX(u0), ca.SX(v0)
        for _ in range(N_STEPS_VDP):
            u2, v2 = u2 + DT * v2, v2 + DT * (mu2 * (1 - u2 * u2) * v2 - u2)

        cost2 = (u2 - VDP_TARGET_U) ** 2 + v2 ** 2
        casadi_fn = ca.Function('f', [params_sym], [cost2, ca.gradient(cost2, params_sym)])

        casadi_cost, casadi_grad = casadi_fn(p_true)
        # both AD backends must agree with each other, not just each with FD
        np.testing.assert_allclose(
            np.asarray(casadi_grad).flatten(), aadc_grad, rtol=GRADIENT_RTOL,
            err_msg='the CasADi and AADC gradients disagree on the same pipeline')

        t0 = time.time()
        for _ in range(200):
            casadi_fn(p_true)
        t_casadi = (time.time() - t0) / 200

    t0 = time.time()
    _central_difference(coupled_numpy, p_true)
    t_fd = time.time() - t0

    print(f'\n{"=" * 68}')
    print(f'Coupled CellML pipeline: Lotka-Volterra -> Van der Pol '
          f'({N_STEPS_LV + N_STEPS_VDP} steps on one tape)')
    print(f'cost = {aadc_cost:.6e}   gradient = {aadc_grad}')
    print(f'{"backend":<12} {"per-eval (ms)":>15}   {"vs AADC":>10}')
    print(f'{"AADC":<12} {t_aadc * 1000:>15.3f}   {"-":>10}')
    if t_casadi is not None:
        print(f'{"CasADi":<12} {t_casadi * 1000:>15.3f}   {t_casadi / t_aadc:>9.1f}x')
    print(f'{"finite diff":<12} {t_fd * 1000:>15.1f}   {t_fd / t_aadc:>9.0f}x')
    print(f'{"=" * 68}')
