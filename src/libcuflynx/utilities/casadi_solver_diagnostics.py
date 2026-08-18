"""
CasADi ODE solver diagnostics for stiffness and adjoint-gradient troubleshooting.

Intended for DEBUG/test use when investigating CVODES adjoint failures (e.g.
CV_TOO_MUCH_WORK) on stiff circulatory models.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import numpy as np


def write_structured_debug_log(
    message: str,
    data: dict[str, Any],
    hypothesis_id: str,
    location: str,
    log_path: str,
    session_id: str = "casadi_diag",
) -> None:
    entry = {
        "sessionId": session_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry) + "\n")


def _log(log_path: str, message: str, data: dict[str, Any], hypothesis_id: str, location: str,
         session_id: str = "casadi_diag") -> None:
    write_structured_debug_log(message, data, hypothesis_id, location, log_path, session_id)


def _get_model_variable(sim_helper, name: str, default: Optional[float] = None) -> Optional[float]:
    idx = sim_helper.var_name_to_idx.get(name)
    if idx is None:
        return default
    return float(sim_helper.variables[idx])


def _numerical_jacobian_at_t0(sim_helper, eps: float = 1e-6):
    import casadi as ca

    x0 = np.array(sim_helper.states, dtype=float)
    p0 = np.array(sim_helper.variables, dtype=float)
    rates_func = ca.Function(
        "rates_check", [sim_helper.states_symb, sim_helper.variables_symb], [sim_helper.rates_symb]
    )
    n_states = sim_helper.STATE_COUNT
    f0 = np.array(rates_func(ca.DM(x0), ca.DM(p0))).flatten()
    jacobian = np.zeros((n_states, n_states))
    for col in range(n_states):
        x_perturbed = x0.copy()
        x_perturbed[col] += eps
        f_perturbed = np.array(rates_func(ca.DM(x_perturbed), ca.DM(p0))).flatten()
        jacobian[:, col] = (f_perturbed - f0) / eps
    eigenvalues = np.linalg.eigvals(jacobian)
    real_parts = np.real(eigenvalues)
    nonzero = real_parts[np.abs(real_parts) > 1e-12]
    min_nonzero = float(np.min(np.abs(nonzero))) if len(nonzero) else 0.0
    max_abs = float(np.max(np.abs(real_parts)))
    stiffness_ratio = max_abs / (min_nonzero + 1e-30)
    return jacobian, f0, eigenvalues, real_parts, stiffness_ratio


def _stiff_eigenvalue_state_decomposition(sim_helper, jacobian: np.ndarray, real_parts: np.ndarray, top_n: int = 3):
    from scipy.linalg import eig

    _, eigenvectors = eig(jacobian)
    sorted_idx = np.argsort(np.abs(real_parts))[::-1]
    top_modes = []
    n_states = sim_helper.STATE_COUNT
    for rank in range(min(top_n, len(sorted_idx))):
        eig_idx = sorted_idx[rank]
        eigenvector = eigenvectors[:, eig_idx]
        eigenvector_abs = np.abs(eigenvector)
        dominant_state_idx = int(np.argmax(eigenvector_abs))
        dominant_state = sim_helper.state_idx_to_name.get(dominant_state_idx, f"state_{dominant_state_idx}")
        top_components = sorted(
            [
                (sim_helper.state_idx_to_name.get(j, f"state_{j}"), float(eigenvector_abs[j]))
                for j in range(n_states)
            ],
            key=lambda item: -item[1],
        )[:5]
        top_modes.append({
            "eigenvalue_real": float(real_parts[eig_idx]),
            "dominant_state": dominant_state,
            "top5_state_components": top_components,
        })
    return top_modes


def _required_inertance_from_stiff_modes(
    sim_helper,
    top_modes: list[dict[str, Any]],
    real_parts: np.ndarray,
) -> dict[str, Any]:
    dt_val = float(sim_helper.dt)
    max_steps = sim_helper.solver_info.get("max_num_steps", 50000)
    bdf_stability = 5.0
    lambda_target = max_steps * bdf_stability / dt_val

    r_par = _get_model_variable(sim_helper, "R_par", 1.0664e7)
    r_t = _get_model_variable(sim_helper, "R_T_systemic_T", 1.1e8)

    lam_par = None
    lam_sys_t = None
    for mode in top_modes:
        dom = mode["dominant_state"]
        lam = abs(mode["eigenvalue_real"])
        if "par/v" in dom and lam_par is None:
            lam_par = lam
        if "systemic_T/v" in dom and lam_sys_t is None:
            lam_sys_t = lam

    if lam_par is None:
        lam_par = float(np.max(np.abs(real_parts)))
    if lam_sys_t is None:
        lam_sys_t = float(np.max(np.abs(real_parts)))

    i_par_current = r_par / lam_par if lam_par else None
    i_t_current = r_t / (2.0 * lam_sys_t) if lam_sys_t else None
    i_par_needed = r_par / lambda_target if lambda_target else None
    i_t_needed = r_t / (2.0 * lambda_target) if lambda_target else None

    result = {
        "lambda_target_s_inv": lambda_target,
        "R_par": r_par,
        "R_T_systemic_T": r_t,
        "lam_par_from_jacobian": lam_par,
        "lam_sysT_from_jacobian": lam_sys_t,
        "I_par_current_Js2_per_m6": i_par_current,
        "I_T_current_Js2_per_m6": i_t_current,
        "I_par_needed_Js2_per_m6": i_par_needed,
        "I_T_needed_Js2_per_m6": i_t_needed,
    }
    if i_par_current and i_par_needed:
        result["I_par_scale_up_factor"] = i_par_needed / i_par_current
    if i_t_current and i_t_needed:
        result["I_T_scale_up_factor"] = i_t_needed / i_t_current
    return result


def diagnose_casadi_solver_after_forward(
    runner,
    baseline_vals,
    *,
    log_path: str,
    session_id: str = "casadi_diag",
) -> None:
    """
    Run CasADi forward-pass stiffness diagnostics after a successful forward simulation.

    Call after ``get_cost_ca`` (or any forward run) so ``state_traj_dm`` is populated.
    """
    sim_helper = runner.param_id.sim_helper
    param_names = runner.param_id.param_id_info["param_names"]

    _log(log_path, "baseline param values", {
        "param_names": [str(name) for name in param_names],
        "param_vals": [
            float(val) if not isinstance(val, list) else [float(x) for x in val]
            for val in baseline_vals
        ],
    }, "H2", "casadi_solver_diagnostics:baseline_vals", session_id)

    traj = np.array(sim_helper.state_traj_dm)
    traj_stats = {}
    for state_idx, state_name in sim_helper.state_idx_to_name.items():
        row = traj[state_idx, :]
        traj_stats[state_name] = {
            "min": float(np.min(row)),
            "max": float(np.max(row)),
            "range": float(np.max(row) - np.min(row)),
        }
    _log(log_path, "forward state trajectory stats", traj_stats, "H1_H4",
         "casadi_solver_diagnostics:traj_stats", session_id)

    try:
        jacobian, f0, _, real_parts, stiffness_ratio = _numerical_jacobian_at_t0(sim_helper)
        _log(log_path, "numerical Jacobian stiffness at t=0", {
            "max_abs_real_eig": float(np.max(np.abs(real_parts))),
            "min_nonzero_abs_real_eig": float(
                np.min(np.abs(real_parts[np.abs(real_parts) > 1e-12]))
            ) if np.any(np.abs(real_parts) > 1e-12) else 0.0,
            "stiffness_ratio": float(stiffness_ratio),
            "all_real_parts": sorted([float(v) for v in real_parts], key=abs, reverse=True)[:10],
            "f0_max": float(np.max(np.abs(f0))),
        }, "H4", "casadi_solver_diagnostics:jacobian_t0", session_id)

        top_modes = _stiff_eigenvalue_state_decomposition(sim_helper, jacobian, real_parts)
        _log(log_path, "stiff eigenvalue state decomposition", {"top3_stiff_modes": top_modes},
             "H_STIFF_STATES", "casadi_solver_diagnostics:eigvec_decomp", session_id)

        inertance_info = _required_inertance_from_stiff_modes(sim_helper, top_modes, real_parts)
        _log(log_path, "required inertance values for adjoint", inertance_info,
             "H_REQUIRED_I", "casadi_solver_diagnostics:required_inertance", session_id)
    except Exception as exc:
        _log(log_path, "Jacobian computation failed", {"error": str(exc)},
             "H4", "casadi_solver_diagnostics:jacobian_error", session_id)

    _diagnose_loose_tolerance_adjoint(sim_helper, log_path, session_id)
    _diagnose_large_max_num_steps_adjoint(runner, log_path, session_id)
    _diagnose_forward_mode_sensitivity(sim_helper, log_path, session_id)
    _diagnose_collocation_gradient(sim_helper, log_path, session_id)
    _diagnose_floor_in_rates(sim_helper, log_path, session_id)


def log_casadi_gradient_diagnostic(
    gradient,
    error: Optional[BaseException] = None,
    *,
    log_path: str,
    session_id: str = "casadi_diag",
) -> None:
    """Log the result of a CasADi adjoint gradient call."""
    if error is not None:
        _log(log_path, "get_jac_cost_ca raised exception", {"error": str(error)[:500]},
             "H2_H5", "casadi_solver_diagnostics:gradient_exception", session_id)
        return

    grad_finite = bool(np.all(np.isfinite(gradient)))
    grad_nonzero = bool(not np.all(gradient == 0))
    _log(log_path, "get_jac_cost_ca result", {
        "gradient": [float(val) for val in gradient],
        "all_finite": grad_finite,
        "nonzero": grad_nonzero,
    }, "H2_H5", "casadi_solver_diagnostics:gradient_result", session_id)


def _diagnose_floor_in_rates(sim_helper, log_path: str, session_id: str) -> None:
    try:
        rates_str = str(sim_helper.rates_symb)
        has_floor = "floor" in rates_str.lower()
        _log(log_path, "symbolic rates floor check", {
            "has_floor_in_rates_symb": has_floor,
            "rates_symb_length": len(rates_str),
            "floor_count": rates_str.lower().count("floor"),
        }, "H1", "casadi_solver_diagnostics:floor_check", session_id)
    except Exception as exc:
        _log(log_path, "floor check failed", {"error": str(exc)},
             "H1", "casadi_solver_diagnostics:floor_check_fail", session_id)


def _diagnose_loose_tolerance_adjoint(sim_helper, log_path: str, session_id: str) -> None:
    try:
        import casadi as ca

        t0 = time.time()
        ode = {"x": sim_helper.states_symb, "p": sim_helper.variables_symb, "ode": sim_helper.rates_symb}
        opts = sim_helper._build_integrator_opts()
        opts["reltolB"] = 1e-3
        opts["abstolB"] = 1e-5
        opts["max_num_steps"] = 500000
        integrator = ca.integrator("F_lt", sim_helper.solve_ivp_method, ode, 0, sim_helper.dt, opts)
        n_steps = int(max(0, len(sim_helper.t_eval) - 1))
        mapped = integrator.mapaccum(n_steps)
        result = mapped(x0=sim_helper.states_symb, p=sim_helper.variables_symb)
        cost = ca.sum1(ca.vec(result["xf"][:, sim_helper.pre_steps:]))
        grad_symb = ca.gradient(cost, sim_helper.variables_symb)
        grad_func = ca.Function("gf_lt", [sim_helper.states_symb, sim_helper.variables_symb], [grad_symb])
        x0_dm = ca.DM(np.array(sim_helper._numeric_x0, dtype=float))
        p0_dm = ca.DM(np.array(sim_helper.variables, dtype=float))
        grad_val = np.array(grad_func(x0_dm, p0_dm)).flatten()
        _log(log_path, "loose-tolerance adjoint result", {
            "success": bool(np.all(np.isfinite(grad_val))),
            "grad_finite": bool(np.all(np.isfinite(grad_val))),
            "elapsed_s": round(time.time() - t0, 2),
        }, "H_LOOSE_TOL", "casadi_solver_diagnostics:loose_tol_result", session_id)
    except Exception as exc:
        _log(log_path, "loose-tolerance adjoint FAILED", {"error": str(exc)[:400]},
             "H_LOOSE_TOL", "casadi_solver_diagnostics:loose_tol_fail", session_id)


def _diagnose_large_max_num_steps_adjoint(runner, log_path: str, session_id: str) -> None:
    try:
        import casadi as ca

        t0 = time.time()
        sim_helper = runner.param_id.sim_helper
        ode = {"x": sim_helper.states_symb, "p": sim_helper.variables_symb, "ode": sim_helper.rates_symb}
        opts = sim_helper._build_integrator_opts()
        opts["max_num_steps"] = 500000
        integrator = ca.integrator("F_big", sim_helper.solve_ivp_method, ode, 0, sim_helper.dt, opts)
        mapped = integrator.mapaccum(int(max(0, len(sim_helper.t_eval) - 1)))
        result = mapped(x0=sim_helper.states_symb, p=sim_helper.variables_symb)
        cost = ca.sum1(ca.vec(result["xf"][:, -1]))
        grad_symb = ca.gradient(cost, sim_helper.variables_symb)
        grad_func = ca.Function("gf_big", [sim_helper.states_symb, sim_helper.variables_symb], [grad_symb])
        x0_dm = ca.DM(np.array(sim_helper.states, dtype=float))
        p0_dm = ca.DM(np.array(sim_helper.variables, dtype=float))
        grad_val = np.array(grad_func(x0_dm, p0_dm)).flatten()
        finite = grad_val[np.isfinite(grad_val)]
        _log(log_path, "adjoint with 500k max_num_steps result", {
            "success": bool(np.all(np.isfinite(grad_val))),
            "grad_finite": bool(np.all(np.isfinite(grad_val))),
            "grad_nonzero": bool(not np.all(grad_val == 0)),
            "grad_max_abs": float(np.max(np.abs(finite))) if len(finite) else None,
            "elapsed_s": round(time.time() - t0, 2),
        }, "H3", "casadi_solver_diagnostics:adjoint_500k", session_id)
    except Exception as exc:
        _log(log_path, "adjoint with 500k max_num_steps FAILED", {"error": str(exc)[:500]},
             "H3", "casadi_solver_diagnostics:adjoint_500k_fail", session_id)


def _diagnose_forward_mode_sensitivity(sim_helper, log_path: str, session_id: str) -> None:
    try:
        import casadi as ca

        t0 = time.time()
        ode = {"x": sim_helper.states_symb, "p": sim_helper.variables_symb, "ode": sim_helper.rates_symb}
        opts = sim_helper._build_integrator_opts()
        opts["max_num_steps"] = 50000
        opts["fsens_err_con"] = True
        integrator = ca.integrator("F_fsa", sim_helper.solve_ivp_method, ode, 0, sim_helper.dt, opts)
        mapped = integrator.mapaccum(int(max(0, len(sim_helper.t_eval) - 1)))
        result = mapped(x0=sim_helper.states_symb, p=sim_helper.variables_symb)
        cost = ca.sum1(ca.vec(result["xf"][:, -1]))
        jac_symb = ca.jacobian(cost, sim_helper.variables_symb)
        jac_func = ca.Function("jf_fsa", [sim_helper.states_symb, sim_helper.variables_symb], [jac_symb])
        x0_dm = ca.DM(np.array(sim_helper.states, dtype=float))
        p0_dm = ca.DM(np.array(sim_helper.variables, dtype=float))
        grad_val = np.array(jac_func(x0_dm, p0_dm)).flatten()
        finite = grad_val[np.isfinite(grad_val)]
        _log(log_path, "FSA (forward-mode) gradient result", {
            "success": bool(np.all(np.isfinite(grad_val))),
            "grad_finite": bool(np.all(np.isfinite(grad_val))),
            "grad_max_abs": float(np.max(np.abs(finite))) if len(finite) else None,
            "elapsed_s": round(time.time() - t0, 2),
            "method": "ca.jacobian (forward-mode SX)",
        }, "H_FSA", "casadi_solver_diagnostics:fsa_result", session_id)
    except Exception as exc:
        _log(log_path, "FSA (forward-mode) gradient FAILED", {"error": str(exc)[:500]},
             "H_FSA", "casadi_solver_diagnostics:fsa_fail", session_id)


def _diagnose_collocation_gradient(sim_helper, log_path: str, session_id: str) -> None:
    try:
        import casadi as ca

        t0 = time.time()
        ode = {"x": sim_helper.states_symb, "p": sim_helper.variables_symb, "ode": sim_helper.rates_symb}
        opts = {
            "collocation_scheme": "radau",
            "interpolation_order": 3,
            "number_of_finite_elements": 1,
        }
        integrator = ca.integrator("F_col", "collocation", ode, 0, sim_helper.dt, opts)
        n_steps = int(max(0, len(sim_helper.t_eval) - 1))
        mapped = integrator.mapaccum(n_steps)
        result = mapped(x0=sim_helper.states_symb, p=sim_helper.variables_symb)
        traj = ca.horzcat(sim_helper.states_symb, result["xf"])
        cost = ca.sum1(ca.vec(traj[:, sim_helper.pre_steps:]))
        grad_symb = ca.gradient(cost, sim_helper.variables_symb)
        grad_func = ca.Function("gf_col", [sim_helper.states_symb, sim_helper.variables_symb], [grad_symb])
        x0_dm = ca.DM(np.array(sim_helper._numeric_x0, dtype=float))
        p0_dm = ca.DM(np.array(sim_helper.variables, dtype=float))
        grad_val = np.array(grad_func(x0_dm, p0_dm)).flatten()
        finite = grad_val[np.isfinite(grad_val)]
        _log(log_path, "collocation gradient result", {
            "success": bool(np.all(np.isfinite(grad_val))),
            "grad_finite": bool(np.all(np.isfinite(grad_val))),
            "grad_nonzero": bool(not np.all(grad_val == 0)),
            "grad_max_abs": float(np.max(np.abs(finite))) if len(finite) else None,
            "elapsed_s": round(time.time() - t0, 2),
            "n_steps": n_steps,
        }, "H_COLLOCATION", "casadi_solver_diagnostics:collocation_result", session_id)
    except Exception as exc:
        _log(log_path, "collocation gradient FAILED", {"error": str(exc)[:500]},
             "H_COLLOCATION", "casadi_solver_diagnostics:collocation_fail", session_id)
