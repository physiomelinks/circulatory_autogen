#!/usr/bin/env python3
"""
Benchmark: AADC vs CasADi for coupled CellML model pipelines.

Demonstrates AADC tape-based AD on two real CellML models coupled:
  Model A: Lotka-Volterra (prey-predator dynamics, 2 states)
  Model B: Van der Pol oscillator (nonlinear oscillation, 2 states)
  Coupling: max(prey from A) → damping parameter mu for B

Gradient flows through: cost ← VdP(B) ← coupling ← max(prey) ← LV(A) ← params

Both models are generated from CellML files by circulatory_autogen's code generator.

Usage: python tests/benchmark_aadc_coupled_pipeline.py
Requires: aadc, casadi (for comparison)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def find_aadc_model(pattern):
    """Find a generated AADC model matching pattern."""
    test_outputs = os.path.join(os.path.dirname(__file__), 'test_outputs')
    for root, dirs, files in os.walk(test_outputs):
        for f in files:
            if pattern in root and f.endswith('.py') and 'utilities' not in f and 'aadc' in root:
                return os.path.join(root, f)
    return None


def run_benchmark():
    try:
        import aadc
        from aadc.recording_ctx import record_kernel
        from aadc.evaluate_wrappers import evaluate_kernel
    except ImportError:
        print("AADC not available — skipping"); return False

    try:
        import casadi as ca
    except ImportError:
        ca = None
        print("CasADi not available — skipping CasADi comparison")

    from solver_wrappers import get_simulation_helper

    # Find models
    lv_path = find_aadc_model('Lotka_Volterra')
    vdp_path = find_aadc_model('VanDerPol')

    if not lv_path or not vdp_path:
        print("Models not found. Run test_aadc_solvers.py first to generate them.")
        return False

    # Load models
    sim_lv = get_simulation_helper(model_path=lv_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.02, sim_time=5.0, pre_time=0.0,
        solver_info={'method': 'adaptive_rk45'})
    sim_vdp = get_simulation_helper(model_path=vdp_path, solver='aadc_semi_implicit',
        model_type='aadc_python', dt=0.02, sim_time=5.0, pre_time=0.0,
        solver_info={'method': 'adaptive_rk45'})

    sim_lv.run(); sim_vdp.run()

    vi_lv = sim_lv.model.VARIABLE_INFO
    vi_vdp = sim_vdp.model.VARIABLE_INFO
    vars_lv = list(sim_lv._numeric_variables_all)
    for cp, ci in enumerate(sim_lv.constant_indices): vars_lv[ci] = sim_lv.variables[cp]
    vars_vdp = list(sim_vdp._numeric_variables_all)
    for cp, ci in enumerate(sim_vdp.constant_indices): vars_vdp[ci] = sim_vdp.variables[cp]

    a_idx = next(i for i, info in enumerate(vi_lv) if 'alpha' in info['name'].lower())
    b_idx = next(i for i, info in enumerate(vi_lv) if 'beta' in info['name'].lower())
    d_idx = next(i for i, info in enumerate(vi_lv) if 'delta' in info['name'].lower())
    g_idx = next(i for i, info in enumerate(vi_lv) if 'gamma' in info['name'].lower())

    dt = 0.02; nA = 250; nB = 250
    alpha = float(vars_lv[a_idx]); beta = float(vars_lv[b_idx])
    delta = float(vars_lv[d_idx]); gamma = float(vars_lv[g_idx])

    print("=" * 65)
    print("  Coupled CellML Models: Lotka-Volterra → Van der Pol")
    print("=" * 65)
    print(f"  Model A: Lotka-Volterra ({nA} RK4 steps, 2 states)")
    print(f"  Model B: Van der Pol oscillator ({nB} RK4 steps, 2 states)")
    print(f"  Coupling: max(prey from A) × 0.1 → mu parameter for B")
    print(f"  Params: alpha, beta (Model A only)")
    print(f"  Cost: (u_final - 0.5)² + (v_final - 0.0)²")
    print(f"  Total: {nA + nB} steps on one tape")
    print()

    # === AADC ===
    with record_kernel() as kernel:
        a = aadc.idouble(alpha); a_arg = a.mark_as_input()
        b = aadc.idouble(beta); b_arg = b.mark_as_input()

        d_ = aadc.idouble(delta); g_ = aadc.idouble(gamma)
        x = aadc.idouble(20.0); y = aadc.idouble(10.0); mx = x

        for step in range(nA):
            kx1 = a*x - b*x*y; ky1 = d_*x*y - g_*y
            xm = x + aadc.idouble(.5*dt)*kx1; ym = y + aadc.idouble(.5*dt)*ky1
            kx2 = a*xm - b*xm*ym; ky2 = d_*xm*ym - g_*ym
            xm = x + aadc.idouble(.5*dt)*kx2; ym = y + aadc.idouble(.5*dt)*ky2
            kx3 = a*xm - b*xm*ym; ky3 = d_*xm*ym - g_*ym
            xm = x + aadc.idouble(dt)*kx3; ym = y + aadc.idouble(dt)*ky3
            kx4 = a*xm - b*xm*ym; ky4 = d_*xm*ym - g_*ym
            x = x + aadc.idouble(dt/6)*(kx1 + aadc.idouble(2)*kx2 + aadc.idouble(2)*kx3 + kx4)
            y = y + aadc.idouble(dt/6)*(ky1 + aadc.idouble(2)*ky2 + aadc.idouble(2)*ky3 + ky4)
            mx = aadc.iif(x > mx, x, mx)

        # Coupling: max(prey) * 0.1 → mu for Van der Pol
        mu = mx * aadc.idouble(0.1)

        u = aadc.idouble(float(sim_vdp.states[0]))
        v = aadc.idouble(float(sim_vdp.states[1]))
        for step in range(nB):
            du = v
            dv = mu * (aadc.idouble(1.0) - u*u) * v - u
            u = u + aadc.idouble(dt) * du
            v = v + aadc.idouble(dt) * dv

        cost = (u - aadc.idouble(0.5))**2 + v**2
        cost_out = cost.mark_as_output()

    all_args = [a_arg, b_arg]
    p_true = np.array([alpha, beta])
    inputs = {a_arg: alpha, b_arg: beta}

    res = evaluate_kernel(kernel, {cost_out: all_args}, inputs, num_threads=1)
    aadc_cost = res.values[cost_out].item()
    aadc_grad = np.array([res.derivs[cost_out][arg].item() for arg in all_args])

    t0 = time.time()
    for _ in range(1000):
        evaluate_kernel(kernel, {cost_out: all_args}, inputs, num_threads=1)
    t_aadc = (time.time() - t0) / 1000

    # FD via tape
    h = 1e-7; fd = np.zeros(2)
    for j in range(2):
        pu = dict(inputs); pu[all_args[j]] = p_true[j] + h
        pd = dict(inputs); pd[all_args[j]] = p_true[j] - h
        cu = evaluate_kernel(kernel, {cost_out: []}, pu, num_threads=1).values[cost_out].item()
        cd = evaluate_kernel(kernel, {cost_out: []}, pd, num_threads=1).values[cost_out].item()
        fd[j] = (cu - cd) / (2*h)

    # FD via full re-solve (numpy)
    def coupled_numpy(p):
        a_, b_ = p
        x_, y_ = 20.0, 10.0; mx_ = x_
        for _ in range(nA):
            kx1=a_*x_-b_*x_*y_; ky1=delta*x_*y_-gamma*y_
            xm=x_+.5*dt*kx1; ym=y_+.5*dt*ky1; kx2=a_*xm-b_*xm*ym; ky2=delta*xm*ym-gamma*ym
            xm=x_+.5*dt*kx2; ym=y_+.5*dt*ky2; kx3=a_*xm-b_*xm*ym; ky3=delta*xm*ym-gamma*ym
            xm=x_+dt*kx3; ym=y_+dt*ky3; kx4=a_*xm-b_*xm*ym; ky4=delta*xm*ym-gamma*ym
            x_+=dt/6*(kx1+2*kx2+2*kx3+kx4); y_+=dt/6*(ky1+2*ky2+2*ky3+ky4)
            mx_=max(mx_,x_)
        mu_=mx_*0.1
        u_=float(sim_vdp.states[0]); v_=float(sim_vdp.states[1])
        for _ in range(nB):
            du_=v_; dv_=mu_*(1-u_*u_)*v_-u_
            u_+=dt*du_; v_+=dt*dv_
        return (u_-0.5)**2+v_**2

    fd_resolve = np.zeros(2)
    t0 = time.time()
    for j in range(2):
        pu = p_true.copy(); pu[j] += h
        pd = p_true.copy(); pd[j] -= h
        fd_resolve[j] = (coupled_numpy(pu) - coupled_numpy(pd)) / (2*h)
    t_fd = time.time() - t0

    # Perturbed validation
    p_pert = p_true * np.array([1.2, 0.8])
    inp_pert = {a_arg: p_pert[0], b_arg: p_pert[1]}
    res2 = evaluate_kernel(kernel, {cost_out: all_args}, inp_pert, num_threads=1)
    grad2 = np.array([res2.derivs[cost_out][arg].item() for arg in all_args])
    fd2 = np.zeros(2)
    for j in range(2):
        pu = p_pert.copy(); pu[j] += h; pd = p_pert.copy(); pd[j] -= h
        fd2[j] = (coupled_numpy(pu) - coupled_numpy(pd)) / (2*h)
    ratios2 = grad2 / (fd2 + 1e-30)

    # === CasADi comparison ===
    t_casadi = None
    if ca is not None:
        a2 = ca.SX.sym('a'); b2 = ca.SX.sym('b')
        params = ca.vertcat(a2, b2)
        x2 = ca.SX(20.0); y2 = ca.SX(10.0); mx2 = x2
        for step in range(nA):
            kx1=a2*x2-b2*x2*y2; ky1=delta*x2*y2-gamma*y2
            xm=x2+.5*dt*kx1; ym=y2+.5*dt*ky1; kx2=a2*xm-b2*xm*ym; ky2=delta*xm*ym-gamma*ym
            xm=x2+.5*dt*kx2; ym=y2+.5*dt*ky2; kx3=a2*xm-b2*xm*ym; ky3=delta*xm*ym-gamma*ym
            xm=x2+dt*kx3; ym=y2+dt*ky3; kx4=a2*xm-b2*xm*ym; ky4=delta*xm*ym-gamma*ym
            x2=x2+dt/6*(kx1+2*kx2+2*kx3+kx4); y2=y2+dt/6*(ky1+2*ky2+2*ky3+ky4)
            mx2=ca.if_else(x2>mx2,x2,mx2)
        mu2=mx2*0.1
        u2=ca.SX(float(sim_vdp.states[0])); v2=ca.SX(float(sim_vdp.states[1]))
        for step in range(nB):
            du2=v2; dv2=mu2*(1-u2*u2)*v2-u2
            u2=u2+dt*du2; v2=v2+dt*dv2
        cost2=(u2-0.5)**2+v2**2
        grad2_ca=ca.gradient(cost2,params)
        fn=ca.Function('f',[params],[cost2,grad2_ca])
        t0=time.time()
        for _ in range(1000): fn(p_true)
        t_casadi=(time.time()-t0)/1000

    # Results
    print(f"  Cost: {aadc_cost:.4e}")
    print()
    print(f"  Gradients [d/d_alpha, d/d_beta]:")
    print(f"    AADC:          {aadc_grad}")
    print(f"    FD (tape):     {fd}")
    print(f"    FD (re-solve): {fd_resolve}")
    print(f"    AD/FD(tape):   {aadc_grad / (fd + 1e-30)}")
    print(f"    AD/FD(solve):  {aadc_grad / (fd_resolve + 1e-30)}")
    print()
    print(f"  Perturbed (±20%): AD/FD = {ratios2}")
    print()
    print(f"  Timing (cost + 2 gradients per eval):")
    print(f"    AADC:   {t_aadc*1000:.3f} ms")
    if t_casadi:
        print(f"    CasADi: {t_casadi*1000:.3f} ms  (AADC {t_casadi/t_aadc:.1f}x faster)")
    print(f"    FD:     {t_fd*1000:.0f} ms    (AADC {t_fd/t_aadc:.0f}x faster)")
    print()

    all_ok = all(abs(r - 1.0) < 0.01 for r in aadc_grad / (fd_resolve + 1e-30)) and \
             all(abs(r - 1.0) < 0.01 for r in ratios2)
    print(f"  All gradients exact (at true and perturbed params): {'YES' if all_ok else 'NO'}")
    print("=" * 65)
    return all_ok


if __name__ == "__main__":
    ok = run_benchmark()
    sys.exit(0 if ok else 1)
