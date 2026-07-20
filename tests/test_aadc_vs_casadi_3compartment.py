"""
AADC vs CasADi comparison on the stiff 3compartment model.

Compares the two automatic-differentiation backends on three axes:
  1. Forward-solve accuracy vs the Myokit CVODE reference.
  2. Gradient accuracy vs finite differences (FD).
  3. Wall-clock speed of the forward solve.

CasADi uses its stiff ``method='bdf'``; AADC uses ``method='semi_implicit'``, which is its
only stiff-capable *tape-consistent* method. The two are therefore different algorithms,
and the tests below say so rather than pretending to a like-for-like comparison.

Both backends generate from the same CellML, so their state names are identical
(``component/variable``); the Myokit reference uses ``component_module.variable`` and is
matched component-wise.

Notes / current status (see PR #251):
  - CasADi's ``method='bdf'`` (symbolic implicit BDF2 with a rootfinder per step) matches
    CVODE on this stiff model, and its own gradient matches FD — the BDF graph is fully
    differentiable, so AD works directly on bdf (no fallback to semi_implicit_euler).
    These tests pass.
  - AADC's ``method='bdf'`` was **removed**: it handed the solve to ``scipy.solve_ivp``
    with AADC supplying only the RHS and Jacobian, so the trajectory never reached the
    tape, and the AD tape had no bdf branch — ``do_ad`` silently recorded rk4 instead,
    making the cost and the gradient different functions. These tests now use
    ``method='semi_implicit'``: first-order with on-tape diagonal damping, so its forward
    solve and its gradient are of the same function, at the cost of real accuracy against
    CVODE (quantified in the individual tests).
  - AADC records a tape and is therefore **license-gated** (Matlogica). Without a license
    it raises ``RuntimeError: AADC License check failed``; those tests take the
    ``aadc_licensed`` fixture and skip when no license is present, so an unlicensed
    environment (including CI) stays green.

Note: the AADC licence check must be performed before ``mpi4py.MPI`` is imported, which
``tests/conftest.py`` takes care of — see ``_validate_aadc_license`` there.
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
def models_3compartment(tmp_path_factory):
    """Generate the casadi_python and aadc_python 3compartment models once for the module."""
    base = str(tmp_path_factory.mktemp("aadc_vs_casadi_3c"))
    resources = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))

    def _gen(model_type, solver, solver_info, subdir):
        d = os.path.join(base, subdir)
        os.makedirs(d, exist_ok=True)
        inp = get_default_inp_data_dict("3compartment", "3compartment_parameters.csv", resources)
        inp.update({
            "model_type": model_type,
            "generated_models_dir": d,
            "solver": solver,
            "solver_info": solver_info,
        })
        assert generate_with_new_architecture(False, inp), f"generation failed for {model_type}"
        return {
            "py": os.path.join(d, "3compartment", "3compartment.py"),
            "cellml": os.path.join(d, "3compartment", "3compartment.cellml"),
        }

    return {
        "casadi": _gen("casadi_python", "casadi_integrator", {"method": "bdf"}, "ca"),
        "aadc": _gen("aadc_python", "aadc_semi_implicit", {"method": "semi_implicit"}, "aadc"),
    }


# ---- helpers ----
def _norm_state_key(name):
    """Normalise a state name to (component, variable) for cross-backend matching.

    Myokit:  component_module.variable   CasADi/AADC/Python: component/variable
    """
    if '/' in name:
        comp, var = name.split('/', 1)
    elif '.' in name:
        comp, var = name.split('.', 1)
    else:
        return (None, name.lower())
    return (comp.replace('_module', '').lower(), var.lower())


def _max_state_rel_l2_vs_cvode(cvode_helper, other_helper):
    """Max over matched ODE states of the whole-trajectory relative-L2 error (%) between
    a CVODE_myokit reference helper and another backend's helper."""
    cv_names = cvode_helper.get_all_variable_names()
    cv_res = cvode_helper.get_all_results(flatten=False)
    cv_map = {_norm_state_key(nm): np.asarray(cv_res[i][0]).flatten() for i, nm in enumerate(cv_names)}

    worst, worst_var, compared = 0.0, None, 0
    for sname in other_helper.state_name_to_idx:
        key = _norm_state_key(sname)
        if key not in cv_map:
            continue
        a = np.asarray(other_helper.get_results([[sname]], flatten=True)[0]).flatten()
        b = cv_map[key]
        L = min(len(a), len(b))
        if L < 2:
            continue
        a, b = a[:L], b[:L]
        denom = np.linalg.norm(b)
        if denom < 1e-12:
            continue
        err = np.linalg.norm(a - b) / denom * 100.0
        compared += 1
        if err > worst:
            worst, worst_var = err, sname
    return worst, worst_var, compared


def _cvode_reference(cellml_path):
    ref = get_simulation_helper(
        model_path=cellml_path, solver='CVODE_myokit', model_type='cellml_only',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0,
        solver_info={'MaximumStep': 1e-4, 'rtol': 1e-8, 'atol': 1e-10},
    )
    ref.run()
    return ref


# ---- 1. Forward-solve accuracy ----
@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_casadi_pretime_algebraic_output_is_post_warmup(models_3compartment):
    """With pre_time>0, casadi *algebraic*-variable outputs must come from the
    post-warmup sim window, not the initial transient.

    Regression for the _post_process slicing bug: it zipped the sim-time vector
    (``tSim = t_eval[pre_steps:]``) against the *first* columns of the full-horizon
    state trajectory, so algebraic vars (e.g. aortic pressure) were evaluated at the
    t≈0 warmup states instead of the settled sim window — while states were already
    sliced with ``[pre_steps:]``. A perturbed initial LV volume creates a transient
    that decays during pre_time; the aortic-pressure peak over the sim window must then
    match the CVODE reference (which it didn't before the fix), and must differ from
    the no-warmup (pre_time=0) peak (proving the warmup is actually reflected).
    """
    paths = models_3compartment["casadi"]
    SIM = 2.0
    INIT = [['heart/q_lv_init']]  # constant that sets the initial LV volume

    def _peak(pre, factor):
        h = get_simulation_helper(
            model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
            dt=0.01, sim_time=SIM, pre_time=pre, solver_info={'method': 'bdf'},
        )
        p0 = float(h.get_init_param_vals(INIT)[0])
        h.set_param_vals(INIT, [[p0 * factor]])  # overfill the LV -> decaying transient
        h.run()
        return float(np.asarray(h.get_results([['aortic_root/u']], flatten=True)[0]).flatten().max())

    F = 8.0  # large overfill -> a transient that decays slowly over many seconds
    u_cold = _peak(0.0, F)   # no warmup  -> sim window sits in the early transient
    u_short = _peak(5.0, F)  # 5 s warmup -> partly decayed
    u_long = _peak(12.0, F)  # 12 s warmup -> decayed further
    print(f"\npre_time algebraic-var peaks: cold(pre=0)={u_cold:.0f} short(pre=5)={u_short:.0f} long(pre=12)={u_long:.0f}")
    # The algebraic output must reflect the warmup: more pre_time -> the transient has
    # decayed more -> a lower peak in the sim window. Before the fix _post_process always
    # used the first (t≈0) columns, so all three were identical regardless of pre_time.
    assert u_cold > u_short * 1.1, "warmup must lower the transient peak (cold > short)"
    assert u_short > u_long * 1.02, "longer warmup must decay further (short > long)"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_forward_casadi_bdf_vs_cvode(models_3compartment):
    """CasADi BDF (symbolic implicit BDF2 + rootfinder) should match the CVODE reference."""
    paths = models_3compartment["casadi"]
    ref = _cvode_reference(paths["cellml"])

    sim = get_simulation_helper(
        model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'bdf'},
    )
    sim.run()

    max_pct, worst_var, compared = _max_state_rel_l2_vs_cvode(ref, sim)
    print(f"\nCasADi BDF vs CVODE: max rel-L2 {max_pct:.3f}% on {worst_var} ({compared} states)")
    assert compared > 0, "No states matched between CVODE and CasADi"
    assert max_pct < 2.0, f"CasADi BDF deviates from CVODE by {max_pct:.3f}% on {worst_var}"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("dt", [0.005, 0.0073, 0.01])
def test_3compartment_casadi_bdf_substep_robust_vs_cvode(models_3compartment, dt):
    """BDF must stay robust at output dt's where a *single* implicit step would straddle
    a valve switch (``if_else``/``fmax``) and the Newton solve diverges
    (``rootfinder process failed``). These exact dt's failed before internal sub-stepping.

    Regression for the CUFLynx forward protocol-run failure at default settings: the run
    must complete, stay finite, and match the CVODE reference (sub-stepping caps the
    internal step so the implicit solve stays in Newton's basin).
    """
    paths = models_3compartment["casadi"]
    ref = get_simulation_helper(
        model_path=paths["cellml"], solver='CVODE_myokit', model_type='cellml_only',
        dt=dt, sim_time=1.0, pre_time=0.0,
        solver_info={'MaximumStep': 1e-4, 'rtol': 1e-8, 'atol': 1e-10},
    )
    ref.run()
    sim = get_simulation_helper(
        model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=dt, sim_time=1.0, pre_time=0.0, solver_info={'method': 'bdf'},
    )
    assert sim.run() is True
    assert np.all(np.isfinite(sim.state_traj_dm[:, -1])), f"BDF produced non-finite states at dt={dt}"

    # Primary check is "didn't diverge": the pre-fix failure either raised or (when the
    # rootfinder was told to ignore failure) returned ~100% garbage. A loose bound well
    # below that, tolerant of coarse-dt cross-backend sampling of sharp valve transients,
    # confirms the solve actually converged.
    max_pct, worst_var, compared = _max_state_rel_l2_vs_cvode(ref, sim)
    print(f"\nCasADi BDF (dt={dt}) vs CVODE: max rel-L2 {max_pct:.3f}% on {worst_var}")
    assert compared > 0
    assert max_pct < 10.0, f"CasADi BDF deviates from CVODE by {max_pct:.3f}% at dt={dt}"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_casadi_rk_no_abstol_option(models_3compartment):
    """Non-SUNDIALS plugins (``rk``/``collocation``) must not be handed ``reltol``/``abstol``;
    CasADi rejects them with "Unknown option: abstol". Only ``cvodes``/``idas`` get them.

    Regression for the CUFLynx ``method='rk'`` crash. (rk is explicit, so on the stiff
    3compartment model the result may be non-finite — that's inherent to explicit
    integration, not the option bug; here we only assert the option crash is gone.)
    """
    paths = models_3compartment["casadi"]
    rk = get_simulation_helper(
        model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=0.001, sim_time=0.1, pre_time=0.0, solver_info={'method': 'rk'},
    )
    opts = rk._build_integrator_opts()
    assert 'abstol' not in opts and 'reltol' not in opts, "rk must not receive SUNDIALS tolerances"
    rk.run()  # must not raise "Unknown option: abstol"

    cvodes = get_simulation_helper(
        model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=0.001, sim_time=0.1, pre_time=0.0, solver_info={'method': 'cvodes'},
    )
    cv_opts = cvodes._build_integrator_opts()
    assert 'abstol' in cv_opts and 'reltol' in cv_opts, "cvodes should still receive tolerances"


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_bdf_is_refused_as_a_removed_method(aadc_licensed, models_3compartment):
    """AADC ``method='bdf'`` must raise, not silently fall back to another integrator.

    bdf was removed: it ran the solve in ``scipy.solve_ivp`` with AADC supplying only the RHS
    and its Jacobian, so the trajectory never reached the tape, and the AD tape had no bdf
    branch -- ``do_ad`` silently recorded rk4, making the cost and the gradient different
    functions. Its RHS kernel was also taped once at t=0 with the integrator discarding t
    (issue #268), so a time-dependent model integrated its t=0 right-hand side throughout
    while ``solve_ivp`` still reported success.

    This guards the *removal*, not the old NotImplementedError: an unrecognised method used to
    fall through to adaptive_rk45, so a config still carrying 'bdf' would have been silently
    integrated with an explicit RK45 -- the worst choice for the stiff models bdf was used for.
    """
    pytest.importorskip("aadc")
    paths = models_3compartment["aadc"]
    helper = get_simulation_helper(
        model_path=paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'bdf'},
    )
    with pytest.raises(ValueError, match="was removed"):
        helper.run()


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_forward_aadc_semi_implicit_vs_cvode(aadc_licensed, models_3compartment):
    """AADC ``semi_implicit`` should match the CVODE reference on the stiff 3compartment model.

    **This test currently FAILS, deliberately.** It previously ran ``method='bdf'``, which met
    the 5% bar but has since been removed: bdf handed the solve to ``scipy.solve_ivp`` with
    AADC supplying only the RHS and Jacobian, so the trajectory never reached the tape, and the
    AD tape had no bdf branch -- ``do_ad`` silently recorded rk4 instead, making the cost and
    the gradient different functions. ``semi_implicit`` is AADC's only stiff-capable
    *tape-consistent* method, and on this model it is not accurate:

        dt=1e-3  ->   35.1%      dt=5e-4  ->  183.2%
        dt=2e-4  ->  173.3%      dt=1e-4  ->  279.7%

    It does not merely miss the bar, it *diverges under step refinement*. The scheme is
    ``y += dt*f/(1 + dt*lam)``; as dt shrinks the damping factor ``1/(1+dt*lam)`` tends to 1
    and it degenerates into explicit forward Euler on a system whose stiffness probe measures
    ``max|Re(eig(J))|*dt ~ 4.5e10``. So there is no step size at which this passes.

    The assertion is left strict rather than relaxed to ~35% or marked xfail, so the gap stays
    visible: AADC has no usable stiff forward path on 3compartment. (``benchmarks/
    benchmark_specs.py`` already excludes AADC from the 3compartment benchmark for a related
    reason -- see issue #258.) Note CI does not catch this: AADC is licence-gated and the
    ``aadc_licensed`` fixture skips the whole test in an unlicensed environment.

    License-gated: raises ``AADC License check failed`` without a Matlogica license.
    """
    pytest.importorskip("aadc")
    paths = models_3compartment["aadc"]
    ref = _cvode_reference(paths["cellml"])

    sim = get_simulation_helper(
        model_path=paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'semi_implicit'},
    )
    sim.run()  # license check happens here (on-tape damping records a tape)

    max_pct, worst_var, compared = _max_state_rel_l2_vs_cvode(ref, sim)
    print(f"\nAADC semi_implicit vs CVODE: max rel-L2 {max_pct:.3f}% on {worst_var} "
          f"({compared} states)")
    assert compared > 0, "No states matched between CVODE and AADC"
    assert max_pct < 5.0, (
        f"AADC semi_implicit deviates from CVODE by {max_pct:.3f}% on {worst_var}. "
        f"Expected to fail at ~35%: semi_implicit is not convergent on this stiff model "
        f"(see docstring). Kept strict so the limitation stays visible."
    )


# ---- 2. Gradient accuracy vs FD ----
def _pick_param_and_state(helper):
    pname = next((n for n in helper.var_name_to_idx if 'e_lv_a' in n.lower()),
                 list(helper.var_name_to_idx.keys())[0])
    sname = next((n for n in helper.state_name_to_idx if 'q_lv' in n.lower()),
                 list(helper.state_name_to_idx.keys())[0])
    return pname, sname


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_gradient_vs_fd_casadi(models_3compartment):
    """CasADi symbolic gradient (semi_implicit_euler differentiable path) vs FD.

    cost = (final value of a state)^2; d(cost)/d(param) by CasADi AD vs central FD.
    """
    import casadi as ca
    paths = models_3compartment["casadi"]

    def _helper():
        return get_simulation_helper(
            model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
            dt=0.01, sim_time=0.3, pre_time=0.0, solver_info={'method': 'semi_implicit_euler'},
        )

    h = _helper()
    pname, sname = _pick_param_and_state(h)
    p0 = float(h.get_init_param_vals([[pname]])[0])
    sidx = h.state_name_to_idx[sname]

    # AD: differentiate cost through the symbolic semi-implicit Euler trajectory.
    h._create_param_subset([[pname]], [p0])
    h.run()
    cost_symb = h.state_traj_symb[sidx, -1] ** 2
    jac = ca.gradient(cost_symb, h.variables_symb_subset)
    jac_func = ca.Function('jac_cost', [h.states_symb, h.variables_symb], [jac])
    g_ad = float(np.array(jac_func(h.states, h.variables)).flatten()[0])

    # FD on the same scheme.
    def cost_at(pv):
        hh = _helper()
        hh.set_param_vals([[pname]], [[pv]])
        hh.run()
        v = np.asarray(hh.get_results([[sname]], flatten=True)[0]).flatten()[-1]
        return float(v) ** 2

    step = abs(p0) * 1e-6 if p0 != 0 else 1e-6
    g_fd = (cost_at(p0 + step) - cost_at(p0 - step)) / (2 * step)

    print(f"\nCasADi gradient vs FD: AD={g_ad:.6e} FD={g_fd:.6e}")
    if abs(g_fd) > 1e-30:
        ratio = g_ad / g_fd
        assert abs(ratio - 1.0) < 0.01, f"CasADi AD/FD ratio = {ratio:.6f} (expected ~1.0)"
    else:
        assert abs(g_ad) < 1e-12, f"FD≈0 but AD={g_ad:.6e}"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_gradient_vs_fd_casadi_bdf(models_3compartment):
    """CasADi symbolic gradient through the implicit **BDF** path vs FD.

    The BDF solver is a symbolic CasADi graph (rootfinder per step), so it supports
    AD: ``state_traj_symb`` is populated and ``d(cost)/d(param)`` by CasADi AD must
    match central finite differences. Same cost = (final value of a state)^2 as the
    semi-implicit gradient test, but on ``method='bdf'`` — proving the former scipy
    BDF guard (do_ad rejected) is gone and AD now works on bdf.
    """
    import casadi as ca
    paths = models_3compartment["casadi"]

    def _helper():
        return get_simulation_helper(
            model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
            dt=0.01, sim_time=0.3, pre_time=0.0, solver_info={'method': 'bdf'},
        )

    h = _helper()
    pname, sname = _pick_param_and_state(h)
    p0 = float(h.get_init_param_vals([[pname]])[0])
    sidx = h.state_name_to_idx[sname]

    # AD: differentiate cost through the symbolic implicit-BDF trajectory.
    h._create_param_subset([[pname]], [p0])
    assert h._do_ad is True
    h.run()
    assert h.state_traj_symb is not None  # symbolic graph exists -> AD is possible
    cost_symb = h.state_traj_symb[sidx, -1] ** 2
    jac = ca.gradient(cost_symb, h.variables_symb_subset)
    jac_func = ca.Function('jac_cost', [h.states_symb, h.variables_symb], [jac])
    g_ad = float(np.array(jac_func(h.states, h.variables)).flatten()[0])

    # FD on the same scheme.
    def cost_at(pv):
        hh = _helper()
        hh.set_param_vals([[pname]], [[pv]])
        hh.run()
        v = np.asarray(hh.get_results([[sname]], flatten=True)[0]).flatten()[-1]
        return float(v) ** 2

    step = abs(p0) * 1e-6 if p0 != 0 else 1e-6
    g_fd = (cost_at(p0 + step) - cost_at(p0 - step)) / (2 * step)

    print(f"\nCasADi BDF gradient vs FD: AD={g_ad:.6e} FD={g_fd:.6e}")
    if abs(g_fd) > 1e-30:
        ratio = g_ad / g_fd
        assert abs(ratio - 1.0) < 0.01, f"CasADi BDF AD/FD ratio = {ratio:.6f} (expected ~1.0)"
    else:
        assert abs(g_ad) < 1e-12, f"FD≈0 but AD={g_ad:.6e}"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_local_sa_ad_bdf_vs_fd(models_3compartment):
    """Local-sensitivity-style AD through bdf via the ``get_results`` path vs FD.

    Mirrors how CUFLynx local SA (gradient_method='AD') builds features: put the
    bdf helper in AD mode (``_create_param_subset``), ``run()``, then pull an
    observable out with ``get_results`` (which returns symbolic SX while ``_do_ad``)
    and reduce it over the whole trajectory (here ``mean``, as an obs operation
    would). The jacobian of that feature w.r.t. the parameter must be finite and
    match central FD — the regression guard for the bdf local-SA path. Uses a
    nonzero ``pre_time`` so the pre-steps slicing in ``get_results`` is exercised too.
    """
    import casadi as ca
    paths = models_3compartment["casadi"]

    def _helper():
        return get_simulation_helper(
            model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
            dt=0.01, sim_time=0.3, pre_time=0.2, solver_info={'method': 'bdf'},
        )

    h = _helper()
    pname, sname = _pick_param_and_state(h)
    p0 = float(h.get_init_param_vals([[pname]])[0])

    # AD: feature = mean over the (sim-time) trajectory of an observable, obtained
    # through get_results — exactly the SA feature-extraction surface.
    h._create_param_subset([[pname]], [p0])
    assert h._do_ad is True
    h.run()
    obs_symb = h.get_results([[sname]])[0][0]   # symbolic SX (1, n_steps) time series (AD mode)
    feature = ca.sum2(obs_symb) / obs_symb.numel()  # mean over the trajectory
    jac = ca.jacobian(feature, h.variables_symb_subset)
    jac_func = ca.Function('jac_feat', [h.states_symb, h.variables_symb], [jac])
    g_ad = float(np.array(jac_func(h.states, h.variables)).flatten()[0])
    assert np.isfinite(g_ad)

    # FD on the same feature with a fresh (numeric) helper.
    def feature_at(pv):
        hh = _helper()
        hh.set_param_vals([[pname]], [[pv]])
        hh.run()
        arr = np.asarray(hh.get_results([[sname]], flatten=True)[0]).flatten()
        return float(np.mean(arr))

    step = abs(p0) * 1e-6 if p0 != 0 else 1e-6
    g_fd = (feature_at(p0 + step) - feature_at(p0 - step)) / (2 * step)

    print(f"\nLocal-SA bdf AD vs FD (mean feature): AD={g_ad:.6e} FD={g_fd:.6e}")
    if abs(g_fd) > 1e-30:
        ratio = g_ad / g_fd
        assert abs(ratio - 1.0) < 0.01, f"bdf local-SA AD/FD ratio = {ratio:.6f} (expected ~1.0)"
    else:
        assert abs(g_ad) < 1e-12, f"FD≈0 but AD={g_ad:.6e}"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_gradient_vs_fd_aadc(aadc_licensed, models_3compartment):
    """AADC tape gradient vs FD on the same tape.

    License-gated: raises ``AADC License check failed`` without a Matlogica license.
    """
    aadc = pytest.importorskip("aadc")
    paths = models_3compartment["aadc"]

    sim = get_simulation_helper(
        model_path=paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=0.01, sim_time=0.3, pre_time=0.0, solver_info={'method': 'semi_implicit'},
    )
    sim.run()

    vi = sim.model.VARIABLE_INFO
    pidx = next(i for i, info in enumerate(vi) if 'e_lv_a' in info['name'].lower())
    sidx = next(i for i, info in enumerate(sim.model.STATE_INFO) if 'q_lv' in info['name'].lower())
    sim._ad_param_names = [vi[pidx]['name']]
    sim._ad_param_var_indices = [pidx]

    def cost_fn(st, p):
        return st[sidx] * st[sidx]

    g_ad = sim.compute_gradient_tape(cost_fn)[0]  # license check happens here

    pv = float(sim._numeric_variables_all[pidx])
    h = abs(pv) * 1e-5 if pv != 0 else 1e-5
    workers, funcs, rc, ap = sim._aad_workers, sim._tape_funcs, sim._tape_r_cost, sim._tape_a_p
    cp = float(np.asarray(aadc.evaluate(funcs, {rc: []}, {ap[0]: pv + h}, workers)[0][rc]).flat[0])
    cm = float(np.asarray(aadc.evaluate(funcs, {rc: []}, {ap[0]: pv - h}, workers)[0][rc]).flat[0])
    g_fd = (cp - cm) / (2 * h)

    print(f"\nAADC gradient vs FD: AD={g_ad:.6e} FD={g_fd:.6e}")
    if abs(g_fd) > 1e-30:
        ratio = g_ad / g_fd
        assert abs(ratio - 1.0) < 0.01, f"AADC AD/FD ratio = {ratio:.6f} (expected ~1.0)"
    else:
        assert abs(g_ad) < 1e-10, f"FD≈0 but AD={g_ad:.6e}"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_aadc_run_warns_stiff(aadc_licensed, models_3compartment):
    """The AADC backend must loudly warn that 3compartment is stiff before integrating.

    Stiffness is what makes AADC's fixed-step tape integrators inaccurate/unstable on this
    model, so run() probes the first second of dynamics and warns. This guards that the
    warning actually fires (and names the recommended stiff-capable backends).
    """
    import warnings as _warnings
    paths = models_3compartment["aadc"]
    sim = get_simulation_helper(
        model_path=paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=0.01, sim_time=0.3, pre_time=0.0, solver_info={'method': 'semi_implicit'},
    )
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        sim.run()  # licence check + stiffness probe happen here
    messages = [str(w.message) for w in caught]
    stiff_warnings = [m for m in messages if "stiffness check" in m.lower() and "stiff" in m.lower()]
    assert stiff_warnings, f"expected a loud stiffness warning; got warnings: {messages}"
    # the warning must point the user at a stiff-capable backend
    assert any("casadi" in m.lower() or "fsa" in m.lower() or "forward-sensitivity" in m.lower()
               for m in stiff_warnings), stiff_warnings


# ---- 3. Speed ----
@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_speed_casadi_bdf_vs_aadc_semi_implicit(aadc_licensed, models_3compartment):
    """Wall-clock comparison of the stiff forward solve: CasADi bdf vs AADC semi_implicit.

    Both are timed and printed; the test asserts only that each solve produces finite
    states, so it reports rather than gates.

    **Time alone is meaningless here — read it with the accuracy.** These are different
    algorithms and they do not land anywhere near the same error:

      * CasADi ``bdf``: *fixed*-step symbolic implicit BDF2 (internal step capped at
        ``solver_info['max_step']``, default 1e-3) — ~0.52% relative-L2 vs the CVODE
        reference, in ~0.19s.
      * AADC ``semi_implicit``: first-order Euler with on-tape diagonal damping — ~35%
        vs the same reference, in ~3.1s.

    So on this stiff model CasADi is both ~67x more accurate and ~16x faster, and the
    ratio below is not a backend-technology result: AADC's remaining stiff-capable method
    is simply not convergent here (see
    ``test_3compartment_forward_aadc_semi_implicit_vs_cvode`` for the dt sweep). This
    previously timed AADC ``method='bdf'``, which was accurate but has been removed — it
    solved in scipy, off-tape, while do_ad silently taped rk4. The accuracy of each run is
    printed next to its time so the number is never quoted bare.
    """
    ca_paths = models_3compartment["casadi"]
    aadc_paths = models_3compartment["aadc"]
    ref = _cvode_reference(ca_paths["cellml"])

    sim_ca = get_simulation_helper(
        model_path=ca_paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'bdf'},
    )
    t0 = time.perf_counter()
    sim_ca.run()
    t_casadi = time.perf_counter() - t0
    err_casadi, _, _ = _max_state_rel_l2_vs_cvode(ref, sim_ca)
    assert np.all(np.isfinite(sim_ca.state_traj_dm[:, -1])), "CasADi BDF produced non-finite states"

    sim_aadc = get_simulation_helper(
        model_path=aadc_paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'semi_implicit'},
    )
    t0 = time.perf_counter()
    sim_aadc.run()  # license check happens here
    t_aadc = time.perf_counter() - t0
    err_aadc, _, _ = _max_state_rel_l2_vs_cvode(ref, sim_aadc)
    assert np.all(np.isfinite(sim_aadc.state_traj[:, -1])), "AADC BDF produced non-finite states"

    # Always report time WITH the accuracy it bought — the raw ratio is meaningless alone
    # (fixed-step CasADi and adaptive AADC land at very different errors; see docstring).
    print(f"\n  CasADi BDF (fixed step) : {t_casadi:6.3f}s  at {err_casadi:.3f}% rel-L2 vs CVODE")
    print(f"  AADC BDF (adaptive)     : {t_aadc:6.3f}s  at {err_aadc:.3f}% rel-L2 vs CVODE")
    print("  NOT iso-accuracy — do not quote this ratio without the errors above.")


# ---- 4. Algebraic post-processing ----

@pytest.mark.integration
def test_3compartment_casadi_post_process_matches_per_timestep_reference(models_3compartment):
    """The vectorised algebraic post-processing must equal the per-timestep computation.

    ``_post_process`` builds the algebraic map ``(t, x, p) -> vars`` once and evaluates it
    over the whole grid with ``ca.Function.map``. It used to traverse the model equations
    symbolically once per output step in a Python loop, which produced the same numbers but
    cost ~9.1s on 3compartment — >99% of ``run()``, against a 0.09s ODE solve — and made
    CasADi look an order of magnitude slower than it is in the AADC comparison.

    This pins the refactor: recompute the algebraic variables the slow, obvious way and
    require the mapped result to match exactly.
    """
    import copy as _copy
    import casadi as ca

    paths = models_3compartment["casadi"]
    sim = get_simulation_helper(
        model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=0.01, sim_time=0.2, pre_time=0.1, solver_info={'method': 'bdf'},
    )
    sim.run()

    # Reference: evaluate the model once per output time, as the old implementation did.
    var_names = list(sim.var_name_to_idx.keys())
    state_cols = ca.horzsplit(sim.state_traj_symb, 1)[sim.pre_steps:]
    ref_cols = []
    for ti, state_vec in zip(sim.tSim, state_cols):
        rates = [0.0] * sim.STATE_COUNT
        vars_copy = _copy.copy(sim.variables_all_symb)
        sim.model.compute_rates(ti, state_vec, rates, vars_copy)
        sim.model.compute_variables(ti, state_vec, rates, vars_copy)
        ref_cols.append(ca.vertcat(*[vars_copy[sim.var_name_to_idx[n]] for n in var_names]))
    ref_symb = ca.horzcat(*ref_cols)
    ref_func = ca.Function('ref', [sim.states_symb, sim.variables_symb], [ref_symb])
    ref = np.array(ref_func(ca.DM(sim._x0_numeric()), ca.DM(sim.variables)))

    assert sim.var_traj_dm.shape == ref.shape, (
        f"post-processed algebraic trajectory has shape {sim.var_traj_dm.shape}, "
        f"per-timestep reference has {ref.shape}"
    )
    np.testing.assert_allclose(sim.var_traj_dm, ref, rtol=1e-12, atol=1e-12)


# ---- 5. The fair (iso-accuracy) speed comparison ----

# There used to be an _ISO_CASADI_MAX_STEP = 1e-5 here: the CasADi step size chosen so its
# error landed in the same ballpark as AADC bdf's ~0.003%, since fixed-step BDF2 degrades to
# ~first order on this model (the valve switches are non-smooth) and needed h ~100x below the
# output dt to get there. It is gone with AADC bdf itself. AADC semi_implicit sits at ~35%,
# so matching it would mean coarsening CasADi rather than refining it, and the 1e-5 graph
# cost ~2.1GB (one rootfinder node per internal step, scaling as sim_time/h) to chase an
# accuracy no AADC method now reaches.


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_casadi_bdf_vs_aadc_semi_implicit_iso_accuracy(
        aadc_licensed, models_3compartment):
    """Compare CasADi bdf and AADC semi_implicit **at matched accuracy** — the only
    comparison that means anything.

    **This test currently FAILS, deliberately**, on the accuracy-parity assertion. Its
    original premise is gone. It used to run AADC ``method='bdf'``, which reached ~0.003%
    on this model, and the interesting question was how far CasADi's fixed step had to
    shrink to match it (h=1e-5, and CasADi was then slower — a step-control result, not an
    AD-technology one). That method has been removed: it solved in ``scipy.solve_ivp`` with
    AADC supplying only the RHS and Jacobian, so the trajectory never reached the tape,
    while ``do_ad`` silently taped rk4 instead.

    AADC's remaining stiff-capable method, ``semi_implicit``, lands at ~35% here and gets
    *worse* under step refinement (see
    ``test_3compartment_forward_aadc_semi_implicit_vs_cvode``), so there is no iso-accuracy
    point to find: reaching parity would mean making CasADi ~70x *coarser*, not finer, and
    comparing two wrong answers. CasADi therefore runs at its default bdf configuration
    (~0.52%, ~0.19s) rather than the old h=1e-5 tuning, which cost ~2.1GB of symbolic graph
    to chase an accuracy AADC no longer reaches.

    Left failing rather than relaxed or deleted so the regression stays visible: AADC has no
    usable stiff path on 3compartment. Note CI does not catch this — AADC is licence-gated
    and ``aadc_licensed`` skips the test in an unlicensed environment.
    """
    ca_paths = models_3compartment["casadi"]
    aadc_paths = models_3compartment["aadc"]
    ref = _cvode_reference(ca_paths["cellml"])

    sim_ca = get_simulation_helper(
        model_path=ca_paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0,
        solver_info={'method': 'bdf'},
    )
    t0 = time.perf_counter()
    sim_ca.run()
    t_casadi = time.perf_counter() - t0
    err_casadi, _, _ = _max_state_rel_l2_vs_cvode(ref, sim_ca)

    sim_aadc = get_simulation_helper(
        model_path=aadc_paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'semi_implicit'},
    )
    t0 = time.perf_counter()
    sim_aadc.run()
    t_aadc = time.perf_counter() - t0
    err_aadc, _, _ = _max_state_rel_l2_vs_cvode(ref, sim_aadc)

    print(f"\n  iso-accuracy comparison (3compartment, {_SIM_TIME}s @ dt={_DT}):")
    print(f"    CasADi bdf (fixed step)     : {t_casadi:7.2f}s  at {err_casadi:.5f}% rel-L2")
    print(f"    AADC semi_implicit          : {t_aadc:7.2f}s  at {err_aadc:.5f}% rel-L2")
    print(f"    -> CasADi is {t_aadc / t_casadi:.1f}x faster AND "
          f"{err_aadc / max(err_casadi, 1e-12):.0f}x more accurate")

    assert np.all(np.isfinite(sim_ca.state_traj_dm[:, -1])), "CasADi produced non-finite states"
    assert np.all(np.isfinite(sim_aadc.state_traj[:, -1])), "AADC produced non-finite states"

    # The whole point is that the two are at comparable accuracy — if they drift apart the
    # timing above stops being a like-for-like comparison and the test should say so rather
    # than silently report a meaningless ratio.
    assert err_casadi < 1.0, (
        f"CasADi bdf gave {err_casadi:.5f}% error, expected ~0.52% at its default step"
    )
    assert err_aadc < 1.0, (
        f"AADC semi_implicit gave {err_aadc:.5f}% error, expected ~0.52% to be comparable "
        f"with CasADi. Expected to fail at ~35%: semi_implicit is not convergent on this "
        f"stiff model, so there is no iso-accuracy point. Kept strict so the gap stays "
        f"visible rather than being relaxed away."
    )
    ratio = max(err_casadi, err_aadc) / max(min(err_casadi, err_aadc), 1e-12)
    assert ratio < 10.0, (
        f"errors are not comparable (CasADi {err_casadi:.5f}% vs AADC {err_aadc:.5f}%, "
        f"{ratio:.1f}x apart) — the timing comparison is not iso-accuracy"
    )
