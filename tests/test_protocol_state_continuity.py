"""
Regression tests for state continuity across protocol sub-experiments.

A multi-sub-experiment protocol must carry the model state forward from one
sub-experiment to the next, exactly as if the whole horizon had been run as a
single segment.  The OpenCOR and Python (solve_ivp) backends do this correctly;
the Myokit backend historically did NOT, because
``MyokitSimulationHelper.update_times`` called ``simulation.reset()`` (which
reverts Myokit to its *default*/initial state) without restoring the carried
state.  Every sub-experiment after the first therefore restarted from the model
initial conditions.

This reproduces the bug with a simple, public model (Lotka-Volterra) so it can
live in circulatory_autogen CI.  It was first observed on a private glucose
metabolism model run with multi-meal protocols, where Myokit diverged from
OpenCOR by 30-100%.

Backends:
- Myokit self-consistency (always runs): a ``[T/2, T/2]`` two-sub-experiment run
  must equal a ``[T]`` single segment.
- Myokit vs Python/solve_ivp (always runs; cross-backend check that works in CI,
  since the Python backend ships with circulatory_autogen and carries state forward
  correctly).
- Myokit vs OpenCOR (skipped when OpenCOR is unavailable, e.g. in CI).
"""
import os
import sys

import numpy as np
import pytest

from libcuflynx.protocol_runners.protocol_runner import ProtocolRunner

# Total horizon and the sub-experiment split used to expose the bug.
_SIM_TIME = 10.0
_DT = 0.1
# Tight tolerances so the only meaningful difference between a single segment and
# a split protocol is state continuity, not CVODE cold-restart error (re-deriving
# BDF order/step at the segment boundary). At these tolerances a correct backend
# matches to ~0%, while the state-reset bug still diverges by tens of percent.
_SOLVER_INFO = {"MaximumStep": 0.01, "MaximumNumberOfSteps": 50000, "rtol": 1e-8, "atol": 1e-10}
# Python (solve_ivp) backend uses a stiff method with matching tight tolerances so
# its trajectory tracks the Myokit/OpenCOR CVODE solution to ~0%.
_PY_SOLVER_INFO = {"method": "BDF", "rtol": 1e-10, "atol": 1e-12, "max_step": 0.01}
# Lotka-Volterra oscillates, so the state at t=T/2 is far from the initial
# condition; a backend that resets at the sub-experiment boundary diverges hard.
_STATE_BASES = ["x", "y"]

# Same-backend self-consistency: only integrator cold-restart noise should remain.
_SELF_TOL_PCT = 0.1
# Cross-backend (Myokit vs OpenCOR) compares two different adaptive integrators,
# so a looser bound is appropriate; still far below the tens-of-percent bug.
_CROSS_TOL_PCT = 1.0


@pytest.fixture(scope="function")
def lotka_volterra_cellml(generated_cellml_model_factory):
    """Lotka-Volterra CellML via the shared factory (copies the committed model)."""
    return generated_cellml_model_factory("Lotka_Volterra", "Lotka_Volterra_parameters.csv")


@pytest.fixture(scope="function")
def lotka_volterra_python(lotka_volterra_cellml, temp_generated_models_dir):
    """Lotka-Volterra Python model (.py) generated from the CellML for the solve_ivp backend."""
    from libcuflynx.generators.PythonGenerator import PythonGenerator

    generator = PythonGenerator(
        lotka_volterra_cellml,
        output_dir=os.path.join(temp_generated_models_dir, "Lotka_Volterra_py"),
        module_name="Lotka_Volterra",
    )
    return generator.generate()


def _run_segments(model_path, solver, sim_times, solver_info=_SOLVER_INFO, model_type=None):
    """Run a single-experiment protocol with the given sub-experiment durations.

    Uses ProtocolRunner, which already concatenates the sub-experiment segments
    (dropping the duplicated boundary sample) and exercises the real protocol loop.
    Returns {base_var: np.ndarray} for the single experiment.
    """
    runner = ProtocolRunner(
        model_path,
        inp_data_dict={"dt": _DT, "solver_info": solver_info},
        solver=solver,
        model_type=model_type,
    )
    protocol_info = {
        "pre_times": [0.0],
        "sim_times": [list(sim_times)],
        "params_to_change": {},
    }
    _, res_all, _ = runner.run_protocols(model_path, protocol_info)
    var2idx = runner.get_var2idx_dict()

    def idx(base):
        # Backend-agnostic: Myokit uses 'Component_module.var', OpenCOR/Python use
        # 'Component/var'. Match the trailing variable name either way.
        for name, i in var2idx.items():
            if name.endswith(f".{base}") or name.endswith(f"/{base}"):
                return i
        raise KeyError(f"{base} not found in {solver} variables: {list(var2idx)[:6]}...")

    return {base: np.asarray(res_all[0][idx(base)], dtype=float) for base in _STATE_BASES}


def _max_rel_error(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-9)
    return float(np.nanmax(np.abs(a - b) / denom * 100))


def _assert_agreement(reference, other, tol_pct, message):
    failures = []
    for base in _STATE_BASES:
        err = _max_rel_error(reference[base], other[base])
        print(f"{base}: ref_end={reference[base][-1]:.6g} other_end={other[base][-1]:.6g} max_rel={err:.4f}%")
        if err > tol_pct:
            failures.append(f"{base}: {err:.4f}%")
    if failures:
        pytest.fail(message + " Offending variables (> " + f"{tol_pct}%): " + "; ".join(failures))


@pytest.mark.integration
@pytest.mark.solver
def test_myokit_state_continuity_across_subexperiments(lotka_volterra_cellml):
    """A [T/2, T/2] two-sub-experiment Myokit run must equal a [T] single segment.

    Pure-Myokit check (no OpenCOR needed) so it runs in circulatory_autogen CI.
    Fails when update_times() resets the carried state between sub-experiments.
    """
    single = _run_segments(lotka_volterra_cellml, "CVODE_myokit", [_SIM_TIME])
    split = _run_segments(lotka_volterra_cellml, "CVODE_myokit", [_SIM_TIME / 2, _SIM_TIME / 2])
    _assert_agreement(
        single, split, _SELF_TOL_PCT,
        "Myokit loses state across sub-experiment boundaries — a split protocol diverges "
        "from the equivalent single segment (update_times resets state).",
    )


@pytest.mark.integration
@pytest.mark.solver
def test_myokit_vs_python_state_continuity(lotka_volterra_cellml, lotka_volterra_python):
    """Myokit (CVODE) and Python (solve_ivp) must agree on a multi-sub-experiment protocol.

    Cross-backend check that runs in circulatory_autogen CI (no OpenCOR needed): the
    Python backend carries state across sub-experiment boundaries correctly, so it is
    an independent reference for the Myokit fix. Catches the case where Myokit is
    internally self-consistent but still wrong.
    """
    split = [_SIM_TIME / 2, _SIM_TIME / 2]
    myokit = _run_segments(lotka_volterra_cellml, "CVODE_myokit", split)
    python = _run_segments(
        lotka_volterra_python, "solve_ivp", split,
        solver_info=_PY_SOLVER_INFO, model_type="python",
    )
    _assert_agreement(
        myokit, python, _CROSS_TOL_PCT,
        "Myokit and Python/solve_ivp disagree on a multi-sub-experiment protocol.",
    )


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.need_opencor
def test_myokit_vs_opencor_state_continuity(lotka_volterra_cellml):
    """Myokit and OpenCOR must agree on a multi-sub-experiment protocol.

    Mirrors the original bug report framing. Skipped only when OpenCOR is genuinely
    unavailable (the construction is attempted before the comparison so a real
    OpenCOR error is not silently swallowed by a name-resolution failure).
    """
    split = [_SIM_TIME / 2, _SIM_TIME / 2]
    try:
        opencor = _run_segments(lotka_volterra_cellml, "CVODE_opencor", split)
    except (RuntimeError, ImportError) as e:
        pytest.skip(f"OpenCOR backend unavailable: {e}")
    myokit = _run_segments(lotka_volterra_cellml, "CVODE_myokit", split)
    _assert_agreement(
        myokit, opencor, _CROSS_TOL_PCT,
        "Myokit and OpenCOR disagree on a multi-sub-experiment protocol.",
    )
