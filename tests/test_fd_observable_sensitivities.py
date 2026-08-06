"""Finite-difference observable sensitivities (issue #338).

The analytic arms (CasADi jacobian, Myokit CVODES) only exist for two backends,
so local sensitivity analysis was unavailable on AADC and on the plain scipy
backend -- the user was told to run a global Sobol SA instead, which is a
different analysis at num_samples*(2M+2) simulations rather than 2M.

FD is opt-in by name, never a silent fallback: it costs 2M simulations and its
accuracy depends on a step size the analytic arms do not have, so a caller must
always know which produced their numbers. No model/MPI needed here -- the
feature evaluation is stubbed, since what is under test is the differencing and
the dispatch, not the solver.
"""
import numpy as np
import pytest

from param_id import fd_backend


class _Pid:
    """Enough of OpencorParamID for the FD arm: parameter metadata, observable
    labels, and a feature function of the parameter vector."""

    def __init__(self, feature_fn, names=("a/x", "b/y"), mins=(0.0, 0.0), maxs=(10.0, 10.0),
                 fail_at=None):
        self.param_id_info = {
            "param_names": [[n] for n in names],
            "param_mins": np.array(mins, dtype=float),
            "param_maxs": np.array(maxs, dtype=float),
        }
        self.obs_info = {"const_idx_to_obs_idx": [0]}
        self._feature_fn = feature_fn
        self._fail_at = fail_at
        self.calls = 0

    def _observable_label(self, obs_idx):
        return f"obs{obs_idx}"

    def get_cost_obs_and_pred_from_params(self, param_vals, reset=True, only_one_exp=-1):
        self.calls += 1
        if self._fail_at is not None and self._fail_at(param_vals):
            return 0.0, [None], []
        return 0.0, [np.asarray(param_vals, dtype=float)], []

    def get_obs_output_dict(self, operands):
        return {"const": [self._feature_fn(operands)]}


# ---------------------------------------------------------------------------
# The differencing
# ---------------------------------------------------------------------------
def test_it_recovers_a_known_derivative():
    """f(p) = 3*x + 2*y  ->  df/dx = 3, df/dy = 2."""
    pid = _Pid(lambda p: 3.0 * p[0] + 2.0 * p[1])
    out = fd_backend.observable_feature_sensitivities(pid, [1.0, 1.0])
    assert out["obs0"]["a/x"] == pytest.approx(3.0, rel=1e-6)
    assert out["obs0"]["b/y"] == pytest.approx(2.0, rel=1e-6)


def test_it_is_central_not_forward():
    """A central difference is exact for a quadratic; a forward one is not.
    f = x^2 at x=2 -> 4 exactly."""
    pid = _Pid(lambda p: p[0] ** 2)
    out = fd_backend.observable_feature_sensitivities(pid, [2.0, 0.0])
    assert out["obs0"]["a/x"] == pytest.approx(4.0, rel=1e-9)


def test_it_costs_two_simulations_per_parameter_plus_one():
    """2M for M parameters, plus the nominal convergence check."""
    pid = _Pid(lambda p: p[0] + p[1])
    fd_backend.observable_feature_sensitivities(pid, [1.0, 1.0])
    assert pid.calls == 2 * 2 + 1


# ---------------------------------------------------------------------------
# Step size
# ---------------------------------------------------------------------------
def test_the_step_is_relative_to_the_parameter():
    """Parameters span orders of magnitude, so one absolute step cannot suit
    them all."""
    assert fd_backend._step(1000.0, 0.0, 1.0, 1e-3) == pytest.approx(1.0)
    assert fd_backend._step(0.001, 0.0, 1.0, 1e-3) == pytest.approx(1e-6)


def test_a_zero_parameter_takes_its_scale_from_its_range():
    """Zero has no scale of its own, and a zero step would divide by zero."""
    assert fd_backend._step(0.0, -5.0, 5.0, 1e-3) == pytest.approx(1e-2)


def test_a_zero_parameter_with_a_degenerate_range_still_steps():
    assert fd_backend._step(0.0, 2.0, 2.0, 1e-3) == pytest.approx(1e-3)


def test_a_derivative_is_recovered_at_a_zero_valued_parameter():
    pid = _Pid(lambda p: 5.0 * p[0], mins=(-1.0,), maxs=(1.0,), names=("a/x",))
    out = fd_backend.observable_feature_sensitivities(pid, [0.0])
    assert out["obs0"]["a/x"] == pytest.approx(5.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Failure is reported, not disguised
# ---------------------------------------------------------------------------
def test_a_parameter_whose_runs_fail_is_none_not_zero():
    """Zero is a real answer -- "this parameter does not matter" -- and must not
    be indistinguishable from a solve that never converged."""
    pid = _Pid(lambda p: p[0] + p[1], fail_at=lambda p: p[1] != 1.0)
    out = fd_backend.observable_feature_sensitivities(pid, [1.0, 1.0])
    assert out["obs0"]["a/x"] == pytest.approx(1.0, rel=1e-6)
    assert out["obs0"]["b/y"] is None


def test_a_failed_nominal_run_raises():
    pid = _Pid(lambda p: p[0], fail_at=lambda p: True)
    with pytest.raises(RuntimeError, match="nominal simulation failed"):
        fd_backend.observable_feature_sensitivities(pid, [1.0, 1.0])


# ---------------------------------------------------------------------------
# Dispatch: opt-in, never silent
# ---------------------------------------------------------------------------
def test_fd_is_selected_by_name():
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    stub = _Pid(lambda p: 3.0 * p[0])
    for attr in ("param_id_info", "obs_info"):
        setattr(pid, attr, getattr(stub, attr))
    pid._observable_label = stub._observable_label
    pid.get_cost_obs_and_pred_from_params = stub.get_cost_obs_and_pred_from_params
    pid.get_obs_output_dict = stub.get_obs_output_dict
    pid.model_type = "aadc_python"  # no analytic arm at all

    out = pid.get_observable_sensitivities([1.0, 1.0], gradient_method="FD")
    assert out["obs0"]["a/x"] == pytest.approx(3.0, rel=1e-6)


def test_without_fd_an_analytic_less_backend_still_raises():
    """The default must not quietly become FD: a result computed a different way,
    at a different cost and accuracy, is not the same result."""
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "aadc_python"
    with pytest.raises(NotImplementedError, match="AADC"):
        pid.get_observable_sensitivities([1.0])


def test_the_aadc_message_now_points_at_fd():
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "aadc_python"
    with pytest.raises(NotImplementedError, match="gradient_method 'FD'"):
        pid.get_observable_sensitivities([1.0])


def test_an_unknown_gradient_method_is_rejected():
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "casadi_python"
    with pytest.raises(ValueError, match="unknown gradient_method 'central'"):
        pid.get_observable_sensitivities([1.0], gradient_method="central")


@pytest.mark.parametrize("alias", [None, "", "analytic", "AUTO"])
def test_the_analytic_aliases_do_not_divert_to_fd(alias):
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "aadc_python"
    with pytest.raises(NotImplementedError):
        pid.get_observable_sensitivities([1.0], gradient_method=alias)


# ---------------------------------------------------------------------------
# The option is declared, so downstream tools can offer it
# ---------------------------------------------------------------------------
def test_gradient_method_is_in_the_analysis_schema():
    from parsers.PrimitiveParsers import ANALYSIS_OPTIONS

    opts = {o["name"]: o for o in ANALYSIS_OPTIONS["sensitivity_analysis"]["options"]}
    assert "gradient_method" in opts
    assert opts["gradient_method"]["choices"] == ["analytic", "FD"]
    assert opts["gradient_method"]["default"] == "analytic"
