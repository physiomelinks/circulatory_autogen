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

from libcuflynx.param_id import fd_backend


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
        # A single experiment of a single sub-experiment. The real object always
        # carries these -- each observable names the experiment it belongs to, and
        # the backend needs them to read the right segment -- so the double does too.
        self.obs_info = {
            "const_idx_to_obs_idx": [0],
            "experiment_idxs": [0],
            "subexperiment_idxs": [0],
        }
        self.protocol_info = {"num_sub_per_exp": [1]}
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
    from libcuflynx.param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    stub = _Pid(lambda p: 3.0 * p[0])
    for attr in ("param_id_info", "obs_info", "protocol_info"):
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
    from libcuflynx.param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "aadc_python"
    with pytest.raises(NotImplementedError, match="AADC"):
        pid.get_observable_sensitivities([1.0])


def test_the_aadc_message_now_points_at_fd():
    from libcuflynx.param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "aadc_python"
    with pytest.raises(NotImplementedError, match="gradient_method 'FD'"):
        pid.get_observable_sensitivities([1.0])


def test_an_unknown_gradient_method_is_rejected():
    from libcuflynx.param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "casadi_python"
    with pytest.raises(ValueError, match="unknown gradient_method 'central'"):
        pid.get_observable_sensitivities([1.0], gradient_method="central")


@pytest.mark.parametrize("alias", [None, "", "analytic", "AUTO"])
def test_the_analytic_aliases_do_not_divert_to_fd(alias):
    from libcuflynx.param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "aadc_python"
    with pytest.raises(NotImplementedError):
        pid.get_observable_sensitivities([1.0], gradient_method=alias)


# ---------------------------------------------------------------------------
# The option is declared, so downstream tools can offer it
# ---------------------------------------------------------------------------
def test_gradient_method_is_in_the_analysis_schema():
    """The arms carry their own names -- AD / FSA, matching gradient_sources() and the Laplace
    gradient_source -- because only an explicit name can be offered, disabled, or reported back
    by a UI. 'analytic' is still accepted in code as a legacy spelling of 'auto', but is no
    longer advertised."""
    from libcuflynx.parsers.PrimitiveParsers import ANALYSIS_OPTIONS

    opts = {o["name"]: o for o in ANALYSIS_OPTIONS["sensitivity_analysis"]["options"]}
    assert "gradient_method" in opts
    assert opts["gradient_method"]["choices"] == ["auto", "AD", "FSA", "FD"]
    assert opts["gradient_method"]["default"] == "auto"


# ---------------------------------------------------------------------------
# Explicit arm names: each reaches the arm it names, or errors naming the
# mismatch -- never a silent reinterpretation (a caller that asked for FSA and
# silently got something else cannot check what ran).
# ---------------------------------------------------------------------------
def test_ad_reaches_the_casadi_arm(monkeypatch):
    from libcuflynx.param_id import paramID

    pid = paramID.OpencorParamID.__new__(paramID.OpencorParamID)
    pid.model_type = "casadi_python"
    pid.param_id_info = {"param_names": [["a/x"]]}
    called = []
    monkeypatch.setattr(paramID.casadi_backend, "get_observable_sensitivities",
                        lambda p, v: called.append(p) or {})
    pid.get_observable_sensitivities([1.0], gradient_method="AD")
    assert called == [pid]


def test_fsa_reaches_the_myokit_arm(monkeypatch):
    from libcuflynx.param_id import paramID

    class _FsaHelper:
        def enable_fsa(self, deps, indeps):
            return []

    pid = paramID.OpencorParamID.__new__(paramID.OpencorParamID)
    pid.model_type = "cellml"
    pid.do_ad = True
    pid.sim_helper = _FsaHelper()
    called = []
    monkeypatch.setattr(paramID.fsa_backend, "observable_feature_sensitivities",
                        lambda p, v: called.append(p) or {})
    pid.get_observable_sensitivities([1.0], gradient_method="FSA")
    assert called == [pid]


def test_ad_on_a_myokit_run_raises_naming_the_mismatch():
    from libcuflynx.param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "cellml"
    pid.solver_info = {"solver": "CVODE_myokit"}
    with pytest.raises(ValueError, match="'AD' needs model_type 'casadi_python'"):
        pid.get_observable_sensitivities([1.0], gradient_method="AD")


def test_fsa_on_a_casadi_run_raises_naming_whats_missing():
    from libcuflynx.param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "casadi_python"
    pid.solver_info = {"solver": "casadi_integrator"}
    pid.sim_helper = object()   # no enable_fsa
    with pytest.raises(ValueError, match="'FSA' is not available") as excinfo:
        pid.get_observable_sensitivities([1.0], gradient_method="FSA")
    msg = str(excinfo.value)
    assert "model_type is 'casadi_python'" in msg
    assert "CVODE_myokit" in msg
    assert "do_ad" in msg


def test_fsa_without_do_ad_names_the_missing_flag():
    from libcuflynx.param_id.paramID import OpencorParamID

    class _FsaHelper:
        def enable_fsa(self, deps, indeps):
            return []

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.model_type = "cellml"
    pid.solver_info = {"solver": "CVODE_myokit"}
    pid.sim_helper = _FsaHelper()
    pid.do_ad = False
    with pytest.raises(ValueError, match="do_ad must be true"):
        pid.get_observable_sensitivities([1.0], gradient_method="FSA")


# ---------------------------------------------------------------------------
# The step size belongs to the caller
#
# It is not a tuning detail: on Lotka-Volterra, moving it from 1e-3 to 1e-2
# changes a sensitivity coefficient by up to 48%, because `max` of an oscillating
# trace is a rough functional. A number that swings the answer that far must not
# be a constant buried in the backend -- which is also why a downstream tool
# passing its own step must be able to get its own numbers.
# ---------------------------------------------------------------------------
def test_the_step_size_reaches_the_backend():
    """f = x^2: the central difference is exact for any h, so a changed h must be
    visible somewhere other than the result -- check the evaluated points."""
    seen = []

    class _Recording(_Pid):
        def get_cost_obs_and_pred_from_params(self, param_vals, reset=True, only_one_exp=-1):
            seen.append(float(np.asarray(param_vals, dtype=float)[0]))
            return super().get_cost_obs_and_pred_from_params(param_vals, reset, only_one_exp)

    pid = _Recording(lambda p: p[0] ** 2, names=("a/x",), mins=(0.0,), maxs=(10.0,))
    fd_backend.observable_feature_sensitivities(pid, [2.0], h=0.25)
    # nominal, then 2 +/- 0.25*2
    assert sorted(seen) == pytest.approx([1.5, 2.0, 2.5])


def test_the_accessor_forwards_the_step():
    from libcuflynx.param_id.paramID import OpencorParamID

    seen = []

    class _Recording(_Pid):
        def get_cost_obs_and_pred_from_params(self, param_vals, reset=True, only_one_exp=-1):
            seen.append(float(np.asarray(param_vals, dtype=float)[0]))
            return super().get_cost_obs_and_pred_from_params(param_vals, reset, only_one_exp)

    stub = _Recording(lambda p: p[0], names=("a/x",), mins=(0.0,), maxs=(10.0,))
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info, pid.obs_info = stub.param_id_info, stub.obs_info
    pid.protocol_info = stub.protocol_info
    pid._observable_label = stub._observable_label
    pid.get_cost_obs_and_pred_from_params = stub.get_cost_obs_and_pred_from_params
    pid.get_obs_output_dict = stub.get_obs_output_dict
    pid.model_type = "aadc_python"

    pid.get_observable_sensitivities([4.0], gradient_method="FD", fd_rel_step=0.5)
    assert sorted(seen) == pytest.approx([2.0, 4.0, 6.0])


def test_omitting_the_step_keeps_the_backend_default():
    from libcuflynx.param_id.paramID import OpencorParamID

    seen = []

    class _Recording(_Pid):
        def get_cost_obs_and_pred_from_params(self, param_vals, reset=True, only_one_exp=-1):
            seen.append(float(np.asarray(param_vals, dtype=float)[0]))
            return super().get_cost_obs_and_pred_from_params(param_vals, reset, only_one_exp)

    stub = _Recording(lambda p: p[0], names=("a/x",), mins=(0.0,), maxs=(10.0,))
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info, pid.obs_info = stub.param_id_info, stub.obs_info
    pid.protocol_info = stub.protocol_info
    pid._observable_label = stub._observable_label
    pid.get_cost_obs_and_pred_from_params = stub.get_cost_obs_and_pred_from_params
    pid.get_obs_output_dict = stub.get_obs_output_dict
    pid.model_type = "aadc_python"

    pid.get_observable_sensitivities([1.0], gradient_method="FD")
    assert sorted(seen) == pytest.approx([0.999, 1.0, 1.001])


def test_fd_rel_step_is_in_the_analysis_schema():
    from libcuflynx.parsers.PrimitiveParsers import ANALYSIS_OPTIONS

    opts = {o["name"]: o for o in ANALYSIS_OPTIONS["sensitivity_analysis"]["options"]}
    assert opts["fd_rel_step"]["default"] == 1e-3


# ---------------------------------------------------------------------------
# Each observable is differentiated in its own experiment
#
# A data_item names both an experiment and a sub-experiment, and the cost scores
# it against that segment. Evaluating every observable against experiment 0
# differentiates the wrong trace: on SN_simple that gave the two `max` observables
# an identical, near-zero sensitivity, when one of them is the largest in the set.
# ---------------------------------------------------------------------------
class _MultiExpPid(_Pid):
    """Two experiments of one sub-experiment each; observable k lives in experiment k.

    ``get_cost_obs_and_pred_from_params`` returns one entry per segment, and the
    feature of an observable is only right when read from its own.
    """

    def __init__(self):
        super().__init__(lambda p: p, names=("a/x",), mins=(0.0,), maxs=(10.0,))
        self.obs_info = {
            "const_idx_to_obs_idx": [0, 1],
            "experiment_idxs": [0, 1],
            "subexperiment_idxs": [0, 0],
        }
        self.protocol_info = {"num_sub_per_exp": [1, 1]}

    def get_cost_obs_and_pred_from_params(self, param_vals, reset=True, only_one_exp=-1):
        self.calls += 1
        x = float(np.asarray(param_vals, dtype=float)[0])
        if only_one_exp == 0:  # the bug: only experiment 0 was ever simulated
            return 0.0, [("exp0", x), None], []
        return 0.0, [("exp0", x), ("exp1", x)], []

    def get_obs_output_dict(self, operands):
        tag, x = operands
        # Experiment 1 responds ten times as strongly as experiment 0.
        return {"const": [x, 10.0 * x] if tag == "exp1" else [x, x]}


def test_each_observable_is_differentiated_in_its_own_experiment():
    pid = _MultiExpPid()
    out = fd_backend.observable_feature_sensitivities(pid, [1.0])
    # obs0 lives in experiment 0 (slope 1); obs1 in experiment 1 (slope 10).
    assert out["obs0"]["a/x"] == pytest.approx(1.0, rel=1e-6)
    assert out["obs1"]["a/x"] == pytest.approx(10.0, rel=1e-6)


def test_observables_in_different_experiments_are_not_forced_equal():
    """The shape of the old bug: every observable read experiment 0, so any two
    measuring the same feature came back identical."""
    pid = _MultiExpPid()
    out = fd_backend.observable_feature_sensitivities(pid, [1.0])
    assert out["obs0"]["a/x"] != out["obs1"]["a/x"]


def test_all_experiments_are_run():
    """only_one_exp is left at its default so every segment is simulated."""
    pid = _MultiExpPid()
    fd_backend.observable_feature_sensitivities(pid, [1.0])
    assert pid.calls == 3  # nominal + 2 for the single parameter


# ---------------------------------------------------------------------------
# Labels identify one observable each
# ---------------------------------------------------------------------------
def _labeller(names, ops, operands, exps, subs):
    from libcuflynx.param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.obs_info = {
        "names_for_plotting": names, "operations": ops, "operands": operands,
        "experiment_idxs": exps, "subexperiment_idxs": subs, "num_obs": len(names),
    }
    return pid


def test_two_experiments_measuring_the_same_feature_get_distinct_labels():
    """These labels are the keys of the returned dict, so a collision silently
    drops one observable's sensitivities and reports the other's for both."""
    pid = _labeller(["V_max", "V_max"], ["max", "max"], [["a/V"], ["a/V"]], [0, 2], [1, 1])
    labels = [pid._observable_label(0), pid._observable_label(1)]
    assert len(set(labels)) == 2
    assert "exp 0" in labels[0] and "exp 2" in labels[1]


def test_an_unambiguous_label_is_left_alone():
    """A single-experiment study keeps the spelling it already had."""
    pid = _labeller(["V_max", "V_mean"], ["max", "mean"], [["a/V"], ["a/V"]], [0, 0], [0, 0])
    assert pid._observable_label(0) == "V_max (max a/V)"


def test_the_operation_still_disambiguates_before_the_experiment():
    pid = _labeller(["V", "V"], ["max", "mean"], [["a/V"], ["a/V"]], [0, 1], [0, 0])
    assert pid._observable_label(0) == "V (max a/V)"
    assert pid._observable_label(1) == "V (mean a/V)"
