"""Tests for profile likelihood (``libcuflynx.identifiabilty_analysis.ProfileLikelihood``)."""

import json
import os
import shutil
import tempfile

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from _params_for_id import write_params_for_id
from libcuflynx.identifiabilty_analysis.ProfileLikelihood import (
    NLL_THRESHOLD_95,
    ProfileLikelihood,
    _HeldEntry,
    profile_likelihood_options,
)
from libcuflynx.param_id.optimisers import SciPyMinimizeOptimiser
from libcuflynx.param_id.paramID import CVS0DParamID, ParamID
from libcuflynx.parsers.PrimitiveParsers import analysis_options
from libcuflynx.scripts.param_id_run_script import run_param_id
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.utilities.mpi_utils import get_MPI
from libcuflynx.utilities.utility_funcs import Normalise_class


@pytest.fixture(scope="function")
def mpi_comm():
    """Fixture that provides MPI communicator."""
    # mpi4py is optional: get_MPI is a one-rank stand-in without it.
    comm = get_MPI().COMM_WORLD
    return comm


def _ensure_cellml_model_generated(config, mpi_comm):
    """Ensure generated CellML exists before run_param_id (mirrors test_param_id.py)."""
    if config.get("model_type") != "cellml":
        return
    rank = mpi_comm.Get_rank()
    if rank == 0:
        success = generate_with_new_architecture(False, config)
        prefix = config.get("file_prefix", "<unknown>")
        assert success, f"CellML autogeneration failed for {prefix}"
    mpi_comm.Barrier()


# ---- a ParamID with no model behind it -----------------------------------------------------


def _stub_param_id(tmp_dir, method="CMA-ES"):
    """A real ParamID (its make_optimiser, remove_params_by_idx) over a quadratic cost.

    Entries: 0 a group [A, B], 1 a scale modifier on [C, D] (baselines 2, 3), 2 a plain E.
    ``pid.calls`` records every vector the cost and gradient methods are given.
    """
    pid = object.__new__(ParamID)
    mins, maxs = np.array([0.0, 0.5, 0.0]), np.array([10.0, 4.0, 10.0])
    pid.param_id_info = {
        "param_names": [["g/A", "g/B"], ["m/C", "m/D"], ["p/E"]],
        "param_mins": mins.copy(),
        "param_maxs": maxs.copy(),
        "param_prior_types": np.array(["uniform"] * 3),
        "param_names_for_plotting": np.array(["A", "mod", "E"]),
        "param_entry_names": ["grp", "mod", "E"],
        "param_labels": ["g/A+g/B", "mod", "p/E"],
        "modifiers": [
            {
                "index": 1,
                "name": "mod",
                "modifier": "scale",
                "targets": ["m/C", "m/D"],
                "baselines": [2.0, 3.0],
                "inputs": {},
                "resolved_inputs": {},
                "affine": {"a": [2.0, 3.0], "b": [0.0, 0.0]},
            }
        ],
        "modifier_funcs_external_path": None,
    }
    pid.num_params = 3
    pid.param_norm_obj = Normalise_class(mins, maxs)
    pid.param_init = None
    pid.best_param_vals = np.array([5.0, 1.0, 5.0])
    pid.best_cost = 0.25
    pid.optimiser_options = {"num_calls_to_function": 1000, "cost_type": "gaussian_MLE"}
    pid.output_dir = str(tmp_dir)
    pid.param_id_method = method
    pid.DEBUG = True
    pid.do_ad = True
    pid.model_type = "cellml"
    pid.emulates_features = False
    pid.calls = []

    def cost(vals):
        vals = np.array(vals, dtype=float)
        pid.calls.append(vals)
        return float(np.sum(vals**2))

    def gradient(vals):
        vals = np.array(vals, dtype=float)
        pid.calls.append(vals)
        return 2.0 * vals

    pid.get_cost_from_params = cost
    pid.get_cost = cost
    pid.get_gradient = gradient
    pid.get_cost_and_gradient = lambda vals: (cost(vals), 2.0 * np.asarray(vals, dtype=float))
    pid.get_lnlikelihood_lnprior_from_params = lambda vals: -cost(vals)
    return pid


def _profile_for(pid):
    profile = ProfileLikelihood(
        param_id=pid,
        param_id_info=pid.param_id_info,
        output_dir=pid.output_dir,
        num_points=3,
        range_factor=0.1,
        optimiser_options={"num_calls_to_function": 40},
    )
    profile.set_best_param_vals(pid.best_param_vals)
    return profile


class _FakeOptimiser:
    """Stands in for ParamID.make_optimiser: records what it was given, returns ``optimum``."""

    def __init__(self, optimum=None, error=None):
        self.optimum, self.error, self.made = optimum, error, []

    def __call__(self, cost_obj, param_id_info, param_norm_obj, output_dir,
                 optimiser_options=None):
        self.made.append({"cost_obj": cost_obj, "info": param_id_info, "norm": param_norm_obj,
                          "output_dir": output_dir, "options": optimiser_options,
                          "param_init": np.array(cost_obj.param_init, dtype=float)})
        fake = self

        class _Run:
            def run(self):
                if fake.error is not None:
                    raise fake.error
                optimum = fake.optimum(cost_obj) if callable(fake.optimum) else fake.optimum
                self.best_param_vals = np.array(
                    cost_obj.param_init if optimum is None else optimum, dtype=float)

        return _Run()


# ---- one profile point ---------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, options",
    [
        ("CMA-ES", {}),
        ("genetic_algorithm", {}),
        ("genetic_algorithm", {"objective_function": "likelihood"}),
        ("sp_minimize", {}),
    ],
)
@pytest.mark.parametrize("idx, value", [(0, 7.0), (1, 2.0), (2, 3.0)])
def test_the_held_value_reaches_every_cost_call(tmp_path, method, options, idx, value):
    """Every vector the param_id is given is the full one, with the held entry at its value."""
    pid = _stub_param_id(tmp_path, method)
    profile = _profile_for(pid)
    profile.optimiser_options.update(options)
    cost, optimum = profile._optimize_with_fixed_parameter(idx, value)

    assert len(pid.calls) > 2
    for vals in pid.calls:
        assert vals.shape == (3,) and vals[idx] == value
    assert optimum.shape == (2,)
    assert cost == pytest.approx(np.sum(np.insert(optimum, idx, value) ** 2))
    # Nothing on the param_id was reduced or replaced.
    assert pid.num_params == 3 and len(pid.param_id_info["param_names"]) == 3
    assert [m["index"] for m in pid.param_id_info["modifiers"]] == [1]


@pytest.mark.unit
def test_the_optimiser_gets_the_other_entries_and_the_merged_options(tmp_path, monkeypatch):
    pid = _stub_param_id(tmp_path)
    fake = _FakeOptimiser(optimum=[1.5, 2.5])
    monkeypatch.setattr(pid, "make_optimiser", fake, raising=False)
    cost, optimum = _profile_for(pid)._optimize_with_fixed_parameter(1, 2.0)

    made = fake.made[0]
    assert made["info"]["param_names"] == [["g/A", "g/B"], ["p/E"]]
    np.testing.assert_allclose(made["info"]["param_mins"], [0.0, 0.0])
    np.testing.assert_allclose(made["info"]["param_maxs"], [10.0, 10.0])
    np.testing.assert_allclose(made["norm"].unnormalise(np.array([1.0, 1.0])), [10.0, 10.0])
    # ia_options.optimiser_options are merged over the calibration's, which are unchanged.
    assert made["options"] == {"num_calls_to_function": 40, "cost_type": "gaussian_MLE"}
    assert pid.optimiser_options == {"num_calls_to_function": 1000,
                                     "cost_type": "gaussian_MLE"}
    np.testing.assert_allclose(made["param_init"], [5.0, 5.0])
    assert made["output_dir"] is None
    np.testing.assert_allclose(optimum, [1.5, 2.5])
    assert cost == pytest.approx(1.5**2 + 2.0**2 + 2.5**2)


@pytest.mark.unit
@pytest.mark.parametrize("rank", [0, 1])
def test_no_rank_gives_the_optimiser_an_output_dir(tmp_path, monkeypatch, rank):
    pid = _stub_param_id(tmp_path)
    fake = _FakeOptimiser()
    monkeypatch.setattr(pid, "make_optimiser", fake, raising=False)
    profile = _profile_for(pid)
    profile.rank = rank
    profile._optimize_with_fixed_parameter(0, 7.0)
    assert fake.made[0]["output_dir"] is None
    assert not os.path.exists(os.path.join(str(tmp_path), "profile_likelihood"))


def _files_under(root):
    return sorted(os.path.relpath(os.path.join(d, f), root)
                  for d, _, files in os.walk(root) for f in files)


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, options",
    [
        ("CMA-ES", {}),
        ("genetic_algorithm", {}),
        ("sp_minimize", {}),
        ("multi_start_sp_minimize", {"num_starts": 3}),
        ("bayesian", {"num_calls_to_function": 6}),
    ],
)
def test_a_profile_points_optimiser_writes_no_files(tmp_path, monkeypatch, method, options):
    """Every param_id_method's optimiser, given no output_dir, writes nothing anywhere: the
    profile's own JSON and plots are its output."""
    if method == "bayesian":
        pytest.importorskip("skopt")
    out_dir, cwd = tmp_path / "out", tmp_path / "cwd"
    out_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    pid = _stub_param_id(out_dir, method)
    pid.acq_func, pid.n_initial_points, pid.random_state = "EI", 3, 0
    pid.acq_func_kwargs = {}
    profile = _profile_for(pid)
    profile.optimiser_options.update(options)
    cost, optimum = profile._optimize_with_fixed_parameter(2, 3.0)

    assert optimum.shape == (2,) and np.isfinite(cost)
    assert _files_under(tmp_path) == []


@pytest.mark.unit
def test_the_gradient_drops_the_held_component(tmp_path):
    pid = _stub_param_id(tmp_path, "sp_minimize")
    held = _HeldEntry(pid, 1, 2.0, [5.0, 5.0])
    np.testing.assert_allclose(held.get_gradient([3.0, 4.0]), [6.0, 8.0])
    cost, gradient = held.get_cost_and_gradient([3.0, 4.0])
    assert cost == pytest.approx(9.0 + 4.0 + 16.0)
    np.testing.assert_allclose(gradient, [6.0, 8.0])
    for vals in pid.calls:
        np.testing.assert_allclose(vals, [3.0, 2.0, 4.0])

    # L-BFGS-B descends that gradient to the other entries' lower bounds.
    _, optimum = _profile_for(pid)._optimize_with_fixed_parameter(0, 7.0)
    np.testing.assert_allclose(optimum, [0.5, 0.0], atol=1e-6)


@pytest.mark.unit
def test_a_single_entry_profile_is_the_cost_itself(tmp_path, monkeypatch):
    pid = _stub_param_id(tmp_path)
    pid.num_params = 1
    monkeypatch.setattr(pid, "make_optimiser", _FakeOptimiser(error=AssertionError("no fit")),
                        raising=False)
    cost, optimum = _profile_for(pid)._optimize_with_fixed_parameter(0, 3.0)
    assert cost == pytest.approx(9.0) and optimum.size == 0


@pytest.mark.unit
def test_each_profile_point_warm_starts_from_its_neighbours_optimum(tmp_path, monkeypatch):
    """The sweep runs outwards from the best fit, each fit starting at the last one's optimum."""
    pid = _stub_param_id(tmp_path)
    # A recognisable optimum per point: [held, held + 100].
    fake = _FakeOptimiser(optimum=lambda held: [held.value, held.value + 100.0])
    monkeypatch.setattr(pid, "make_optimiser", fake, raising=False)
    profile = _profile_for(pid)
    profile.num_points = 5
    result = profile._profile_single_parameter(2)

    values = result["param_values"]
    centre = int(np.argmin(np.abs(values - 5.0)))
    assert centre == 2
    order = [2, 3, 4, 1, 0]
    expected = {2: [5.0, 1.0], 3: [values[2], values[2] + 100], 4: [values[3], values[3] + 100]}
    expected[1] = [values[2], values[2] + 100]
    expected[0] = [values[1], values[1] + 100]
    starts = [made["param_init"] for made in fake.made]
    assert [made["cost_obj"].value for made in fake.made] == pytest.approx(values[order])
    assert len(starts) == 5
    for point, start in zip(order, starts):
        np.testing.assert_allclose(start, expected[point])


@pytest.mark.unit
def test_the_model_is_left_at_the_best_fit_after_each_parameter(tmp_path, monkeypatch):
    pid = _stub_param_id(tmp_path)
    monkeypatch.setattr(pid, "make_optimiser", _FakeOptimiser(), raising=False)
    profile = _profile_for(pid)
    real = profile._profile_single_parameter

    def profile_and_note_the_last_call(param_idx):
        result = real(param_idx)
        pid.calls.append(("done", param_idx))
        return result

    profile._profile_single_parameter = profile_and_note_the_last_call
    profile.run()
    done = [i for i, call in enumerate(pid.calls) if isinstance(call, tuple)]
    assert [pid.calls[i] for i in done] == [("done", 0), ("done", 1), ("done", 2)]
    for i in done:
        np.testing.assert_allclose(pid.calls[i + 1], [5.0, 1.0, 5.0])


# ---- remove_params_by_idx (plot_param_id_script, MCMC) -------------------------------------


@pytest.mark.unit
def test_remove_params_by_idx_reindexes_modifiers_and_per_entry_lists(tmp_path):
    pid = _stub_param_id(tmp_path)
    pid.remove_params_by_idx([0])
    info = pid.param_id_info
    assert info["param_names"] == [["m/C", "m/D"], ["p/E"]]
    assert list(info["param_names_for_plotting"]) == ["mod", "E"]
    assert info["param_entry_names"] == ["mod", "E"]
    assert info["param_labels"] == ["mod", "p/E"]
    assert [m["index"] for m in info["modifiers"]] == [0]

    pid = _stub_param_id(tmp_path)
    pid.remove_params_by_idx([1])
    assert pid.param_id_info["modifiers"] == []


@pytest.mark.unit
def test_cvs0d_remove_params_by_name_removes_them_from_the_param_id(tmp_path):
    """What plot_param_id_script does with param_names_to_remove.csv."""
    pid = _stub_param_id(tmp_path)
    outer = object.__new__(CVS0DParamID)
    outer.mcmc_instead = False
    outer.param_id = pid
    outer.param_id_info = pid.param_id_info
    outer.remove_params_by_name([["p/E"]])
    assert pid.num_params == 2
    assert outer.param_id_info["param_names"] == [["g/A", "g/B"], ["m/C", "m/D"]]


# ---- the 95% threshold -----------------------------------------------------------------------


@pytest.mark.unit
def test_the_95_percent_interval_uses_the_negative_log_likelihood_scale():
    """cost = -ln L / N, so the bound is a rise of 1.92 / N in the cost, not 3.84."""
    sigma, centre, n_obs = 0.1, 0.5, 10.0
    values = np.linspace(0.0, 1.0, 4001)
    nll = (values - centre) ** 2 / (2 * sigma**2)
    pid = type("P", (), {"_lnlikelihood_denorm_factor": n_obs})()
    profile = ProfileLikelihood(pid, None, "/unused")
    ci = profile.confidence_interval_95(values, nll / n_obs, best_val=centre)
    half_width = sigma * np.sqrt(2 * NLL_THRESHOLD_95)  # 1.96 sigma
    assert NLL_THRESHOLD_95 == pytest.approx(3.841459 / 2, rel=1e-6)
    assert ci["lower"] == pytest.approx(centre - half_width, abs=1e-3)
    assert ci["upper"] == pytest.approx(centre + half_width, abs=1e-3)
    assert ci["cost_threshold"] == pytest.approx(NLL_THRESHOLD_95 / n_obs)

    # A sweep that never rises far enough leaves that side open.
    flat = profile.confidence_interval_95(values, np.where(values < centre, 0.0, nll / n_obs))
    assert flat["lower"] is None and flat["upper"] is not None


# ---- MLE costs -------------------------------------------------------------------------------


def _with_costs(pid, cost_types):
    """Give the stub observables named p_ao, q_lv, ... with these costs; record the costs used."""
    from libcuflynx.funcs.cost_funcs_user import get_cost_funcs_dict_for_mode

    names = ["p_ao", "q_lv", "v_lv"][: len(cost_types)]
    pid.obs_info = {"num_obs": len(cost_types), "cost_type": list(cost_types),
                    "data_item_names": names}
    pid.cost_type = pid.obs_info["cost_type"]
    pid.cost_funcs_dict = get_cost_funcs_dict_for_mode("numpy")
    pid._lnlikelihood_denorm_factor = 4.0
    used = []

    def cost(vals):
        used.append(list(pid.cost_type))
        return float(np.sum(np.asarray(vals) ** 2))

    pid.get_cost_from_params = pid.get_cost = cost
    pid.make_optimiser = _FakeOptimiser()
    return used


@pytest.mark.unit
def test_profile_converts_a_cost_to_its_mle_form_with_a_warning(tmp_path):
    pid = _stub_param_id(tmp_path)
    used = _with_costs(pid, ["MSE", "gaussian_MLE"])
    original = pid.obs_info["cost_type"]
    profile = _profile_for(pid)
    with pytest.warns(UserWarning) as record:
        profile.run()

    converted = [str(w.message) for w in record if "converting cost" in str(w.message)]
    assert converted == [
        "profile likelihood: converting cost 'MSE' to its MLE form 'gaussian_MLE' for p_ao"
    ]
    # The best fit, every re-fit and each reset to the best fit are scored on the MLE form.
    assert len(used) == 1 + 3 * 3 + 3
    assert all(costs == ["gaussian_MLE", "gaussian_MLE"] for costs in used)
    # ... so every profile gets an interval, on the -ln L scale.
    for result in profile.profile_results.values():
        ci = result["ci_95"]
        assert ci is not None
        assert ci["cost_threshold"] == pytest.approx(
            min(np.min(result["costs"]), result["best_cost"]) + NLL_THRESHOLD_95 / 4.0
        )
    # The calibration's own costs are back.
    assert pid.obs_info["cost_type"] is original
    assert original == ["MSE", "gaussian_MLE"]
    assert pid.cost_type is original


@pytest.mark.unit
def test_profile_restores_the_costs_when_a_profile_point_fails(tmp_path):
    pid = _stub_param_id(tmp_path)
    _with_costs(pid, ["MSE"])
    original = pid.obs_info["cost_type"]

    pid.make_optimiser = _FakeOptimiser(error=RuntimeError("solver blew up"))
    with pytest.warns(UserWarning, match="converting cost 'MSE'"):
        with pytest.raises(RuntimeError, match="solver blew up"):
            _profile_for(pid).run()
    assert pid.obs_info["cost_type"] is original
    assert original == ["MSE"]
    assert pid.cost_type is original


@pytest.mark.unit
def test_profile_refuses_a_cost_without_an_mle_form(tmp_path):
    pid = _stub_param_id(tmp_path)
    used = _with_costs(pid, ["gaussian_MLE", "AE"])
    with pytest.raises(ValueError) as excinfo:
        _profile_for(pid).run()
    message = str(excinfo.value)
    assert "'q_lv' (cost 'AE')" in message
    assert "p_ao" not in message
    assert "'MSE' -> 'gaussian_MLE'" in message
    assert used == []
    assert pid.obs_info["cost_type"] == ["gaussian_MLE", "AE"]


# ---- options -------------------------------------------------------------------------------


@pytest.mark.unit
def test_profile_likelihood_options_default_from_the_schema_and_are_checked():
    schema = {o["name"]: o["default"] for o in analysis_options("identifiability_analysis")}
    options = profile_likelihood_options({"method": "profile_likelihood"})
    assert options == {
        "num_points": schema["num_points"],
        "range_factor": schema["range_factor"],
        "optimiser_options": {},
    }
    assert profile_likelihood_options(
        {"num_points": 5, "range_factor": 0.5, "optimiser_options": {"seed": 1}}
    ) == {"num_points": 5, "range_factor": 0.5, "optimiser_options": {"seed": 1}}
    for bad in (
        {"num_points": 1},
        {"num_points": 2.5},
        {"num_points": True},
        {"range_factor": 0},
        {"range_factor": 1.5},
        {"range_factor": "0.2"},
        {"optimiser_options": [1]},
    ):
        with pytest.raises(ValueError):
            profile_likelihood_options(bad)


# ---- construction, saving and plotting -----------------------------------------------------


@pytest.mark.unit
def test_profile_likelihood_initialization():
    param_id_info = {
        "param_names": ["param1", "param2"],
        "param_mins": np.array([0.0, 0.0]),
        "param_maxs": np.array([1.0, 1.0]),
    }
    profile = ProfileLikelihood(
        param_id=None,
        param_id_info=param_id_info,
        output_dir="/tmp/test",
        num_points=20,
        range_factor=0.1,
    )
    assert profile.num_points == 20
    assert profile.range_factor == 0.1
    assert profile.output_dir == "/tmp/test"
    assert profile.param_id_info == param_id_info


@pytest.mark.unit
def test_profile_likelihood_set_best_param_vals():
    profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir="/tmp/test")
    best_params = np.array([0.5, 0.7])
    profile.set_best_param_vals(best_params)
    np.testing.assert_array_equal(profile.best_param_vals, best_params)


@pytest.mark.unit
def test_profile_likelihood_no_best_params():
    profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir="/tmp/test")
    with pytest.raises(ValueError, match="Best parameter values must be set first"):
        profile.run()


def _mock_results(names, costs_1=(2.0, 1.0, 0.5, 1.0, 2.0)):
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    return {
        0: {
            "param_values": values,
            "costs": np.array(costs_1),
            "param_name": names[0],
            "best_val": 0.3,
        },
        1: {
            "param_values": values,
            "costs": np.array([3.0, 1.5, 0.8, 1.5, 3.0]),
            "param_name": names[1],
            "best_val": 0.3,
        },
    }


@pytest.mark.unit
def test_profile_likelihood_save_results():
    with tempfile.TemporaryDirectory() as temp_dir:
        profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir=temp_dir)
        profile.profile_results = _mock_results(["param1", "param2"])
        profile._save_results()

        results_dir = os.path.join(temp_dir, "profile_likelihood")
        with open(os.path.join(results_dir, "summary.json")) as f:
            summary = json.load(f)
        assert summary["num_params"] == 2
        assert summary["parameters"] == ["param1", "param2"]
        assert summary["nll_threshold_95"] == pytest.approx(NLL_THRESHOLD_95)
        for param_idx in [0, 1]:
            param_files = [
                f for f in os.listdir(results_dir) if f.startswith(f"param_{param_idx}_")
            ]
            with open(os.path.join(results_dir, param_files[0])) as f:
                data = json.load(f)
            assert {"param_values", "costs", "param_name", "best_val", "ci_95"} <= set(data)


@pytest.mark.unit
def test_profile_likelihood_plotting():
    with tempfile.TemporaryDirectory() as temp_dir:
        profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir=temp_dir)
        profile.profile_results = _mock_results(["param1", "param2"])
        profile._plot_profiles()
        results_dir = os.path.join(temp_dir, "profile_likelihood")
        assert "profile_likelihood_plots.pdf" in os.listdir(results_dir)
        for param_idx in [0, 1]:
            assert any(
                f.startswith(f"param_{param_idx}_") and f.endswith(".pdf")
                for f in os.listdir(results_dir)
            )


@pytest.mark.unit
def test_plotting_survives_infinite_costs_and_labels_list_names(monkeypatch):
    """A failed point (cost inf) must not break the y-limits; a list name is not "['...']"."""
    labels = []
    original = matplotlib.axes.Axes.set_xlabel

    def record(self, label, *args, **kwargs):
        labels.append(label)
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", record)
    with tempfile.TemporaryDirectory() as temp_dir:
        profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir=temp_dir)
        profile.profile_results = _mock_results(
            [["aortic_root/C"], ["g/A", "g/B"]], costs_1=(np.inf, 1.0, 0.5, 1.0, np.inf)
        )
        profile._plot_profiles()
        files = os.listdir(os.path.join(temp_dir, "profile_likelihood"))
    assert set(labels) == {"aortic_root/C", "g/A+g/B"}
    assert "param_0_aortic_root_C.pdf" in files and "param_1_g_A__g_B.pdf" in files


# ---- end to end ----------------------------------------------------------------------------

_OSCILLATOR_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "funcs_user", "example_model_scipy")
)


@pytest.mark.integration
@pytest.mark.mpi
def test_profile_points_warm_start_through_the_optimiser(
    base_user_inputs, temp_output_dir, temp_generated_models_dir, mpi_comm, monkeypatch
):
    """Each profile point's CMA-ES fit starts at the neighbouring point's optimum, not at the
    model's parameter values (the scipy oscillator; seconds)."""
    rank = mpi_comm.Get_rank()
    resources_dir = os.path.join(temp_output_dir, "resources")
    if rank == 0:
        os.makedirs(resources_dir, exist_ok=True)
        for name in (
            "oscillator_params_for_id.json",
            "oscillator_parameters.json",
            "oscillator_obs_data.json",
        ):
            shutil.copy(os.path.join(_OSCILLATOR_DIR, name), os.path.join(resources_dir, name))
    mpi_comm.Barrier()

    config = base_user_inputs.copy()
    config.update(
        {
            "file_prefix": "oscillator",
            "input_param_file": "oscillator_parameters.json",
            "model_type": "external_python",
            "solver": "external",
            "solver_info": {"solver": "external", "method": "external"},
            "external_model_path": os.path.join(_OSCILLATOR_DIR, "oscillator_model.py"),
            "resources_dir": resources_dir,
            "param_id_method": "CMA-ES",
            "pre_time": 0.0,
            "sim_time": 10.0,
            "dt": 0.05,
            "DEBUG": True,
            "do_uq": False,
            "do_ad": False,
            "plot_predictions": False,
            "model_out_names": ["oscillator/x"],
            "param_id_obs_path": os.path.join(resources_dir, "oscillator_obs_data.json"),
            "param_id_output_dir": temp_output_dir,
            "generated_models_dir": temp_generated_models_dir,
            "debug_optimiser_options": {
                "num_calls_to_function": 40,
                "cost_type": "gaussian_MLE",
                "seed": 0,
            },
            "do_ia": True,
            "ia_options": {
                "method": "profile_likelihood",
                "num_points": 3,
                "range_factor": 0.1,
                "optimiser_options": {"num_calls_to_function": 12},
            },
        }
    )

    starts, budgets = [], []
    real_make = ParamID.make_optimiser

    def spy(self, cost_obj, *args, **kwargs):
        optimiser = real_make(self, cost_obj, *args, **kwargs)
        real_run = optimiser.run

        def run():
            start = np.array(cost_obj.param_init, dtype=float)
            real_run()
            starts.append((start.size, start, np.array(optimiser.best_param_vals, dtype=float)))
            budgets.append(optimiser.optimiser_options.get("num_calls_to_function"))

        optimiser.run = run
        return optimiser

    monkeypatch.setattr(ParamID, "make_optimiser", spy)
    result = run_param_id(config)

    best = np.asarray(result["best_param_vals"], dtype=float)
    calibration, points = starts[0], starts[1:]
    assert calibration[0] == 2 and len(points) == 6
    assert budgets == [40] + [12] * 6
    # Per profiled entry: the first point starts at the best fit, each later one at the
    # optimum of a point already fitted (its neighbour nearer the best fit).
    for p, param_idx in enumerate((0, 1)):
        group = points[3 * p : 3 * p + 3]
        np.testing.assert_allclose(group[0][1], np.delete(best, param_idx))
        np.testing.assert_allclose(group[1][1], group[0][2])
        assert any(np.allclose(group[2][1], g[2]) for g in group[:2])
    assert result["param_id"].param_id.num_params == 2

    # The profile writes its own JSON and plots, and its points' optimisers write nothing, on
    # any rank.
    results_dir = os.path.join(temp_output_dir, "profile_likelihood")
    mpi_comm.Barrier()
    assert not os.path.exists(os.path.join(results_dir, "optimiser"))
    if rank == 0:
        written = sorted(os.listdir(results_dir))
        assert "summary.json" in written and "profile_likelihood_plots.pdf" in written
        assert len([f for f in written if f.startswith("param_") and f.endswith(".json")]) == 2
        assert not [f for f in written if f.startswith("best_") or f.startswith("multi_start")]


# ---- on real models ------------------------------------------------------------------------

_TEST_INPUTS = os.path.join(os.path.dirname(__file__), "test_inputs")
_SCALING_OBS = os.path.join(_TEST_INPUTS, "scaling_obs_data.json")
# k1..k5 in tests/test_inputs/scaling_modifier.cellml.
_SCALING_K = [0.7314, 0.1129, 0.4207, 0.2561, 0.8893]


@pytest.fixture(scope="module")
def scaling_casadi_model(tmp_path_factory):
    pytest.importorskip("casadi")
    from libcuflynx.generators.PythonGenerator import PythonGenerator

    out = str(tmp_path_factory.mktemp("pl_scaling_casadi"))
    return PythonGenerator(
        os.path.join(_TEST_INPUTS, "scaling_modifier.cellml"), output_dir=out, casadi_compat=True
    ).generate()


def _two_sub_obs_data(tmp_path):
    """The scaling obs data over two sub-experiments of 2 s, x1..x3 measured in the second.

    The second sub-experiment starts from the first one's end state, which on the CasADi AD
    path is an expression with every non-calibrated constant baked in.
    """
    with open(_SCALING_OBS) as f:
        obs = json.load(f)
    obs["protocol_info"]["sim_times"] = [[2.0, 2.0]]
    for item in obs["data_items"]:
        item["subexperiment_idx"] = 1 if item["operands"][0] in (
            "scaling/x1", "scaling/x2", "scaling/x3") else 0
    path = os.path.join(str(tmp_path), "scaling_two_sub_obs_data.json")
    with open(path, "w") as f:
        json.dump(obs, f)
    return path


def _scaling_engine(tmp_path, params, backend="myokit", casadi_model=None,
                    obs_path=_SCALING_OBS, **kwargs):
    """The inner ParamID of the five-decay scaling model, calibrating ``params``."""
    params_path = os.path.join(str(tmp_path), f"{backend}_params_for_id.json")
    write_params_for_id(params_path, {"version": 1, "params": params})
    out_dir = os.path.join(str(tmp_path), f"out_{backend}")
    os.makedirs(out_dir, exist_ok=True)
    if backend == "myokit":
        model_path = os.path.join(_TEST_INPUTS, "scaling_modifier.cellml")
        model_type, solver_info = "cellml", {"solver": "CVODE_myokit"}
    else:
        model_path, model_type = casadi_model, "casadi_python"
        solver_info = {"solver": "casadi_integrator", "method": "bdf", "max_step": 0.01,
                       "max_num_steps": 100000}
    outer = CVS0DParamID(
        model_path=model_path, model_type=model_type, file_name_prefix="scaling",
        params_for_id_path=params_path, param_id_obs_path=obs_path, sim_time=4.0,
        pre_time=0.0, dt=0.01, solver_info=solver_info, DEBUG=True,
        param_id_output_dir=out_dir, **kwargs)
    engine = outer.param_id
    engine.output_dir = out_dir
    return engine


@pytest.mark.integration
def test_a_group_and_a_modifier_are_held_whole_in_every_simulation(tmp_path, monkeypatch):
    engine = _scaling_engine(
        tmp_path,
        [
            {"targets": ["scaling/k1", "scaling/k2"], "min": 0.05, "max": 2.0},
            {"name": "mod", "modifies": ["scaling/k3", "scaling/k4"], "modifier": "scale",
             "min": 0.2, "max": 5.0},
            {"targets": ["scaling/k5"], "min": 0.05, "max": 2.0},
        ],
        param_id_method="CMA-ES",
        optimiser_options={"num_calls_to_function": 6},
    )
    helper = engine.sim_helper
    model, simulated = {}, []
    real_set, real_run = helper.set_param_vals, helper.run

    def set_param_vals(names, vals, *args, **kwargs):
        for entry, value in zip(names, vals):
            entry = entry if isinstance(entry, (list, tuple)) else [entry]
            value = value if isinstance(value, (list, tuple, np.ndarray)) else [value] * len(entry)
            model.update(zip(entry, value))
        return real_set(names, vals, *args, **kwargs)

    def run(*args, **kwargs):
        simulated.append(dict(model))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(helper, "set_param_vals", set_param_vals)
    monkeypatch.setattr(helper, "run", run)

    profile = ProfileLikelihood(engine, engine.param_id_info, str(tmp_path))
    profile.set_best_param_vals([0.5, 1.0, 0.8])
    for idx, value, expected in (
        (0, 0.3, {"scaling/k1": 0.3, "scaling/k2": 0.3}),
        (1, 2.0, {"scaling/k3": 2.0 * _SCALING_K[2], "scaling/k4": 2.0 * _SCALING_K[3]}),
    ):
        simulated.clear()
        profile._optimize_with_fixed_parameter(idx, value)
        assert len(simulated) > 3
        for values in simulated:
            for name, held in expected.items():
                assert values[name] == pytest.approx(held, rel=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["myokit", "casadi"])
def test_the_gradient_backends_see_each_held_value(backend, tmp_path, monkeypatch, request):
    """Cost and gradient at every profile point match a re-simulation at the held value.

    The calibration has already set up each backend over all entries (the CasADi graph, the
    FSA entry -> member map); a profile point must not be answered from that setup with a
    stale held value or with another entry's members. The reference is a second engine that
    never ran AD (on casadi_python the numeric cost is not available once it has).
    """
    casadi_model = request.getfixturevalue("scaling_casadi_model") if backend == "casadi" \
        else None
    params = [{"targets": [f"scaling/k{i}"], "min": 0.05, "max": 2.0} for i in (1, 2, 3)]
    obs_path = _two_sub_obs_data(tmp_path)
    engine = _scaling_engine(tmp_path, params, backend=backend, casadi_model=casadi_model,
                             obs_path=obs_path, param_id_method="sp_minimize", do_ad=True)
    os.makedirs(tmp_path / "reference", exist_ok=True)
    reference = _scaling_engine(tmp_path / "reference", params, backend=backend,
                                casadi_model=casadi_model, obs_path=obs_path,
                                param_id_method="sp_minimize")
    best = np.array(_SCALING_K[:3])
    engine.get_gradient(best)  # as the calibration's gradient-based fit did

    held = {}
    checks = []

    def reference_cost(x):
        return reference.get_cost_from_params(np.insert(x, 0, held["value"]))

    def check_instead_of_fitting(self):
        x = np.asarray(self.param_id_obj.param_init, dtype=float)
        h = 1e-6 * x
        fd = [(reference_cost(x + h[i] * e) - reference_cost(x - h[i] * e)) / (2 * h[i])
              for i, e in enumerate(np.eye(len(x)))]
        checks.append((self.param_id_obj.get_cost(x), reference_cost(x),
                       self.param_id_obj.get_gradient(x), fd))
        self.best_param_vals, self.best_cost = x, checks[-1][1]
        self.init_gradient = self.best_gradient = None

    monkeypatch.setattr(SciPyMinimizeOptimiser, "run", check_instead_of_fitting)
    profile = ProfileLikelihood(engine, engine.param_id_info, str(tmp_path))
    profile.set_best_param_vals(best)
    for value in (0.3, 1.5):
        held["value"] = value
        profile._optimize_with_fixed_parameter(0, value)

    assert len(checks) == 2
    for cost, expected, gradient, fd in checks:
        assert float(cost) == pytest.approx(expected, rel=1e-4)
        np.testing.assert_allclose(np.ravel(gradient), fd, rtol=1e-3, atol=1e-8)


@pytest.mark.integration
def test_profile_likelihood_runs_on_an_emulator(base_user_inputs, resources_dir, tmp_path):
    """The emulator is given the whole calibrated vector at every point, as in calibration."""
    from test_emulator_solver_helper import _config, _write_bundle

    config = _config(base_user_inputs, resources_dir, tmp_path, param_id_method="CMA-ES")
    _write_bundle(config)
    engine = CVS0DParamID.init_from_dict(config).param_id
    assert engine.emulates_features
    thetas = []
    real_set_theta = engine.sim_helper.set_theta

    def set_theta(theta):
        thetas.append(np.array(theta, dtype=float))
        return real_set_theta(theta)

    engine.sim_helper.set_theta = set_theta
    mins = np.asarray(engine.param_id_info["param_mins"], dtype=float)
    maxs = np.asarray(engine.param_id_info["param_maxs"], dtype=float)
    profile = ProfileLikelihood(
        engine, engine.param_id_info, str(tmp_path / "pl"), num_points=3, range_factor=0.1,
        optimiser_options={"num_calls_to_function": 8},
    )
    profile.set_best_param_vals(0.5 * (mins + maxs))
    profile.run()

    assert len(profile.profile_results) == 2
    for result in profile.profile_results.values():
        assert np.all(np.isfinite(result["costs"]))
    assert len(thetas) > 2 * 3 and all(theta.shape == (2,) for theta in thetas)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_profile_likelihood_3compartment_succeeds(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm
):
    """Calibrate, then profile both parameters through ``do_ia`` on every rank."""
    rank = mpi_comm.Get_rank()

    params_for_id_path = os.path.join(
        temp_output_dir, "3compartment_profile_likelihood_params_for_id.json"
    )
    if rank == 0:
        write_params_for_id(
            params_for_id_path,
            {
                "version": 1,
                "params": [
                    {
                        "targets": ["aortic_root/C"],
                        "min": 1e-9,
                        "max": 5e-8,
                        "param_type": "const",
                        "name_for_plotting": "C_{ao}",
                    },
                    {
                        "targets": ["global/q_lv_init"],
                        "min": 200e-6,
                        "max": 1500e-6,
                        "param_type": "const",
                        "name_for_plotting": "q_{sbv}",
                    },
                ],
            },
        )
    mpi_comm.Barrier()

    num_points = 3
    config = base_user_inputs.copy()
    config.update(
        {
            "file_prefix": "3compartment",
            "input_param_file": "3compartment_parameters.json",
            "params_for_id_file": params_for_id_path,
            "model_type": "cellml",
            "solver": "CVODE_myokit",
            "param_id_method": "genetic_algorithm",
            "pre_time": 20,
            "sim_time": 2,
            "dt": 0.01,
            "DEBUG": True,
            "plot_predictions": False,
            "do_uq": False,
            "do_ia": True,
            # Bounded: 2 parameters x 3 points, each a short GA re-fit.
            "ia_options": {
                "method": "profile_likelihood",
                "num_points": num_points,
                "range_factor": 0.1,
                "optimiser_options": {"num_calls_to_function": 40},
            },
            "solver_info": {"MaximumStep": 0.001},
            "param_id_obs_path": os.path.join(resources_dir, "3compartment_obs_data.json"),
            "param_id_output_dir": temp_output_dir,
            "generated_models_dir": temp_generated_models_dir,
            "debug_optimiser_options": {
                "num_calls_to_function": 60,
                "max_patience": 500,
                "cost_type": "gaussian_MLE",
            },
        }
    )

    _ensure_cellml_model_generated(config, mpi_comm)
    result = run_param_id(config)

    # The profiles leave the calibration as they found it, inner and outer.
    inner = result["param_id"].param_id
    expected_names = [["aortic_root/C"], ["global/q_lv_init"]]
    assert inner.param_id_info["param_names"] == expected_names
    assert result["param_id"].param_id_info["param_names"] == expected_names
    assert inner.num_params == 2
    assert inner.optimiser_options["num_calls_to_function"] == 60

    if rank == 0:
        results_dir = os.path.join(temp_output_dir, "profile_likelihood")
        with open(os.path.join(results_dir, "summary.json")) as f:
            summary = json.load(f)
        assert summary["num_params"] == 2
        assert summary["num_points"] == num_points
        mins = [1e-9, 200e-6]
        maxs = [5e-8, 1500e-6]
        for param_idx in range(2):
            json_files = [
                f
                for f in os.listdir(results_dir)
                if f.startswith(f"param_{param_idx}_") and f.endswith(".json")
            ]
            assert len(json_files) == 1
            with open(os.path.join(results_dir, json_files[0])) as f:
                profile = json.load(f)
            values = np.array(profile["param_values"])
            assert len(values) == num_points
            assert np.all(values >= mins[param_idx]) and np.all(values <= maxs[param_idx])
            assert np.all(np.isfinite(profile["costs"]))
            # gaussian_MLE is a likelihood, so an interval threshold is recorded.
            assert profile["ci_95"]["cost_threshold"] == pytest.approx(
                min(min(profile["costs"]), profile["best_cost"])
                + NLL_THRESHOLD_95 / profile["nll_per_cost"]
            )

    mpi_comm.Barrier()
