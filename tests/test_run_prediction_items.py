"""Running an obs_data's prediction items without a calibration (issue #569).

``run_prediction_items`` parses the obs_data, applies its protocol and evaluates the items in one
executor pass; ``join_segments`` is the one join of sub-experiment results, used by
``ProtocolRunner.run_protocols`` too.
"""
import csv
import json

import numpy as np
import pytest

from libcuflynx.protocol_runners import ProtocolRunner, run_prediction_items
from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor

# dx/dt = u, y = u
_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="probe">
  <units name="per_second"><unit units="second" exponent="-1"/></units>
  <component name="comp">
    <variable name="t" units="second"/>
    <variable name="u" units="dimensionless" initial_value="0"/>
    <variable name="y" units="dimensionless"/>
    <variable name="x" units="dimensionless" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><ci>y</ci><ci>u</ci></apply>
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><times/><ci>u</ci><cn cellml:units="per_second">1</cn></apply></apply>
    </math>
  </component>
</model>"""

_DT = 0.1
_SOLVER_INFO = {"MaximumStep": 0.01}

# u = 0 for 1 s, then u = 2t for 1 s, so x = t^2 over sub-experiment 1 and x_max = 1.
_PROTOCOL = {"pre_times": [0.5], "sim_times": [[1.0, 1.0]],
             "params_to_change": {"comp/u": [[0.0, "ramp"]]},
             "protocol_traces": {"ramp": {"t": [0.0, 1.0], "values": [0.0, 2.0]}}}

_ITEMS = [
    {"data_item_name": "u_sub0", "operands": ["comp/y"], "unit": "dimensionless",
     "experiment_idx": 0, "subexperiment_idx": 0},
    {"data_item_name": "x_sub1", "operands": ["comp/x"], "unit": "dimensionless",
     "experiment_idx": 0},
    {"data_item_name": "x_max", "operands": ["comp/x"], "unit": "dimensionless",
     "experiment_idx": 0, "operation": "max", "data_type": "constant", "value": 1.2,
     "std": 0.1},
    {"data_item_name": "y_early", "operands": ["comp/y"], "unit": "dimensionless",
     "experiment_idx": 0, "operation": "mean_in_range",
     "operation_kwargs": {"start_frac": 0.0, "end_frac": 0.5}, "data_type": "constant"},
]


def _obs(items=_ITEMS):
    return {"protocol_info": json.loads(json.dumps(_PROTOCOL)), "data_items": [],
            "prediction_items": json.loads(json.dumps(items))}


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "probe.cellml"
    path.write_text(_CELLML)
    return str(path)


class _ScalarHelper:
    """Steps like a solver helper, but every result is a scalar."""
    dt = _DT

    def update_times(self, dt, start_time, sim_time, pre_time):
        self.tSim = np.linspace(start_time + pre_time, start_time + pre_time + sim_time,
                                int(sim_time / dt) + 1)

    def set_param_vals(self, *args, **kwargs):
        pass

    def reset_states(self):
        pass

    def reset_and_clear(self):
        pass

    def run(self):
        return True

    def get_all_results(self, flatten=False):
        return [2.5]


@pytest.mark.unit
def test_join_pads_scalars_to_their_own_segment():
    """dt = 0.1, sub-experiments of 1.0 and 0.3 s: 11 + 3 - 1 samples, as t has."""
    runner = object.__new__(ProtocolRunner)
    runner.inp_data_dict, runner.dt, runner.MaximumStep = {}, _DT, 0.01
    runner.sim_helper = _ScalarHelper()
    runner._executor = ProtocolExecutor(runner.sim_helper)
    t, res, _ = runner.run_protocols(None, {"pre_times": [0.0], "sim_times": [[1.0, 0.3]],
                                            "params_to_change": {}})
    assert len(t[0]) == 13
    assert len(res[0][0]) == len(t[0])
    assert np.all(np.asarray(res[0][0]) == 2.5)


@pytest.mark.integration
def test_run_protocols_result_variables_and_segments(model):
    runner = ProtocolRunner(model, inp_data_dict={"dt": _DT, "solver_info": _SOLVER_INFO},
                            solver="CVODE_myokit")
    t, res, _, segments = runner.run_protocols(protocol_info=_obs()["protocol_info"],
                                               result_variables=["comp/x", "comp/y"],
                                               return_segments=True)
    x0, x1 = (np.ravel(segments[(0, sub)][0][0]) for sub in (0, 1))
    assert len(res[0]) == 2 and len(res[0][0]) == len(t[0]) == 21
    np.testing.assert_array_equal(res[0][0], np.concatenate([x0, x1[1:]]))
    assert res[0][0][-1] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.integration
def test_run_prediction_items_features_equal_hand_sliced_segment(model):
    from libcuflynx.param_id.operation_funcs import get_operation_funcs_dict_for_mode

    result = run_prediction_items(model, _obs(), _DT, _SOLVER_INFO)
    funcs = get_operation_funcs_dict_for_mode("numpy")
    sub1 = result.segments[(0, 1)]
    assert result.features["x_max"] == float(np.max(sub1["comp/x"]))
    assert result.features["x_max"] == pytest.approx(1.0, abs=1e-3)
    assert result.features["y_early"] == float(
        funcs["mean_in_range"](sub1["comp/y"], start_frac=0.0, end_frac=0.5))

    # traces over their own segment, time from the end of the warm-up
    t0, u0 = result.traces["u_sub0"]
    t1, x1 = result.traces["x_sub1"]
    assert t0[0] == pytest.approx(0.0) and t0[-1] == pytest.approx(1.0)
    assert t1[0] == pytest.approx(1.0) and t1[-1] == pytest.approx(2.0)
    assert np.all(u0 == 0.0)
    np.testing.assert_allclose(x1, (t1 - 1.0) ** 2, atol=1e-3)
    np.testing.assert_allclose(result.t_by_exp[0], np.linspace(0.0, 2.0, 21))


@pytest.mark.integration
def test_run_prediction_items_validation_for_items_with_values(model):
    result = run_prediction_items(model, _obs(), _DT, _SOLVER_INFO)
    items = result.validation["items"]
    assert [item["data_item_name"] for item in items] == ["x_max"]
    assert items[0]["model"][0] == result.features["x_max"]
    assert items[0]["rmse"] == pytest.approx(abs(result.features["x_max"] - 1.2))


@pytest.mark.integration
def test_run_prediction_items_unresolved_names_reported_together(model):
    items = [dict(_ITEMS[0], operands=["comp/nope_a"]),
             dict(_ITEMS[1], operands=["comp/nope_b"]), _ITEMS[2]]
    with pytest.raises(ValueError, match="comp/nope_a.*comp/nope_b"):
        run_prediction_items(model, _obs(items), _DT, _SOLVER_INFO)


@pytest.mark.integration
def test_run_predictions_script_writes_outputs(model, tmp_path):
    from libcuflynx.scripts.run_predictions_script import main

    obs_path = tmp_path / "obs_data.json"
    obs_path.write_text(json.dumps(_obs()))
    out = tmp_path / "out"
    assert main([model, str(obs_path), "--dt", str(_DT),
                 "--solver-info", json.dumps(_SOLVER_INFO), "--output-dir", str(out)]) == 0

    traces = np.load(out / "traces.npz")
    assert sorted(traces.files) == ["u_sub0", "x_sub1"]
    assert traces["x_sub1"].shape == (2, 11)
    with open(out / "features.csv") as fh:
        rows = {row["data_item_name"]: float(row["value"]) for row in csv.DictReader(fh)}
    assert sorted(rows) == ["x_max", "y_early"]
    assert rows["x_max"] == pytest.approx(1.0, abs=1e-3)
    with open(out / "validation_results.json") as fh:
        assert [i["data_item_name"] for i in json.load(fh)["items"]] == ["x_max"]
