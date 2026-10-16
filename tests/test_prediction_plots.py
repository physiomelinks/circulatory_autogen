"""``prediction_plots``: a feature against a feature or an input, across experiments (#568).

The structural rules run when obs_data is parsed; drawing checks that each item is a scalar
feature and takes model values from ``run_prediction_items`` (prediction items) or the
calibration's cost path (data_items, in ``save_prediction_data``).
"""
import copy
import json
import os

import numpy as np
import pytest

from libcuflynx.param_id import prediction_plots
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.protocol_runners import run_prediction_items
from test_prediction_features import mpi_comm  # noqa: F401  (fixture)

# dx/dt = a, dy/dt = b
_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="two_inputs">
  <component name="main">
    <variable name="t" units="second"/>
    <variable name="a" units="dimensionless" initial_value="0"/>
    <variable name="b" units="dimensionless" initial_value="0"/>
    <variable name="x" units="dimensionless" initial_value="0"/>
    <variable name="y" units="dimensionless" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><divide/><ci>a</ci><cn cellml:units="second">1</cn></apply></apply>
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>y</ci></apply>
        <apply><divide/><ci>b</ci><cn cellml:units="second">1</cn></apply></apply>
    </math>
  </component>
</model>"""

# Four experiments: a over two values, b over two values. Sub-experiment 0 holds a and b at 0;
# sub-experiment 1 (1 s) sets them, so x_end = a and y_end = b.
_A, _B = [1.0, 2.0, 1.0, 2.0], [3.0, 3.0, 5.0, 5.0]


def _item(name, operand, exp, label, **extra):
    return dict({"data_item_name": f"{name}_e{exp}", "operands": [operand],
                 "unit": "dimensionless", "experiment_idx": exp, "subexperiment_idx": 1,
                 "item_name_for_plotting": label, "operation": "max",
                 "data_type": "constant"}, **extra)


def _obs(plots):
    return {
        "protocol_info": {
            "pre_times": [0.0] * 4, "sim_times": [[0.5, 1.0]] * 4,
            "params_to_change": {"main/a": [[0.0, a] for a in _A],
                                 "main/b": [[0.0, b] for b in _B]}},
        "data_items": [],
        "prediction_items": (
            [_item("x_end", "main/x", e, "x_end", value=_A[e] + 0.1, std=0.05)
             for e in range(4)]
            + [_item("y_end", "main/y", e, "y_end") for e in range(4)]),
        "prediction_plots": plots,
    }


_VS_INPUT = {"name": "x_end vs a", "kind": "feature_vs_input",
             "x": {"params_to_change": "main/a", "subexperiment_idx": 1}, "y": "x_end",
             "series": {"params_to_change": "main/b", "subexperiment_idx": 1}}
_VS_FEATURE = {"name": "x_end vs y_end", "kind": "feature_vs_feature", "x": "y_end",
               "y": "x_end", "series": None}


def _parse(obs):
    return ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=copy.deepcopy(obs))


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "two_inputs.cellml"
    path.write_text(_CELLML)
    return str(path)


_BAD = {
    "rule1": ([dict(_VS_INPUT), dict(_VS_INPUT)], "names are unique"),
    "rule2": ([dict(_VS_INPUT, kind="feature_vs_feature")], "names its x group as a string"),
    "rule3": ([dict(_VS_FEATURE, y="no_such_group")], "no item has item_name_for_plotting"),
    "rule4": ("drop_y_end_e3", "every point needs both"),
    "rule5": ([dict(_VS_INPUT, x={"params_to_change": "main/a", "subexperiment_idx": 2})],
              "out of range"),
    "rule5_trace": ("trace_input", "not a number"),
}


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(_BAD))
def test_bad_plot_rejected(case):
    plots, message = _BAD[case]
    obs = _obs([_VS_FEATURE])
    if plots == "drop_y_end_e3":
        obs["prediction_items"] = [i for i in obs["prediction_items"]
                                   if i["data_item_name"] != "y_end_e3"]
    elif plots == "trace_input":
        obs["protocol_info"]["params_to_change"]["main/a"][2][1] = "ramp"
        obs["protocol_info"]["protocol_traces"] = {"ramp": {"t": [0, 1], "values": [0, 1]}}
        obs["prediction_plots"] = [_VS_INPUT]
    else:
        obs["prediction_plots"] = plots
    with pytest.raises(prediction_plots.PredictionPlotError, match=message):
        _parse(obs)


@pytest.mark.unit
def test_a_plot_with_an_unknown_key_is_refused_by_the_schema():
    obs = _obs([_VS_FEATURE])
    obs["prediction_plots"] = [dict(_VS_INPUT, colour="red")]
    with pytest.raises(ValueError, match="'colour' was unexpected"):
        _parse(obs)


@pytest.mark.unit
def test_a_group_with_two_items_in_one_experiment_is_rejected():
    obs = _obs([_VS_FEATURE])
    obs["prediction_items"].append(_item("x_again", "main/x", 0, "x_end"))
    with pytest.raises(prediction_plots.PredictionPlotError, match="two items for experiment 0"):
        _parse(obs)


@pytest.mark.unit
def test_default_is_empty_list():
    obs = _obs(None)
    del obs["prediction_plots"]
    parsed = ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=obs)
    assert parsed["prediction_plots"] == []


@pytest.mark.integration
def test_feature_vs_input_with_series_draws(model, tmp_path):
    result = run_prediction_items(model, _obs([_VS_INPUT]), 0.1, {"MaximumStep": 0.01},
                                  plot_dir=str(tmp_path / "plots"))
    assert [os.path.basename(p) for p in result.plots] == ["x_end_vs_a.png"]
    assert os.path.getsize(result.plots[0]) > 0

    parsed = _parse(_obs([_VS_INPUT]))
    points = prediction_plots.plot_points(
        _VS_INPUT, 0, prediction_plots.item_groups(parsed["gt_df"], parsed["prediction_info"]),
        parsed["protocol_info"], result.features, prediction_info=parsed["prediction_info"])
    assert [(p["series"], p["x"]) for p in points] == [(3.0, 1.0), (3.0, 2.0), (5.0, 1.0),
                                                       (5.0, 2.0)]
    np.testing.assert_allclose([p["y"] for p in points], [1.0, 2.0, 1.0, 2.0], atol=1e-3)
    assert [p["data"] for p in points] == [1.1, 2.1, 1.1, 2.1]


@pytest.mark.integration
def test_feature_vs_feature_draws(model, tmp_path):
    result = run_prediction_items(model, _obs([_VS_FEATURE]), 0.1, {"MaximumStep": 0.01},
                                  plot_dir=str(tmp_path / "plots"))
    assert [os.path.basename(p) for p in result.plots] == ["x_end_vs_y_end.png"]
    parsed = _parse(_obs([_VS_FEATURE]))
    points = prediction_plots.plot_points(
        _VS_FEATURE, 0, prediction_plots.item_groups(parsed["gt_df"], parsed["prediction_info"]),
        parsed["protocol_info"], result.features, prediction_info=parsed["prediction_info"])
    # sorted by x (y_end = b), one point per experiment
    np.testing.assert_allclose([p["x"] for p in points], [3.0, 3.0, 5.0, 5.0], atol=1e-3)
    assert sorted(p["experiment_idx"] for p in points) == [0, 1, 2, 3]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_data_item_group_shows_fit_and_data(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path):
    """A calibrated data_item group: the fit from the cost path, the data as error bars."""
    from libcuflynx.param_id.paramID import CVS0DParamID
    from test_prediction_features import _benchmark_config, _generate, BENCHMARK_OBS

    with open(os.path.join(resources_dir, BENCHMARK_OBS)) as f:
        doc = json.load(f)
    x_item = doc["data_items"][0]
    doc["protocol_info"] = {"pre_times": [0.0, 0.0], "sim_times": [[8.0], [8.0]],
                            "params_to_change": {"benchmark/x_init": [[0.0], [2.0]]}}
    doc["data_items"] = [
        dict(x_item, data_item_name="x_ss_e0", item_name_for_plotting="x_SS",
             experiment_idx=0, value=1.0),
        dict(x_item, data_item_name="x_ss_e1", item_name_for_plotting="x_SS",
             experiment_idx=1, value=1.2)]
    doc["prediction_plots"] = [{"name": "x_SS vs x_init", "kind": "feature_vs_input",
                                "x": {"params_to_change": "benchmark/x_init",
                                      "subexperiment_idx": 0},
                                "y": "x_SS"}]
    obs_path = tmp_path / "plot_obs_data.json"
    obs_path.write_text(json.dumps(doc))
    config = _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                               temp_generated_models_dir, str(obs_path))
    _generate(config, mpi_comm)
    if mpi_comm.Get_rank() != 0:
        return
    pid = CVS0DParamID.init_from_dict(config)
    pid.set_best_param_vals(np.array([0.5, 1.0]))
    pid.save_prediction_data()
    assert os.path.getsize(os.path.join(pid.output_dir, "x_SS_vs_x_init.png")) > 0

    values = pid._data_item_feature_values(pid.prediction_plots)
    assert values == {"x_ss_e0": pytest.approx(0.5, abs=0.02),
                      "x_ss_e1": pytest.approx(0.5, abs=0.02)}
    points = prediction_plots.plot_points(
        pid.prediction_plots[0], 0, prediction_plots.item_groups(pid.gt_df, pid.prediction_info),
        pid.protocol_info, values, prediction_info=pid.prediction_info)
    assert [(p["x"], p["data"], p["std"]) for p in points] == [(0.0, 1.0, 0.1), (2.0, 1.2, 0.1)]


@pytest.mark.unit
def test_schema_accepts_example():
    jsonschema = pytest.importorskip("jsonschema")
    from libcuflynx.schemas import OBS_DATA_SCHEMA, load_schema

    schema = load_schema(OBS_DATA_SCHEMA)
    jsonschema.validate(_obs([_VS_INPUT, _VS_FEATURE]), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_obs([dict(_VS_INPUT, kind="feature_vs_time")]), schema)
