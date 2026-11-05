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


def _set_trace(obs):
    """Experiment 2 sets main/a to the trace "ramp" in sub-experiment 1 (1 s long)."""
    obs["protocol_info"]["params_to_change"]["main/a"][2][1] = "ramp"
    obs["protocol_info"]["protocol_traces"] = {
        "ramp": {"t": [0.0, 0.5, 1.0], "values": [0.0, 3.0, 1.0]}}


def _vs_trace(reduce=None, **extra):
    x = {"params_to_change": "main/a", "subexperiment_idx": 1}
    if reduce is not None:
        x["reduce"] = reduce
    return dict({"name": "x_end vs a", "kind": "feature_vs_input", "x": x, "y": "x_end"},
                **extra)


_BAD = {
    "rule1": ([dict(_VS_INPUT), dict(_VS_INPUT)], "names are unique"),
    "rule2": ([dict(_VS_INPUT, kind="feature_vs_feature")], "names its x group as a string"),
    "rule3": ([dict(_VS_FEATURE, y="no_such_group")], "no item has item_name_for_plotting"),
    "rule4": ("drop_y_end_e3", "every point needs both"),
    "rule5": ([dict(_VS_INPUT, x={"params_to_change": "main/a", "subexperiment_idx": 2})],
              "out of range"),
    "x_key_not_in_protocol": (
        [dict(_VS_INPUT, x={"params_to_change": "main/c", "subexperiment_idx": 1})],
        "'main/c' is not a params_to_change key"),
    "series_key_not_in_protocol": (
        [dict(_VS_INPUT, series={"params_to_change": "main/c", "subexperiment_idx": 1})],
        "'main/c' is not a params_to_change key"),
    "reduce_at_after_the_sub_experiment": ("trace_at_2s", "after the end of sub-experiment 1"),
}


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(_BAD))
def test_bad_plot_rejected(case):
    plots, message = _BAD[case]
    obs = _obs([_VS_FEATURE])
    if plots == "drop_y_end_e3":
        obs["prediction_items"] = [i for i in obs["prediction_items"]
                                   if i["data_item_name"] != "y_end_e3"]
    elif plots == "trace_at_2s":
        _set_trace(obs)
        obs["prediction_plots"] = [_vs_trace({"at": 2.0})]
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
    """A calibrated data_item group: the fit from the cost path, the data as error bars; and a
    time course at the best fit."""
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
                                "y": "x_SS"},
                               {"name": "x over time", "kind": "input_and_output",
                                "input": {"params_to_change": "benchmark/x_init"},
                                "output": "benchmark/x", "experiments": [1]}]
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
    assert os.path.getsize(os.path.join(pid.output_dir, "x_over_time_exp_1.png")) > 0

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


# --- reduce: a trace input made one number --------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("reduce, expected", [
    ("final", 3.0), ("min", 1.0), ("max", 3.0),
    # held at 1 until 0.2 s, linear to 3 at 0.6 s, held: 0.2 + 0.8 + 1.2
    ("mean", 2.2), ({"at": 0.4}, 2.0), ({"at": 0.0}, 1.0)])
def test_reduce_trace_over_the_sub_experiment(reduce, expected):
    assert prediction_plots.reduce_trace([0.2, 0.6], [1.0, 3.0], 1.0, reduce) == \
        pytest.approx(expected)


@pytest.mark.unit
def test_reduce_trace_stops_at_the_end_of_the_sub_experiment():
    """A trace longer than its sub-experiment is cut where the sub-experiment ends."""
    assert prediction_plots.reduce_trace([0.0, 2.0], [0.0, 4.0], 1.0, "final") == 2.0
    assert prediction_plots.reduce_trace([0.0, 2.0], [0.0, 4.0], 1.0, "max") == 2.0
    assert prediction_plots.reduce_trace([0.0, 2.0], [0.0, 4.0], 1.0, "mean") == 1.0


def _points(obs, plot):
    parsed = _parse(obs)
    groups = prediction_plots.item_groups(parsed["gt_df"], parsed["prediction_info"])
    values = {f"x_end_e{e}": 10.0 + e for e in range(4)}
    return prediction_plots.plot_points(plot, 0, groups, parsed["protocol_info"], values,
                                        prediction_info=parsed["prediction_info"])


@pytest.mark.unit
@pytest.mark.parametrize("reduce, expected", [
    ("final", 1.0), ("mean", 1.75), ("max", 3.0), ("min", 0.0), ({"at": 0.25}, 1.5)])
def test_a_trace_input_with_reduce_is_one_point(reduce, expected):
    obs = _obs([_vs_trace(reduce)])
    _set_trace(obs)
    points = {p["experiment_idx"]: p["x"] for p in _points(obs, _vs_trace(reduce))}
    assert points == {0: 1.0, 1: 2.0, 2: pytest.approx(expected), 3: 2.0}


@pytest.mark.unit
def test_reduce_leaves_a_number_as_it_is():
    obs = _obs([_vs_trace("max")])
    points = {p["experiment_idx"]: p["x"] for p in _points(obs, _vs_trace("max"))}
    assert points == {0: 1.0, 1: 2.0, 2: 1.0, 3: 2.0}


@pytest.mark.unit
def test_a_trace_input_without_reduce_parses_and_its_point_is_skipped():
    obs = _obs([_vs_trace()])
    _set_trace(obs)
    with pytest.warns(prediction_plots.PredictionPlotWarning,
                      match=r"experiment 2 has no point.*'ramp'.*\"reduce\""):
        points = _points(obs, _vs_trace())
    assert sorted(p["experiment_idx"] for p in points) == [0, 1, 3]


@pytest.mark.unit
def test_a_series_trace_without_reduce_skips_the_point():
    plot = dict(_VS_INPUT, x={"params_to_change": "main/b", "subexperiment_idx": 1},
                series={"params_to_change": "main/a", "subexperiment_idx": 1})
    obs = _obs([plot])
    _set_trace(obs)
    with pytest.warns(prediction_plots.PredictionPlotWarning,
                      match="experiment 2 has no point: its series input"):
        points = _points(obs, plot)
    assert sorted(p["experiment_idx"] for p in points) == [0, 1, 3]


@pytest.mark.unit
@pytest.mark.parametrize("reduce", ["median", {"at": -1.0}, {"at": 0.5, "of": 1}, {"t": 1.0}])
def test_a_malformed_reduce_is_refused(reduce):
    obs = _obs([_vs_trace(reduce)])
    _set_trace(obs)
    with pytest.raises(ValueError, match="reduce"):
        _parse(obs)
    # and without the schema, by check_prediction_plots itself
    parsed_obs = _obs([])
    _set_trace(parsed_obs)
    parsed = _parse(parsed_obs)
    with pytest.raises(prediction_plots.PredictionPlotError, match="reduce"):
        prediction_plots.check_prediction_plots(
            [_vs_trace(reduce)], parsed["gt_df"], parsed["prediction_info"],
            parsed["protocol_info"])


# --- the schema, kind by kind -----------------------------------------------------------------

_IO = {"name": "a and x", "kind": "input_and_output", "input": {"params_to_change": "main/a"},
       "output": "main/x"}
_OVI = {"name": "x vs a", "kind": "output_vs_input", "input": {"params_to_change": "main/a"},
        "output": "main/x", "experiments": [0, 2]}

_SCHEMA_REFUSES = {
    "feature_vs_input without y": dict(_VS_INPUT, y=None),
    "feature_vs_feature without x": {"name": "n", "kind": "feature_vs_feature", "y": "x_end"},
    "feature_vs_input with output": dict(_VS_INPUT, output="main/x"),
    "feature_vs_feature with experiments": dict(_VS_FEATURE, experiments=[0]),
    "input_and_output without output": {k: v for k, v in _IO.items() if k != "output"},
    "input_and_output without input": {k: v for k, v in _IO.items() if k != "input"},
    "input_and_output with x": dict(_IO, x="y_end"),
    "output_vs_input with series": dict(_OVI, series=None),
    "output_vs_input with y": dict(_OVI, y="x_end"),
    "input with subexperiment_idx": dict(_IO, input={"params_to_change": "main/a",
                                                     "subexperiment_idx": 1}),
    "empty experiments": dict(_OVI, experiments=[]),
    "negative experiment": dict(_OVI, experiments=[-1]),
    "empty output": dict(_IO, output=" "),
    "unknown kind": dict(_IO, kind="output_vs_time"),
    "reduce not a mode": _vs_trace("median"),
}


def _validator():
    jsonschema = pytest.importorskip("jsonschema")
    from libcuflynx.schemas import OBS_DATA_SCHEMA, load_schema
    return jsonschema, load_schema(OBS_DATA_SCHEMA)


@pytest.mark.unit
def test_schema_accepts_every_kind():
    jsonschema, schema = _validator()
    plots = [_VS_INPUT, _VS_FEATURE, _IO, _OVI, _vs_trace("mean"), _vs_trace({"at": 0.5})]
    jsonschema.validate(_obs([dict(p, name=str(i)) for i, p in enumerate(plots)]), schema)


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(_SCHEMA_REFUSES))
def test_schema_refuses(case):
    jsonschema, schema = _validator()
    plot = {k: v for k, v in _SCHEMA_REFUSES[case].items() if v is not None or k == "series"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_obs([plot]), schema)


# --- time-course kinds: parse-time rules ------------------------------------------------------

_TIME_COURSE_BAD = {
    "input key not in params_to_change": (dict(_IO, input={"params_to_change": "main/c"}),
                                          "'main/c' is not a params_to_change key"),
    "experiment out of range": (dict(_OVI, experiments=[0, 4]),
                                r"experiments \[4\] are out of range"),
    "output with an empty level": (dict(_IO, output="SN//V"), "is not a variable name"),
    "output ending in /": (dict(_IO, output="main/"), "is not a variable name"),
    "duplicate name": ([_IO, dict(_OVI, name=_IO["name"])], "names are unique"),
}


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(_TIME_COURSE_BAD))
def test_time_course_plot_refused_at_parse(case):
    plots, message = _TIME_COURSE_BAD[case]
    obs = _obs(plots if isinstance(plots, list) else [plots])
    with pytest.raises(prediction_plots.PredictionPlotError, match=message):
        _parse(obs)


@pytest.mark.unit
def test_time_course_plots_parse():
    parsed = _parse(_obs([_IO, _OVI]))
    assert [p["kind"] for p in parsed["prediction_plots"]] == ["input_and_output",
                                                                "output_vs_input"]


@pytest.mark.unit
def test_a_time_course_plot_needs_the_recorded_time_courses(tmp_path):
    parsed = _parse(_obs([_IO]))
    with pytest.raises(prediction_plots.PredictionPlotError, match="record_time_courses"):
        prediction_plots.plot_prediction_plots(
            parsed["prediction_plots"], parsed["gt_df"], parsed["prediction_info"],
            parsed["protocol_info"], {}, str(tmp_path))


# --- time-course kinds: drawn from a model run ------------------------------------------------

# A nested vessel SN/soma/membrane: dV/dt = I, J = I; and a global constant g.
_NESTED_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="nested">
  <units name="per_second"><unit units="second" exponent="-1"/></units>
  <component name="SN__soma__membrane">
    <variable name="t" units="second"/>
    <variable name="I" units="dimensionless" initial_value="0"/>
    <variable name="J" units="dimensionless"/>
    <variable name="V" units="dimensionless" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><ci>J</ci><ci>I</ci></apply>
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>V</ci></apply>
        <apply><times/><ci>I</ci><cn cellml:units="per_second">1</cn></apply></apply>
    </math>
  </component>
  <component name="parameters_global">
    <variable name="g" units="dimensionless" initial_value="2"/>
  </component>
</model>"""

_I = "SN/soma/membrane/I"
_V = "SN/soma/membrane/V"


def _nested_obs(plots):
    """Experiment 0: I = 0, then the ramp 0 -> 2 over 1 s (V ends at 1). Experiment 1: I = 1,
    then 3 (V ends at 4), with V's data as a series in its sub-experiment 1."""
    return {
        "protocol_info": {
            "pre_times": [0.0, 0.0], "sim_times": [[1.0, 1.0], [1.0, 1.0]],
            "params_to_change": {_I: [[0.0, "ramp"], [1.0, 3.0]]},
            "protocol_traces": {"ramp": {"t": [0.0, 1.0], "values": [0.0, 2.0]}}},
        "data_items": [{"data_item_name": "V_trace", "operands": [_V], "unit": "dimensionless",
                        "experiment_idx": 1, "subexperiment_idx": 1, "data_type": "series",
                        "value": [1.0, 1.3, 1.6], "std": 0.05, "obs_dt": 0.1}],
        "prediction_items": [{"data_item_name": "V_max", "operands": [_V],
                              "unit": "dimensionless", "experiment_idx": 0, "operation": "max",
                              "data_type": "constant"}],
        "prediction_plots": plots,
    }


_NESTED_IO = {"name": "Membrane", "kind": "input_and_output", "input": {"params_to_change": _I},
              "output": _V}
_NESTED_OVI = {"name": "V vs I", "kind": "output_vs_input", "input": {"params_to_change": _I},
               "output": _V, "experiments": [1]}


@pytest.fixture
def nested_model(tmp_path):
    path = tmp_path / "nested.cellml"
    path.write_text(_NESTED_CELLML)
    return str(path)


def _record(model, plots):
    from libcuflynx.protocol_runners import ProtocolRunner
    parsed = _parse(_nested_obs(plots))
    runner = ProtocolRunner(model, inp_data_dict={"dt": 0.1,
                                                  "solver_info": {"MaximumStep": 0.01}})
    runner._apply_protocol_info(parsed["protocol_info"])
    courses = prediction_plots.record_time_courses(runner._executor, parsed["protocol_info"],
                                                   parsed["prediction_plots"])
    return parsed, courses


@pytest.mark.integration
def test_record_time_courses_resolves_nested_flat_and_global_outputs(nested_model):
    flat = "SN__soma__membrane/V"
    plots = [_NESTED_IO, dict(_NESTED_OVI, name="flat", output=flat),
             dict(_NESTED_OVI, name="algebraic", output="SN/soma/membrane/J"),
             dict(_NESTED_OVI, name="global", output="g")]
    parsed, courses = _record(nested_model, plots)
    assert sorted(courses) == sorted([_V, flat, "SN/soma/membrane/J", "g"])
    assert sorted(courses[_V]) == [0, 1]
    t, v = courses[_V][0]
    np.testing.assert_allclose(t, np.linspace(0.0, 2.0, 21), atol=1e-9)
    np.testing.assert_allclose(v[[0, 10, 15, 20]], [0.0, 0.0, 0.25, 1.0], atol=1e-3)
    np.testing.assert_allclose(courses[_V][1][1][[10, 20]], [1.0, 4.0], atol=1e-3)
    np.testing.assert_allclose(courses[flat][1][1], courses[_V][1][1])
    np.testing.assert_allclose(courses["SN/soma/membrane/J"][1][1][[5, 15]], [1.0, 3.0])
    np.testing.assert_allclose(courses["g"][1][1], np.full(21, 2.0))

    # the input at the same times; a boundary belongs to the sub-experiment it ends
    protocol_info = parsed["protocol_info"]
    np.testing.assert_allclose(
        prediction_plots.input_at(protocol_info, _I, 0, [0.0, 1.0, 1.5, 2.0]),
        [0.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        prediction_plots.input_at(protocol_info, _I, 1, [0.5, 1.0, 1.1]), [1.0, 1.0, 3.0])
    t_in, v_in = prediction_plots.input_time_course(protocol_info, _I, 1)
    assert list(t_in) == [0.0, 1.0, 1.0, 2.0] and list(v_in) == [1.0, 1.0, 3.0, 3.0]


@pytest.mark.integration
def test_an_output_the_model_does_not_have_is_refused_before_running(nested_model):
    with pytest.raises(prediction_plots.PredictionPlotError,
                       match=r"model does not have: 'SN/soma/membrane/W' \(plot 'V vs I'\)"):
        _record(nested_model, [dict(_NESTED_OVI, output="SN/soma/membrane/W")])


@pytest.mark.integration
def test_input_and_output_and_output_vs_input_draw(nested_model, tmp_path):
    result = run_prediction_items(nested_model, _nested_obs([_NESTED_IO, _NESTED_OVI]), 0.1,
                                  {"MaximumStep": 0.01}, plot_dir=str(tmp_path / "plots"))
    assert [os.path.basename(p) for p in result.plots] == [
        "Membrane_exp_0.png", "Membrane_exp_1.png", "V_vs_I.png"]
    assert all(os.path.getsize(p) > 0 for p in result.plots)

    # the data overlaid on experiment 1: V as a series, from the start of sub-experiment 1
    parsed = _parse(_nested_obs([_NESTED_IO]))
    [(name, t, value, std)] = prediction_plots.series_data(parsed["gt_df"], _V, 1,
                                                           parsed["protocol_info"])
    assert name == "V_trace"
    np.testing.assert_allclose(t, [1.0, 1.1, 1.2])
    np.testing.assert_allclose(std, [0.05] * 3)
    assert prediction_plots.series_data(parsed["gt_df"], _V, 0, parsed["protocol_info"]) == []
