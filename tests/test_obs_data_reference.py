"""The optional ``data_reference`` of a data_item or prediction_item: where the datum comes from.

Spelled as the parameters-file column. libcuflynx accepts it as a string, never requires it,
and carries it to the results (``error_vec_data_references.npy``, ``validation_results.json``),
the plots that draw the item and the methods text. Requiring it is left to the module library's
own checks.
"""
import copy
import json
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from libcuflynx.param_id import prediction_plots, validation
from libcuflynx.param_id.plot_outputs import ParamIDPlotOutputs
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.reporting.methods import observables_text
from libcuflynx.utilities.obs_data_helpers import (ObsDataCreator, data_reference_line,
                                                   obs_item_references)

DATA_REF = "Smith2020, Table 2"
PRED_REF = "doi:10.1000/held-out"


def _doc(data_ref=None, pred_ref=None):
    data_item = {"data_item_name": "c0", "operands": ["main/c"], "data_type": "constant",
                 "unit": "dimensionless", "value": 1.0, "std": 0.1}
    pred_item = {"data_item_name": "v_end", "operands": ["main/v"], "unit": "m3",
                 "data_type": "constant", "value": 4.0, "std": 1.0}
    if data_ref is not None:
        data_item["data_reference"] = data_ref
    if pred_ref is not None:
        pred_item["data_reference"] = pred_ref
    return {"data_items": [data_item], "prediction_items": [pred_item],
            "protocol_info": {"pre_times": [0.0], "sim_times": [[2.0]]}}


def _parse(doc):
    return ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=copy.deepcopy(doc))


def _obs_info(parsed, tmp_path):
    return ObsAndParamDataParser().process_obs_info(parsed["gt_df"], str(tmp_path), 0.01)


@pytest.mark.unit
def test_a_data_item_and_a_prediction_item_carry_their_reference(tmp_path):
    parsed = _parse(_doc(DATA_REF, PRED_REF))
    assert obs_item_references(_obs_info(parsed, tmp_path)) == [DATA_REF]
    assert obs_item_references(parsed["prediction_info"]) == [PRED_REF]


@pytest.mark.unit
def test_the_reference_is_optional(tmp_path):
    parsed = _parse(_doc())
    assert obs_item_references(_obs_info(parsed, tmp_path)) == [None]
    assert obs_item_references(parsed["prediction_info"]) == [None]


@pytest.mark.unit
def test_items_may_mix_set_and_unset_references(tmp_path):
    doc = _doc(DATA_REF)
    doc["data_items"].append(dict(doc["data_items"][0], data_item_name="c1"))
    del doc["data_items"][1]["data_reference"]
    assert obs_item_references(_obs_info(_parse(doc), tmp_path)) == [DATA_REF, None]


@pytest.mark.unit
@pytest.mark.parametrize("where", ["data_items", "prediction_items"])
@pytest.mark.parametrize("bad", [3, ["Smith2020"], {"key": "Smith2020"}])
def test_a_reference_that_is_not_a_string_is_refused(where, bad):
    doc = _doc()
    doc[where][0]["data_reference"] = bad
    with pytest.raises(ValueError, match="data_reference"):
        _parse(doc)


@pytest.mark.unit
def test_the_schema_accepts_the_reference():
    jsonschema = pytest.importorskip("jsonschema")
    from libcuflynx.schemas import OBS_DATA_SCHEMA, load_schema
    jsonschema.validate(_doc(DATA_REF, PRED_REF), load_schema(OBS_DATA_SCHEMA))


@pytest.mark.unit
def test_the_builder_writes_the_reference(tmp_path):
    obs = ObsDataCreator()
    obs.add_protocol_info([0.0], [[2.0]], {})
    obs.add_data_item({"data_item_name": "c0", "operands": ["main/c"], "unit": "dimensionless",
                       "value": 1.0, "std": 0.1, "data_reference": DATA_REF})
    obs.add_prediction_item("main/v", "m3", 0, data_reference=PRED_REF)
    parsed = _parse(obs.get_obs_data_dict())
    assert obs_item_references(_obs_info(parsed, tmp_path)) == [DATA_REF]
    assert obs_item_references(parsed["prediction_info"]) == [PRED_REF]


@pytest.mark.unit
def test_a_prediction_items_reference_reaches_validation_results(tmp_path):
    pred = _parse(_doc(pred_ref=PRED_REF))["prediction_info"]
    t = np.linspace(0.0, 2.0, 21)
    results = validation.validation_results(pred, {0: t}, [t])
    path = validation.write_validation_results(results, str(tmp_path))
    with open(path) as f:
        assert json.load(f)["items"][0]["data_reference"] == PRED_REF

    unset = _parse(_doc())["prediction_info"]
    assert validation.validation_results(unset, {0: t}, [t])["items"][0]["data_reference"] \
        is None


class _Client:
    def __init__(self, out, obs_info):
        self.output_dir = str(out)
        self.obs_info = obs_info


@pytest.mark.unit
def test_a_data_items_reference_is_saved_beside_the_error_vectors(tmp_path):
    doc = _doc(DATA_REF)
    doc["data_items"].append(dict(doc["data_items"][0], data_item_name="c1"))
    del doc["data_items"][1]["data_reference"]
    plotter = ParamIDPlotOutputs.__new__(ParamIDPlotOutputs)
    plotter.client = _Client(tmp_path, _obs_info(_parse(doc), tmp_path))
    plotter.save_error_vectors(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    references = np.load(os.path.join(tmp_path, "error_vec_data_references.npy"))
    assert list(references) == [DATA_REF, ""]


@pytest.mark.unit
def test_a_reconstruction_page_names_the_sources_of_its_data(tmp_path, monkeypatch):
    from test_reconstruction_page_grouping import _run
    items = [("u_ar_max", "u_ar", 0), ("u_ar_min", "u_ar", 0), ("v_max", "v", 0)]
    pages, _, _ = _run(tmp_path, items, 1, monkeypatch,
                       data_references=[DATA_REF, DATA_REF, None])
    assert [page["source_line"] for page in pages] == [f"Data: {DATA_REF}", ""]


@pytest.mark.unit
def test_the_source_line_lists_each_reference_once():
    assert data_reference_line(["A", None, "B", "A", "  "]) == "Data: A; B"
    assert data_reference_line([None, ""]) is None


def _feature_obs():
    items = [{"data_item_name": f"x_e{e}", "operands": ["main/x"], "unit": "dimensionless",
              "experiment_idx": e, "item_name_for_plotting": "x_end", "operation": "max",
              "data_type": "constant", "value": float(e), "std": 0.1,
              "data_reference": f"Lab{e}"} for e in range(2)]
    return {"protocol_info": {"pre_times": [0.0, 0.0], "sim_times": [[1.0], [1.0]],
                              "params_to_change": {"main/a": [[1.0], [2.0]]}},
            "prediction_items": items,
            "prediction_plots": [{"name": "x_end vs a", "kind": "feature_vs_input",
                                  "x": {"params_to_change": "main/a", "subexperiment_idx": 0},
                                  "y": "x_end"}]}


@pytest.mark.unit
def test_a_feature_plot_names_the_sources_of_its_data(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    texts = []
    savefig = Figure.savefig

    def _capture(fig, *args, **kwargs):
        texts.extend(t.get_text() for t in fig.texts)
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _capture)
    parsed = _parse(_feature_obs())
    prediction_plots.plot_prediction_plots(
        parsed["prediction_plots"], parsed["gt_df"], parsed["prediction_info"],
        parsed["protocol_info"], {"x_e0": 0.0, "x_e1": 1.0}, str(tmp_path))
    assert texts == ["Data: Lab0; Lab1"]


@pytest.mark.unit
def test_the_methods_text_lists_the_references():
    class _Symbols:
        def variable(self, name):
            return name

    text, _ = observables_text(_doc(DATA_REF, PRED_REF), _Symbols(), "gaussian_MLE")
    assert f"Sources: {DATA_REF}; {PRED_REF}." in text
