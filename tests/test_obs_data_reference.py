"""The optional ``data_reference`` of a data_item or prediction_item: where the datum comes from.

Spelled as the parameters-file column. libcuflynx accepts it as a string, never requires it,
and carries it to the results (``error_vec_data_references.npy``, ``validation_results.json``),
the plots that draw the item and the methods text. Requiring it is left to the module library's
own checks. An item's ``source`` is read as its ``data_reference``, with a warning.
"""
import copy
import json
import os
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from libcuflynx.param_id import prediction_plots, validation
from libcuflynx.param_id.plot_outputs import ParamIDPlotOutputs
from libcuflynx.parsers import PrimitiveParsers
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.reporting.methods import _load_obs_data, _obs_cited_keys, observables_text
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


def _source_warnings(caught):
    return [w for w in caught if "'source'" in str(w.message)]


@pytest.mark.unit
@pytest.mark.parametrize("data,pred", [("Smith2020 lab notes", "Jones2019 notes"), (None, None)])
def test_source_is_read_as_the_data_reference_on_both_item_kinds(tmp_path, data, pred):
    doc = _doc()
    doc["data_items"][0]["source"] = data
    doc["prediction_items"][0]["source"] = pred
    with pytest.warns(UserWarning, match="has been replaced by 'data_reference'"):
        parsed = _parse(doc)
    assert obs_item_references(_obs_info(parsed, tmp_path)) == [data]
    assert obs_item_references(parsed["prediction_info"]) == [pred]
    assert "source" not in parsed["gt_df"].columns


@pytest.mark.unit
@pytest.mark.parametrize("where,ref", [("data_items", DATA_REF), ("prediction_items", PRED_REF)])
def test_the_data_reference_wins_over_source(tmp_path, where, ref):
    doc = _doc(DATA_REF, PRED_REF)
    doc[where][0]["source"] = "lab notes"
    with pytest.warns(UserWarning, match="has been replaced by 'data_reference'"):
        parsed = _parse(doc)
    assert obs_item_references(_obs_info(parsed, tmp_path)) == [DATA_REF]
    assert obs_item_references(parsed["prediction_info"]) == [PRED_REF]


@pytest.mark.unit
def test_a_null_data_reference_does_not_hide_a_source(tmp_path):
    doc = _doc()
    doc["data_items"][0].update(data_reference=None, source="lab notes")
    with pytest.warns(UserWarning, match="has been replaced by 'data_reference'"):
        assert obs_item_references(_obs_info(_parse(doc), tmp_path)) == ["lab notes"]


@pytest.mark.unit
def test_reading_source_leaves_the_given_document_alone():
    doc = _doc()
    doc["data_items"][0]["source"] = "lab notes"
    original = copy.deepcopy(doc)
    with pytest.warns(UserWarning):
        ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=doc)
    assert doc == original


@pytest.mark.unit
def test_source_warns_once_per_file_with_the_number_of_items(tmp_path):
    def _write(name, data_items, prediction_items):
        doc = _doc()
        doc["data_items"] = [dict(doc["data_items"][0], data_item_name=f"c{i}", source="notes")
                             for i in range(data_items)]
        doc["prediction_items"] = [dict(doc["prediction_items"][0], data_item_name=f"p{i}",
                                        source="notes") for i in range(prediction_items)]
        path = tmp_path / name
        path.write_text(json.dumps(doc))
        return str(path)

    first, second = _write("a_obs_data.json", 3, 2), _write("b_obs_data.json", 1, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for path in (first, second):
            ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=path)
    messages = [str(w.message) for w in _source_warnings(caught)]
    assert len(messages) == 2
    assert messages[0].startswith(f"{first}: 5 item(s) use 'source'")
    assert messages[1].startswith(f"{second}: 1 item(s) use 'source'")
    assert all("replaced by 'data_reference'" in message for message in messages)


@pytest.mark.unit
def test_source_warns_on_rank_0_only(monkeypatch):
    doc = _doc()
    doc["data_items"][0]["source"] = "lab notes"
    monkeypatch.setattr(PrimitiveParsers, "rank", 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed = _parse(doc)
    assert _source_warnings(caught) == []
    assert parsed["gt_df"]["data_reference"].tolist() == ["lab notes"]


@pytest.mark.unit
def test_an_item_without_source_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _parse(_doc(DATA_REF, PRED_REF))
    assert _source_warnings(caught) == []


@pytest.mark.unit
@pytest.mark.parametrize("where", ["data_items", "prediction_items"])
@pytest.mark.parametrize("bad", [3, ["Smith2020"], {"key": "Smith2020"}])
def test_a_source_that_is_not_a_string_is_refused(where, bad):
    doc = _doc()
    doc[where][0]["source"] = bad
    with pytest.raises(ValueError, match="source"):
        _parse(doc)


@pytest.mark.unit
def test_source_does_not_stand_in_for_a_series_value():
    """source names an origin, as data_reference does; the samples are still needed."""
    doc = _doc()
    item = doc["data_items"][0]
    del item["value"]
    item.update(data_type="series", obs_dt=0.1, std=[0.1], source="lab notes")
    with pytest.raises(ValueError, match="'value' is a required property"):
        _parse(doc)


@pytest.mark.unit
def test_a_reference_does_not_stand_in_for_a_series_value():
    """A series needs its samples, in value or value_path; naming their origin is not enough."""
    doc = _doc(DATA_REF)
    item = doc["data_items"][0]
    del item["value"]
    item.update(data_type="series", obs_dt=0.1, std=[0.1])
    with pytest.raises(ValueError, match="'value' is a required property"):
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


@pytest.mark.unit
def test_the_methods_text_cites_a_source_as_a_data_reference(tmp_path):
    """The methods document reads the file directly; a source reaches it as a data_reference,
    so a BibTeX key in it is cited."""
    class _Symbols:
        def variable(self, name):
            return name

    doc = _doc(pred_ref=PRED_REF)
    doc["data_items"][0]["source"] = DATA_REF
    path = tmp_path / "x_obs_data.json"
    path.write_text(json.dumps(doc))
    with pytest.warns(UserWarning, match="has been replaced by 'data_reference'"):
        obs = _load_obs_data(str(path))
    entries = {"Smith2020": "@article{Smith2020,}"}
    text, _ = observables_text(obs, _Symbols(), "gaussian_MLE", entries)
    assert rf"Sources: \cite{{Smith2020}}, Table 2; {PRED_REF}." in text
    assert _obs_cited_keys(obs, entries) == {"Smith2020"}


@pytest.mark.unit
def test_the_methods_text_cites_a_data_reference_as_a_parameters_is():
    """A word of an item's data_reference that is a BibTeX key is cited, as cite() does for a
    parameter's; the other words stay text."""
    class _Symbols:
        def variable(self, name):
            return name

    doc = _doc(DATA_REF, "Jones2019_b, Fig. 3")
    text, _ = observables_text(doc, _Symbols(), "gaussian_MLE",
                               {"Smith2020": "@article{Smith2020,}", "Jones2019_b": "@misc{x,}"})
    assert r"Sources: \cite{Jones2019_b}, Fig. 3; \cite{Smith2020}, Table 2." in text
