"""Held-out data in prediction_items is validated against the calibrated model's prediction."""
import json

import numpy as np
import pytest

from libcuflynx.param_id import validation
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser


def _parsed(prediction_items):
    doc = {"data_items": [{"data_item_name": "c0", "operands": ["main/c"], "data_type": "constant",
                           "unit": "dimensionless", "value": 1.0, "std": 0.1}],
           "prediction_items": prediction_items,
           "protocol_info": {"pre_times": [0.0], "sim_times": [[2.0]]}}
    return ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=doc, pre_time=0.0, sim_time=2.0)


@pytest.mark.unit
def test_a_series_is_compared_at_its_observation_times(tmp_path):
    pred = _parsed([{"data_item_name": "y_validation", "operands": ["main/y"], "unit": "mV",
                     "data_type": "series", "value": [0.0, 1.0, 2.0, 3.0, 9.0], "std": 0.5, "obs_dt": 0.5},
                    {"data_item_name": "z", "operands": ["main/z"], "unit": "mV"}])["prediction_info"]
    t = np.linspace(0.0, 1.5, 151)          # the run ends at 1.5: the point at t = 2 is not reached
    res = validation.validation_results(pred, {0: t}, [2.0 * t, np.zeros_like(t)])
    (item,) = res["items"]                    # z has no data, so it is not validated
    assert item["data_item_name"] == "y_validation"
    assert item["t"] == [0.0, 0.5, 1.0, 1.5]
    assert item["model"] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert item["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert item["within_2std"] == 1.0
    path = validation.write_validation_results(res, str(tmp_path))
    assert json.load(open(path))["items"][0]["n_points"] == 4


@pytest.mark.unit
def test_a_constant_is_compared_with_the_end_of_the_experiment():
    pred = _parsed([{"data_item_name": "v_end", "operands": ["main/v"], "unit": "m3",
                     "data_type": "constant", "value": 4.0, "std": 1.0}])["prediction_info"]
    t = np.linspace(0.0, 2.0, 21)
    (item,) = validation.validation_results(pred, {0: t}, [t])["items"]
    assert item["model"] == [2.0]
    assert item["rmse"] == pytest.approx(2.0)
    assert item["mean_abs_z"] == pytest.approx(2.0)
    assert item["nrmse"] == pytest.approx(0.5)


@pytest.mark.unit
def test_no_held_out_data_writes_no_file(tmp_path):
    pred = _parsed([{"data_item_name": "z", "operands": ["main/z"], "unit": "mV"}])["prediction_info"]
    res = validation.validation_results(pred, {0: np.linspace(0, 1, 3)}, [np.zeros(3)])
    assert res == {"items": []}
    assert validation.write_validation_results(res, str(tmp_path)) is None


def _series(std, value=(0.0, 1.0, 2.0)):
    return {"data_item_name": "y", "operands": ["main/y"], "unit": "mV", "data_type": "series",
            "value": list(value), "std": std, "obs_dt": 0.5}


@pytest.mark.unit
@pytest.mark.parametrize('std, expected', [(0.5, [0.5, 0.5, 0.5]), ([0.5], [0.5, 0.5, 0.5]),
                                           ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])])
def test_a_held_out_series_std_is_one_number_or_one_per_point(std, expected):
    assert _parsed([_series(std)])["prediction_info"]["stds"] == [expected]


@pytest.mark.unit
@pytest.mark.parametrize('item, match', [
    # a list shorter than the series used to raise an IndexError while scoring
    (_series([0.1, 0.2]), 'series of 3 values but has 2 stds'),
    (_series([0.1, 0.2, 0.3, 0.4]), 'series of 3 values but has 4 stds'),
    # a zero std used to drop the z-scores of the whole item, silently
    (_series([0.1, 0.0, 0.3]), 'finite and > 0'),
    (_series(0.0), 'finite and > 0'),
    (_series(-1.0), 'finite and > 0'),
    ({"data_item_name": "c", "operands": ["main/c"], "unit": "m3", "data_type": "constant",
      "value": 4.0, "std": 0.0}, 'finite and > 0'),
    ({"data_item_name": "c", "operands": ["main/c"], "unit": "m3", "data_type": "constant",
      "value": 4.0, "std": [1.0]}, 'its std must be a single number'),
])
def test_a_bad_held_out_std_is_a_parse_error_naming_the_item(item, match):
    with pytest.raises(ValueError, match=match):
        _parsed([item])


@pytest.mark.unit
def test_a_held_out_item_without_a_std_gets_no_z_scores():
    item = _series(None)
    del item["std"]
    pred = _parsed([item])["prediction_info"]
    t = np.linspace(0.0, 1.0, 11)
    (result,) = validation.validation_results(pred, {0: t}, [2.0 * t])["items"]
    assert result["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert result["mean_abs_z"] is None and result["within_2std"] is None


@pytest.mark.unit
def test_every_point_of_a_series_gets_a_z_score():
    pred = _parsed([_series([0.5, 1.0, 2.0], value=(0.0, 2.0, 2.0))])["prediction_info"]
    t = np.linspace(0.0, 1.0, 11)
    (result,) = validation.validation_results(pred, {0: t}, [2.0 * t])["items"]
    # model 0, 1, 2 against data 0, 2, 2: |z| = 0, 1, 0
    assert result["mean_abs_z"] == pytest.approx(1.0 / 3.0)
    assert result["within_2std"] == 1.0
