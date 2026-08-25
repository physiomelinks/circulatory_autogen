"""Putting a protocol_info into an obs_data document.

``fill_protocol_info`` lives in ``utilities.obs_data_helpers`` rather than beside
the ``.mmt`` reader that first needed it, because every key it writes is that
module's vocabulary -- and that module is where those names get migrated when
they change. These tests go with it.
"""

import json

import pytest

from libcuflynx.parsers.MyokitParsers import protocol_info_from_events
from libcuflynx.utilities.obs_data_helpers import fill_protocol_info


def _protocol_info():
    info, _notes = protocol_info_from_events(
        [{"level": 1.0, "start": 100.0, "length": 2.0,
          "period": 1000.0, "multiplier": 0}],
        name="engine/pace",
        duration=2000.0,
    )
    return info


def test_a_bare_list_of_data_items_becomes_the_object_form():
    """CA's other accepted obs_data shape, and what several studies ship."""
    info = _protocol_info()
    out = fill_protocol_info([{"data_item_name": "x"}], info)
    assert out["data_items"] == [{"data_item_name": "x"}]
    assert out["protocol_info"] == info


def test_hand_written_labels_survive_a_reconversion():
    info = _protocol_info()
    existing = {"protocol_info": {"experiment_labels": ["1 Hz pacing"],
                                  "experiment_colors": ["b"]}}
    out = fill_protocol_info(existing, info)
    assert out["protocol_info"]["experiment_labels"] == ["1 Hz pacing"]
    assert out["protocol_info"]["experiment_colors"] == ["b"]
    assert out["protocol_info"]["sim_times"] == info["sim_times"]


def test_labels_that_no_longer_fit_the_schedule_are_replaced():
    info = _protocol_info()
    existing = {"protocol_info": {"experiment_labels": ["a", "b"]}}
    out = fill_protocol_info(existing, info)
    assert out["protocol_info"]["experiment_labels"] == info["experiment_labels"]


def test_the_result_is_json_serialisable():
    info = _protocol_info()
    assert json.loads(json.dumps(fill_protocol_info(None, info)))
