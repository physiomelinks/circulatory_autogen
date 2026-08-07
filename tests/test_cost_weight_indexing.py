"""Weights reach the observable they were written for (issues #349, #350).

Two indexing bugs, both silent.

#349: process_protocol_and_weights builds each scaled_weight_<type>_from_exp_sub
as a full-length vector over *all* data_items, zero where the row is not that
type. cost_calc reads it with the per-type compacted counter (const_idx,
series_idx, ...). The two agree only when the items of a type occupy the leading
rows, so an obs_data that interleaves types scores the wrong weights -- and an
observable whose weight reads as 0 is dropped from the cost while still being
counted in the denominator.

#350: _build_param_id_info_from_df decides a whole row's gen-name shape from
vessel_name[0] alone, so a row mixing 'global' with named vessels loses every
vessel after the first.

No model, solver or MPI: the parsing and the cost arithmetic are what is wrong.
"""
import io

import numpy as np
import pandas as pd
import pytest

from parsers.PrimitiveParsers import ObsAndParamDataParser


# ---------------------------------------------------------------------------
# #349 -- weights are matched to their observable
# ---------------------------------------------------------------------------
def _obs_data(items):
    return {"data_items": items,
            "protocol_info": {"pre_times": [0.0], "sim_times": [[1.0]]}}


def _series_item(name, value=2.0, weight=1.0):
    return {"variable": f"a/{name}", "data_type": "series", "unit": "mV",
            "operands": [f"a/{name}"], "value": [value, value], "std": 1.0,
            "weight": weight, "obs_dt": 0.01, "name_for_plotting": name}


def _const_item(name, value=5.0, weight=1.0, std=1.0):
    return {"variable": f"a/{name}", "data_type": "constant", "unit": "mV",
            "operands": [f"a/{name}"], "value": value, "std": std,
            "weight": weight, "name_for_plotting": name}


def _parsed(items, tmp_path):
    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(obs_data_dict=_obs_data(items),
                                        pre_time=0.0, sim_time=1.0)
    obs_info = parser.process_obs_info(gt_df=parsed["gt_df"], output_dir=str(tmp_path), dt=0.01)
    protocol_info = parser.process_protocol_and_weights(
        gt_df=parsed["gt_df"], protocol_info=parsed["protocol_info"], dt=0.01)
    return obs_info, protocol_info


def test_a_trailing_constants_weight_is_not_read_from_another_row(tmp_path):
    """The bug in isolation. Two series then a constant: the constant is const_idx
    0, but its weight lives at row 2, and row 0 belongs to a series -- so reading
    by the compacted index finds a zero and drops it."""
    items = [_series_item("s1"), _series_item("s2"), _const_item("c1", weight=7.0)]
    obs_info, protocol_info = _parsed(items, tmp_path)

    weights = protocol_info["scaled_weight_const_from_exp_sub"][0][0]
    const_to_obs = obs_info["const_idx_to_obs_idx"]

    # One constant, and its weight is the 7.0 the user wrote.
    assert list(const_to_obs) == [2]
    assert weights[const_to_obs[0]] == pytest.approx(7.0)
    # Read the way cost_calc reads it, the constant's weight is a series' zero.
    assert weights[0] == 0.0


def test_a_trailing_constant_is_actually_scored(tmp_path):
    """End to end through cost_calc: the constant is 3 sigma from its target, so
    a weighted MSE must be non-zero. Dropped, the cost is exactly zero while the
    denominator still counts it -- a fit that looks perfect because the term
    vanished."""
    from param_id.paramID import OpencorParamID

    items = [_series_item("s1", weight=0.0), _series_item("s2", weight=0.0),
             _const_item("c1", value=5.0, std=1.0, weight=1.0)]
    obs_info, protocol_info = _parsed(items, tmp_path)

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.obs_info = obs_info
    pid.protocol_info = protocol_info
    pid.cost_type = obs_info["cost_type"]
    pid._num_weighted_obs_by_exp_sub = None
    from parsers.PrimitiveParsers import scriptFunctionParser
    pid.cost_funcs_dict = scriptFunctionParser().get_cost_funcs_dict("numpy")
    pid.cost_funcs_dict_symbolic = pid.cost_funcs_dict

    # The model says 8.0 where the data says 5.0, with std 1 -> a real cost.
    obs_dict = {"const": np.array([8.0]), "series": None,
                "amp": None, "phase": None, "val_for_prob_dist": None}
    cost = pid.cost_calc(obs_dict, exp_idx=0, sub_idx=0)

    assert cost > 0.0, (
        "the constant contributed nothing: its weight was read from another "
        "data_item's row, so a mis-fitting observable is silently free"
    )


def test_the_weights_are_positional_by_data_item_not_by_type(tmp_path):
    """States the invariant the fix relies on, so a future change to how the
    vectors are built cannot quietly re-break the read side."""
    items = [_series_item("s1"), _const_item("c1", weight=7.0), _series_item("s2")]
    obs_info, protocol_info = _parsed(items, tmp_path)

    weights = protocol_info["scaled_weight_const_from_exp_sub"][0][0]
    # One entry per data_item, not per constant.
    assert len(weights) == len(items)
    # Non-zero only at the constant's own row.
    assert [i for i, w in enumerate(weights) if w != 0] == [1]


# ---------------------------------------------------------------------------
# #350 -- a row mixing 'global' with named vessels
# ---------------------------------------------------------------------------
def _info(csv):
    return ObsAndParamDataParser()._build_param_id_info_from_df(pd.read_csv(io.StringIO(csv)))


def test_a_global_row_with_named_vessels_keeps_every_vessel():
    """param_names and param_names_for_gen are positional partners; dropping a
    vessel from one leaves the two lists no longer describing the same thing."""
    info = _info("vessel_name,param_name,min,max\nglobal a,k,1,2\n")

    assert info["param_names"] == [["global/k", "a/k"]]
    assert info["param_names_for_gen"] == [["k", "k_a"]]


def test_the_two_name_lists_stay_the_same_length():
    info = _info("vessel_name,param_name,min,max\nglobal a b,k,1,2\n")
    assert len(info["param_names"][0]) == len(info["param_names_for_gen"][0])


def test_a_global_only_row_is_unchanged():
    """The common case must keep the name it has always had."""
    info = _info("vessel_name,param_name,min,max\nglobal,k,1,2\n")
    assert info["param_names_for_gen"] == [["k"]]


def test_a_row_without_global_is_unchanged():
    info = _info("vessel_name,param_name,min,max\na b,k,1,2\n")
    assert info["param_names_for_gen"] == [["k_a", "k_b"]]


def test_a_series_is_scored_too(tmp_path):
    """Exercises the *series* branch of cost_calc, which the constant tests leave
    at None.

    Its absence is what let a mistake through: the numeric series loop assigned
    obs_idx one line *after* the weight read, so changing the read to use obs_idx
    raised UnboundLocalError on the first iteration -- or, worse, silently reused
    the const loop's stale obs_idx when a constant came first. CI's param_id job
    caught it; nothing here did.
    """
    from param_id.paramID import OpencorParamID
    from parsers.PrimitiveParsers import scriptFunctionParser

    # A constant first, so a stale obs_idx would be in scope and plausible.
    items = [_const_item("c1", value=5.0, weight=0.0),
             _series_item("s1", value=2.0, weight=1.0)]
    obs_info, protocol_info = _parsed(items, tmp_path)

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.obs_info = obs_info
    pid.protocol_info = protocol_info
    pid.cost_type = obs_info["cost_type"]
    pid._num_weighted_obs_by_exp_sub = None
    pid.dt = 0.01
    pid.cost_funcs_dict = scriptFunctionParser().get_cost_funcs_dict("numpy")
    pid.cost_funcs_dict_symbolic = pid.cost_funcs_dict

    # Model says 4.0 where the series says 2.0 -> a real, non-zero cost.
    obs_dict = {"const": np.array([5.0]), "series": [np.array([4.0, 4.0])],
                "amp": None, "phase": None, "val_for_prob_dist": None}
    cost = pid.cost_calc(obs_dict, exp_idx=0, sub_idx=0)

    assert cost > 0.0, "the series contributed nothing"
