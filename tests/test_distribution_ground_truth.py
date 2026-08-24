"""Issue #421: a distribution is a ground truth, not a data_type.

``prob_dist`` used to be a fourth ``data_type`` beside ``constant``, ``series`` and
``frequency``. The other three describe the shape of the *simulated data* -- a scalar, a trace,
an FFT. ``prob_dist`` did not: it described the shape of the *ground truth*, while the feature it
produced was an ordinary scalar, filed into a parallel vector by an otherwise identical branch.

The cost of that split was not tidiness. Anything that works on scalar features indexes by
``const_idx_to_obs_idx``, so a ``prob_dist`` item had no slot in it and was invisible to the
emulator and to FD sensitivities -- which is what made the bimodal UQ case impossible to emulate.

These tests pin the replacement: such an item is a ``constant`` whose ``cost_type`` scores it
against ``prob_dist_params``, chosen from the cost func's own signature.
"""
import json

import numpy as np
import pytest

from libcuflynx.funcs import cost_funcs_user
from libcuflynx.param_id.cost_kwargs import call_cost_func, ground_truth_param_name
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.utilities.obs_data_helpers import VALID_DATA_TYPES


DATA_POINTS = [1.05, 0.99, 1.10, 0.95, 1.02, 3.95, 4.10, 4.02, 3.90, 4.08]


def _obs_data(items):
    return {
        "protocol_info": {"pre_times": [0.0], "sim_times": [[1.0]], "params_to_change": {}},
        "prediction_items": [],
        "data_items": items,
    }


def _kde_item(**overrides):
    item = {
        "data_item_name": "benchmark/x",
        "trace_name_for_plotting": "x_{SS}",
        "data_type": "constant",
        "operation": "steady_state_avg",
        "operands": ["benchmark/x"],
        "unit": "dimensionless",
        "weight": 1.0,
        "cost_type": "kernel_density_estimation",
        "prob_dist_params": {"data_points": DATA_POINTS},
    }
    item.update(overrides)
    return item


def _gaussian_item(**overrides):
    item = {
        "data_item_name": "benchmark/y",
        "trace_name_for_plotting": "y_{SS}",
        "data_type": "constant",
        "operation": "steady_state_avg",
        "operands": ["benchmark/y"],
        "unit": "dimensionless",
        "weight": 1.0,
        "cost_type": "gaussian_MLE",
        "value": 0.33,
        "std": 0.1,
    }
    item.update(overrides)
    return item


def _parse(items, tmp_path):
    path = tmp_path / "obs_data.json"
    path.write_text(json.dumps(_obs_data(items)))
    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(param_id_obs_path=str(path), pre_time=0.0, sim_time=1.0,
                                        model_type="cellml")
    return parser.process_obs_info(gt_df=parsed["gt_df"], output_dir=str(tmp_path), dt=0.01)


# ---------------------------------------------------------------------------
# the data_type is gone
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_prob_dist_is_not_an_advertised_data_type():
    """CUFLynx populates its data_type menu from this tuple, so a removed type must leave it."""
    assert "prob_dist" not in VALID_DATA_TYPES
    assert VALID_DATA_TYPES == ("constant", "series", "frequency")


@pytest.mark.unit
def test_an_obs_data_still_using_prob_dist_is_told_what_to_write_instead(tmp_path):
    """Silently ignoring it would score the item with an unweighted, unread cost."""
    with pytest.raises(ValueError) as excinfo:
        _parse([_kde_item(data_type="prob_dist", value=1.0, std=0.1)], tmp_path)
    message = str(excinfo.value)
    assert '"constant"' in message
    assert "prob_dist_params" in message


# ---------------------------------------------------------------------------
# a distribution-scored item is an ordinary scalar observable
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_kde_item_is_a_constant_feature(tmp_path):
    """The point of #421: it occupies a slot in the constant index map, which is what the
    emulator, observable_features and the FD sensitivities are built on."""
    obs_info = _parse([_kde_item(), _gaussian_item()], tmp_path)

    assert obs_info["data_types"] == ["constant", "constant"]
    assert list(obs_info["const_idx_to_obs_idx"]) == [0, 1]


@pytest.mark.unit
def test_the_distribution_is_indexed_by_data_item_row(tmp_path):
    """Not by a compacted counter of its own: cost_type and the weight vectors are indexed by
    row, and disagreeing index spaces is exactly the #349 bug."""
    obs_info = _parse([_gaussian_item(), _kde_item()], tmp_path)

    params = obs_info["ground_truth_prob_dist_params"]
    assert len(params) == 2
    assert params[0] is None, "a value/std item has no distribution"
    assert params[1]["data_points"] == DATA_POINTS


# ---------------------------------------------------------------------------
# every item needs *a* ground truth, and only the one it uses
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_distribution_item_needs_no_value_or_std(tmp_path):
    """They would be numbers nothing reads. The fixtures carried "value": 1, "std": 0.1 next to
    a kernel_density_estimation cost that never looked at either."""
    obs_info = _parse([_kde_item()], tmp_path)

    assert np.isnan(obs_info["ground_truth_const"][0])
    assert obs_info["ground_truth_prob_dist_params"][0]["data_points"] == DATA_POINTS


@pytest.mark.unit
def test_an_item_with_neither_a_value_nor_a_distribution_is_rejected(tmp_path):
    """Relaxing value/std must not become 'a ground truth is optional' -- an item with no target
    contributes a nan to the cost, which propagates and looks like a solver failure."""
    with pytest.raises(ValueError) as excinfo:
        _parse([_gaussian_item(value=None, std=None)], tmp_path)
    message = str(excinfo.value)
    assert "benchmark/y" in message
    assert "prob_dist_params" in message


@pytest.mark.unit
def test_an_obs_data_with_no_data_items_is_still_valid(tmp_path):
    """A protocol-only obs_data says how to drive the model without yet saying what to measure --
    what an obs_data generated from a model's own protocol looks like before its targets are
    added. The schema loop is skipped for it, so none of the columns the ground-truth checks
    read exist. Only the parse is exercised, which is as far as a protocol-only file goes."""
    path = tmp_path / "obs_data.json"
    path.write_text(json.dumps(_obs_data([])))
    parsed = ObsAndParamDataParser().parse_obs_data_json(
        param_id_obs_path=str(path), pre_time=0.0, sim_time=1.0, model_type="cellml")

    assert len(parsed["gt_df"]) == 0


@pytest.mark.unit
def test_a_value_item_still_needs_its_std(tmp_path):
    with pytest.raises(ValueError, match="std"):
        _parse([_gaussian_item(std=None)], tmp_path)


# ---------------------------------------------------------------------------
# which ground truth a cost gets follows from its signature
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("cost_name, expected", [
    ("gaussian_MLE", "desired_mean"),
    ("MSE", "desired_mean"),
    ("AE", "desired_mean"),
    ("kernel_density_estimation", "prob_dist_params"),
    ("multimodal_gaussian", "prob_dist_params"),
    ("poisson_MLE", "prob_dist_params"),
])
def test_the_ground_truth_is_read_off_the_cost_signature(cost_name, expected):
    """#84 generalised the *keyword* arguments across cost shapes; the positional ground truth
    stayed hardcoded per data_type. This is the same rule extended to that slot."""
    funcs = cost_funcs_user.get_cost_funcs_dict_for_mode("numpy")
    assert ground_truth_param_name(funcs[cost_name]) == expected


@pytest.mark.unit
def test_a_user_cost_taking_neither_falls_back_to_the_scalar_ground_truth():
    """A cost with only an output -- a barrier, an absolute bound -- must not be mistaken for a
    distribution cost."""
    def bounded(output, weight):
        return abs(output) * weight

    assert ground_truth_param_name(bounded) == "desired_mean"


# ---------------------------------------------------------------------------
# the cost itself
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_kde_cost_is_lowest_at_each_mode(tmp_path):
    """The whole reason a KDE target exists: a bimodal ground truth must score *both* modes as
    good, which no (mean, std) pair can express -- their mean, 2.5, is where no measurement
    landed."""
    obs_info = _parse([_kde_item(cost_kwargs={"bandwidth": 0.1})], tmp_path)
    kde = cost_funcs_user.get_cost_funcs_dict_for_mode("numpy")["kernel_density_estimation"]

    def cost(x):
        return call_cost_func(kde, x, obs_info["ground_truth_prob_dist_params"][0],
                              std=obs_info["std_const_vec"][0], weight=1.0,
                              cost_kwargs=obs_info["cost_kwargs"][0])

    assert cost(1.0) < cost(2.5)
    assert cost(4.0) < cost(2.5)
    assert cost(1.0) == pytest.approx(cost(4.0), abs=0.5), \
        "the two modes carry equal weight in the samples, so neither is preferred"


@pytest.mark.unit
def test_the_bandwidth_is_a_cost_kwarg_not_part_of_the_ground_truth(tmp_path):
    """It tunes the comparison rather than stating the data, so it can be swept without editing
    the measurements -- and a typo in it is rejected by the #84 validation rather than ignored."""
    obs_info = _parse([_kde_item(cost_kwargs={"bandwidth": 0.02})], tmp_path)
    assert obs_info["cost_kwargs"][0] == {"bandwidth": 0.02}
    assert "bandwidth" not in obs_info["ground_truth_prob_dist_params"][0]

    kde = cost_funcs_user.get_cost_funcs_dict_for_mode("numpy")["kernel_density_estimation"]
    params = obs_info["ground_truth_prob_dist_params"][0]
    # A narrow kernel makes the gap between the modes deeper: the density there is further from
    # any sample. If bandwidth were being dropped, both would return the same number.
    narrow = call_cost_func(kde, 2.5, params, weight=1.0, cost_kwargs={"bandwidth": 0.02})
    wide = call_cost_func(kde, 2.5, params, weight=1.0, cost_kwargs={"bandwidth": 0.5})
    assert narrow > wide


@pytest.mark.unit
def test_cost_calc_scores_a_distribution_item_in_the_constant_loop(tmp_path):
    """End to end through the cost, mixing both ground truth shapes in one obs_data: the KDE
    item must be scored, and scored against its samples rather than against the nan standing in
    for the value it does not have."""
    from libcuflynx.param_id.paramID import ParamID
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser, scriptFunctionParser

    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(
        obs_data_dict=_obs_data([_kde_item(cost_kwargs={"bandwidth": 0.1}), _gaussian_item()]),
        pre_time=0.0, sim_time=1.0)
    obs_info = parser.process_obs_info(gt_df=parsed["gt_df"], output_dir=str(tmp_path), dt=0.01)
    protocol_info = parser.process_protocol_and_weights(
        gt_df=parsed["gt_df"], protocol_info=parsed["protocol_info"], dt=0.01)

    pid = ParamID.__new__(ParamID)
    pid.obs_info = obs_info
    pid.protocol_info = protocol_info
    pid.cost_type = obs_info["cost_type"]
    pid._num_weighted_obs_by_exp_sub = None
    pid.cost_funcs_dict = scriptFunctionParser().get_cost_funcs_dict("numpy")
    pid.cost_funcs_dict_symbolic = pid.cost_funcs_dict

    def cost_at(x):
        return pid.cost_calc({"const": np.array([x, 0.33]), "series": None,
                              "amp": None, "phase": None}, exp_idx=0, sub_idx=0)

    assert np.isfinite(cost_at(1.0)), 'the KDE item was scored against a nan ground truth'
    assert cost_at(2.5) > cost_at(1.0), 'the KDE item is not contributing to the cost'
    assert cost_at(4.0) < cost_at(2.5), 'the far mode is not being scored as a good fit'


@pytest.mark.unit
def test_a_distribution_cost_cannot_be_differentiated_symbolically(tmp_path):
    """Its density is built from numbers -- scipy's gaussian_kde cannot take a symbol. Silently
    returning something would be worse than raising: the nan standing in for `value` would
    propagate into a gradient that looks like a failed solve."""
    from libcuflynx.param_id.paramID import ParamID
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser, scriptFunctionParser

    ca = pytest.importorskip('casadi')

    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(obs_data_dict=_obs_data([_kde_item()]),
                                        pre_time=0.0, sim_time=1.0)
    obs_info = parser.process_obs_info(gt_df=parsed["gt_df"], output_dir=str(tmp_path), dt=0.01)
    protocol_info = parser.process_protocol_and_weights(
        gt_df=parsed["gt_df"], protocol_info=parsed["protocol_info"], dt=0.01)

    pid = ParamID.__new__(ParamID)
    pid.obs_info = obs_info
    pid.protocol_info = protocol_info
    pid.cost_type = obs_info["cost_type"]
    pid._num_weighted_obs_by_exp_sub = None
    pid.cost_funcs_dict = scriptFunctionParser().get_cost_funcs_dict("numpy")
    pid.cost_funcs_dict_symbolic = scriptFunctionParser().get_cost_funcs_dict("casadi")

    with pytest.raises(NotImplementedError, match='do_ad'):
        pid.cost_calc({"const": ca.SX.sym('x', 1, 1), "series": None,
                       "amp": None, "phase": None},
                      exp_idx=0, sub_idx=0, is_symbolic=True)


@pytest.mark.unit
def test_an_unknown_cost_kwarg_is_rejected(tmp_path):
    obs_info = _parse([_kde_item(cost_kwargs={"bandwith": 0.1})], tmp_path)
    from libcuflynx.param_id.cost_kwargs import validate_cost_kwargs

    funcs = cost_funcs_user.get_cost_funcs_dict_for_mode("numpy")
    with pytest.raises(ValueError, match="bandwith"):
        validate_cost_kwargs(obs_info, funcs, obs_info["cost_type"])
