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

from funcs_user import cost_funcs_user
from param_id.cost_kwargs import call_cost_func, ground_truth_param_name
from parsers.PrimitiveParsers import ObsAndParamDataParser
from utilities.obs_data_helpers import VALID_DATA_TYPES


DATA_POINTS = [1.05, 0.99, 1.10, 0.95, 1.02, 3.95, 4.10, 4.02, 3.90, 4.08]


def _obs_data(items):
    return {
        "protocol_info": {"pre_times": [0.0], "sim_times": [[1.0]], "params_to_change": {}},
        "prediction_items": [],
        "data_items": items,
    }


def _kde_item(**overrides):
    item = {
        "variable": "benchmark/x",
        "name_for_plotting": "x_{SS}",
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
        "variable": "benchmark/y",
        "name_for_plotting": "y_{SS}",
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
                                        model_type="cellml_only")
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
    assert "y_{SS}" in message
    assert "prob_dist_params" in message


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
def test_an_unknown_cost_kwarg_is_rejected(tmp_path):
    obs_info = _parse([_kde_item(cost_kwargs={"bandwith": 0.1})], tmp_path)
    from param_id.cost_kwargs import validate_cost_kwargs

    funcs = cost_funcs_user.get_cost_funcs_dict_for_mode("numpy")
    with pytest.raises(ValueError, match="bandwith"):
        validate_cost_kwargs(obs_info, funcs, obs_info["cost_type"])
