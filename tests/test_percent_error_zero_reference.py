"""A percent error against a reference value of 0 is NaN (Q-0099).

A no-data item (value 0, a huge std) that a ratio reads as one of its terms reported about
-5e19 % from dividing by ``value + 1e-10``. Undefined is what it is, so the vectors hold NaN,
the bar pages leave the bar out and keep their axis from the other items.

No model: the error-vector and plotting code run on stand-ins.
"""
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from libcuflynx.param_id import plot_outputs
from libcuflynx.param_id.plot_outputs import ParamIDPlotOutputs


class _ParamID:
    best_param_vals = np.array([1.0])

    def __init__(self, consts):
        self.consts = consts

    def get_cost_and_obs_from_params(self, _params):
        return 0.0, [object()]

    def obs_output_dicts_per_segment(self, operands_list):
        return [{"const": self.consts} for _ in operands_list]


class _Client:
    def __init__(self, tmp_path, consts, ground_truth):
        n = len(ground_truth)
        self.param_id = _ParamID(np.asarray(consts, dtype=float))
        self.obs_info = {
            "num_obs": n, "const_idx_to_obs_idx": list(range(n)),
            "experiment_idxs": [0] * n, "subexperiment_idxs": [0] * n,
            "ground_truth_const": np.asarray(ground_truth, dtype=float),
            "std_const_vec": np.ones(n), "item_names_for_plotting": [f"a{i}" for i in range(n)],
        }
        self.protocol_info = {"num_sub_per_exp": [1], "pre_times": [0.0]}
        self.plot_dir = str(tmp_path)
        self.file_name_prefix = "m"
        self.param_id_obs_file_prefix = "obs"


@pytest.mark.unit
def test_percent_error_is_nan_against_a_zero_reference():
    percent_error = plot_outputs.percent_error
    assert np.isnan(percent_error(3.0, 0.0))
    assert percent_error(1.1, 1.0) == pytest.approx(10.0)
    assert percent_error(-0.5, -1.0) == pytest.approx(-50.0)


@pytest.mark.unit
def test_the_error_vectors_hold_nan_for_a_zero_reference(tmp_path):
    plotter = ParamIDPlotOutputs(_Client(tmp_path, consts=[2.0, 0.51], ground_truth=[0.0, 0.5]))
    percent, std = plotter.emulator_error_vectors()
    assert np.isnan(percent[0])
    assert percent[1] == pytest.approx(2.0)
    assert std[0] == pytest.approx(2.0), "the std error is unchanged"


@pytest.mark.unit
def test_the_bar_pages_draw_with_a_nan_entry(tmp_path):
    client = _Client(tmp_path, consts=[0.0, 0.0], ground_truth=[0.0, 1.0])
    plotter = ParamIDPlotOutputs(client)
    names = np.array(["$a0$", "$a1$"])
    plotter.plot_percent_error_bar_pages(names, np.array([np.nan, -20.0]))
    assert os.path.exists(os.path.join(str(tmp_path), "error_bars_m_obs_0.png"))
