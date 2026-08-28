"""The posterior median is written as something that can be loaded, not just reported.

``mcmc_statistics.json`` has always carried the median, keyed by *display* name
(``g_{Na}``) inside a per-parameter summary next to quartiles and a 95% interval. That is a
report. Nothing can feed it back into a model: the display names are not the model's, and
the shape is not one any parameter loader reads.

So ``save_mcmc_statistics`` could promise "both are now on disk to choose from" while only
one of the two -- ``best_param_vals.npy`` -- was in a loadable form, and choosing the median
meant reading JSON and retyping fourteen numbers.

That is not a cosmetic gap. A calibration best is an argmin found by whatever drove the
search, and when a surrogate drove it the argmin can sit where the surrogate is wrong: one
SN_full cpvt run reported a best fit whose resting potential was -19 mV against data at -75,
while the posterior median from the same chain sat at -80.
"""
import csv
import json
import os

import numpy as np
import pytest

from libcuflynx.param_id.paramID import MCMC


class _Writer:
    """The parts of MCMC ``save_posterior_point_estimates`` touches, and nothing else.

    Bound off the real class rather than reimplemented, so the test exercises the shipped
    method: driving a whole MCMC would need a model, a protocol and a chain to reach four
    lines of file writing.
    """

    def __init__(self, output_dir, members_per_slot, rank=0):
        self.output_dir = output_dir
        self.param_id_info = {"param_names": members_per_slot}
        self.rank = rank

    save_posterior_point_estimates = MCMC.save_posterior_point_estimates


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def writer(tmp_path):
    return _Writer(str(tmp_path), [["soma_SN/g_Na"], ["soma_SN/g_leak_K"],
                                   ["soma_SN/V_mid_w"]])


def test_both_estimates_are_written_in_both_formats(writer, tmp_path):
    writer.save_posterior_point_estimates([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    for label in ("median", "mean"):
        assert os.path.isfile(tmp_path / f"posterior_{label}_params.csv")
        assert os.path.isfile(tmp_path / f"posterior_{label}_param_vals.npy")


def test_the_csv_carries_names_so_it_does_not_depend_on_order(writer, tmp_path):
    """The columns a parameter loader reads: vessel_name, param_name, value. Self-describing,
    so loading it into a model does not require the columns to line up positionally."""
    writer.save_posterior_point_estimates([1.5, 2.5, 3.5], [0.0, 0.0, 0.0])
    rows = read_csv(tmp_path / "posterior_median_params.csv")
    assert [r["vessel_name"] for r in rows] == ["soma_SN"] * 3
    assert [r["param_name"] for r in rows] == ["g_Na", "g_leak_K", "V_mid_w"]
    assert [float(r["value"]) for r in rows] == [1.5, 2.5, 3.5]


def test_the_npy_matches_best_param_vals_in_shape_and_order(writer, tmp_path):
    """A bare array in optimiser order, so anything reading best_param_vals.npy reads this."""
    writer.save_posterior_point_estimates([1.5, 2.5, 3.5], [4.5, 5.5, 6.5])
    assert np.allclose(np.load(tmp_path / "posterior_median_param_vals.npy"), [1.5, 2.5, 3.5])
    assert np.allclose(np.load(tmp_path / "posterior_mean_param_vals.npy"), [4.5, 5.5, 6.5])


def test_the_median_and_the_mean_do_not_overwrite_each_other(writer, tmp_path):
    writer.save_posterior_point_estimates([1.0, 1.0, 1.0], [9.0, 9.0, 9.0])
    med = [float(r["value"]) for r in read_csv(tmp_path / "posterior_median_params.csv")]
    mean = [float(r["value"]) for r in read_csv(tmp_path / "posterior_mean_params.csv")]
    assert med == [1.0, 1.0, 1.0] and mean == [9.0, 9.0, 9.0]


def test_a_modifier_slot_writes_every_member_at_the_slot_value(tmp_path):
    """One slot, several model parameters, one value -- matching how best_param_vals.npy is
    read back, where each member qname takes the slot's value."""
    w = _Writer(str(tmp_path), [["a/g_x", "b/g_x", "c/g_x"], ["soma_SN/g_Na"]])
    w.save_posterior_point_estimates([2.0, 7.0], [0.0, 0.0])
    rows = read_csv(tmp_path / "posterior_median_params.csv")
    assert len(rows) == 4
    assert [float(r["value"]) for r in rows] == [2.0, 2.0, 2.0, 7.0]
    assert [r["vessel_name"] for r in rows] == ["a", "b", "c", "soma_SN"]


def test_a_name_without_a_vessel_is_kept_whole(tmp_path):
    """Rather than dropped, so the file still accounts for every slot."""
    w = _Writer(str(tmp_path), [["bare_name"]])
    w.save_posterior_point_estimates([3.0], [3.0])
    rows = read_csv(tmp_path / "posterior_median_params.csv")
    assert rows[0]["vessel_name"] == "" and rows[0]["param_name"] == "bare_name"


def test_full_precision_is_kept(writer, tmp_path):
    """A parameter set rounded on the way to disk is a different parameter set. Spans and
    conductances here run to 1e-5, where %.6g would lose real digits."""
    values = [1.2345678901234567e-05, 2.0, 3.0]
    writer.save_posterior_point_estimates(values, values)
    rows = read_csv(tmp_path / "posterior_median_params.csv")
    assert float(rows[0]["value"]) == values[0]


def test_only_rank_zero_writes(tmp_path):
    """Every rank runs this; one file, written once."""
    w = _Writer(str(tmp_path), [["soma_SN/g_Na"]], rank=1)
    w.save_posterior_point_estimates([1.0], [1.0])
    assert not os.path.isfile(tmp_path / "posterior_median_params.csv")


def test_more_slots_than_values_stops_rather_than_raising(tmp_path):
    """A truncated array should give what it can, not an IndexError in the last step of a
    run that has already spent hours sampling."""
    w = _Writer(str(tmp_path), [["a/x"], ["b/y"], ["c/z"]])
    w.save_posterior_point_estimates([1.0, 2.0], [1.0, 2.0])
    rows = read_csv(tmp_path / "posterior_median_params.csv")
    assert [r["param_name"] for r in rows] == ["x", "y"]
