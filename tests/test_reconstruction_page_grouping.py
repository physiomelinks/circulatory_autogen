"""One reconstruction page per trace, not per feature (issue #515).

Before #466 a data_item's ``variable`` was both its identity and the name of the model
variable it reduced, so ``max`` and ``min`` of the same pressure shared a ``variable`` and
were grouped onto one page. #466 split those jobs and made ``data_item_name`` unique by
construction -- and the page grouping still keyed on it, so every feature got a page of its
own and the merged figures disappeared.

The key is ``trace_name_for_plotting``: the series a feature is measured on, allowed to
repeat, and already what the y-axis is labelled with. Subexperiment is deliberately not part
of it, so a trace measured on several subexperiments is drawn as one figure with a segment
each.

No model and no MPI: the plotting method is driven directly with a synthetic obs_info.
"""
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from libcuflynx.param_id.plot_outputs import ParamIDPlotOutputs


N_STEPS = 4
DT = 0.25
SUB_SIM_TIME = 1.0


class _Client:
    def __init__(self, plot_dir, obs_info, protocol_info):
        self.obs_info = obs_info
        self.protocol_info = protocol_info
        self.plot_dir = str(plot_dir)
        self.output_dir = str(plot_dir)
        self.file_name_prefix = "model"
        self.param_id_obs_file_prefix = "obs"
        self.dt = DT


def _obs_info(items):
    """``items`` are ``(data_item_name, trace_name_for_plotting, subexperiment_idx)``."""
    num_obs = len(items)
    return {
        "num_obs": num_obs,
        "data_item_names": [name for name, _, _ in items],
        "trace_names_for_plotting": [trace for _, trace, _ in items],
        "item_names_for_plotting": [f"{name} (max)" for name, _, _ in items],
        "experiment_idxs": [0] * num_obs,
        "subexperiment_idxs": [sub for _, _, sub in items],
        "data_types": ["constant"] * num_obs,
        "operations": ["max"] * num_obs,
        "operation_kwargs": [{} for _ in range(num_obs)],
        "units": ["J_per_m3"] * num_obs,
        "plot_type": ["horizontal"] * num_obs,
        "plot_colors": ["tab:blue"] * num_obs,
        "ground_truth_const": np.ones(num_obs),
        "std_const_vec": np.ones(num_obs),
        "weight_const_vec": np.ones(num_obs),
        "ground_truth_prob_dist_params": [None] * num_obs,
        "ground_truth_series": [],
        "obs_dt": [],
        "freqs": [None] * num_obs,
    }


def _protocol_info(num_sub):
    return {
        "num_experiments": 1,
        "num_sub_per_exp": [num_sub],
        "sim_times": [[SUB_SIM_TIME] * num_sub],
        "pre_times": [0.0],
        "experiment_colors": ["tab:green"],
    }


def _run(tmp_path, items, num_sub, monkeypatch):
    """Drive ``plot_reconstruction_pages`` and capture each page's axes before it closes."""
    obs_info = _obs_info(items)
    protocol_info = _protocol_info(num_sub)

    plotter = ParamIDPlotOutputs.__new__(ParamIDPlotOutputs)
    plotter.client = _Client(tmp_path, obs_info, protocol_info)

    pages = []

    def _capture(plot_dir, prefix, obs_stub, plot_idx, fig, axs, fig_phase, axs_phase, phase):
        pages.append(
            {
                "xlim": axs.get_xlim(),
                "labels": [line.get_label() for line in axs.get_lines()],
                "ylabel": axs.get_ylabel(),
                "n_lines": len(axs.get_lines()),
            }
        )

    monkeypatch.setattr(
        ParamIDPlotOutputs, "_save_reconstruction_figure_bundle", staticmethod(_capture)
    )

    num_obs = obs_info["num_obs"]
    # One time grid per subexperiment, laid end to end along the experiment's timeline --
    # the same absolute axis _compute_subexperiment_time_axes builds.
    tSim_per_sub_count = [
        np.linspace(sub * SUB_SIM_TIME, (sub + 1) * SUB_SIM_TIME, N_STEPS + 1)
        for sub in range(num_sub)
    ]
    # A reconstruction per item, per subexperiment: series_per_sub[II] is item II's trace.
    list_of_all_series = [
        np.ones((num_obs, N_STEPS + 1)) * (sub + 1) for sub in range(num_sub)
    ]
    list_of_obs_dicts = [
        {
            "const": np.ones(num_obs),
            "series": np.ones((num_obs, N_STEPS + 1)),
            "amp": [],
            "phase": [],
        }
        for _ in range(num_sub)
    ]

    percent, std, phase_err = plotter.plot_reconstruction_pages(
        False,
        list_of_obs_dicts,
        list_of_all_series,
        tSim_per_sub_count,
        [SUB_SIM_TIME * num_sub],
        [N_STEPS] * num_sub,
    )
    return pages, percent, std


def test_two_features_of_one_trace_share_a_page(tmp_path, monkeypatch):
    """The regression in the issue: max and min of one pressure are one figure, not two."""
    pages, _, _ = _run(
        tmp_path,
        [("u_ar_max", "u_ar", 0), ("u_ar_min", "u_ar", 0)],
        num_sub=1,
        monkeypatch=monkeypatch,
    )
    assert len(pages) == 1


def test_distinct_traces_stay_on_separate_pages(tmp_path, monkeypatch):
    """Grouping merges a trace's features; it does not merge two different traces."""
    pages, _, _ = _run(
        tmp_path,
        [("u_ar_max", "u_ar", 0), ("u_ar_min", "u_ar", 0), ("q_lv_mean", "q_lv", 0)],
        num_sub=1,
        monkeypatch=monkeypatch,
    )
    assert len(pages) == 2
    assert sorted(page["ylabel"] for page in pages) == [
        "$q_lv$ $[kPa]$",
        "$u_ar$ $[kPa]$",
    ]


def test_a_trace_measured_on_two_subexperiments_is_one_page(tmp_path, monkeypatch):
    """The issue's "even across different subexperiments"."""
    pages, _, _ = _run(
        tmp_path,
        [("u_ar_max_sub0", "u_ar", 0), ("u_ar_max_sub1", "u_ar", 1)],
        num_sub=2,
        monkeypatch=monkeypatch,
    )
    assert len(pages) == 1


def test_both_subexperiment_segments_are_drawn(tmp_path, monkeypatch):
    """Merging the pages is only half of it -- the combined trace has to be drawn.

    A single "already plotted" flag per page would draw the first segment and silently
    drop the second, which looks like a correct plot of the wrong window.
    """
    pages, _, _ = _run(
        tmp_path,
        [("u_ar_max_sub0", "u_ar", 0), ("u_ar_max_sub1", "u_ar", 1)],
        num_sub=2,
        monkeypatch=monkeypatch,
    )
    outputs = [lab for lab in pages[0]["labels"] if lab == "output"]
    segments = [lab for lab in pages[0]["labels"] if lab in ("output", "_nolegend_")]
    assert len(segments) == 2, "expected one trace segment per subexperiment"
    assert len(outputs) == 1, "the trace should take one legend entry, not one per segment"


def test_the_x_axis_spans_every_subexperiment_the_trace_was_measured_on(
    tmp_path, monkeypatch
):
    pages, _, _ = _run(
        tmp_path,
        [("u_ar_max_sub0", "u_ar", 0), ("u_ar_max_sub1", "u_ar", 1)],
        num_sub=2,
        monkeypatch=monkeypatch,
    )
    lo, hi = pages[0]["xlim"]
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(2 * SUB_SIM_TIME)


def test_a_subexperiment_nothing_was_measured_on_is_still_left_out(
    tmp_path, monkeypatch
):
    """The #474 behaviour this must not undo: a settle window the ground truth says
    nothing about does not stretch the axis."""
    pages, _, _ = _run(
        tmp_path,
        [("u_ar_max_sub1", "u_ar", 1)],
        num_sub=2,
        monkeypatch=monkeypatch,
    )
    lo, hi = pages[0]["xlim"]
    assert lo == pytest.approx(SUB_SIM_TIME)
    assert hi == pytest.approx(2 * SUB_SIM_TIME)


def test_an_item_with_no_trace_name_falls_back_to_its_own_name(tmp_path, monkeypatch):
    """An item built from other items has no operand, so its trace name defaults to its
    own name -- it gets a page of its own rather than joining an unrelated group.

    The parser fills this in (``default_trace_names_for_plotting``), so an empty one only
    reaches here from a hand-assembled obs_info. It still must not collapse every such item
    onto one nameless page, nor label that page '$$'.
    """
    pages, _, _ = _run(
        tmp_path,
        [("u_ar_max", "u_ar", 0), ("difference", "", 0)],
        num_sub=1,
        monkeypatch=monkeypatch,
    )
    assert len(pages) == 2
    assert sorted(page["ylabel"] for page in pages) == [
        "$difference$ $[kPa]$",
        "$u_ar$ $[kPa]$",
    ]


def test_the_error_vectors_stay_one_entry_per_item(tmp_path, monkeypatch):
    """Pages are grouped; the error vectors are not. Entry i still belongs to
    data_items[i], which every consumer of these artefacts depends on (#341)."""
    items = [("u_ar_max", "u_ar", 0), ("u_ar_min", "u_ar", 0), ("q_lv_mean", "q_lv", 0)]
    pages, percent, std = _run(tmp_path, items, num_sub=1, monkeypatch=monkeypatch)
    assert len(pages) == 2
    assert percent.shape == (len(items),)
    assert std.shape == (len(items),)
