"""What ``process_obs_info`` publishes, and who may rely on it.

``tests/test_obs_data_vocabulary.py`` answers a different question -- how keys are *spelled*,
and how an old spelling migrates to a new one. This answers what the parser's output surface
*is*. Its reader is somebody about to rename a key, not somebody about to write an obs_data
file.

It exists because that rename happened (#507) and nothing fast caught it. The published
``obs_info`` is indexed by string literal around three hundred times inside this repository
and thirty-one times inside CUFLynx, with no schema and no version, so the first thing to
notice a key going missing was an hour-long CI job that downloads the last CUFLynx *release*.

Two properties are pinned, and the second is the one that catches real bugs:

**Presence.** Every key below is published unconditionally -- none is content-dependent, which
is worth stating because it is not obvious from the source.

**Index space.** A value is indexed by the data_item row, or by a *compacted* per-type counter,
and the two are not interchangeable. ``obs_cost.py`` in CUFLynx indexes ``experiment_idxs`` by
the row and ``ground_truth_const`` by the compacted constant counter inside one loop body; read
with the wrong counter, a value silently belongs to a different observable. That is CA #349,
and ``tests/test_cost_weight_indexing.py`` pins it for ``protocol_info``.

Deliberately NOT pinned: dtype. ``weight_phase_vec`` is ``dtype=object`` when a frequency
item's weight is a list, and ``ground_truth_phase`` is ``object`` when any frequency item omits
``phase``. Asserting dtype is how this file would become flaky and then ignored.
"""
import tempfile

import numpy as np
import pytest

from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.utilities.obs_data_helpers import (LEGACY_OBS_INFO_KEYS,
                                                   LEGACY_PREDICTION_INFO_KEYS)

# Index spaces. ITEM is the data_item row; the rest are compacted per-type counters.
ITEM, CONST, SERIES, FREQ, SCALAR = "item", "const", "series", "freq", "scalar"

#: key -> (index space, tier, what it is and who reads it)
#:
#: Tier A -- a libcuflynx accessor already covers it, so a rename is invisible to anything
#:           that uses the accessor.
#: Tier B -- public, no accessor. Renaming one needs a LEGACY_OBS_INFO_KEYS entry, a CHANGELOG
#:           line, and a sweep of the downstream readers. **These 18 are the real contract.**
#: Tier C -- published, but internal by intent. Nothing outside libcuflynx should read them.
OBS_INFO_CONTRACT = {
    # --- per data_item ---
    "data_item_names": (ITEM, "A", "the item's identity, and what an operation_kwargs "
                        "reference to another item resolves against (#466). Accessor: "
                        "obs_item_names. CUFLynx emulator bundles are keyed on it."),
    "item_names_for_plotting": (ITEM, "A", "the scalar feature's own label. Accessor: "
                                "obs_item_labels. CUFLynx labels its cost and sensitivity "
                                "panels with it."),
    "trace_names_for_plotting": (ITEM, "A", "the axis label of the trace the item reduces. "
                                 "Accessor: obs_trace_labels. May repeat -- the mean and the "
                                 "max of one trace share it."),
    "operands": (ITEM, "A", "the model variables each item reduces, as a list per item. "
                 "Accessor: obs_operand_lists. Handed to the solver as result variables."),
    "data_types": (ITEM, "B", "'constant' | 'series' | 'frequency'; decides which compacted "
                   "index space an item also appears in."),
    "units": (ITEM, "B", "each item's unit, for axis labels and error reports."),
    "experiment_idxs": (ITEM, "B", "which experiment each item belongs to. CUFLynx "
                        "obs_cost.py and local_sensitivity.py both walk it."),
    "subexperiment_idxs": (ITEM, "B", "which sub-experiment each item belongs to."),
    "operations": (ITEM, "B", "the reduction applied to each item's trace, or None."),
    "cost_type": (ITEM, "B", "the cost function per item. NOTE: paramID rewrites this in "
                  "place for Bayesian runs, so assert it on the parser's output only."),
    "operation_kwargs": (ITEM, "C", "per-item operation arguments; read through "
                         "resolve_operation_kwargs, not directly."),
    "cost_kwargs": (ITEM, "C", "per-item cost arguments; read through pid._cost_kwargs_for."),
    "freqs": (ITEM, "C", "the frequency grid for a frequency item, None otherwise."),
    "plot_colors": (ITEM, "C", "per-item plot colour. Currently always None -- see "
                    "test_plot_colors_never_gets_its_default."),
    "plot_type": (ITEM, "C", "how a constant is drawn over a trace."),
    "ground_truth_prob_dist_params": (ITEM, "C", "distribution parameters for a "
                                      "distribution-costed item."),
    # --- scalar ---
    "num_obs": (SCALAR, "B", "how many data_items there are; the length of every ITEM key."),
    # --- compacted: constants ---
    "ground_truth_const": (CONST, "B", "the measured value of each constant item."),
    "std_const_vec": (CONST, "B", "the std each constant is scored against."),
    "const_idx_to_obs_idx": (CONST, "B", "compacted constant index -> data_item row. CUFLynx "
                             "obs_cost.py and local_sensitivity.py walk this to label the "
                             "cost panel; without it the panel goes silent."),
    "weight_const_vec": (CONST, "C", "pre-scaling weights. Downstream reads "
                         "protocol_info['scaled_weight_const_from_exp_sub'] instead (#349)."),
    # --- compacted: series ---
    "ground_truth_series": (SERIES, "B", "the measured trace of each series item."),
    "std_series_vec": (SERIES, "B", "the std each series is scored against."),
    "series_idx_to_obs_idx": (SERIES, "B", "compacted series index -> data_item row."),
    "obs_dt": (SERIES, "B", "the sample spacing of each series, in seconds. plot_outputs "
               "indexes it by series_idx."),
    "weight_series_vec": (SERIES, "C", "pre-scaling weights; see weight_const_vec."),
    # --- compacted: frequencies ---
    "ground_truth_amp": (FREQ, "B", "the measured amplitude spectrum of each frequency item."),
    "ground_truth_phase": (FREQ, "B", "the measured phase spectrum, where one was given."),
    "std_amp_vec": (FREQ, "B", "the std each amplitude is scored against."),
    "freq_idx_to_obs_idx": (FREQ, "B", "compacted frequency index -> data_item row."),
    "weight_amp_vec": (FREQ, "C", "pre-scaling weights; see weight_const_vec."),
    "weight_phase_vec": (FREQ, "C", "pre-scaling weights; see weight_const_vec."),
}

#: prediction_info is simpler: one definition (_empty_prediction_info), six parallel keys, and
#: the dict itself may legitimately be None. Nothing in CUFLynx reads any of it.
PREDICTION_INFO_CONTRACT = {
    "operands": "the model qnames to record, as a list per item -- not a flat list (#507).",
    "units": "each prediction's unit.",
    "data_item_names": "each prediction's identity, unique across data_items too.",
    "trace_names_for_plotting": "the axis label.",
    "item_names_for_plotting": "the item's own label.",
    "experiment_idxs": "which experiment each prediction belongs to.",
}


# ---------------------------------------------------------------------------
# Fixtures. Built through the parser, never by hand: process_obs_info cannot take a
# hand-assembled gt_df (it reads columns the schema adds), and `output_dir` is required
# because get_ground_truth_values writes .npy unconditionally.
# ---------------------------------------------------------------------------
def _const(name):
    return {"data_item_name": f"a/{name}", "data_type": "constant", "unit": "mV",
            "operands": [f"a/{name}"], "value": 5.0, "std": 1.0, "weight": 1.0,
            "operation": "mean", "trace_name_for_plotting": name}


def _series(name):
    return {"data_item_name": f"s/{name}", "data_type": "series", "unit": "mV",
            "operands": [f"s/{name}"], "value": [1.0] * 5, "std": [1.0] * 5,
            "weight": 1.0, "obs_dt": 0.1, "trace_name_for_plotting": name}


def _freq(name):
    return {"data_item_name": f"f/{name}", "data_type": "frequency", "unit": "m3_per_s",
            "operands": [f"f/{name}"], "operation": "None", "cost_type": "gaussian_MLE",
            "value": [1e-4, 5e-4, 0.0], "std": [0.1, 0.1, 0.1], "weight": [1.0, 1.0, 1.0],
            "frequencies": [0.0, 1.0, 2.0], "phase": [0.0, 1.0, 0.0],
            "trace_name_for_plotting": name}


#: Interleaved on purpose. If the items were grouped by type, a value read with the wrong
#: counter could still land on the right row by coincidence, and the index-space test below
#: would pass while the invariant it exists for was broken.
MIXED_ITEMS = [_const("c0"), _series("s0"), _freq("f0"), _const("c1"), _series("s1"),
               _const("c2")]


@pytest.fixture(scope="module")
def mixed():
    """A parsed obs_info with all four index spaces populated, and their expected sizes."""
    parser = ObsAndParamDataParser()
    doc = {"data_items": MIXED_ITEMS,
           "protocol_info": {"pre_times": [0.0], "sim_times": [[1.0]]}}
    parsed = parser.parse_obs_data_json(obs_data_dict=doc, pre_time=0.0, sim_time=1.0)
    with tempfile.TemporaryDirectory() as out_dir:
        obs_info = parser.process_obs_info(gt_df=parsed["gt_df"], output_dir=out_dir, dt=0.01)
    counts = {
        ITEM: len(MIXED_ITEMS),
        CONST: sum(1 for i in MIXED_ITEMS if i["data_type"] == "constant"),
        SERIES: sum(1 for i in MIXED_ITEMS if i["data_type"] == "series"),
        FREQ: sum(1 for i in MIXED_ITEMS if i["data_type"] == "frequency"),
    }
    return obs_info, counts


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(OBS_INFO_CONTRACT))
def test_obs_info_publishes(key, mixed):
    """Parametrised so the pytest id names the key -- a CI summary line is enough to see
    which one went, without opening a log."""
    obs_info, _ = mixed
    space, tier, why = OBS_INFO_CONTRACT[key]
    remedy = (
        f"Tier {tier}: add {key!r} to LEGACY_OBS_INFO_KEYS mapping it to its replacement, "
        f"add a CHANGELOG entry, and sweep the downstream readers (CUFLynx included)."
        if tier in "AB" else
        f"Tier {tier} (internal): if the removal is deliberate, delete the row from "
        f"OBS_INFO_CONTRACT in this file."
    )
    assert key in obs_info, f"process_obs_info no longer publishes {key!r}. {why}\n{remedy}"


@pytest.mark.unit
def test_obs_info_publishes_nothing_undocumented(mixed):
    """The other half. Without this the table rots into a subset of reality and stops
    describing the surface it claims to."""
    obs_info, _ = mixed
    undocumented = sorted(set(obs_info) - set(OBS_INFO_CONTRACT))
    assert not undocumented, (
        f"process_obs_info publishes {undocumented}, which OBS_INFO_CONTRACT does not "
        f"describe. Add a row for each saying its index space (item/const/series/freq/scalar) "
        f"and whether anything outside libcuflynx may read it.")


# ---------------------------------------------------------------------------
# Index space -- the invariant that breaks things silently
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(OBS_INFO_CONTRACT))
def test_obs_info_index_spaces(key, mixed):
    obs_info, counts = mixed
    space, _tier, why = OBS_INFO_CONTRACT[key]
    value = obs_info[key]

    if space == SCALAR:
        assert isinstance(value, int) and value == counts[ITEM], (
            f"{key!r} should be the data_item count ({counts[ITEM]}), got {value!r}. {why}")
        return

    assert isinstance(value, (list, np.ndarray)), (
        f"{key!r} should be a sequence, got {type(value).__name__}. {why}")
    assert len(value) == counts[space], (
        f"{key!r} is indexed by {space}, so for this obs_data it must have "
        f"{counts[space]} entries, not {len(value)}. {why}\n"
        f"Mixing index spaces is CA #349: a value read with the wrong counter is silently "
        f"a different observable's.")


@pytest.mark.unit
def test_the_compacted_indexes_point_at_the_right_rows(mixed):
    """The maps are the bridge between the spaces, so they are worth checking directly
    rather than only by length."""
    obs_info, _ = mixed
    types = obs_info["data_types"]
    for space, key in ((CONST, "const_idx_to_obs_idx"),
                       (SERIES, "series_idx_to_obs_idx"),
                       (FREQ, "freq_idx_to_obs_idx")):
        expected = {CONST: "constant", SERIES: "series", FREQ: "frequency"}[space]
        rows = list(obs_info[key])
        assert rows == sorted(rows), f"{key} should be ascending, got {rows}"
        for obs_idx in rows:
            assert types[obs_idx] == expected, (
                f"{key} points at row {obs_idx}, whose data_type is {types[obs_idx]!r} "
                f"rather than {expected!r}.")


# ---------------------------------------------------------------------------
# prediction_info
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_prediction_info_publishes_its_contract():
    parser = ObsAndParamDataParser()
    doc = {"data_items": [_const("c0")],
           "prediction_items": [{"data_item_name": "p/y", "operands": ["main/y"],
                                 "unit": "mV", "trace_name_for_plotting": "y"}],
           "protocol_info": {"pre_times": [0.0], "sim_times": [[1.0]]}}
    parsed = parser.parse_obs_data_json(obs_data_dict=doc, pre_time=0.0, sim_time=1.0)
    pred = parsed["prediction_info"]

    assert set(pred) == set(PREDICTION_INFO_CONTRACT), (
        f"prediction_info's key set changed: "
        f"missing {sorted(set(PREDICTION_INFO_CONTRACT) - set(pred))}, "
        f"unexpected {sorted(set(pred) - set(PREDICTION_INFO_CONTRACT))}")
    assert len({len(v) for v in pred.values()}) == 1, (
        f"prediction_info's columns must stay parallel, got "
        f"{ {k: len(v) for k, v in pred.items()} }")
    assert pred["operands"] == [["main/y"]], (
        "operands is a list *per item*, not a flat qname list -- that reshape is the "
        "substance of #507 and normalise_prediction_info has bespoke code for it.")


# ---------------------------------------------------------------------------
# The link to the migration tables
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_published_key_is_a_retired_spelling():
    """The two halves of a rename. obs_data_helpers says the old name is retired; this says
    the parser stopped emitting it. #507 changed the first, and the second was only true by
    accident -- nothing asserted it."""
    assert set(OBS_INFO_CONTRACT) & set(LEGACY_OBS_INFO_KEYS) == set()
    assert set(PREDICTION_INFO_CONTRACT) & set(LEGACY_PREDICTION_INFO_KEYS) == set()


# ---------------------------------------------------------------------------
# Entry points the docstrings promise, and one that does not work
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_hand_built_gt_df_is_accepted(tmp_path):
    """``process_obs_info`` calls itself a public entry point a caller may hand its own
    frame. It read ``prob_dist_params`` with a bare ``[...]``, and the schema adds that
    column -- so every hand-built frame raised KeyError and the invitation was untrue."""
    import pandas as pd

    df = pd.DataFrame([{"data_item_name": "a/x", "data_type": "constant", "unit": "mV",
                        "operands": ["a/x"], "value": 1.0, "std": 1.0, "weight": 1.0,
                        "operation": "mean", "experiment_idx": 0, "subexperiment_idx": 0,
                        "cost_type": "gaussian_MLE"}])
    obs_info = ObsAndParamDataParser().process_obs_info(
        gt_df=df, output_dir=str(tmp_path), dt=0.01)
    assert obs_info["num_obs"] == 1
    assert obs_info["data_item_names"] == ["a/x"]


@pytest.mark.unit
def test_an_obs_data_with_only_protocol_info_still_parses(tmp_path):
    """A study with no observables is a legitimate thing to open -- it just cannot be
    scored. The empty frame carried no columns at all, so every read in process_obs_info
    was a KeyError and such a file could not be loaded."""
    parser = ObsAndParamDataParser()
    doc = {"data_items": [], "protocol_info": {"pre_times": [0.0], "sim_times": [[1.0]]}}
    parsed = parser.parse_obs_data_json(obs_data_dict=doc, pre_time=0.0, sim_time=1.0)
    obs_info = parser.process_obs_info(
        gt_df=parsed["gt_df"], output_dir=str(tmp_path), dt=0.01)

    assert obs_info["num_obs"] == 0
    # Still the whole surface -- a reader should not have to special-case the empty study.
    assert set(obs_info) == set(OBS_INFO_CONTRACT)


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason=(
    "plot_colors can never be anything but None: the schema creates the column with value "
    "None for every row, so the `.get(\"plot_color\", <cycle>)` default in process_obs_info "
    "is never reached. Recorded rather than fixed -- making the cycle work would change the "
    "colours of every existing plot, which is a decision, not a bugfix."))
def test_plot_colors_never_gets_its_default(mixed):
    obs_info, _ = mixed
    assert any(c is not None for c in obs_info["plot_colors"])
