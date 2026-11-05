"""Tests for protocol_shapes -- Myokit-style pacing events as an alternative to
hand-written protocol_traces.

The interesting assertions are the ones tying the expansion to Myokit itself: a
shape written with the same five numbers as a .mmt [[protocol]] line has to
produce the same waveform Myokit would, or the two representations agree only by
coincidence.
"""

import numpy as np
import pytest
from bisect import bisect_right as _bisect

from libcuflynx.utilities.protocol_shapes import (
    ProtocolShapeError,
    expand_shape,
    materialise_shapes,
    normalise_shape,
    validate_trace_references,
)

# The stimulus from resources/br-1977.mmt: 1 unit, 2 long, every 1000, forever.
STIM = {"level": 1.0, "start": 100, "length": 2, "period": 1000, "multiplier": 0}


def _shape(**over):
    event = dict(STIM)
    event.update(over)
    return {"events": [event]}


def _protocol(shape=None, sim_times=None, **extra):
    info = {
        "pre_times": [0.0],
        "sim_times": sim_times or [[2000.0]],
        "params_to_change": {"engine/pace": [["stim"]]},
        "protocol_shapes": {"stim": shape if shape is not None else _shape()},
    }
    info.update(extra)
    return info


def _levels_at(trace, times):
    """Sample the trace the way myokit.TimeSeriesProtocol would: linearly."""
    import numpy as np

    return list(np.interp(times, trace["t"], trace["values"]))


# ---------------------------------------------------------------------------
# The waveform
# ---------------------------------------------------------------------------
def test_a_periodic_event_repeats_for_as_long_as_the_subexperiment_runs():
    info = materialise_shapes(_protocol())
    trace = info["protocol_traces"]["stim"]
    # Two beats in 2000 with a period of 1000.
    assert _levels_at(trace, [50, 101, 500, 1101, 1500]) == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_the_stimulus_sits_exactly_where_the_event_says():
    trace = materialise_shapes(_protocol())["protocol_traces"]["stim"]
    assert _levels_at(trace, [99.9, 100.5, 101.9, 102.5]) == [0.0, 1.0, 1.0, 0.0]


def test_a_multiplier_bounds_the_repeats():
    info = materialise_shapes(_protocol(_shape(multiplier=1)))
    trace = info["protocol_traces"]["stim"]
    assert _levels_at(trace, [101, 1101]) == [1.0, 0.0]  # first beat only


def test_no_period_means_a_single_event():
    info = materialise_shapes(_protocol(_shape(period=0, multiplier=0)))
    trace = info["protocol_traces"]["stim"]
    assert _levels_at(trace, [101, 1101]) == [1.0, 0.0]


def test_the_baseline_is_the_value_outside_every_event():
    info = materialise_shapes(_protocol({"events": [STIM], "baseline": -80.0}))
    trace = info["protocol_traces"]["stim"]
    assert _levels_at(trace, [50, 101, 500]) == [-80.0, 1.0, -80.0]


def test_the_trace_covers_the_whole_subexperiment():
    trace = materialise_shapes(_protocol())["protocol_traces"]["stim"]
    assert trace["t"][0] == 0.0
    assert trace["t"][-1] == pytest.approx(2000.0)


def test_the_times_strictly_increase():
    """myokit.TimeSeriesProtocol requires it, and a duplicate instant is what a
    naive square-wave expansion produces."""
    trace = materialise_shapes(_protocol())["protocol_traces"]["stim"]
    assert all(b > a for a, b in zip(trace["t"], trace["t"][1:]))


def test_edges_stay_short_next_to_a_brief_stimulus():
    """TimeSeriesProtocol interpolates linearly, so an edge is a ramp. A 2-long
    stimulus inside a 2000-long beat must not be ramped away."""
    trace = materialise_shapes(_protocol())["protocol_traces"]["stim"]
    rise = [b - a for a, b in zip(trace["t"], trace["t"][1:])]
    assert min(rise) <= 2 * 1e-3 * 1.01
    # Full amplitude is reached well inside the event, not at its end.
    assert _levels_at(trace, [100.1]) == [1.0]


def test_several_events_in_one_shape():
    """A .mmt [[protocol]] table can have many lines."""
    shape = {
        "events": [
            {"level": 1.0, "start": 100, "length": 2, "period": 0, "multiplier": 0},
            {"level": 2.0, "start": 500, "length": 5, "period": 0, "multiplier": 0},
        ]
    }
    trace = materialise_shapes(_protocol(shape))["protocol_traces"]["stim"]
    assert _levels_at(trace, [101, 300, 502, 800]) == [1.0, 0.0, 2.0, 0.0]


# ---------------------------------------------------------------------------
# Agreement with Myokit -- the point of using its vocabulary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "event,duration",
    [
        (STIM, 2000.0),
        ({"level": 1.0, "start": 100, "length": 2, "period": 1000, "multiplier": 3}, 5000.0),
        ({"level": -2.5, "start": 0, "length": 0.5, "period": 30, "multiplier": 0}, 120.0),
        ({"level": 1.0, "start": 10, "length": 5, "period": 0, "multiplier": 0}, 50.0),
    ],
)
def test_the_waveform_is_the_one_myokit_would_produce(event, duration):
    """Same five numbers, same stimulus. Built independently: Myokit expands its
    own Protocol, we expand ours, and the two are sampled at the same instants."""
    myokit = pytest.importorskip("myokit")

    protocol = myokit.Protocol()
    protocol.schedule(
        event["level"], event["start"], event["length"], event["period"], event["multiplier"]
    )
    log_for_interval = getattr(protocol, "log_for_interval", None)
    if log_for_interval is None:  # older Myokit
        log_for_interval = protocol.create_log_for_interval
    reference = log_for_interval(0, duration, for_drawing=False)

    ours = expand_shape(normalise_shape({"events": [event]}, name="s"), duration, name="s")

    # Sample midway through every stretch Myokit reports, where both
    # representations are unambiguously at their held value. Myokit's log is a
    # list of change points -- a step function, held forward -- so it is sampled
    # as one; ours is a point table for linear interpolation.
    times = list(reference["time"])
    levels = list(reference["pace"])
    midpoints = [(a + b) / 2 for a, b in zip(times, times[1:])]
    held = [levels[max(0, _bisect(times, t) - 1)] for t in midpoints]

    assert midpoints, "the reference protocol produced no stretches to compare"
    assert _levels_at(ours, midpoints) == pytest.approx(held, abs=1e-9)


# ---------------------------------------------------------------------------
# Sizing: a shape takes its length from the sub-experiment that uses it
# ---------------------------------------------------------------------------
def test_the_shape_is_sized_by_the_subexperiment_that_uses_it():
    info = materialise_shapes(_protocol(sim_times=[[5000.0]]))
    assert info["protocol_traces"]["stim"]["t"][-1] == pytest.approx(5000.0)
    # 5000 with a period of 1000 is five beats.
    assert _levels_at(info["protocol_traces"]["stim"], [101, 1101, 2101, 3101, 4101]) == [1.0] * 5


def test_an_explicit_duration_overrides_the_subexperiment():
    shape = {"events": [STIM], "duration": 1500.0}
    info = materialise_shapes(_protocol(shape))
    assert info["protocol_traces"]["stim"]["t"][-1] == pytest.approx(1500.0)


def test_a_shape_used_at_two_different_lengths_is_refused():
    """Silently picking one of them would give a protocol that is wrong in half
    the experiments and looks right in the file."""
    info = _protocol(sim_times=[[2000.0], [3000.0]])
    info["pre_times"] = [0.0, 0.0]
    info["params_to_change"] = {"engine/pace": [["stim"], ["stim"]]}
    with pytest.raises(ProtocolShapeError, match="different lengths"):
        materialise_shapes(info)


def test_a_shape_nobody_uses_is_refused_rather_than_guessed_at():
    info = _protocol()
    info["params_to_change"] = {}
    with pytest.raises(ProtocolShapeError, match="never used"):
        materialise_shapes(info)


# ---------------------------------------------------------------------------
# Shapes and traces are alternatives
# ---------------------------------------------------------------------------
def test_a_name_defined_as_both_a_shape_and_a_trace_is_refused():
    info = _protocol()
    info["protocol_traces"] = {"stim": {"t": [0, 1], "values": [0, 1]}}
    with pytest.raises(ProtocolShapeError, match="both"):
        materialise_shapes(info)


def test_hand_written_traces_are_left_alone():
    info = _protocol()
    info["protocol_traces"] = {"other": {"t": [0.0, 1.0], "values": [0.0, 1.0]}}
    out = materialise_shapes(info)
    assert out["protocol_traces"]["other"] == {"t": [0.0, 1.0], "values": [0.0, 1.0]}
    assert "stim" in out["protocol_traces"]


def test_expanding_twice_changes_nothing():
    """protocol_info is passed between the parser, the runner and the helper, so
    each of them expanding it must be safe."""
    first = materialise_shapes(_protocol())
    once = dict(first["protocol_traces"]["stim"])
    twice = materialise_shapes(first)["protocol_traces"]["stim"]
    assert twice == once


def test_a_protocol_with_no_shapes_is_untouched():
    info = {"pre_times": [0.0], "sim_times": [[1.0]], "params_to_change": {"a/b": [[1.0]]}}
    assert materialise_shapes(dict(info)) == info


# ---------------------------------------------------------------------------
# Refusals -- each names the field it is complaining about
# ---------------------------------------------------------------------------
def test_length_and_duration_are_the_same_field_under_two_names():
    """`Length` is the .mmt column, `duration` is Myokit's Python API."""
    a = expand_shape(normalise_shape({"events": [STIM]}, name="s"), 2000.0, name="s")
    b_event = {k: v for k, v in STIM.items() if k != "length"}
    b_event["duration"] = STIM["length"]
    b = expand_shape(normalise_shape({"events": [b_event]}, name="s"), 2000.0, name="s")
    assert a == b


def test_giving_both_length_and_duration_is_refused():
    event = dict(STIM)
    event["duration"] = 2
    with pytest.raises(ProtocolShapeError, match="only one"):
        normalise_shape({"events": [event]}, name="s")


def test_a_missing_length_is_refused():
    event = {k: v for k, v in STIM.items() if k != "length"}
    with pytest.raises(ProtocolShapeError, match="needs a 'length'"):
        normalise_shape({"events": [event]}, name="s")


def test_a_missing_level_is_refused():
    event = {k: v for k, v in STIM.items() if k != "level"}
    with pytest.raises(ProtocolShapeError, match="needs a 'level'"):
        normalise_shape({"events": [event]}, name="s")


def test_a_misspelled_field_is_named_rather_than_ignored():
    event = dict(STIM)
    event["lenght"] = 2
    with pytest.raises(ProtocolShapeError, match="lenght"):
        normalise_shape({"events": [event]}, name="s")


def test_an_unknown_shape_type_is_refused():
    with pytest.raises(ProtocolShapeError, match="expected one of"):
        normalise_shape({"type": "sine", "events": [STIM]}, name="s")


def test_an_empty_event_list_is_refused():
    with pytest.raises(ProtocolShapeError, match="non-empty"):
        normalise_shape({"events": []}, name="s")


def test_overlapping_events_are_refused_as_myokit_refuses_them():
    shape = {
        "events": [
            {"level": 1.0, "start": 100, "length": 50, "period": 0},
            {"level": 2.0, "start": 120, "length": 10, "period": 0},
        ]
    }
    with pytest.raises(ProtocolShapeError, match="overlap"):
        materialise_shapes(_protocol(shape))


def test_a_stimulus_that_never_fires_in_the_run_is_refused():
    """A flat line that looks like a protocol and does nothing is the failure
    that is hardest to notice."""
    with pytest.raises(ProtocolShapeError, match="fires nothing"):
        materialise_shapes(_protocol(_shape(start=5000)))


def test_a_negative_length_is_refused():
    with pytest.raises(ProtocolShapeError, match="must last some time"):
        normalise_shape({"events": [dict(STIM, length=-1)]}, name="s")


def test_a_fractional_multiplier_is_refused():
    with pytest.raises(ProtocolShapeError, match="whole number"):
        normalise_shape({"events": [dict(STIM, multiplier=1.5)]}, name="s")


def test_repeats_without_a_period_are_refused():
    with pytest.raises(ProtocolShapeError, match="no period"):
        normalise_shape({"events": [dict(STIM, period=0, multiplier=3)]}, name="s")


def test_a_bare_list_of_events_is_accepted_as_the_mmt_table_shorthand():
    shape = normalise_shape([STIM], name="s")
    assert shape["events"][0]["level"] == 1.0


# ---------------------------------------------------------------------------
# Dangling references
# ---------------------------------------------------------------------------
def test_a_reference_to_nothing_is_caught_before_the_solver_reaches_it():
    info = {
        "pre_times": [0.0],
        "sim_times": [[1.0]],
        "params_to_change": {"engine/pace": [["typo"]]},
        "protocol_traces": {"stim": {"t": [0, 1], "values": [0, 1]}},
    }
    with pytest.raises(ProtocolShapeError, match="typo"):
        validate_trace_references(info)


def test_valid_references_pass():
    info = materialise_shapes(_protocol())
    validate_trace_references(info)  # must not raise


# ---------------------------------------------------------------------------
# Through the obs_data parser
# ---------------------------------------------------------------------------
def test_the_parser_accepts_protocol_shapes_and_returns_traces(tmp_path):
    import json

    import libcuflynx.parsers.PrimitiveParsers as primitive_parsers

    obs = {
        "protocol_info": {
            "pre_times": [0.0],
            "sim_times": [[2000.0]],
            "params_to_change": {"engine/pace": [["stim"]]},
            "protocol_shapes": {"stim": {"events": [STIM]}},
        },
        "data_items": [],
    }
    path = tmp_path / "obs_data.json"
    path.write_text(json.dumps(obs))

    parser = primitive_parsers.ObsAndParamDataParser()
    result = parser.parse_obs_data_json(param_id_obs_path=str(path))
    protocol_info = result[1] if isinstance(result, tuple) else result["protocol_info"]
    assert "stim" in protocol_info["protocol_traces"]
    assert protocol_info["protocol_traces"]["stim"]["t"][-1] == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# End to end: a shape and the sub-experiments it replaces must simulate alike
# ---------------------------------------------------------------------------
PACED_MMT = """[[model]]
name: paced
membrane.V = -80

[engine]
time = 0 bind time
pace = 0 bind pace

[membrane]
dot(V) = -0.05 * V + 50 * engine.pace
"""


@pytest.fixture
def paced_cellml(tmp_path):
    """A minimal model driven by `pace`, exported to CellML for the solver."""
    myokit = pytest.importorskip("myokit")
    from myokit.formats.cellml import CellML2Exporter

    mmt = tmp_path / "paced.mmt"
    mmt.write_text(PACED_MMT)
    model, _, _ = myokit.load(str(mmt))
    path = tmp_path / "paced.cellml"
    CellML2Exporter().model(str(path), model)
    return str(path)


def _run(model_path, protocol_info):
    from libcuflynx.protocol_runners.protocol_runner import ProtocolRunner

    inp = {"dt": 0.1, "solver_info": {"MaximumStep": 0.01}, "model_type": "cellml"}
    runner = ProtocolRunner(
        model_path=model_path, inp_data_dict=inp, solver="CVODE_myokit", model_type="cellml"
    )
    t_list, res_list, _ = runner.run_protocols(model_path, protocol_info=protocol_info)
    idx = runner.get_var2idx_dict()
    key = next(k for k in idx if k.endswith(".V") or k.endswith("/V"))
    return t_list[0], res_list[0][idx[key]]


@pytest.mark.integration
def test_a_shape_simulates_the_same_as_the_subexperiments_it_replaces(paced_cellml):
    """The claim the feature rests on. One sub-experiment driven by a shape and
    five sub-experiments holding the same levels are two spellings of one
    protocol; if they simulated differently, the shape would be a new protocol
    wearing the old one's numbers."""
    import numpy as np

    shaped = {
        "pre_times": [0.0],
        "sim_times": [[500.0]],
        "params_to_change": {"engine/pace": [["stim"]]},
        "protocol_shapes": {
            "stim": {
                "events": [
                    {"level": 1.0, "start": 100, "length": 2, "period": 200, "multiplier": 0}
                ]
            }
        },
    }
    expanded = {
        "pre_times": [0.0],
        "sim_times": [[100.0, 2.0, 198.0, 2.0, 198.0]],
        "params_to_change": {"engine/pace": [[0.0, 1.0, 0.0, 1.0, 0.0]]},
    }

    t_shape, v_shape = _run(paced_cellml, shaped)
    t_sub, v_sub = _run(paced_cellml, expanded)

    # Compared on a common grid: the two runs report at different instants
    # because the sub-experiment version restarts the solver at every boundary.
    grid = np.linspace(0, 500, 501)
    on_shape = np.interp(grid, t_shape, v_shape)
    on_sub = np.interp(grid, t_sub, v_sub)
    assert np.max(np.abs(on_shape - on_sub)) < 1e-3 * max(1.0, np.ptp(on_sub))


@pytest.mark.integration
def test_the_runner_reports_the_variables_it_returns_after_a_pace_rebind(paced_cellml):
    """Binding `pace` rebuilds the simulation and reorders the result rows. The
    runner's name->index map is built before that, so leaving it stale made every
    variable read as its neighbour -- a wrong answer with no error anywhere."""
    import numpy as np

    shaped = {
        "pre_times": [0.0],
        "sim_times": [[500.0]],
        "params_to_change": {"engine/pace": [["stim"]]},
        "protocol_shapes": {
            "stim": {"events": [{"level": 1.0, "start": 100, "length": 2, "period": 200}]}
        },
    }
    _t, v = _run(paced_cellml, shaped)
    # V starts at -80 and is driven upward by the stimulus; the pace variable
    # itself only ever sits between 0 and 1, so reading the wrong row is obvious.
    assert np.min(v) < -1.0
    assert np.max(v) > 1.0


# ---------------------------------------------------------------------------
# ramp -- the one editor shape that is not a square event
# ---------------------------------------------------------------------------
def test_a_ramp_sweeps_across_the_subexperiment():
    info = materialise_shapes(_protocol({"type": "ramp", "from": 0.0, "to": 5.0}))
    trace = info["protocol_traces"]["stim"]
    assert trace == {"t": [0.0, 2000.0], "values": [0.0, 5.0]}


def test_a_ramp_is_sized_by_its_subexperiment_too():
    info = materialise_shapes(
        _protocol({"type": "ramp", "from": 1.0, "to": 2.0}, sim_times=[[7.5]])
    )
    assert info["protocol_traces"]["stim"]["t"] == [0.0, 7.5]


def test_a_ramp_can_go_downwards():
    info = materialise_shapes(_protocol({"type": "ramp", "from": 10.0, "to": -10.0}))
    assert _levels_at(info["protocol_traces"]["stim"], [0, 1000, 2000]) == [10.0, 0.0, -10.0]


def test_a_ramp_needs_both_ends():
    with pytest.raises(ProtocolShapeError, match="needs 'to'"):
        normalise_shape({"type": "ramp", "from": 0.0}, name="s")


def test_a_ramp_rejects_the_pacing_fields_rather_than_ignoring_them():
    """Silently dropping 'events' would give a ramp where the user wrote a
    stimulus and expected one."""
    with pytest.raises(ProtocolShapeError, match="does not apply"):
        normalise_shape({"type": "ramp", "from": 0.0, "to": 1.0, "events": [STIM]}, name="s")


# ----------------------------------------------------------------------------------------
# Mixing a hand-written trace and a shape, and the one-paced-variable limit
# ----------------------------------------------------------------------------------------

def test_a_trace_and_a_shape_can_drive_different_variables():
    """One variable driven by a hand-written trace, another by a shape, in one protocol.

    The two forms are alternatives *per name*, not mutually exclusive per file: a name is
    defined in protocol_traces or protocol_shapes, but a protocol may use both for different
    variables. Expansion merges the shape-derived traces into whatever traces are already
    there, so params_to_change can refer to either kind the same way.
    """
    info = {
        "pre_times": [0.0],
        "sim_times": [[2000.0], [2000.0]],
        "params_to_change": {
            "engine/pace": [["stim"], [0.0]],        # sub-exp 0: shape-derived
            "membrane/i_ext": [[0.0], ["ramp"]],     # sub-exp 1: hand-written trace
        },
        "protocol_shapes": {"stim": _shape()},
        "protocol_traces": {"ramp": {"t": [0.0, 1000.0, 2000.0],
                                     "values": [0.0, 1.0, 0.0]}},
    }
    out = materialise_shapes(info)
    traces = out["protocol_traces"]

    assert set(traces) == {"stim", "ramp"}
    # the hand-written one is untouched
    assert traces["ramp"] == {"t": [0.0, 1000.0, 2000.0], "values": [0.0, 1.0, 0.0]}
    # the shape expanded into a real waveform, distinct from it
    assert traces["stim"]["t"] and traces["stim"] != traces["ramp"]
    # and neither reference is reported as dangling
    validate_trace_references(out)


def test_a_shape_and_a_trace_may_drive_the_same_variable_in_different_subexperiments():
    """Switching which waveform drives one variable between sub-experiments is allowed.

    Driving one variable from a shape in one sub-experiment and a hand-written trace in the
    next is allowed, as is driving several variables at once (see the Myokit tests below).
    """
    info = {
        "pre_times": [0.0],
        "sim_times": [[2000.0, 2000.0]],
        "params_to_change": {"engine/pace": [["stim", "ramp"]]},
        "protocol_shapes": {"stim": _shape()},
        "protocol_traces": {"ramp": {"t": [0.0, 2000.0], "values": [0.0, 1.0]}},
    }
    out = materialise_shapes(info)
    assert set(out["protocol_traces"]) == {"stim", "ramp"}
    validate_trace_references(out)


# dx/dt = a, dy/dt = b (issue #559)
_TWO_INPUTS_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="two_inputs">
  <component name="main">
    <variable name="t" units="second"/>
    <variable name="a" units="dimensionless" initial_value="0"/>
    <variable name="b" units="dimensionless" initial_value="0"/>
    <variable name="x" units="dimensionless" initial_value="0"/>
    <variable name="y" units="dimensionless" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><divide/><ci>a</ci><cn cellml:units="second">1</cn></apply></apply>
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>y</ci></apply>
        <apply><divide/><ci>b</ci><cn cellml:units="second">1</cn></apply></apply>
    </math>
  </component>
</model>"""

_TWO_TRACES = {"ta": {"t": [0, 0.5, 1.0], "values": [1, 1, 2]},
               "tb": {"t": [0, 0.5, 1.0], "values": [3, 3, 4]}}


def _two_input_helper(tmp_path):
    from libcuflynx.solver_wrappers.myokit_helper import SimulationHelper

    path = tmp_path / "two_inputs.cellml"
    path.write_text(_TWO_INPUTS_CELLML)
    return SimulationHelper(str(path), dt=0.1, sim_time=1.0,
                            solver_info={"MaximumStep": 0.01}, pre_time=0.0)


def _run_two_traced_inputs(helper):
    from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor

    protocol_info = {"pre_times": [0.0], "sim_times": [[1.0]],
                     "params_to_change": {"main/a": [["ta"]], "main/b": [["tb"]]},
                     "protocol_traces": _TWO_TRACES}
    helper.set_protocol_info(protocol_info)
    ok, by_sub, _, _ = ProtocolExecutor(helper).run_protocol(
        protocol_info, result_variables=["main/x", "main/y"])
    assert ok
    x, y = (np.ravel(np.asarray(r, dtype=float)) for r in by_sub[(0, 0)])
    return x, y


@pytest.mark.integration
@pytest.mark.solver
def test_myokit_drives_two_traced_inputs_in_one_subexperiment(tmp_path):
    """Each traced input follows its own trace, through its own binding label (#559)."""
    helper = _two_input_helper(tmp_path)
    x, y = _run_two_traced_inputs(helper)
    assert x[-1] == pytest.approx(1.25, abs=1e-3)
    assert y[-1] == pytest.approx(3.25, abs=1e-3)
    assert helper.trace_labels == {"main.a": "pace", "main.b": "pace_1"}


@pytest.mark.integration
@pytest.mark.solver
def test_imported_pace_binding_keeps_its_label(tmp_path):
    """A variable that arrives bound to ``pace`` (as a .mmt import's stimulus does) keeps
    that label; the other traced input takes the next free one."""
    helper = _two_input_helper(tmp_path)
    helper.qname_to_var["main.b"].set_binding("pace")
    x, y = _run_two_traced_inputs(helper)
    assert helper.trace_labels == {"main.a": "pace_1", "main.b": "pace"}
    assert x[-1] == pytest.approx(1.25, abs=1e-3)
    assert y[-1] == pytest.approx(3.25, abs=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("module_name,backend_label", [
    ("libcuflynx.solver_wrappers.casadi_python_solver_helper", "CasADi"),
    ("libcuflynx.solver_wrappers.aadc_python_solver_helper", "AADC"),
    ("libcuflynx.solver_wrappers.python_solver_helper", "python"),
])
def test_non_myokit_backends_refuse_protocol_traces_explicitly(module_name, backend_label):
    """Only the Myokit backend implements protocol_traces.

    The others previously assigned the trace *name* straight into their numeric parameter
    vector (CasADi and AADC) or named the wrong backend in the error (python said 'OpenCOR').
    A silently corrupted parameter vector surfaces somewhere numeric, far from the protocol
    that caused it, so each backend now refuses a string value where the mistake is made.
    """
    import importlib, inspect
    src = inspect.getsource(importlib.import_module(module_name).SimulationHelper.set_param_vals)
    assert "isinstance(val, str)" in src or "type(val) == str" in src, (
        f"{backend_label} backend does not check for a protocol trace name")
    assert "NotImplementedError" in src
    assert "CVODE_myokit" in src, (
        f"{backend_label} error should point at the solver that does support traces")


# dx/dt = u, y = u (issue #556)
_PROBE_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="probe">
  <units name="per_second"><unit units="second" exponent="-1"/></units>
  <component name="comp">
    <variable name="t" units="second"/>
    <variable name="u" units="dimensionless" initial_value="0"/>
    <variable name="y" units="dimensionless"/>
    <variable name="x" units="dimensionless" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><ci>y</ci><ci>u</ci></apply>
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><times/><ci>u</ci><cn cellml:units="per_second">1</cn></apply></apply>
    </math>
  </component>
</model>"""

_PULSE_SHAPE = {"events": [{"level": 1, "start": 1, "length": 1}]}  # u = 1 on [1, 2]
_PULSE_TRACE = {"t": [0.0, 0.999, 1.0, 2.0, 2.001, 4.0], "values": [0, 0, 1, 1, 0, 0]}
_RAMP_SHAPE = {"type": "ramp", "from": 0, "to": 4}  # u = t over the 4 s sub-experiment


def _probe_helper(tmp_path):
    from libcuflynx.solver_wrappers.myokit_helper import SimulationHelper

    path = tmp_path / "probe.cellml"
    path.write_text(_PROBE_CELLML)
    return SimulationHelper(str(path), dt=0.25, sim_time=4.0,
                            solver_info={"MaximumStep": 0.01}, pre_time=0.0)


def _run_probe(helper, protocol_info):
    from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor

    helper.set_protocol_info(protocol_info)
    ok, by_sub, _, t_by_exp = ProtocolExecutor(helper).run_protocol(
        protocol_info, result_variables=["comp/y", "comp/x"])
    assert ok
    u, x = (np.ravel(np.asarray(r, dtype=float)) for r in by_sub[(0, 0)])
    return np.asarray(t_by_exp[0], dtype=float), u, x


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.parametrize("kind", ["shape", "trace", "ramp"])
def test_first_subexperiment_input_is_timed_from_its_own_start(kind, tmp_path):
    """With a warm-up, an input's t = 0 is still the start of its sub-experiment (#556)."""
    info = {"pre_times": [2.0], "sim_times": [[4.0]], "params_to_change": {"comp/u": [["in"]]}}
    if kind == "trace":
        info["protocol_traces"] = {"in": _PULSE_TRACE}
    else:
        info["protocol_shapes"] = {"in": _RAMP_SHAPE if kind == "ramp" else _PULSE_SHAPE}
    t, u, x = _run_probe(_probe_helper(tmp_path), info)

    at = {float(v): i for i, v in enumerate(np.round(t, 6))}
    assert x[0] == pytest.approx(0.0, abs=1e-6)  # the warm-up held u at its first value, 0
    if kind == "ramp":
        assert u[at[1.5]] == pytest.approx(1.5, abs=1e-6)
        assert x[-1] == pytest.approx(8.0, abs=1e-3)
    else:
        assert u[at[0.5]] == pytest.approx(0.0, abs=1e-6)
        assert u[at[1.5]] == pytest.approx(1.0, abs=1e-6)
        assert x[-1] == pytest.approx(1.0, abs=1e-2)


@pytest.mark.integration
@pytest.mark.solver
def test_offline_pre_time_holds_traced_input_at_first_value(tmp_path):
    """The offline warm-up drives a traced input at experiment 0's first value, not at 0."""
    info = {"pre_times": [0.0], "sim_times": [[1.0]], "params_to_change": {"comp/u": [["in"]]},
            "protocol_traces": {"in": {"t": [0.0, 1.0], "values": [1.0, 1.0]}}}
    helper = _probe_helper(tmp_path)
    helper.set_protocol_info(info)
    helper.run_offline_pre_and_set_default_state(3.0)
    _, _, x = _run_probe(helper, info)
    assert x[0] == pytest.approx(3.0, abs=1e-3)
    assert x[-1] == pytest.approx(4.0, abs=1e-3)
