"""Reading a Myokit ``.mmt``: its ``[[model]]`` as CellML, its ``[[protocol]]``
as ``protocol_info``.

This used to live downstream, in CUFLynx's ``apps/api``. It belongs here: what
the protocol half produces is *this* package's vocabulary -- the five event
fields are ``utilities.protocol_shapes``' own -- and an agreement maintained by
convention across two repositories is one nothing enforces.
"""

import json
import os
import tempfile

import pytest

from libcuflynx.parsers.MyokitParsers import (
    DEFAULT_BEATS,
    MmtProtocolError,
    MyokitImportError,
    cellml_from_model,
    cellml_from_myokit,
    is_myokit_filename,
    looks_like_myokit,
    pace_variable,
    protocol_info_from_events,
    protocol_info_from_mmt,
)
from libcuflynx.utilities import protocol_shapes

myokit = pytest.importorskip("myokit")


MMT = b"""[[model]]
name: tiny
membrane.V = -80

[engine]
time = 0 bind time
pace = 0 bind pace

[membrane]
dot(V) = 0.1 * engine.pace

[[protocol]]
# Level  Start    Length   Period   Multiplier
1.0      100      2        1000     0
"""

STUB = b"""[[model]]
name: stub

[engine]
time = 0 bind time
"""


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def test_recognises_the_extension():
    assert is_myokit_filename("lr-1991.mmt")
    assert not is_myokit_filename("model.cellml")


def test_recognises_the_content_not_the_name():
    assert looks_like_myokit(MMT)
    assert not looks_like_myokit(b'<?xml version="1.0"?><model/>')


# ---------------------------------------------------------------------------
# [[model]] -> CellML
# ---------------------------------------------------------------------------
def test_the_model_converts_and_a_copy_is_kept():
    with tempfile.TemporaryDirectory() as td:
        cellml, saved = cellml_from_myokit(MMT, filename="tiny.mmt", out_dir=td)
        assert cellml.lstrip().startswith(b"<?xml")
        assert saved == os.path.join(td, "tiny.cellml")
        assert open(saved, "rb").read() == cellml


def test_without_an_output_directory_the_conversion_is_still_returned():
    cellml, saved = cellml_from_myokit(MMT, filename="tiny.mmt")
    assert cellml and saved is None


def test_an_unreadable_file_is_refused_with_myokits_reason():
    with pytest.raises(MyokitImportError, match="could not read"):
        cellml_from_myokit(b"[[model]]\nthis is not a model\n", filename="x.mmt")


def test_a_stub_model_is_refused_at_the_door():
    """Myokit ships files whose ``[[model]]`` exists only to demonstrate a
    protocol. Those import "successfully" with nothing to integrate, and the
    emptiness would otherwise show up much later as a run with no outputs."""
    with pytest.raises(MyokitImportError, match="no state variables"):
        cellml_from_myokit(STUB, filename="stub.mmt")


def test_an_in_memory_model_converts_by_the_same_route():
    """The EasyML reader builds a model rather than reading a .mmt, and has to
    reach the same last step."""
    model = myokit.parse_model(MMT.decode().split("[[protocol]]")[0])
    assert cellml_from_model(model, stem="tiny").lstrip().startswith(b"<?xml")


def test_converting_nothing_is_refused():
    with pytest.raises(MyokitImportError, match="nothing to convert"):
        cellml_from_model(None)


# ---------------------------------------------------------------------------
# [[protocol]] -> protocol_info
# ---------------------------------------------------------------------------
def test_the_events_cross_over_unchanged():
    info, notes = protocol_info_from_mmt(MMT, filename="tiny.mmt")
    assert info["params_to_change"] == {"engine/pace": [["engine_pace"]]}
    events = info["protocol_shapes"]["engine_pace"]["events"]
    assert events == [
        {"level": 1.0, "start": 100.0, "length": 2.0, "period": 1000.0, "multiplier": 0}
    ]
    assert notes  # the protocol is indefinite; the cut has to be reported


def test_an_indefinite_protocol_is_cut_to_two_beats_and_says_so():
    info, notes = protocol_info_from_mmt(MMT, filename="tiny.mmt")
    assert info["sim_times"] == [[1000.0 * DEFAULT_BEATS]]
    assert "repeats indefinitely" in notes[0]


def test_a_named_duration_wins_and_needs_no_note():
    info, notes = protocol_info_from_mmt(MMT, filename="tiny.mmt", duration=250.0)
    assert info["sim_times"] == [[250.0]]
    assert notes == []


def test_the_pre_time_and_label_are_carried_through():
    info, _ = protocol_info_from_mmt(
        MMT, filename="tiny.mmt", pre_time=5.0, label="1 Hz pacing")
    assert info["pre_times"] == [5.0]
    assert info["experiment_labels"] == ["1 Hz pacing"]


def test_the_default_label_names_the_period():
    info, _ = protocol_info_from_mmt(MMT, filename="tiny.mmt")
    assert info["experiment_labels"] == ["pacing, period 1000"]


def test_the_paced_variable_is_spelled_cas_way():
    model, _protocol, _script = myokit.parse(MMT.decode())
    assert pace_variable(model) == "engine/pace"


def test_a_model_with_nothing_bound_to_pace_is_refused():
    unpaced = MMT.replace(b"pace = 0 bind pace", b"pace = 0")
    with pytest.raises(MmtProtocolError, match="bound to `pace`"):
        protocol_info_from_mmt(unpaced, filename="x.mmt")


def test_a_file_with_no_protocol_is_refused():
    no_protocol = MMT.split(b"[[protocol]]")[0]
    with pytest.raises(MmtProtocolError, match="no \\[\\[protocol\\]\\] events"):
        protocol_info_from_mmt(no_protocol, filename="x.mmt")


def test_a_stimulus_of_amplitude_zero_is_refused():
    """Myokit ships examples that declare one because the file is about the
    model's own currents; converting one gives a protocol that applies nothing."""
    flat = MMT.replace(b"1.0      100", b"0        100")
    with pytest.raises(MmtProtocolError, match="amplitude 0"):
        protocol_info_from_mmt(flat, filename="x.mmt")


def test_a_stimulus_that_never_fires_in_the_window_is_refused():
    with pytest.raises(MmtProtocolError, match="fires nothing"):
        protocol_info_from_mmt(MMT, filename="x.mmt", duration=10.0)


def test_beats_below_one_is_refused():
    with pytest.raises(MmtProtocolError, match="at least 1"):
        protocol_info_from_mmt(MMT, filename="x.mmt", beats=0)


# ---------------------------------------------------------------------------
# The agreement with protocol_shapes, which is the reason this lives here
# ---------------------------------------------------------------------------
def test_the_events_are_valid_protocol_shapes():
    info, _ = protocol_info_from_mmt(MMT, filename="tiny.mmt")
    for name, shape in info["protocol_shapes"].items():
        normalised = protocol_shapes.normalise_shape(shape, name=name)
        assert normalised["events"]


def test_the_converted_protocol_materialises_into_traces():
    """The end of the road for a converted protocol: an actual waveform, reached
    by the same call CA's own pipeline makes."""
    info, _ = protocol_info_from_mmt(MMT, filename="tiny.mmt", duration=2000.0)
    out = protocol_shapes.materialise_shapes(info)
    trace = out["protocol_traces"]["engine_pace"]
    assert max(trace["values"]) == pytest.approx(1.0)
    assert min(trace["values"]) == pytest.approx(0.0)
    assert max(trace["t"]) == pytest.approx(2000.0)
    protocol_shapes.validate_trace_references(out)


def test_an_event_protocol_shapes_would_reject_is_refused_here_too():
    with pytest.raises(MmtProtocolError, match="cannot be expressed as a shape"):
        protocol_info_from_events(
            [{"level": float("inf"), "start": 0.0, "length": 1.0,
              "period": 0.0, "multiplier": 0}],
            name="engine/pace",
            duration=10.0,
        )


def test_events_can_come_from_somewhere_other_than_a_protocol_object():
    """The EasyML reader synthesises a stimulus rather than reading one."""
    info, _ = protocol_info_from_events(
        [{"level": -80.0, "start": 5.0, "length": 1.0,
          "period": 1000.0, "multiplier": 0}],
        name="Tiny/i_stim",
        duration=2000.0,
    )
    assert info["params_to_change"] == {"Tiny/i_stim": [["Tiny_i_stim"]]}


def test_no_events_at_all_is_refused():
    with pytest.raises(MmtProtocolError, match="no protocol events"):
        protocol_info_from_events([], name="a/b", duration=1.0)
