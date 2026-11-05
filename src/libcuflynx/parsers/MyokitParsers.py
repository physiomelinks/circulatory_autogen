"""Read a Myokit ``.mmt`` file: its ``[[model]]`` as CellML, its ``[[protocol]]``
as ``protocol_info``.

Everything downstream of a model assumes CellML -- the metadata parser, the
``component/variable`` naming that ``params_for_id`` and obs ``operands`` are
written in, and the generators. Rather than teach each of those about ``.mmt``,
a Myokit model is converted once on the way in and the rest of the pipeline
never knows the difference.

The two halves of a ``.mmt`` are deliberately read separately:

``[[model]]``
    exported to CellML by :func:`cellml_from_myokit`, **without** its protocol.
    Baking Myokit's stimulus into the exported CellML would give the model two
    sources of pacing that disagree, since here the protocol comes from
    obs_data's ``protocol_info``.

``[[protocol]]``
    read by :func:`protocol_info_from_mmt` into that ``protocol_info``. Without
    this the protocol the user actually wrote sits in the ``.mmt`` unused, to be
    re-entered by hand -- which is where transcription errors live.

The two formats say the same thing in different shapes:

    Myokit   a list of events: (level, start, length, period, multiplier),
             i.e. a stimulus waveform defined by when it fires.
    CA       one sub-experiment, with ``protocol_shapes`` holding those same five
             fields under those same names, which CA expands into the point
             table its solvers want.

Putting the result *into* an obs_data document is
:func:`libcuflynx.utilities.obs_data_helpers.fill_protocol_info`, which is
where obs_data's own vocabulary lives.

So the events copy across unchanged -- see
:mod:`libcuflynx.utilities.protocol_shapes`, whose field vocabulary is Myokit's
on purpose and which this module validates against rather than restating. The
alternative -- slicing the run into a sub-experiment per constant stretch of the
waveform -- describes the same stimulus, but describes it in a form that cannot
be read back: five durations and five levels do not announce that they are a
1 Hz stimulus, so the period cannot be edited afterwards, only recomputed.

A periodic Myokit protocol usually runs forever (``multiplier=0``), while a CA
experiment has a finite length -- so an indefinite protocol still needs a number
of beats, which is a choice and not a conversion. It defaults to 2: one beat
cannot show that a model returns to its diastolic state, and two can. The events
keep their own ``multiplier``, so it is the sub-experiment's length that decides
how many stimuli land.

Myokit is a required dependency of this package, but it is still imported inside
the functions that need it: this module is also reached by callers that only
want the filename predicates, and a bare checkout with a broken myokit build
should fail at conversion with a sentence rather than at import with a
traceback.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from libcuflynx.utilities import protocol_shapes

#: Myokit's own extension. ``.txt`` is deliberately not accepted: it would make
#: any stray text file look like a model.
MYOKIT_SUFFIXES = (".mmt",)

#: Beats an indefinite protocol is cut to when the caller names no duration.
DEFAULT_BEATS = 2


class MyokitImportError(ValueError):
    """A Myokit model that could not be read or exported (surface as HTTP 422)."""


class MmtProtocolError(ValueError):
    """A .mmt whose protocol cannot be expressed as protocol_info (surface as 422)."""


def is_myokit_filename(name: str) -> bool:
    return Path(str(name or "")).suffix.lower() in MYOKIT_SUFFIXES


def looks_like_myokit(data: bytes) -> bool:
    """Whether ``data`` is an ``.mmt`` file, judged by its own section headers.

    Content rather than extension, so a model dropped with the wrong name is
    still recognised -- and, more importantly, so an XML file named ``.mmt`` is
    not fed to the Myokit parser.
    """
    try:
        head = data[:4096].decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001 - undecodable is not a Myokit model
        return False
    if head.lstrip().startswith("<"):
        return False  # XML: CellML, SBML, or an OMEX manifest
    # An .mmt is a sectioned file; [[model]] is the one every model has.
    return "[[model]]" in head


# --------------------------------------------------------------------------
# [[model]] -> CellML
# --------------------------------------------------------------------------


def cellml_from_myokit(data: bytes, *, filename: str, out_dir: str | None = None) -> tuple[bytes, str | None]:
    """Convert a Myokit ``.mmt`` to CellML 2.0.

    Returns ``(cellml_bytes, saved_path_or_None)``. ``saved_path`` is where the
    converted file was kept for the user; None when no output directory was
    given, in which case the conversion is still returned but not persisted.
    """
    try:
        import myokit  # noqa: PLC0415 - heavy, imported on use
        from myokit.formats.cellml import CellML2Exporter  # noqa: PLC0415
    except ImportError as exc:
        raise MyokitImportError(
            "Myokit is not installed, so a .mmt model cannot be converted to CellML. "
            "Install myokit, or export the model to CellML yourself and use that."
        ) from exc

    stem = Path(filename).stem or "model"
    with tempfile.TemporaryDirectory() as td:
        mmt_path = Path(td) / f"{stem}.mmt"
        mmt_path.write_bytes(data)
        try:
            # Only the [[model]] section is imported; see the module docstring
            # for why the protocol is left where it is. load() rather than
            # load_model() so a malformed protocol/script section still yields a
            # clear error rather than a parse failure attributed to the model.
            model, _protocol, _script = myokit.load(str(mmt_path))
        except Exception as exc:  # noqa: BLE001 - myokit raises several types
            raise MyokitImportError(f"could not read the Myokit model: {exc}") from exc
        cellml = cellml_from_model(model, stem=stem, exporter=CellML2Exporter)

    saved = None
    if out_dir:
        try:
            target = Path(out_dir) / f"{stem}.cellml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cellml)
            saved = str(target)
        except OSError:
            # Keeping a copy is a convenience; failing to would be a poor reason
            # to reject a model that converted successfully.
            saved = None
    return cellml, saved


def cellml_from_model(model, *, stem: str = "model", exporter=None) -> bytes:
    """Export an in-memory ``myokit.Model`` to CellML 2.0 bytes.

    Split out from :func:`cellml_from_myokit` because a model does not have to
    have come from a ``.mmt``: :mod:`libcuflynx.parsers.EasyMLParsers` builds one
    from an EasyML ``.model`` file and needs the same last step. The emptiness
    checks belong here rather than in the ``.mmt`` reader for the same reason --
    a model with nothing to integrate is worth catching whatever produced it.
    """
    if model is None:
        raise MyokitImportError(
            "that file has no model section, so there is nothing to convert."
        )
    if exporter is None:
        from myokit.formats.cellml import CellML2Exporter as exporter  # noqa: PLC0415

    # Myokit's example set includes files whose [[model]] is a stub -- just a
    # time variable -- because the file exists to demonstrate a protocol or a
    # script (fink-2009-protocol.mmt is one). Those import "successfully" as a
    # model with nothing to integrate, and the emptiness only shows up later as
    # a simulation with no outputs. Say so at the door instead.
    try:
        n_states = model.count_states()
    except Exception:  # noqa: BLE001 - odd model object; let the export decide
        n_states = None
    if n_states == 0:
        raise MyokitImportError(
            "that model has no state variables, so there is nothing to simulate. "
            "Myokit ships protocol- and script-demonstration files whose "
            "[[model]] section is only a stub -- this looks like one of those "
            "rather than a model."
        )

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / f"{stem}.cellml"
        try:
            # No protocol argument: the exported CellML is the model alone.
            exporter().model(str(out_path), model)
        except Exception as exc:  # noqa: BLE001 - export failures are varied
            raise MyokitImportError(f"could not export the model to CellML: {exc}") from exc
        return out_path.read_bytes()


# --------------------------------------------------------------------------
# [[protocol]] -> protocol_info
# --------------------------------------------------------------------------


def _load(data: bytes, filename: str):
    try:
        import myokit  # noqa: PLC0415 - heavy, imported on use
    except ImportError as exc:  # pragma: no cover - myokit is present in CI
        raise MmtProtocolError(
            "Myokit is not installed, so a .mmt protocol cannot be read."
        ) from exc

    stem = Path(filename).stem or "model"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"{stem}.mmt"
        path.write_bytes(data)
        try:
            model, protocol, _script = myokit.load(str(path))
        except Exception as exc:  # noqa: BLE001 - myokit raises several types
            raise MmtProtocolError(f"could not read the Myokit file: {exc}") from exc
    return model, protocol


def pace_variable(model) -> str:
    """The CA-style ``component/variable`` name of the paced variable.

    Myokit marks it with ``bind pace``; that binding is the only thing in the
    file that says which variable the protocol drives, so a model without one
    gives the levels nowhere to go.
    """
    if model is None:
        raise MmtProtocolError(
            "that .mmt has no [[model]] section, so there is no way to tell which "
            "variable the protocol drives. Add a variable with `bind pace`, or "
            "write the protocol_info by hand."
        )
    bound = [v for v in model.variables(deep=True) if v.binding() == "pace"]
    if not bound:
        raise MmtProtocolError(
            "no variable in that .mmt is bound to `pace`, so there is nothing for "
            "the protocol to drive. Myokit applies such a protocol through a "
            "simulation's own pacing input rather than through the model, which "
            "CA has no equivalent of -- add `bind pace` to the stimulus "
            "variable, or write the protocol_info by hand."
        )
    var = bound[0]
    if var.is_state():
        raise MmtProtocolError(
            f"`pace` is bound to {var.qname()}, which is a state variable. The "
            "protocol is driven by setting a parameter between sub-experiments, "
            "and a state is integrated rather than set."
        )
    # Myokit qualifies with a dot, CA and Myokit-in-CA with a slash. The CellML
    # export keeps both names, so this is a spelling change, not a mapping.
    return var.qname().replace(".", "/")


def _duration(protocol, beats: int, duration: float | None) -> tuple[float, list[str]]:
    notes: list[str] = []
    if duration is not None:
        if duration <= 0:
            raise MmtProtocolError("duration must be greater than zero.")
        return float(duration), notes

    characteristic = float(protocol.characteristic_time())
    if protocol.is_infinite():
        if beats < 1:
            raise MmtProtocolError("beats must be at least 1.")
        total = characteristic * beats
        notes.append(
            f"the protocol repeats indefinitely, so it was cut to {beats} "
            f"beat(s) of {characteristic:g} = {total:g}. Pass a duration or a "
            f"beat count to change that."
        )
        return total, notes
    if characteristic <= 0:
        raise MmtProtocolError(
            "that protocol has no duration, so there is nothing to simulate."
        )
    return characteristic, notes


def protocol_info_from_mmt(
    data: bytes,
    *,
    filename: str = "model.mmt",
    beats: int = DEFAULT_BEATS,
    duration: float | None = None,
    pre_time: float = 0.0,
    label: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build a one-experiment ``protocol_info`` from a .mmt's ``[[protocol]]``.

    Returns ``(protocol_info, notes)``. ``notes`` records the choices the
    conversion had to make -- how long an indefinite protocol was run for, above
    all -- because those are the parts a user may want to overrule and would
    otherwise have to infer from the numbers.
    """
    model, protocol = _load(data, filename)
    if protocol is None or not protocol.events():
        raise MmtProtocolError(
            "that .mmt has no [[protocol]] events, so there is no protocol to "
            "convert. The model may simply be unpaced."
        )

    name = pace_variable(model)
    total, notes = _duration(protocol, beats, duration)
    return protocol_info_from_events(
        [_event_fields(e) for e in protocol.events()],
        name=name,
        duration=total,
        notes=notes,
        pre_time=pre_time,
        label=label,
    )


def _event_fields(event) -> dict[str, Any]:
    """One Myokit event in ``protocol_shapes`` vocabulary."""
    return {
        "level": float(event.level()),
        "start": float(event.start()),
        "length": float(event.duration()),
        "period": float(event.period() or 0),
        "multiplier": int(event.multiplier() or 0),
    }


def protocol_info_from_events(
    events: list[dict[str, Any]],
    *,
    name: str,
    duration: float,
    notes: list[str] | None = None,
    pre_time: float = 0.0,
    label: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """One sub-experiment driving ``name`` with ``events``.

    Takes plain event dicts rather than a ``myokit.Protocol`` so that a caller
    which has events but no Myokit protocol object -- the EasyML importer, whose
    stimulus is synthesised rather than read -- reaches the same schedule by the
    same route.
    """
    notes = list(notes or [])
    events = [dict(e) for e in events]
    if not events:
        raise MmtProtocolError("there are no protocol events to convert.")

    # Myokit ships examples whose stimulus has amplitude zero because the file is
    # about the model's own currents rather than about pacing -- dn-1985-if-gna
    # declares `0 10 0.5 1000 0`. Converting one gives a protocol_info that looks
    # like a stimulus and applies none.
    if all(e["level"] == 0 for e in events):
        raise MmtProtocolError(
            "that protocol's only stimulus has amplitude 0, so it never changes "
            "anything -- the model is effectively unpaced and there is no "
            "protocol worth converting."
        )
    if all(e["start"] >= duration for e in events):
        raise MmtProtocolError(
            f"that protocol fires nothing within the {duration:g} it would be run "
            f"over -- its first event starts later than that."
        )

    if label is None:
        period = events[0]["period"]
        label = f"pacing, period {period:g}" if period else "protocol"

    trace_name = name.replace("/", "_")
    shape = {"events": events}
    # Validate through the shape vocabulary itself rather than trusting that the
    # five field names still line up. They are the same names by design, which is
    # exactly the kind of agreement that rots silently when it is only a comment.
    try:
        protocol_shapes.normalise_shape(shape, name=trace_name)
    except protocol_shapes.ProtocolShapeError as exc:
        raise MmtProtocolError(f"that protocol cannot be expressed as a shape: {exc}") from exc

    return (
        {
            "pre_times": [float(pre_time)],
            "sim_times": [[duration]],
            "params_to_change": {name: [[trace_name]]},
            "protocol_shapes": {trace_name: shape},
            "experiment_labels": [label],
            "experiment_colors": ["r"],
        },
        notes,
    )
