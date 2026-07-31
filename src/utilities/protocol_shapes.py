"""Declarative pacing shapes that expand into ``protocol_traces``.

``protocol_traces`` lets a ``params_to_change`` entry be a string key naming an
arbitrary ``{t, values}`` waveform. That is completely general and completely
unwritable by hand: a 1 Hz stimulus over ten beats is a forty-point table in
which every number has to be right, and changing the period means rebuilding the
table rather than editing a field.

``protocol_shapes`` is the same waveform written the way Myokit's ``[[protocol]]``
section writes it -- one line per event, in the same five columns::

    # Level  Start  Length  Period  Multiplier
    1.0      100    2       1000    0

becomes::

    "protocol_shapes": {
        "stim": {"events": [{"level": 1.0, "start": 100, "length": 2,
                             "period": 1000, "multiplier": 0}]}
    }

The two are alternatives, not layers: a name is defined in one or the other, and
whichever you use, ``params_to_change`` refers to it the same way. Shapes are
expanded into ``protocol_traces`` before anything downstream looks at them, so
solvers, plotting and the rest of the pipeline keep seeing the traces they
already understand and need no knowledge of shapes at all.

The field names are Myokit's on purpose. A model imported from a ``.mmt`` carries
its protocol in exactly this vocabulary, and a user who has written one should
not have to learn a second name for ``Length``.
"""

from __future__ import annotations

# Recognised shape types. Only pacing today; the key exists so a later shape
# (ramp, sine) is an addition rather than a breaking change to the format.
PACING = "pacing"
SHAPE_TYPES = (PACING,)

# `Length` in a .mmt's column header, `duration` in Myokit's Python API. Both are
# accepted: a user coming from either should not have to look this up.
LENGTH_KEYS = ("length", "duration")

EVENT_KEYS = {"level", "start", "period", "multiplier", *LENGTH_KEYS}
SHAPE_KEYS = {"type", "events", "baseline", "duration"}

# A generated trace is square, but myokit.TimeSeriesProtocol interpolates
# linearly between the points it is given -- so an edge is a very short ramp
# rather than a discontinuity. This is the fraction of the shortest interval in
# the waveform that an edge is allowed to take.
EDGE_FRACTION = 1e-3


class ProtocolShapeError(ValueError):
    """A protocol_shapes entry that cannot be turned into a trace."""


def _number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolShapeError(f"{what} must be a number, got {value!r}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ProtocolShapeError(f"{what} must be finite, got {value!r}")
    return value


def normalise_shape(shape, *, name):
    """Validate one ``protocol_shapes`` entry and return it in canonical form.

    Accepts the canonical ``{"events": [...]}`` mapping and, as a shorthand, a
    bare list of events -- which is what a ``.mmt`` protocol table looks like
    when transcribed directly.
    """
    if isinstance(shape, (list, tuple)):
        shape = {"events": list(shape)}
    if not isinstance(shape, dict):
        raise ProtocolShapeError(
            f"protocol_shapes['{name}'] must be a mapping or a list of events, "
            f"got {type(shape).__name__}"
        )

    unknown = sorted(set(shape) - SHAPE_KEYS)
    if unknown:
        raise ProtocolShapeError(
            f"protocol_shapes['{name}'] has unknown keys {unknown}; "
            f"expected any of {sorted(SHAPE_KEYS)}"
        )

    shape_type = shape.get("type", PACING)
    if shape_type not in SHAPE_TYPES:
        raise ProtocolShapeError(
            f"protocol_shapes['{name}'] has type '{shape_type}'; "
            f"expected one of {list(SHAPE_TYPES)}"
        )

    events = shape.get("events")
    if not isinstance(events, (list, tuple)) or not events:
        raise ProtocolShapeError(
            f"protocol_shapes['{name}'] needs a non-empty 'events' list -- one "
            f"entry per line of a .mmt [[protocol]] table"
        )

    normalised = []
    for i, raw in enumerate(events):
        if not isinstance(raw, dict):
            raise ProtocolShapeError(
                f"protocol_shapes['{name}'].events[{i}] must be a mapping, "
                f"got {type(raw).__name__}"
            )
        unknown = sorted(set(raw) - EVENT_KEYS)
        if unknown:
            raise ProtocolShapeError(
                f"protocol_shapes['{name}'].events[{i}] has unknown keys {unknown}; "
                f"expected any of {sorted(EVENT_KEYS)}"
            )
        where = f"protocol_shapes['{name}'].events[{i}]"

        given_length = [k for k in LENGTH_KEYS if k in raw]
        if not given_length:
            raise ProtocolShapeError(f"{where} needs a 'length' (the .mmt column) or 'duration'")
        if len(given_length) > 1:
            raise ProtocolShapeError(
                f"{where} gives both 'length' and 'duration'; they are the same "
                f"field under two names, so give only one"
            )
        length = _number(raw[given_length[0]], f"{where}['{given_length[0]}']")
        if length <= 0:
            raise ProtocolShapeError(f"{where} has length {length}; an event must last some time")

        if "level" not in raw:
            raise ProtocolShapeError(f"{where} needs a 'level'")
        level = _number(raw["level"], f"{where}['level']")
        start = _number(raw.get("start", 0.0), f"{where}['start']")
        if start < 0:
            raise ProtocolShapeError(f"{where} has start {start}; it cannot be before the run")
        period = _number(raw.get("period", 0.0), f"{where}['period']")
        if period < 0:
            raise ProtocolShapeError(f"{where} has period {period}; it cannot be negative")
        multiplier = _number(raw.get("multiplier", 0.0), f"{where}['multiplier']")
        if multiplier < 0 or multiplier != int(multiplier):
            raise ProtocolShapeError(
                f"{where} has multiplier {multiplier}; it must be a non-negative whole number "
                f"(0 means repeat for as long as the sub-experiment lasts)"
            )
        if period == 0 and multiplier > 1:
            raise ProtocolShapeError(
                f"{where} repeats {int(multiplier)} times but has no period, so every "
                f"repeat would land on top of the first one"
            )
        normalised.append(
            {
                "level": level,
                "start": start,
                "length": length,
                "period": period,
                "multiplier": int(multiplier),
            }
        )

    canonical = {
        "type": shape_type,
        "events": normalised,
        "baseline": _number(shape.get("baseline", 0.0), f"protocol_shapes['{name}']['baseline']"),
    }
    if "duration" in shape:
        duration = _number(shape["duration"], f"protocol_shapes['{name}']['duration']")
        if duration <= 0:
            raise ProtocolShapeError(
                f"protocol_shapes['{name}'] has duration {duration}; it must be positive"
            )
        canonical["duration"] = duration
    return canonical


def _occurrences(event, duration):
    """When the event fires, within ``[0, duration)``.

    Myokit's rules: no period means it happens once; multiplier 0 with a period
    means it repeats indefinitely, which here means "for as long as this
    sub-experiment runs".
    """
    start, period, multiplier = event["start"], event["period"], event["multiplier"]
    if period == 0:
        return [start] if start < duration else []

    times = []
    when = start
    # An indefinite event is bounded by the sub-experiment; a counted one by its
    # multiplier. The `len(times)` guard also stops a pathologically small period
    # from generating an unbounded list.
    limit = multiplier if multiplier else None
    max_points = int(duration / period) + 2
    while when < duration and (limit is None or len(times) < limit):
        times.append(when)
        when += period
        if limit is None and len(times) > max_points:
            break
    return times


def _intervals(events, duration, *, name):
    """``(start, end, level)`` for every occurrence, checked for overlap."""
    spans = []
    for event in events:
        for start in _occurrences(event, duration):
            spans.append((start, min(start + event["length"], duration), event["level"]))
    spans.sort()
    for (a_start, a_end, _), (b_start, _, _) in zip(spans, spans[1:]):
        if b_start < a_end:
            raise ProtocolShapeError(
                f"protocol_shapes['{name}'] has events that overlap: one runs from "
                f"{a_start:g} to {a_end:g} while another starts at {b_start:g}. "
                f"Myokit rejects overlapping events too -- the value during the "
                f"overlap would be ambiguous."
            )
    return spans


def expand_shape(shape, duration, *, name):
    """Turn a normalised shape into a ``{"t": [...], "values": [...]}`` trace."""
    duration = _number(duration, f"duration for protocol_shapes['{name}']")
    if duration <= 0:
        raise ProtocolShapeError(
            f"protocol_shapes['{name}'] is used over a sub-experiment of length "
            f"{duration:g}; there is no time to pace anything in"
        )

    baseline = shape["baseline"]
    spans = _intervals(shape["events"], duration, name=name)
    if not spans:
        # Every occurrence fell outside the run. Say so rather than returning a
        # flat line that looks like a protocol and does nothing.
        raise ProtocolShapeError(
            f"protocol_shapes['{name}'] fires nothing within the {duration:g} it is "
            f"run over -- check 'start' against the sub-experiment's sim_time"
        )

    # An edge has to be short relative to the shortest feature in the waveform,
    # or a 2 ms stimulus in a 2000 ms beat gets ramped away.
    features = [end - start for start, end, _ in spans]
    features += [b[0] - a[1] for a, b in zip(spans, spans[1:]) if b[0] > a[1]]
    features += [spans[0][0]] if spans[0][0] > 0 else []
    features += [duration - spans[-1][1]] if spans[-1][1] < duration else []
    smallest = min(f for f in features if f > 0)
    eps = max(smallest * EDGE_FRACTION, duration * 1e-12)

    t = [0.0]
    values = [baseline]

    def push(when, value):
        when = min(max(when, 0.0), duration)
        if when > t[-1]:
            t.append(when)
            values.append(value)
        else:
            # Two edges landing on the same instant: the later one wins, which is
            # what a square wave means at a transition.
            values[-1] = value

    for start, end, level in spans:
        if start > 0:
            push(start, baseline)
        push(min(start + eps, duration), level)
        if end > start + eps:
            push(end, level)
        if end < duration:
            push(min(end + eps, duration), baseline)
    push(duration, baseline if spans[-1][1] < duration else spans[-1][2])
    return {"t": t, "values": values}


def _string_leaves(value):
    """Every string leaf of a params_to_change entry, with its (exp, sub) index."""
    found = []
    if isinstance(value, (list, tuple)):
        for exp_idx, row in enumerate(value):
            if isinstance(row, (list, tuple)):
                for sub_idx, leaf in enumerate(row):
                    if isinstance(leaf, str):
                        found.append((exp_idx, sub_idx, leaf))
            elif isinstance(row, str):
                found.append((exp_idx, 0, row))
    return found


def materialise_shapes(protocol_info):
    """Expand ``protocol_shapes`` into ``protocol_traces``, in place.

    Called before anything reads the traces, so the rest of the pipeline never
    has to know shapes exist. Idempotent: a protocol_info that has already been
    expanded passes through untouched, which matters because it is handed around
    between the parser, the runner and the solver helpers.

    A shape is expanded over the sub-experiment that refers to it, so its length
    comes from ``sim_times`` rather than having to be repeated in the shape. A
    shape used by sub-experiments of different lengths is ambiguous and is
    refused unless it names its own ``duration``.
    """
    if not isinstance(protocol_info, dict):
        return protocol_info

    shapes_raw = protocol_info.get("protocol_shapes") or {}
    if not shapes_raw:
        return protocol_info
    if not isinstance(shapes_raw, dict):
        raise ProtocolShapeError(
            f"protocol_shapes must be a mapping of name -> shape, got {type(shapes_raw).__name__}"
        )

    traces = protocol_info.get("protocol_traces")
    if not isinstance(traces, dict):
        traces = {}

    shapes = {name: normalise_shape(shape, name=name) for name, shape in shapes_raw.items()}

    sim_times = protocol_info.get("sim_times") or []
    durations = {}
    for param, value in (protocol_info.get("params_to_change") or {}).items():
        for exp_idx, sub_idx, leaf in _string_leaves(value):
            if leaf not in shapes:
                continue
            try:
                duration = float(sim_times[exp_idx][sub_idx])
            except (IndexError, TypeError, ValueError):
                raise ProtocolShapeError(
                    f"params_to_change['{param}'][{exp_idx}][{sub_idx}] uses shape "
                    f"'{leaf}', but sim_times has no matching sub-experiment"
                ) from None
            durations.setdefault(leaf, {}).setdefault(duration, []).append(
                f"{param}[{exp_idx}][{sub_idx}]"
            )

    for name, shape in shapes.items():
        if "duration" in shape:
            duration = shape["duration"]
        else:
            seen = durations.get(name)
            if not seen:
                raise ProtocolShapeError(
                    f"protocol_shapes['{name}'] is never used by params_to_change, so "
                    f"there is no sub-experiment to size it against. Reference it, "
                    f"remove it, or give it an explicit 'duration'."
                )
            if len(seen) > 1:
                lengths = ", ".join(
                    f"{d:g} (from {', '.join(refs)})" for d, refs in sorted(seen.items())
                )
                raise ProtocolShapeError(
                    f"protocol_shapes['{name}'] is used over sub-experiments of "
                    f"different lengths: {lengths}. Give it an explicit 'duration', "
                    f"or split it into one shape per length."
                )
            duration = next(iter(seen))
        generated = expand_shape(shape, duration, name=name)
        # A name defined as both a shape and a hand-written trace is ambiguous --
        # except when the "hand-written" one is this function's own output from an
        # earlier pass, which is how expansion stays idempotent across the parser,
        # the runner and the solver helper all calling it on the same dict.
        if name in traces and traces[name] != generated:
            raise ProtocolShapeError(
                f"'{name}' is defined in both protocol_shapes and protocol_traces, "
                f"and they disagree. They are alternatives -- define each name in "
                f"one or the other."
            )
        traces[name] = generated

    protocol_info["protocol_traces"] = traces
    return protocol_info


def validate_trace_references(protocol_info):
    """Fail early on a ``params_to_change`` string that names nothing.

    Without this the mistake surfaces at solve time, inside whichever
    sub-experiment happens to reach it first, long after the file was read.
    """
    if not isinstance(protocol_info, dict):
        return
    traces = protocol_info.get("protocol_traces") or {}
    shapes = protocol_info.get("protocol_shapes") or {}
    known = set(traces) | set(shapes)
    missing = []
    for param, value in (protocol_info.get("params_to_change") or {}).items():
        for exp_idx, sub_idx, leaf in _string_leaves(value):
            if leaf not in known:
                missing.append(f"  params_to_change['{param}'][{exp_idx}][{sub_idx}] -> '{leaf}'")
    if missing:
        raise ProtocolShapeError(
            "params_to_change refers to traces that are defined nowhere:\n"
            + "\n".join(missing)
            + f"\nDefined names: {sorted(known) if known else 'none'}"
        )
