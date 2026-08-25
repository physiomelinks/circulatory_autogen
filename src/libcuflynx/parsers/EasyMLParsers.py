"""Read an EasyML ``.model`` file -- openCARP's ionic-model language -- as a
Myokit model, and from there as CellML.

EasyML is what openCARP's published ionic models are written in. It is a small,
non-Turing-complete language: every statement ends in a semicolon, every value is
a float, order does not matter, and a name may be assigned only once. What makes
it more than a list of equations is the *implicit* part, which ``limpet_fe.py``
supplies when it generates C and which this module has to supply too:

``diff_X`` is generated for gating variables
    A Hodgkin-Huxley gate is written by giving either ``alpha_X``/``beta_X`` (or
    the short ``a_X``/``b_X``) or ``tau_X``/``X_inf``. The state equation itself
    is *not* written -- and must not be, per openCARP's own documentation -- so
    a reader that only looks for ``diff_X`` sees a model with most of its states
    missing.

``X_init`` is generated if absent
    for those same gates, at the steady state implied by the pair.

**there is no membrane equation**
    A published model declares ``V; .nodal(); .external(Vm);`` and
    ``Iion; .nodal(); .external();``. In openCARP the tissue solver owns V and
    integrates it; the model file only says what ``Iion`` is. Imported as
    written, such a model has no ``dV/dt`` and cannot be run standalone at all,
    so this module synthesises ``dot(V) = -(Iion + i_stim)`` -- see
    :func:`parse_easyml` for the sign and unit convention -- plus a stimulus
    variable for the protocol to drive.

The ``.method()`` groups are **read and reported, never executed**. They say how
openCARP would step each state (``rush_larsen`` for gates, ``markov_be`` for
Markov chains, ``cvode`` for the rest), which is a discretisation choice for a
fixed-step tissue solver and not part of the model: the ODE system is the same
either way, and here it is integrated as one system. Silently dropping that
would be the wrong kind of quiet, so every non-CVODE group comes back as a
warning naming the states it covered.

Nothing here imports openCARP, and nothing from openCARP is vendored: it is
distributed under the openCARP Academic Public License, which is neither
OSI-approved nor compatible with this package's Apache-2.0 licence. This is an
independent reader of the file format, written against Myokit's public API so it
can be offered upstream as the importer Myokit's ``formats/easyml`` package does
not yet have.

Units
    EasyML records none. Its conventions are V in mV, time in ms, currents in
    A/F and concentrations in mM -- which is why ``dot(V) = -Iion`` needs no
    capacitance term, A/F and mV/ms being the same thing. Variables are
    therefore left dimensionless rather than annotated from a guess: a partial
    annotation makes every equation that mixes an annotated and an unannotated
    term inconsistent, and libCellML reports those. The convention is recorded
    in the model's metadata instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from libcuflynx.parsers.MyokitParsers import (
    MyokitImportError,
    cellml_from_model,
    protocol_info_from_events,
)

#: openCARP's extension for an EasyML model.
EASYML_SUFFIXES = (".model",)

#: The name given to the synthesised stimulus current. Not ``I_stim``: EasyML
#: files are free to use that, and a collision would silently redirect the
#: protocol into one of the model's own variables.
STIMULUS_NAME = "i_stim"

#: The time variable. ``t`` is reserved in EasyML (absolute time in ms).
TIME_NAME = "t"

#: Integration methods that mean "as accurately as the solver can", so restating
#: them in a warning would be noise. Everything else is a fixed-step scheme
#: chosen for tissue-scale cost and is worth reporting.
_ACCURATE_METHODS = {"cvode", "sundials"}

_UNIT_CONVENTION = "V in mV, t in ms, currents in A/F, concentrations in mM"


class EasyMLImportError(ValueError):
    """An EasyML file that could not be read (surface as HTTP 422)."""


def _myokit():
    """Myokit, or an :class:`EasyMLImportError` saying it is not installed.

    Myokit is a required dependency of this package, so this is not an ordinary
    optional-import dance -- but the light tiers that only parse (CUFLynx's unit
    tests, a config-only run) install neither it nor libCellML. Reading an EasyML
    file needs it, and "No module named 'myokit'" from three frames inside an
    expression walker is a worse way to learn that than a sentence.
    """
    try:
        import myokit  # noqa: PLC0415
    except ImportError as exc:
        raise EasyMLImportError(
            "Myokit is not installed, so an EasyML .model file cannot be read. "
            "Install myokit, or convert the model to CellML yourself and use that."
        ) from exc
    return myokit


def is_easyml_filename(name: str) -> bool:
    return Path(str(name or "")).suffix.lower() in EASYML_SUFFIXES


def looks_like_easyml(data: bytes) -> bool:
    """Whether ``data`` is an EasyML model, judged by its own markup.

    Content rather than extension, because ``.model`` is a generic suffix -- the
    cardiac-geometry files in other tools use it too -- so recognising one by
    name alone would hand unrelated files to this parser.
    """
    try:
        head = data[:8192].decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001 - undecodable is not a model file
        return False
    if head.lstrip().startswith("<"):
        return False  # XML: CellML, SBML, or an OMEX manifest
    if ";" not in head:
        return False
    markers = (".method(", ".param(", ".trace(", ".external(", ".nodal(",
               ".regional(", "diff_")
    if any(m in head for m in markers):
        return True
    # A model with no markup at all is still an EasyML file if it declares an
    # initial value the language's way.
    return re.search(r"\b\w+_init\s*=", head) is not None


# ---------------------------------------------------------------------------
# Tokenising
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op><<|>>|<=|>=|==|!=|&&|\|\||[-+*/%^<>!?:=(){},;.])
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass
class _Token:
    kind: str
    text: str
    line: int


def _tokenise(text: str) -> list[_Token]:
    out: list[_Token] = []
    pos = 0
    line = 1
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise EasyMLImportError(
                f"line {line}: cannot read {text[pos:pos + 20]!r}"
            )
        kind = m.lastgroup
        chunk = m.group()
        if kind not in ("space", "comment"):
            out.append(_Token(kind, chunk, line))
        line += chunk.count("\n")
        pos = m.end()
    return out


# ---------------------------------------------------------------------------
# Expressions
#
# Parsed into plain tuples rather than straight into Myokit expressions: a name
# cannot become a `myokit.Name` until its Variable exists, and EasyML is
# order-independent, so nothing can be resolved until the whole file is read.
# ---------------------------------------------------------------------------

# ('num', value) ('name', text) ('call', fname, [args]) ('un', op, a)
# ('bin', op, a, b) ('if', cond, then, else)

_BINARY_LEVELS = (
    ("||",),
    ("&&",),
    ("==", "!="),
    ("<", ">", "<=", ">="),
    ("+", "-"),
    ("*", "/", "%"),
)


class _ExprParser:
    def __init__(self, tokens: list[_Token], pos: int):
        self.toks = tokens
        self.i = pos

    def peek(self) -> _Token | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _at(self, *texts: str) -> bool:
        tok = self.peek()
        return tok is not None and tok.text in texts

    def _take(self) -> _Token:
        tok = self.peek()
        if tok is None:
            raise EasyMLImportError("the file ends in the middle of an expression")
        self.i += 1
        return tok

    def _expect(self, text: str) -> _Token:
        tok = self._take()
        if tok.text != text:
            raise EasyMLImportError(
                f"line {tok.line}: expected {text!r} but found {tok.text!r}"
            )
        return tok

    def parse(self):
        return self._ternary()

    def _ternary(self):
        cond = self._binary(0)
        if self._at("?"):
            self._take()
            then = self._ternary()
            self._expect(":")
            other = self._ternary()
            return ("if", cond, then, other)
        return cond

    def _binary(self, level: int):
        if level >= len(_BINARY_LEVELS):
            return self._unary()
        ops = _BINARY_LEVELS[level]
        node = self._binary(level + 1)
        while self._at(*ops):
            op = self._take().text
            node = ("bin", op, node, self._binary(level + 1))
        return node

    def _unary(self):
        if self._at("-", "+", "!"):
            op = self._take().text
            return ("un", op, self._unary())
        return self._power()

    def _power(self):
        base = self._primary()
        if self._at("^"):
            self._take()
            return ("bin", "^", base, self._unary())
        return base

    def _primary(self):
        tok = self._take()
        if tok.text == "(":
            node = self._ternary()
            self._expect(")")
            return node
        if tok.kind == "number":
            return ("num", float(tok.text))
        if tok.kind == "name":
            if self._at("("):
                self._take()
                args = []
                if not self._at(")"):
                    args.append(self._ternary())
                    while self._at(","):
                        self._take()
                        args.append(self._ternary())
                self._expect(")")
                return ("call", tok.text, args)
            return ("name", tok.text)
        raise EasyMLImportError(
            f"line {tok.line}: {tok.text!r} cannot start an expression"
        )


def _names_in(node, out: set[str]) -> None:
    kind = node[0]
    if kind == "name":
        out.add(node[1])
    elif kind == "call":
        for a in node[2]:
            _names_in(a, out)
    elif kind == "un":
        _names_in(node[2], out)
    elif kind == "bin":
        _names_in(node[2], out)
        _names_in(node[3], out)
    elif kind == "if":
        for a in node[1:]:
            _names_in(a, out)


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass
class _Group:
    names: list[str]
    markup: list[tuple[str, list[str]]] = field(default_factory=list)


@dataclass
class _Parsed:
    """What the file literally says, before any EasyML semantics are applied."""

    assignments: dict[str, Any] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    markup: dict[str, list[tuple[str, list[str]]]] = field(default_factory=dict)
    groups: list[_Group] = field(default_factory=list)
    header: dict[str, str] = field(default_factory=dict)
    referenced: set[str] = field(default_factory=set)


def _header_fields(text: str) -> dict[str, str]:
    """``key: value`` lines from the leading block comment.

    Myokit's exporter writes the model's metadata there, and openCARP's own
    published files carry a reference to the paper the model comes from in the
    same shape. It is the only place a file states its own name.
    """
    m = re.match(r"\s*/\*(.*?)\*/", text, re.DOTALL)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        got = re.match(r"\s*([A-Za-z][\w ]*?)\s*:\s*(\S.*?)\s*$", line)
        if got:
            key = got.group(1).strip()
            out.setdefault(key, got.group(2).strip())
    return out


class _StatementParser:
    def __init__(self, text: str):
        self.toks = _tokenise(text)
        self.i = 0
        self.out = _Parsed(header=_header_fields(text))
        self._subject: str | list[str] | None = None

    # -- token helpers ----------------------------------------------------
    def peek(self, ahead: int = 0) -> _Token | None:
        j = self.i + ahead
        return self.toks[j] if j < len(self.toks) else None

    def _take(self) -> _Token:
        tok = self.peek()
        if tok is None:
            raise EasyMLImportError("the file ends in the middle of a statement")
        self.i += 1
        return tok

    def _expect(self, text: str) -> _Token:
        tok = self._take()
        if tok.text != text:
            raise EasyMLImportError(
                f"line {tok.line}: expected {text!r} but found {tok.text!r}"
            )
        return tok

    def _at(self, text: str, ahead: int = 0) -> bool:
        tok = self.peek(ahead)
        return tok is not None and tok.text == text

    def _expression(self):
        sub = _ExprParser(self.toks, self.i)
        node = sub.parse()
        self.i = sub.i
        return node

    # -- statements -------------------------------------------------------
    def parse(self) -> _Parsed:
        while self.peek() is not None:
            self._statement()
        return self.out

    def _statement(self) -> None:
        tok = self.peek()
        if tok.text == ";":
            self._take()
            return
        if tok.text == ".":
            self._markup_statement()
            return
        if tok.kind == "name" and tok.text == "group":
            self._group_statement()
            return
        if tok.kind == "name" and self._is_conditional(tok):
            self._conditional_statement()
            return
        if tok.kind == "name":
            self._name_statement()
            return
        raise EasyMLImportError(
            f"line {tok.line}: {tok.text!r} cannot start a statement"
        )

    def _is_conditional(self, tok: _Token) -> bool:
        """Whether ``if``/``elif``/``else`` here opens a block or names a variable.

        Judged by what follows rather than by the word alone, because EasyML does
        not reserve these names and real files use them: the Noble 1962 funny
        current is ``i_f``, whose component exports as a variable literally named
        ``if``. Deciding on the word would turn ``if = ifNa + ifK;`` into a syntax
        error twenty tokens later.
        """
        if tok.text in ("if", "elif"):
            return self._at("(", 1)
        if tok.text == "else":
            return self._at("{", 1) or (self._at("if", 1) and self._at("(", 2))
        return False

    def _name_statement(self) -> None:
        name = self._take().text
        if self._at("="):
            self._take()
            if self._at(";"):
                # Myokit's own EasyML exporter emits `Iion = ;` for a model it
                # found no membrane currents in, so this is a real file that
                # exists rather than a hypothetical. Naming the variable beats
                # "';' cannot start an expression" twenty lines later.
                raise EasyMLImportError(
                    f"{name} is assigned nothing: the statement is `{name} = ;`. "
                    f"If this file came from Myokit's EasyML exporter, that is "
                    f"what it writes when it cannot identify the model's "
                    f"membrane currents."
                )
            node = self._expression()
            self._assign(name, node)
        self._subject = name
        # `V; .nodal();` and `x = 1; .param();` both leave the subject set, so
        # the markup statements that follow know what they are talking about.
        while self._at(";"):
            self._take()
            if self._at("."):
                self._markup_statement()
                continue
            break

    def _assign(self, name: str, node) -> None:
        if name in self.out.assignments:
            raise EasyMLImportError(
                f"{name} is assigned more than once; EasyML allows each name "
                f"a single assignment."
            )
        self.out.assignments[name] = node
        self.out.order.append(name)
        _names_in(node, self.out.referenced)

    def _markup_statement(self) -> None:
        self._expect(".")
        tok = self._take()
        if tok.kind != "name":
            raise EasyMLImportError(f"line {tok.line}: expected a markup name")
        args: list[str] = []
        if self._at("("):
            self._take()
            while not self._at(")"):
                args.append(self._take().text)
            self._expect(")")
        if self._at(";"):
            self._take()
        entry = (tok.text, [a for a in args if a != ","])
        subject = self._subject
        if subject is None:
            raise EasyMLImportError(
                f"line {tok.line}: .{tok.text}() does not follow a variable or group"
            )
        if isinstance(subject, list):
            self.out.groups[-1].markup.append(entry)
        else:
            self.out.markup.setdefault(subject, []).append(entry)

    def _group_statement(self) -> None:
        self._take()  # 'group'
        self._expect("{")
        names: list[str] = []
        while not self._at("}"):
            tok = self._take()
            if tok.text == ";":
                continue
            if tok.kind != "name":
                raise EasyMLImportError(
                    f"line {tok.line}: a group lists variable names; found {tok.text!r}"
                )
            names.append(tok.text)
        self._expect("}")
        self.out.groups.append(_Group(names))
        self._subject = names
        while self._at("."):
            self._markup_statement()
        if self._at(";"):
            self._take()

    def _conditional_statement(self) -> None:
        """``if (c) { ... } elif (c) { ... } else { ... }`` as nested ternaries.

        EasyML's blocks are the one imperative-looking thing in the language, but
        single assignment still holds: a name may be assigned in each branch, and
        that is exactly a conditional expression. Anything else -- a name set in
        some branches and not others, with no value to fall back on -- has no
        expression form, and is refused rather than given a silent default.
        """
        branches: list[tuple[Any | None, dict[str, Any]]] = []
        while True:
            keyword = self._take().text
            cond = None
            if keyword in ("if", "elif"):
                self._expect("(")
                cond = self._expression()
                self._expect(")")
            branches.append((cond, self._block()))
            nxt = self.peek()
            if nxt is not None and nxt.text in ("elif", "else") and keyword != "else":
                continue
            break

        assigned: list[str] = []
        for _cond, body in branches:
            for name in body:
                if name not in assigned:
                    assigned.append(name)

        has_else = branches[-1][0] is None
        for name in assigned:
            missing = [i for i, (_c, b) in enumerate(branches) if name not in b]
            if (missing or not has_else) and name not in self.out.assignments:
                raise EasyMLImportError(
                    f"{name} is assigned in only some branches of a conditional "
                    f"and has no value outside it, so there is nothing for the "
                    f"other branches to leave it at."
                )
            # Build from the last branch backwards, so `if/elif/elif/else`
            # nests the way it reads.
            node = self.out.assignments.pop(name, None)
            if node is not None:
                self.out.order.remove(name)
            for cond, body in reversed(branches):
                value = body.get(name, node)
                if cond is None:
                    node = value
                else:
                    node = ("if", cond, value, node)
            self._assign(name, node)

    def _block(self) -> dict[str, Any]:
        self._expect("{")
        body: dict[str, Any] = {}
        while not self._at("}"):
            tok = self._take()
            if tok.text == ";":
                continue
            if tok.kind != "name":
                raise EasyMLImportError(
                    f"line {tok.line}: only assignments are supported inside a "
                    f"conditional block; found {tok.text!r}"
                )
            self._expect("=")
            node = self._expression()
            body[tok.text] = node
            _names_in(node, self.out.referenced)
        self._expect("}")
        return body


# ---------------------------------------------------------------------------
# EasyML semantics: gates, states, the membrane equation
# ---------------------------------------------------------------------------

_INIT_SUFFIX = "_init"
_DIFF_PREFIX = "diff_"

#: The two ways EasyML writes a Hodgkin-Huxley gate, longest spelling first.
#: The short forms are accepted because openCARP does, but they are ambiguous --
#: ``a_1``/``b_1`` are perfectly ordinary names -- so a candidate only counts as
#: a gate if its state is otherwise undefined (see :func:`_find_gates`).
_ALPHA_PREFIXES = ("alpha_", "a_")
_BETA_PREFIXES = ("beta_", "b_")


@dataclass
class Gate:
    """One state whose equation EasyML leaves implicit."""

    state: str
    kind: str          # 'alpha_beta' or 'tau_inf'
    first: str         # alpha_X  / X_inf
    second: str        # beta_X   / tau_X


def _find_gates(parsed: _Parsed) -> dict[str, Gate]:
    assigned = parsed.assignments
    inits = {n[: -len(_INIT_SUFFIX)] for n in assigned if n.endswith(_INIT_SUFFIX)}
    derivs = {n[len(_DIFF_PREFIX):] for n in assigned if n.startswith(_DIFF_PREFIX)}

    def candidate(state: str) -> bool:
        # The state's own equation must be absent -- that is what makes the gate
        # implicit -- and something must point at the state, or the "gate" is
        # only a pair of similarly named ordinary variables.
        if not state or state in assigned or state in derivs:
            return False
        return state in inits or state in parsed.referenced

    gates: dict[str, Gate] = {}
    for name in assigned:
        for prefix in _ALPHA_PREFIXES:
            if not name.startswith(prefix):
                continue
            state = name[len(prefix):]
            if state in gates or not candidate(state):
                continue
            beta = next(
                (b + state for b in _BETA_PREFIXES if b + state in assigned), None)
            if beta is not None:
                gates[state] = Gate(state, "alpha_beta", name, beta)
        if name.startswith("tau_"):
            state = name[4:]
            if state in gates or not candidate(state):
                continue
            if state + "_inf" in assigned:
                gates[state] = Gate(state, "tau_inf", state + "_inf", name)
    return gates


def _membrane_names(parsed: _Parsed) -> tuple[str | None, str | None]:
    """``(V, Iion)`` as this file names them.

    Taken from the ``.external()`` markup rather than from the spelling: the
    markup is what openCARP itself keys on, and a model is free to call its
    membrane potential ``Vm`` or its current sum something else.
    """
    v_name = i_name = None
    for name, entries in parsed.markup.items():
        for fname, args in entries:
            if fname != "external":
                continue
            if args and args[0] in ("Vm", "V"):
                v_name = v_name or name
            elif not args:
                i_name = i_name or name
    if v_name is None:
        v_name = next((n for n in ("V", "Vm") if n in parsed.markup), None)
    if i_name is None and "Iion" in parsed.assignments:
        i_name = "Iion"
    return v_name, i_name


def _group_markup(parsed: _Parsed) -> tuple[dict[str, str], set[str], set[str], list[str]]:
    """Read the ``group {...}.markup()`` blocks and the per-variable markup.

    Returns ``(methods, parameters, traces, warnings)``.
    """
    methods: dict[str, str] = {}
    parameters: set[str] = set()
    traces: set[str] = set()
    warnings: list[str] = []
    known = {"method", "param", "trace", "nodal", "external", "regional", "units",
             "lb", "ub", "flag", "store"}

    def apply(names: list[str], fname: str, args: list[str]) -> None:
        if fname == "method":
            for n in names:
                methods[n] = args[0] if args else "unspecified"
        elif fname == "param":
            parameters.update(names)
        elif fname == "trace":
            traces.update(names)
        elif fname not in known:
            warnings.append(
                f"ignored unrecognised markup .{fname}() on "
                f"{', '.join(names[:3])}{'...' if len(names) > 3 else ''}"
            )

    for group in parsed.groups:
        for fname, args in group.markup:
            apply(group.names, fname, args)
    for name, entries in parsed.markup.items():
        for fname, args in entries:
            apply([name], fname, args)
    return methods, parameters, traces, warnings


# ---------------------------------------------------------------------------
# Building the Myokit model
# ---------------------------------------------------------------------------

_UNARY = {}
_BINARY = {}
_FUNCS = {}


def _init_expression_tables():
    myokit = _myokit()

    if _BINARY:
        return
    _UNARY.update({"-": myokit.PrefixMinus, "+": myokit.PrefixPlus, "!": myokit.Not})
    _BINARY.update({
        "+": myokit.Plus, "-": myokit.Minus, "*": myokit.Multiply,
        "/": myokit.Divide, "%": myokit.Remainder, "^": myokit.Power,
        "==": myokit.Equal, "!=": myokit.NotEqual, "<": myokit.Less,
        ">": myokit.More, "<=": myokit.LessEqual, ">=": myokit.MoreEqual,
        "&&": myokit.And, "||": myokit.Or,
    })
    _FUNCS.update({
        "exp": myokit.Exp, "log": myokit.Log, "log10": myokit.Log10,
        "sqrt": myokit.Sqrt, "sin": myokit.Sin, "cos": myokit.Cos,
        "tan": myokit.Tan, "asin": myokit.ASin, "acos": myokit.ACos,
        "atan": myokit.ATan, "floor": myokit.Floor, "ceil": myokit.Ceil,
        "abs": myokit.Abs, "fabs": myokit.Abs, "pow": myokit.Power,
    })


def _to_myokit(node, resolve):
    """One parsed expression as a Myokit expression.

    The function table is the inverse of ``myokit.formats.easyml``'s expression
    writer, which is a C writer: ``pow(a, b)`` rather than ``a**b``, ``fabs``,
    and ``expm1(x)`` for ``exp(x) - 1``. Reading ``expm1`` back as
    ``exp(x) - 1`` is what makes the export/import round trip exact.
    """
    myokit = _myokit()

    _init_expression_tables()
    kind = node[0]
    if kind == "num":
        return myokit.Number(node[1])
    if kind == "name":
        return resolve(node[1])
    if kind == "un":
        return _UNARY[node[1]](_to_myokit(node[2], resolve))
    if kind == "bin":
        return _BINARY[node[1]](_to_myokit(node[2], resolve), _to_myokit(node[3], resolve))
    if kind == "if":
        return myokit.If(*(_to_myokit(a, resolve) for a in node[1:]))

    fname, args = node[1], [_to_myokit(a, resolve) for a in node[2]]
    if fname == "expm1":
        return myokit.Minus(myokit.Exp(args[0]), myokit.Number(1))
    if fname == "log1p":
        return myokit.Log(myokit.Plus(myokit.Number(1), args[0]))
    if fname in ("max", "min"):
        if len(args) != 2:
            raise EasyMLImportError(f"{fname}() takes two arguments")
        cmp = myokit.More if fname == "max" else myokit.Less
        return myokit.If(cmp(args[0], args[1]), args[0], args[1])
    if fname not in _FUNCS:
        raise EasyMLImportError(
            f"{fname}() is not a function this reader knows. If openCARP accepts "
            f"it, it needs adding to the translation table rather than guessing."
        )
    return _FUNCS[fname](*args)


@dataclass
class EasyMLResult:
    """A read EasyML file: the model, and everything the file said about it."""

    model: Any                       # myokit.Model
    warnings: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    traces: list[str] = field(default_factory=list)
    methods: dict[str, str] = field(default_factory=dict)
    v_name: str | None = None
    stimulus_name: str | None = None
    synthesised_membrane: bool = False


def _model_name(header: dict[str, str], filename: str) -> str:
    raw = header.get("name") or Path(filename).stem or "easyml_model"
    cleaned = re.sub(r"\W+", "_", raw).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "model_" + cleaned
    return cleaned


def parse_easyml(data: bytes | str, *, filename: str = "model.model") -> EasyMLResult:
    """Read an EasyML ``.model`` file into a :class:`myokit.Model`.

    The membrane equation is synthesised when the file declares V as external,
    which is what every published openCARP model does::

        dot(V) = -(Iion + i_stim)

    with no capacitance term: EasyML's currents are in A/F, V is in mV and time
    is in ms, and A/F is mV/ms. The sign is openCARP's -- an inward (negative)
    current depolarises -- so a depolarising stimulus is a **negative**
    ``i_stim``.
    """
    myokit = _myokit()

    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    parsed = _StatementParser(text).parse()
    if not parsed.assignments:
        raise EasyMLImportError(
            "that file has no equations in it, so there is no model to read."
        )

    methods, parameters, traces, warnings = _group_markup(parsed)
    v_name, i_name = _membrane_names(parsed)
    gates = _find_gates(parsed)

    inits = {
        n[: -len(_INIT_SUFFIX)]: parsed.assignments[n]
        for n in parsed.assignments
        if n.endswith(_INIT_SUFFIX)
    }
    derivatives = {
        n[len(_DIFF_PREFIX):]: parsed.assignments[n]
        for n in parsed.assignments
        if n.startswith(_DIFF_PREFIX)
    }

    # Everything the file assigns that is a real variable: not an initial value,
    # not a derivative (those are the *equations* of states, not variables).
    plain = [
        n for n in parsed.order
        if not n.endswith(_INIT_SUFFIX) and not n.startswith(_DIFF_PREFIX)
    ]

    states = list(derivatives) + [g for g in gates if g not in derivatives]
    synthesise = v_name is not None and v_name not in parsed.assignments and v_name not in states
    if synthesise:
        if i_name is None:
            raise EasyMLImportError(
                f"{v_name} is declared external, so this file expects openCARP's "
                f"solver to integrate it -- but it defines no Iion for the "
                f"membrane equation to use. A .model that only adds to a current "
                f"(openCARP calls it a plugin) is not a model on its own."
            )
        states.append(v_name)

    unknown_inits = [n for n in inits if n not in states]
    if unknown_inits:
        warnings.append(
            "initial values were given for "
            + ", ".join(sorted(unknown_inits))
            + ", which this file does not define as states; they were ignored."
        )

    # ---- build ----------------------------------------------------------
    model = myokit.Model(_model_name(parsed.header, filename))
    comp_name = model.name()
    component = model.add_component(comp_name)

    variables: dict[str, Any] = {}
    for name in plain + [s for s in states if s not in plain]:
        if name in variables:
            continue
        variables[name] = component.add_variable(name)

    time = component.add_variable_allow_renaming(TIME_NAME)
    time.set_rhs(0)
    time.set_binding("time")
    variables.setdefault(TIME_NAME, time)

    stim = None
    if synthesise:
        stim = component.add_variable_allow_renaming(STIMULUS_NAME)
        stim.set_rhs(0)
        variables[stim.name()] = stim

    missing = sorted(
        n for n in parsed.referenced
        if n not in variables and n not in inits and n != TIME_NAME
    )
    if missing:
        hint = ""
        if "dt" in missing:
            hint = (
                " `dt` is openCARP's own timestep, which only exists inside its "
                "fixed-step solver; a model that reads it cannot be integrated "
                "adaptively."
            )
        raise EasyMLImportError(
            "these names are used but never defined: " + ", ".join(missing) + "." + hint
        )

    def resolve(name: str):
        var = variables.get(name)
        if var is None:
            raise EasyMLImportError(f"{name} is used but never defined.")
        return myokit.Name(var)

    # Constant initial values, so a gate's steady state can be evaluated below.
    for state in states:
        var = variables[state]
        node = inits.get(state)
        value = 0.0
        if node is not None:
            try:
                value = _to_myokit(node, resolve).eval()
            except Exception as exc:  # noqa: BLE001 - a non-constant _init
                raise EasyMLImportError(
                    f"{state}{_INIT_SUFFIX} is not a plain number: {exc}"
                ) from exc
        var.promote(value)

    for name in plain:
        if name in states:
            continue
        variables[name].set_rhs(_to_myokit(parsed.assignments[name], resolve))

    for state, node in derivatives.items():
        variables[state].set_rhs(_to_myokit(node, resolve))

    for state, gate in gates.items():
        if state in derivatives:
            continue
        var = variables[state]
        if gate.kind == "tau_inf":
            var.set_rhs(myokit.Divide(
                myokit.Minus(resolve(gate.first), myokit.Name(var)),
                resolve(gate.second)))
        else:
            var.set_rhs(myokit.Minus(
                myokit.Multiply(
                    resolve(gate.first),
                    myokit.Minus(myokit.Number(1), myokit.Name(var))),
                myokit.Multiply(resolve(gate.second), myokit.Name(var))))

    if synthesise:
        total = myokit.Plus(resolve(i_name), myokit.Name(stim))
        variables[v_name].set_rhs(myokit.PrefixMinus(total))

    _set_gate_steady_states(gates, inits, variables, warnings)

    model.meta["desc"] = (
        f"Imported from the EasyML file {Path(filename).name}. "
        f"Unit convention: {_UNIT_CONVENTION}."
    )
    for key, value in parsed.header.items():
        model.meta.setdefault(key.lower().replace(" ", "_"), value)
    if synthesise:
        model.meta["easyml_membrane"] = (
            f"dot({v_name}) = -({i_name} + {stim.name()}), synthesised: the file "
            f"declares {v_name} external, so openCARP's solver owned it."
        )
        warnings.append(
            f"the file declares {v_name} as external, so it carries no membrane "
            f"equation; dot({v_name}) = -({i_name} + {stim.name()}) was added, "
            f"with {stim.name()} for a protocol to drive."
        )

    warnings.extend(_method_warnings(methods))

    try:
        model.validate()
    except Exception as exc:  # noqa: BLE001 - myokit raises its own family
        raise EasyMLImportError(f"the model this file describes is not valid: {exc}") from exc

    return EasyMLResult(
        model=model,
        warnings=warnings,
        parameters=sorted(parameters),
        traces=sorted(traces),
        methods=methods,
        v_name=v_name,
        stimulus_name=stim.name() if stim is not None else None,
        synthesised_membrane=synthesise,
    )


def _set_gate_steady_states(gates, inits, variables, warnings) -> None:
    """Give every gate with no ``X_init`` its steady state.

    openCARP's translator generates the missing initial value rather than
    defaulting it to zero, and a gate started at zero is a different simulation
    for the first few beats. The steady state is read off the pair the gate is
    written with, evaluated at the initial state of everything else -- which is
    exactly what the file already fixed with the other ``_init`` values.
    """
    myokit = _myokit()

    for state, gate in gates.items():
        if state in inits:
            continue
        var = variables[state]
        first, second = variables[gate.first], variables[gate.second]
        if gate.kind == "tau_inf":
            expr = myokit.Name(first)
        else:
            expr = myokit.Divide(
                myokit.Name(first),
                myokit.Plus(myokit.Name(first), myokit.Name(second)))
        try:
            value = expr.eval()
        except Exception as exc:  # noqa: BLE001 - a gate that cannot settle
            warnings.append(
                f"{state} has no {state}{_INIT_SUFFIX} and its steady state could "
                f"not be evaluated ({exc}); it starts at 0."
            )
            continue
        var.set_initial_value(value)
        warnings.append(
            f"{state} has no {state}{_INIT_SUFFIX}; it was started at its steady "
            f"state, {value:g}."
        )


def _method_warnings(methods: dict[str, str]) -> list[str]:
    by_method: dict[str, list[str]] = {}
    for name, method in methods.items():
        if method in _ACCURATE_METHODS:
            continue
        by_method.setdefault(method, []).append(name)
    out = []
    for method, names in sorted(by_method.items()):
        listed = ", ".join(sorted(names)[:6])
        if len(names) > 6:
            listed += f", and {len(names) - 6} more"
        out.append(
            f"openCARP would step {listed} with {method}; here the whole system "
            f"is integrated as one, which is a discretisation difference and not "
            f"a different model."
        )
    return out


# ---------------------------------------------------------------------------
# The same last step as a .mmt: CellML out
# ---------------------------------------------------------------------------


def cellml_from_easyml(
    data: bytes, *, filename: str, out_dir: str | None = None
) -> tuple[bytes, str | None, list[str]]:
    """Convert an EasyML ``.model`` to CellML 2.0.

    Returns ``(cellml_bytes, saved_path_or_None, warnings)``. The warnings are
    the choices this reader had to make -- a synthesised membrane equation, a
    gate started at its steady state, an integration method openCARP would have
    used -- and are meant to be shown, not logged: each one is a place where the
    imported model differs from the file.
    """
    stem = Path(filename).stem or "model"
    result = parse_easyml(data, filename=filename)
    try:
        cellml = cellml_from_model(result.model, stem=stem)
    except MyokitImportError as exc:
        # The model was read; it is the CellML step that refused it. Say so as
        # an EasyML failure, because the file the user handed over is the .model.
        raise EasyMLImportError(str(exc)) from exc

    saved = None
    if out_dir:
        try:
            target = Path(out_dir) / f"{stem}.cellml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cellml)
            saved = str(target)
        except OSError:
            saved = None
    return cellml, saved, result.warnings


#: A stimulus for a model that carries none. openCARP's own single-cell driver
#: supplies one from the command line, so a published .model file has no
#: amplitude of its own to inherit -- these are ordinary single-cell values, and
#: the point of returning them as a *note* is that they are a starting guess.
#: The sign follows the membrane equation: with ``dot(V) = -(Iion + i_stim)`` an
#: inward (negative) current depolarises.
DEFAULT_STIMULUS = {
    "level": -80.0,
    "start": 5.0,
    "length": 1.0,
    "period": 1000.0,
    "multiplier": 0,
}
DEFAULT_BEATS = 2


def protocol_info_from_easyml(
    result: EasyMLResult,
    *,
    beats: int = DEFAULT_BEATS,
    duration: float | None = None,
    pre_time: float = 0.0,
    stimulus: dict[str, Any] | None = None,
    label: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """A default pacing protocol for a model whose membrane equation was synthesised.

    An EasyML model paced by nothing sits at its resting potential forever, so
    the import is only half a study without this. Everything about the stimulus
    is a choice rather than a conversion -- the file says nothing about it -- so
    all of it is overridable and all of it comes back in the notes.
    """
    if not result.synthesised_membrane or result.stimulus_name is None:
        raise EasyMLImportError(
            "that model integrates its own membrane potential, so it is not "
            "clear what a synthesised stimulus would drive."
        )
    event = dict(DEFAULT_STIMULUS)
    event.update(stimulus or {})
    total = float(duration) if duration is not None else event["period"] * beats
    name = f"{result.model.name()}/{result.stimulus_name}"
    info, notes = protocol_info_from_events(
        [event], name=name, duration=total, pre_time=pre_time, label=label)
    notes.append(
        f"the .model file carries no stimulus, so one was added: level "
        f"{event['level']:g} for {event['length']:g} every {event['period']:g}, "
        f"over {total:g}. Check the amplitude against the model you are pacing."
    )
    return info, notes


def import_easyml(
    data: bytes,
    *,
    filename: str,
    out_dir: str | None = None,
    beats: int = DEFAULT_BEATS,
    duration: float | None = None,
    stimulus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything a caller needs from one ``.model`` file, parsed once.

    A published ionic model is not a small file -- Decker 2009 is 700 lines and
    46 states -- so the CellML, the warnings, the parameter list and the default
    protocol all come out of a single read rather than three.

    The protocol is *offered*, never applied: a model is often dropped next to an
    obs_data the user wrote, and replacing that silently would be worse than not
    converting at all. When there is nothing to offer, ``protocol_reason`` says
    why, because "no protocol appeared" is a question and the answer is usually
    one line.
    """
    stem = Path(filename).stem or "model"
    result = parse_easyml(data, filename=filename)
    try:
        cellml = cellml_from_model(result.model, stem=stem)
    except MyokitImportError as exc:
        raise EasyMLImportError(str(exc)) from exc

    saved = None
    if out_dir:
        try:
            target = Path(out_dir) / f"{stem}.cellml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cellml)
            saved = str(target)
        except OSError:
            saved = None

    info: dict[str, Any] | None = None
    notes: list[str] = []
    reason: str | None = None
    try:
        info, notes = protocol_info_from_easyml(
            result, beats=beats, duration=duration, stimulus=stimulus)
    except EasyMLImportError as exc:
        reason = str(exc)

    return {
        "cellml": cellml,
        "cellml_path": saved,
        "model_name": result.model.name(),
        "warnings": result.warnings,
        "parameters": result.parameters,
        "traces": result.traces,
        "methods": result.methods,
        "membrane_variable": result.v_name,
        "stimulus_variable": result.stimulus_name,
        "protocol_info": info,
        "protocol_notes": notes,
        "protocol_reason": reason,
    }
