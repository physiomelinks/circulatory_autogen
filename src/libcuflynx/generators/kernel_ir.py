"""A language-neutral form of a CellML model's rates and their exact Jacobian, for code generators.

The libCellML analyser's equation trees become expressions over a model's states, inputs (constants and
external variables, read from libCellML's one ``variables`` array) and local temporaries (the algebraic
variables and computed constants a block needs, computed in the kernel). From them:

- **Blocks.** The states fall into blocks that do not depend on one another: the connected components of
  the graph joining each rate to the states it reads (through the algebraic variables it needs). A module
  array of unconnected cells gives one block per cell; a circulation gives one block. Each block can be
  integrated on its own, with its own step size.
- **Programs.** Each block's rates are a :class:`Program`: assignments to temporaries (only the algebraic
  variables its rates need, in dependency order) and one expression per rate. Its Jacobian is another
  program, made by forward-mode differentiation of the same trees, with an entry per structurally non-zero
  derivative.
- **Templates.** Blocks whose programs are the same once their states and inputs are numbered locally share
  one :class:`Kernel` (a module array of identical cells needs one kernel, whatever the number of cells).
  An input that every block of a kernel reads from the same variable is marked ``shared``.

An emitter only has to print expressions (:func:`libcuflynx.generators.cpp.batched.c_expression` prints C).
Expressions are tuples::

    ('num', value)              a literal
    ('y', i)                    the block's i-th state
    ('in', k)                   the block's k-th input
    ('t',)                      the variable of integration
    ('tmp', name)               a temporary assigned earlier in the program
    (op, operand, ...)          an operation, op one of OPERATIONS

Booleans are produced by comparisons and logic and consumed by ``select`` (``('select', c, a, b)`` is ``a``
where ``c`` holds, else ``b``) and by logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import libcellml as lc

T = lc.AnalyserEquationAst.Type
VT = lc.AnalyserVariable.Type
ET = lc.AnalyserEquation.Type

#: Every operation an expression can hold, with its number of operands.
OPERATIONS = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'neg': 1, 'pow': 2, 'sqrt': 1, 'exp': 1, 'ln': 1, 'log10': 1,
    'abs': 1, 'sign': 1, 'floor': 1, 'ceil': 1, 'min': 2, 'max': 2, 'rem': 2, 'trunc': 1,
    'sin': 1, 'cos': 1, 'tan': 1, 'sinh': 1, 'cosh': 1, 'tanh': 1, 'asin': 1, 'acos': 1, 'atan': 1,
    'asinh': 1, 'acosh': 1, 'atanh': 1,
    'lt': 2, 'le': 2, 'gt': 2, 'ge': 2, 'eq': 2, 'ne': 2, 'and': 2, 'or': 2, 'xor': 2, 'not': 1,
    'select': 3,
}
BOOLEAN_OPERATIONS = {'lt', 'le', 'gt', 'ge', 'eq', 'ne', 'and', 'or', 'xor', 'not'}

ZERO = ('num', 0.0)
ONE = ('num', 1.0)
NAN = ('num', math.nan)


class KernelIRError(ValueError):
    """The model has something the kernels cannot express (an algebraic loop, an unknown MathML element)."""


@dataclass(frozen=True)
class ModelVariable:
    """A variable of the analysed model: ``kind`` is 'state' (``index`` into the states) or 'constant',
    'computed_constant', 'algebraic' or 'external' (``index`` into libCellML's one variables array)."""
    kind: str
    index: int
    name: str
    component: str
    units: str
    initial: object = None     # a state's initial value: a float, or the name of the variable that sets it

    @property
    def qname(self):
        return f'{self.component}.{self.name}'


@dataclass
class Program:
    """Assignments to temporaries, in order, then the program's results: ``rates[i]`` is the rate of the
    block's i-th state; ``entries`` are the Jacobian's structurally non-zero ``(row, column, expression)``."""
    assignments: list = field(default_factory=list)      # [(name, expression)]
    rates: list = field(default_factory=list)
    entries: list = field(default_factory=list)


@dataclass
class Kernel:
    """The rates and Jacobian of a block of ``n_states`` states reading ``n_inputs`` inputs, shared by every
    block with the same programs. ``shared_inputs`` are inputs every one of those blocks reads from the same
    variable (an emitter may read them once per call)."""
    index: int
    n_states: int
    n_inputs: int
    rates: Program
    jacobian: Program
    shared_inputs: tuple = ()

    @property
    def nonzeros(self):
        return len(self.jacobian.entries)


@dataclass
class Block:
    """States that depend only on one another: the model's ``states`` (indices) in the kernel's order, and
    the model variables (one-array indices) behind the kernel's ``inputs``."""
    kernel: int
    states: list
    inputs: list


@dataclass
class KernelModel:
    """The kernels and blocks of an analysed model, with its variables for a descriptor."""
    states: list            # [ModelVariable], by state index
    variables: list         # [ModelVariable], by one-array index
    voi: ModelVariable
    kernels: list
    blocks: list

    @property
    def state_count(self):
        return len(self.states)

    @property
    def variable_count(self):
        return len(self.variables)


# --------------------------------------------------------------------------------------------------
# The analysed model's variables
# --------------------------------------------------------------------------------------------------

_KINDS = ((VT.CONSTANT, 'constant'), (VT.COMPUTED_CONSTANT, 'computed_constant'),
          (VT.ALGEBRAIC_VARIABLE, 'algebraic'), (VT.EXTERNAL_VARIABLE, 'external'))


def _model_variable(av, kind, index):
    v = av.variable()
    units = v.units().name() if v.units() is not None else ''
    initial = None
    if kind == 'state':
        init = v.initialValue()
        try:
            initial = float(init)
        except (TypeError, ValueError):
            initial = init or None
    return ModelVariable(kind, index, v.name(), v.parent().name(), units, initial)


class _Variables:
    """Maps the analyser's variables to ModelVariables (states by state index, the others by their index in
    the one variables array: constants, computed constants, algebraic, external, as #271 lays it out)."""

    def __init__(self, am):
        self.am = am
        self.states = [_model_variable(s, 'state', s.index()) for s in sorted(am.states(), key=lambda s: s.index())]
        self.variables = []
        self.offset = {}
        for vtype, kind in _KINDS:
            self.offset[vtype] = len(self.variables)
            group = {VT.CONSTANT: am.constants, VT.COMPUTED_CONSTANT: am.computedConstants,
                     VT.ALGEBRAIC_VARIABLE: am.algebraicVariables, VT.EXTERNAL_VARIABLE: am.externalVariables}[vtype]()
            for av in sorted(group, key=lambda a: a.index()):
                self.variables.append(_model_variable(av, kind, len(self.variables)))
        self.voi = _model_variable(am.voi(), 'voi', 0)

    def of(self, variable):
        """The ModelVariable for a variable of the flat model (any of its equivalent variables)."""
        av = self.am.analyserVariable(variable)
        if av is None:
            raise KernelIRError(f'{variable.parent().name()}.{variable.name()} is not in the analysed model')
        t = av.type()
        if t == VT.VARIABLE_OF_INTEGRATION:
            return self.voi
        if t == VT.STATE:
            return self.states[av.index()]
        return self.variables[self.offset[t] + av.index()]


# --------------------------------------------------------------------------------------------------
# Expressions from equation trees
# --------------------------------------------------------------------------------------------------

_UNARY = {T.ABS: 'abs', T.EXP: 'exp', T.LN: 'ln', T.CEILING: 'ceil', T.FLOOR: 'floor', T.SIN: 'sin',
          T.COS: 'cos', T.TAN: 'tan', T.SINH: 'sinh', T.COSH: 'cosh', T.TANH: 'tanh', T.ASIN: 'asin',
          T.ACOS: 'acos', T.ATAN: 'atan', T.ASINH: 'asinh', T.ACOSH: 'acosh', T.ATANH: 'atanh'}
_RECIPROCAL = {T.SEC: 'cos', T.CSC: 'sin', T.COT: 'tan', T.SECH: 'cosh', T.CSCH: 'sinh', T.COTH: 'tanh'}
_INVERSE_RECIPROCAL = {T.ASEC: 'acos', T.ACSC: 'asin', T.ACOT: 'atan', T.ASECH: 'acosh', T.ACSCH: 'asinh',
                       T.ACOTH: 'atanh'}     # asec(x) = acos(1/x), ...
_BINARY = {T.EQ: 'eq', T.NEQ: 'ne', T.LT: 'lt', T.LEQ: 'le', T.GT: 'gt', T.GEQ: 'ge', T.AND: 'and',
           T.OR: 'or', T.XOR: 'xor', T.MIN: 'min', T.MAX: 'max', T.REM: 'rem'}


def _num(value):
    return ('num', float(value))


class _TreeReader:
    """Equation trees to expressions, with each variable reference given by ``ref(ModelVariable)``."""

    def __init__(self, variables, ref):
        self.vars = variables
        self.ref = ref

    def expr(self, ast):
        t = ast.type()
        left, right = ast.leftChild(), ast.rightChild()
        if t == T.CI:
            return self.ref(self.vars.of(ast.variable()))
        if t == T.CN:
            return _num(ast.value())
        if t == T.PLUS:
            return ('add', self.expr(left), self.expr(right)) if right is not None else self.expr(left)
        if t == T.MINUS:
            return ('sub', self.expr(left), self.expr(right)) if right is not None else ('neg', self.expr(left))
        if t == T.TIMES:
            return ('mul', self.expr(left), self.expr(right))
        if t == T.DIVIDE:
            return ('div', self.expr(left), self.expr(right))
        if t == T.POWER:
            return ('pow', self.expr(left), self.expr(right))
        if t == T.ROOT:
            if right is None:
                return ('sqrt', self.expr(left))
            return ('pow', self.expr(right), ('div', ONE, self.expr(left.leftChild())))
        if t == T.LOG:
            if right is None:
                return ('log10', self.expr(left))
            return ('div', ('ln', self.expr(right)), ('ln', self.expr(left.leftChild())))
        if t in _UNARY:
            return (_UNARY[t], self.expr(left))
        if t in _RECIPROCAL:
            return ('div', ONE, (_RECIPROCAL[t], self.expr(left)))
        if t in _INVERSE_RECIPROCAL:
            return (_INVERSE_RECIPROCAL[t], ('div', ONE, self.expr(left)))
        if t in _BINARY:
            return (_BINARY[t], self.expr(left), self.expr(right))
        if t == T.NOT:
            return ('not', self.expr(left))
        if t == T.PIECEWISE:
            return self.piecewise(ast)
        if t == T.PIECE:
            return ('select', self.expr(right), self.expr(left), NAN)
        if t == T.TRUE:
            return ONE
        if t == T.FALSE:
            return ZERO
        if t == T.E:
            return _num(math.e)
        if t == T.PI:
            return _num(math.pi)
        if t == T.INF:
            return _num(math.inf)
        if t == T.NAN:
            return NAN
        raise KernelIRError(f'unsupported MathML element {lc.AnalyserEquationAst.typeAsString(t)}')

    def piecewise(self, ast):
        # PIECEWISE(left=PIECE(value, condition), right=PIECE | OTHERWISE | PIECEWISE | None)
        left, right = ast.leftChild(), ast.rightChild()
        if right is None:
            rest = NAN
        elif right.type() == T.OTHERWISE:
            rest = self.expr(right.leftChild())
        elif right.type() == T.PIECE:
            rest = ('select', self.expr(right.rightChild()), self.expr(right.leftChild()), NAN)
        else:
            rest = self.piecewise(right)
        return ('select', self.expr(left.rightChild()), self.expr(left.leftChild()), rest)


def _uses_rate(ast):
    if ast is None:
        return False
    return ast.type() == T.DIFF or _uses_rate(ast.leftChild()) or _uses_rate(ast.rightChild())


# --------------------------------------------------------------------------------------------------
# Simplifying constructors (structural zeros stay None, so a derivative that is zero is never written)
# --------------------------------------------------------------------------------------------------

def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return ('add', a, b)


def _sub(a, b):
    if b is None:
        return a
    if a is None:
        return ('neg', b)
    return ('sub', a, b)


def _mul(a, b):
    if a is None or b is None:
        return None
    if a == ONE:
        return b
    if b == ONE:
        return a
    return ('mul', a, b)


def _neg(a):
    return None if a is None else ('neg', a)


# --------------------------------------------------------------------------------------------------
# Forward-mode differentiation
# --------------------------------------------------------------------------------------------------

def derivative(e, j, tangent):
    """d e / d y_j, or None where it is structurally zero. ``tangent(name, j)`` is the derivative of the
    temporary ``name`` (an expression, or None when the temporary does not depend on y_j). A piecewise
    expression differentiates its active branch."""
    op = e[0]
    if op == 'y':
        return ONE if e[1] == j else None
    if op == 'tmp':
        return tangent(e[1], j)
    if op in ('num', 'in', 't') or op in BOOLEAN_OPERATIONS or op in ('floor', 'ceil', 'sign', 'trunc'):
        return None
    d = lambda x: derivative(x, j, tangent)     # noqa: E731
    if op == 'add':
        return _add(d(e[1]), d(e[2]))
    if op == 'sub':
        return _sub(d(e[1]), d(e[2]))
    if op == 'neg':
        return _neg(d(e[1]))
    if op == 'mul':
        return _add(_mul(d(e[1]), e[2]), _mul(e[1], d(e[2])))
    if op == 'div':
        u, v = e[1], e[2]
        du, dv = d(u), d(v)
        if du is None and dv is None:
            return None
        if dv is None:
            return ('div', du, v)
        # (du - (u/v) dv) / v: no v^2 to underflow when v is tiny (a concentration near zero in a denominator)
        return ('div', _sub(du, ('mul', ('div', u, v), dv)), v)
    if op == 'pow':
        u, w = e[1], e[2]
        du, dw = d(u), d(w)
        if du is None and dw is None:
            return None
        if dw is None:
            if w == ('num', 2.0):
                return ('mul', ('mul', _num(2.0), u), du)
            return ('mul', ('mul', w, ('pow', u, ('sub', w, ONE))), du)
        inner = _add(('mul', dw, ('ln', u)), None if du is None else ('div', ('mul', w, du), u))
        return ('mul', e, inner)
    if op == 'sqrt':
        du = d(e[1])
        return None if du is None else ('div', du, ('mul', _num(2.0), e))
    if op == 'exp':
        du = d(e[1])
        return None if du is None else ('mul', e, du)
    if op == 'ln':
        du = d(e[1])
        return None if du is None else ('div', du, e[1])
    if op == 'log10':
        du = d(e[1])
        return None if du is None else ('div', du, ('mul', e[1], _num(math.log(10.0))))
    if op == 'abs':
        du = d(e[1])
        return None if du is None else ('mul', ('sign', e[1]), du)
    if op in ('min', 'max'):
        du, dv = d(e[1]), d(e[2])
        if du is None and dv is None:
            return None
        cond = ('le' if op == 'min' else 'ge', e[1], e[2])
        return ('select', cond, du or ZERO, dv or ZERO)
    if op == 'rem':      # rem(u, v) = u - trunc(u/v) v
        du, dv = d(e[1]), d(e[2])
        if dv is None:
            return du
        return _sub(du, ('mul', ('trunc', ('div', e[1], e[2])), dv))
    if op == 'select':
        da, db = d(e[2]), d(e[3])
        if da is None and db is None:
            return None
        return ('select', e[1], da or ZERO, db or ZERO)
    u = e[1]
    du = d(u)
    if du is None:
        return None
    outer = {
        'sin': lambda: ('cos', u),
        'cos': lambda: ('neg', ('sin', u)),
        'tan': lambda: ('div', ONE, ('mul', ('cos', u), ('cos', u))),
        'sinh': lambda: ('cosh', u),
        'cosh': lambda: ('sinh', u),
        'tanh': lambda: ('sub', ONE, ('mul', e, e)),
        'asin': lambda: ('div', ONE, ('sqrt', ('sub', ONE, ('mul', u, u)))),
        'acos': lambda: ('neg', ('div', ONE, ('sqrt', ('sub', ONE, ('mul', u, u))))),
        'atan': lambda: ('div', ONE, ('add', ONE, ('mul', u, u))),
        'asinh': lambda: ('div', ONE, ('sqrt', ('add', ('mul', u, u), ONE))),
        'acosh': lambda: ('div', ONE, ('sqrt', ('sub', ('mul', u, u), ONE))),
        'atanh': lambda: ('div', ONE, ('sub', ONE, ('mul', u, u))),
    }
    if op not in outer:
        raise KernelIRError(f'no derivative for operation {op}')
    return ('mul', outer[op](), du)


def references(e, out):
    """Add to ``out`` the temporaries ('tmp', name), states ('y', i) and inputs ('in', k) ``e`` reads."""
    op = e[0]
    if op in ('tmp', 'y', 'in'):
        out.add(e)
    elif op not in ('num', 't'):
        for x in e[1:]:
            references(x, out)
    return out


def prune(program):
    """Drop temporaries nothing reads, directly or through other temporaries."""
    used = set()
    for e in program.rates:
        references(e, used)
    for _, _, e in program.entries:
        references(e, used)
    kept = []
    for name, e in reversed(program.assignments):
        if ('tmp', name) in used:
            kept.append((name, e))
            references(e, used)
    program.assignments = kept[::-1]
    return program


# --------------------------------------------------------------------------------------------------
# Building the kernels
# --------------------------------------------------------------------------------------------------

class _Equations:
    """The model's ODEs (by state index) and algebraic equations (by one-array index), as expressions over
    model references: ('state', i), ('var', k) for inputs, ('alg', k) for algebraic variables, ('t',).

    Computed constants count as algebraic variables here: a kernel computes those its rates read from the
    constants, so a constant that changes along a protocol trace reaches everything that depends on it."""

    def __init__(self, am, variables):
        def ref(mv):
            if mv.kind == 'voi':
                return ('t',)
            if mv.kind == 'state':
                return ('state', mv.index)
            if mv.kind in ('algebraic', 'computed_constant'):
                return ('alg', mv.index)
            return ('var', mv.index)

        reader = _TreeReader(variables, ref)
        self.ode = {}
        self.algebraic = {}
        for eq in am.analyserEquations():
            t = eq.type()
            if t == ET.NLA:
                raise KernelIRError('the model has an algebraic loop (non-linear algebraic equations): '
                                    'kernels need every algebraic variable in explicit form')
            ast = eq.ast()
            if ast is None:
                continue          # an external variable: an input
            lhs, rhs = ast.leftChild(), ast.rightChild()
            if t == ET.ODE:
                if lhs.type() != T.DIFF:
                    raise KernelIRError('an ODE is not in the form d(state)/dt = expression')
                self.ode[eq.state(0).index()] = reader.expr(rhs)
            else:
                # by the variable it defines, whatever libCellML calls the equation (a computed constant can be
                # defined by a literal, in an equation of type 'constant')
                if lhs is None or lhs.type() != T.CI:
                    raise KernelIRError('an algebraic equation is not in explicit form')
                target = variables.of(lhs.variable())
                if target.kind not in ('algebraic', 'computed_constant'):
                    continue      # a constant (an input), or a state's initial value (libCellML's code sets it)
                if _uses_rate(rhs):
                    # read by outputs only (libCellML's computeVariables computes it), never by a rate
                    self.algebraic[target.index] = None
                    continue
                self.algebraic[target.index] = reader.expr(rhs)
        missing = set(range(len(variables.states))) - set(self.ode)
        if missing:
            raise KernelIRError(f'no ODE found for states {sorted(missing)[:5]}')

    def closure(self, roots):
        """The algebraic variables the expressions ``roots`` read, directly or not, in dependency order."""
        order, seen = [], set()

        def visit(k, stack):
            if k in seen:
                return
            if k in stack:
                raise KernelIRError('algebraic variables depend on one another in a loop')
            e = self.algebraic.get(k)
            if e is None:
                raise KernelIRError(f'algebraic variable {k} is needed by a rate but depends on a rate')
            stack.add(k)
            for r in _model_refs(e):
                if r[0] == 'alg':
                    visit(r[1], stack)
            stack.discard(k)
            seen.add(k)
            order.append(k)

        for e in roots:
            for r in _model_refs(e):
                if r[0] == 'alg':
                    visit(r[1], set())
        return order


def _model_refs(e, out=None):
    """The model references in ``e``, in order of first appearance (so equal structures visit equally)."""
    out = {} if out is None else out
    op = e[0]
    if op in ('state', 'var', 'alg'):
        out.setdefault(e, None)
    elif op not in ('num', 't'):
        for x in e[1:]:
            _model_refs(x, out)
    return out


def _blocks_of(eqs, n_states):
    """Connected components of the graph joining each rate to the states it reads."""
    parent = list(range(n_states))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    alg_states = {}

    def states_of_alg(k):
        if k not in alg_states:
            alg_states[k] = set()          # placeholder: closure() has already refused loops
            found = set()
            for r in _model_refs(eqs.algebraic[k]):
                if r[0] == 'state':
                    found.add(r[1])
                elif r[0] == 'alg':
                    found |= states_of_alg(r[1])
            alg_states[k] = found
        return alg_states[k]

    for i in range(n_states):
        eqs.closure([eqs.ode[i]])          # refuses loops and rate-dependent algebraics a rate needs
        reads = set()
        for r in _model_refs(eqs.ode[i]):
            if r[0] == 'state':
                reads.add(r[1])
            elif r[0] == 'alg':
                reads |= states_of_alg(r[1])
        for j in reads:
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups = {}
    for i in range(n_states):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def _localise(eqs, states):
    """A block's rates as a Program over local names, and the model variables behind its inputs (in order of
    first use). Temporaries are named a0, a1, ... in the order they are assigned."""
    local_state = {s: i for i, s in enumerate(states)}
    inputs, input_index, tmp_name = [], {}, {}
    order = eqs.closure([eqs.ode[s] for s in states])

    def local(e):
        op = e[0]
        if op == 'state':
            if e[1] not in local_state:
                raise KernelIRError('a block reads a state outside it')
            return ('y', local_state[e[1]])
        if op == 'var':
            if e[1] not in input_index:
                input_index[e[1]] = len(inputs)
                inputs.append(e[1])
            return ('in', input_index[e[1]])
        if op == 'alg':
            return ('tmp', tmp_name[e[1]])
        if op in ('num', 't'):
            return e
        return (op,) + tuple(local(x) for x in e[1:])

    program = Program()
    for k in order:
        tmp_name[k] = f'a{len(tmp_name)}'
        program.assignments.append((tmp_name[k], local(eqs.algebraic[k])))
    program.rates = [local(eqs.ode[s]) for s in states]
    return program, inputs


def jacobian_program(rates, n_states):
    """The Jacobian of a rates Program, by forward-mode differentiation: each temporary a gets a tangent
    ``d<a>_<j>`` for each state j it depends on (structurally), and each rate its non-zero entries."""
    deps = {}           # temporary -> states it depends on
    program = Program()

    def states_read(e):
        out = set()
        for r in references(e, set()):
            if r[0] == 'y':
                out.add(r[1])
            elif r[0] == 'tmp':
                out |= deps[r[1]]
        return out

    def tangent(name, j):
        return ('tmp', f'd{name}_{j}') if j in deps.get(name, ()) else None

    for name, e in rates.assignments:
        program.assignments.append((name, e))
        deps[name] = states_read(e)
        for j in sorted(deps[name]):
            program.assignments.append((f'd{name}_{j}', derivative(e, j, tangent) or ZERO))
    for row, e in enumerate(rates.rates):
        for j in sorted(states_read(e)):
            d = derivative(e, j, tangent)
            if d is not None:
                program.entries.append((row, j, d))
    return prune(program)


def build_kernel_model(am):
    """Kernels and blocks for an analysed ODE model (a libcellml AnalyserModel)."""
    if am.type() in (lc.AnalyserModel.Type.DAE, lc.AnalyserModel.Type.NLA):
        raise KernelIRError(f'the model is {lc.AnalyserModel.typeAsString(am.type())}: it has an algebraic loop '
                            '(non-linear algebraic equations), and kernels need every algebraic variable in '
                            'explicit form')
    if am.type() != lc.AnalyserModel.Type.ODE:
        raise KernelIRError(f'the analysed model is {lc.AnalyserModel.typeAsString(am.type())}, not an ODE model')
    variables = _Variables(am)
    eqs = _Equations(am, variables)
    kernels, blocks, by_key = [], [], {}
    for states in _blocks_of(eqs, len(variables.states)):
        rates, inputs = _localise(eqs, states)
        key = repr((len(states), rates.assignments, rates.rates))
        k = by_key.get(key)
        if k is None:
            k = len(kernels)
            by_key[key] = k
            kernels.append(Kernel(k, len(states), len(inputs), rates, jacobian_program(rates, len(states))))
        blocks.append(Block(k, states, inputs))
    for kernel in kernels:
        mine = [b for b in blocks if b.kernel == kernel.index]
        kernel.shared_inputs = tuple(i for i in range(kernel.n_inputs) if len({b.inputs[i] for b in mine}) == 1)
    return KernelModel(variables.states, variables.variables, variables.voi, kernels, blocks)


def output_indices(am, outputs):
    """The one-array indices of the algebraic variables or computed constants named in ``outputs``
    (``component/name``, or a name only one component has), in the order given."""
    by_name = {}
    for v in _Variables(am).variables:
        if v.kind in ('algebraic', 'computed_constant'):
            by_name.setdefault(f'{v.component}/{v.name}', []).append(v.index)
            by_name.setdefault(v.name, []).append(v.index)
    found = []
    for name in outputs:
        k = by_name.get(name, [])
        if len(k) != 1:
            raise KernelIRError(f'output {name!r} names {len(k)} algebraic variables or computed constants, not one')
        found.append(k[0])
    return found


def outputs_program(am, outputs):
    """Assignments that compute the algebraic variables named in ``outputs`` (``component/name``, or a name only
    one component has) and those they read, in dependency order: ``[(one-array index, expression)]`` over the
    model's states ``('y', i)``, the one variables array ``('in', k)`` (inputs and the variables assigned before)
    and time. For a caller that reads a few algebraic variables after every step and the rest only on demand.
    Computed constants are read from the variables, where libCellML's computeComputedConstants keeps them, so an
    output has the same value whichever way it is computed; a computed constant may itself be an output."""
    variables = _Variables(am)
    eqs = _Equations(am, variables)
    roots = []
    for name, k in zip(outputs, output_indices(am, outputs)):
        if variables.variables[k].kind == 'computed_constant':
            continue
        if eqs.algebraic.get(k) is None:
            raise KernelIRError(f'output {name!r} depends on a rate')
        roots.append(('alg', k))

    def global_(e):
        op = e[0]
        if op == 'state':
            return ('y', e[1])
        if op in ('var', 'alg'):
            return ('in', e[1])
        if op in ('num', 't'):
            return e
        return (op,) + tuple(global_(x) for x in e[1:])
    return [(k, global_(eqs.algebraic[k])) for k in eqs.closure(roots) if variables.variables[k].kind == 'algebraic']


def analyse_cellml(path, external_variables=()):
    """Parse, flatten and analyse a CellML file; ``external_variables`` are (component, variable) names to
    make external. Returns (flat model, AnalyserModel)."""
    import os
    import libcuflynx.utilities.libcellml_helper_funcs as cellml
    model = cellml.parse_model(path, False)
    importer = cellml.resolve_imports(model, os.path.dirname(os.path.abspath(path)), False)
    flat = cellml.flatten_model(model, importer)
    analyser = lc.Analyser()
    for component, name in external_variables:
        c = flat.component(component, True)
        if c is None or c.variable(name) is None:
            raise KernelIRError(f'no variable {component}.{name} to make external')
        analyser.addExternalVariable(lc.AnalyserExternalVariable(c.variable(name)))
    analyser.analyseModel(flat)
    if analyser.errorCount():
        raise KernelIRError('libCellML analyser errors:\n  ' + '\n  '.join(
            analyser.error(i).description() for i in range(analyser.errorCount())))
    return flat, analyser.analyserModel()
