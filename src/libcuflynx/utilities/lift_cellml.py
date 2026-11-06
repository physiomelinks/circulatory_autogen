"""Lift a smooth flat CellML 2.0 model to exactly quadratic form, for Operator Inference (Lift & Learn).

A lifting map T adds auxiliary states w_aux = T_aux(s) to the original states s, so that the lifted
state w = (s, w_aux) obeys an ODE that is quadratic in w and affine in the inputs u::

    dw/dt = A w + H (w (x) w) + sum_k N_k w u_k + B u + C

(Qian, Kramer, Peherstorfer & Willcox 2020, Definition 1 and eq. 62). The original states are the first n
entries of w, so the reverse map is the projection w[:n].

Use on a model smoothed by :mod:`libcuflynx.utilities.smooth_cellml` (``cuflynx-smooth``): its smooth forms use
only square roots, reciprocals, products and sums, which lift exactly.

The steps (:func:`lift`):

1. **Read.** The flat model's MathML becomes sympy, one symbol per connected set of variables. Constants
   (no equation, a numeric initial value) are parameters; algebraic chains are substituted into the state
   right-hand sides.
2. **Inputs.** Named variables (``inputs``: module ports, open boundary conditions) stay symbols u. A
   state-free subexpression that depends on time (``sin(w t)``, a pacing piecewise) becomes an input
   given by that expression, and so does a non-smooth subexpression left in the model (an exempt site of
   the smoothing pass, ``floor``): the lifted model computes it from the original states, as the original
   does. A parameter-only subexpression that is not a monomial becomes a derived parameter.
3. **Polynomialize.** QBee's ``polynomialize`` (in a subprocess with a time budget), checked against the
   chain rule; if it fails, a rule table: each non-polynomial node f(g) becomes an auxiliary w = f(g) with
   dw/dt = f'(g) dg/dt rewritten in the auxiliaries -- exp(g): w g'; ln g: g' z with z = 1/g;
   g^(a/b): w = g^(1/b), z = g^(-1/b); 1/D: -w^2 D'; sin/cos as a pair; tanh: (1 - w^2) g'.
   Input-free (Bychkov et al. 2023): the auxiliaries depend on the states only. Where an argument mixes
   states and inputs, the derivative of an input that is a known function of time is added as an input
   (``d<u>_dt``) and reported; otherwise the term becomes an input of its own (reported, not input-free).
4. **Quadratize.** QBee's ``quadratize(input_free=True)`` (subprocess, time budget), else a greedy search that
   adds the monomial removing the most terms of degree > 2 until every term is quadratic (a term with an
   input may carry at most one state factor).
5. **Outputs.** :func:`write_outputs` writes the lifted CellML model (each auxiliary initialised at
   T(s(0))), ``lifting.json`` and ``lifting_functions.py`` (vectorised numpy T, J, inputs and right-hand
   sides). :func:`verify` simulates the lifted and the smoothed model with CVODE_myokit and reports the
   error in the original states, the distance from the manifold max |w - T(s)|, J f = rhs_lifted at sampled
   states and the operator residual; :func:`structure` reports clocks, input kinds, the block-triangular
   pattern and conserved linear functionals.

In Python: :func:`lift_cellml_file` does all of it; ``cuflynx-lift`` is its command.
"""

import argparse
import itertools
import json
import keyword
import math
import multiprocessing
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

try:
    import sympy as sp
except ModuleNotFoundError as e:  # pragma: no cover - checked in a subprocess
    raise ModuleNotFoundError(f'{e}: the lifting needs pip install "libcuflynx[lift]"') from None

from libcuflynx.utilities.smooth_cellml import _C, _M, _args, _op, _read, _tag

CELLML2_NS = 'http://www.cellml.org/cellml/2.0#'
MATHML_NS = 'http://www.w3.org/1998/Math/MathML'
DEFAULT_TIME_BUDGET = 60.0          # s for each QBee stage
DEFAULT_MAX_AUX = 400               # auxiliaries either fallback may add
DEFAULT_MAX_TERMS = 20000           # terms of the polynomial system beyond which no quadratization is attempted
QBEE_MAX_TERMS = 2000               # terms beyond which QBee's branch-and-bound is not tried (its examples are ~12 states)
#: smoothness class of each smoothing rule's original operation, and of its smooth form
SMOOTHNESS = {'abs': 'C0 (Lipschitz, kink at x = 0)', 'maxmin': 'C0 (Lipschitz, kink where a = b)',
              'rectifier': 'C0 (Lipschitz, kink at x = 0)',
              'piecewise': 'C-1 (a jump) unless the branches meet at the switch, then C0',
              'division': 'singular (unbounded where the denominator crosses 0)'}
_NONSMOOTH = (sp.Piecewise, sp.Abs, sp.Max, sp.Min, sp.floor, sp.ceiling, sp.sign, sp.Heaviside, sp.Mod)
_RESERVED = {'np', 'numpy', 'p', 'X', 'W', 'U', 'math', 'sp', 'E', 'I', 'N', 'O', 'S', 'Q', 'pi', 'oo', 'zoo',
             'nan', 'beta', 'gamma', 'zeta', 'lambda', 'Lambda', 'ff', 'rf'}


class LiftError(RuntimeError):
    """The model cannot be read or lifted."""


# ---------------------------------------------------------------------------------------------
# reading: flat CellML MathML -> sympy

_FUNCS = {'exp': sp.exp, 'ln': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'sinh': sp.sinh,
          'cosh': sp.cosh, 'tanh': sp.tanh, 'arcsin': sp.asin, 'arccos': sp.acos, 'arctan': sp.atan,
          'arcsinh': sp.asinh, 'arccosh': sp.acosh, 'arctanh': sp.atanh, 'abs': sp.Abs, 'floor': sp.floor,
          'ceiling': sp.ceiling, 'sec': sp.sec, 'csc': sp.csc, 'cot': sp.cot}
_REL = {'lt': sp.StrictLessThan, 'gt': sp.StrictGreaterThan, 'leq': sp.LessThan, 'geq': sp.GreaterThan,
        'eq': sp.Eq, 'neq': sp.Ne}


def _number(text):
    text = text.strip()
    try:
        return sp.Integer(int(text))
    except ValueError:
        value = float(text)
        r = sp.Rational(value).limit_denominator(10000)
        return r if float(r) == value and r.q <= 1000 else sp.Float(value)


class _Reader:
    """The variables of a flat model, one per connected set, and its equations as sympy."""

    def __init__(self, root):
        self.root = root
        self.parent = {}
        for conn in root.findall(_C + 'connection'):
            c1, c2 = conn.get('component_1'), conn.get('component_2')
            for mv in conn.findall(_C + 'map_variables'):
                a, b = self.find((c1, mv.get('variable_1'))), self.find((c2, mv.get('variable_2')))
                if a != b:
                    self.parent[a] = b
        self.members, self.init, self.units = {}, {}, {}
        for comp in root.findall(_C + 'component'):
            for v in comp.findall(_C + 'variable'):
                key = self.find((comp.get('name'), v.get('name')))
                self.members.setdefault(key, []).append((comp.get('name'), v.get('name')))
                if v.get('initial_value') is not None:
                    self.init.setdefault(key, (comp.get('name'), v.get('initial_value')))
                self.units.setdefault(key, v.get('units'))
        self.state_of, self.alg_of, self.time_key, self.home = {}, {}, None, {}
        self._raw = []
        for comp in root.findall(_C + 'component'):
            name = comp.get('name')
            for m in comp.findall(_M + 'math'):
                for eq in m:
                    if _op(eq) != 'eq' or len(_args(eq)) != 2:
                        raise LiftError(f'{name}: an equation that is not "a = b"')
                    lhs, rhs = _args(eq)
                    if _op(lhs) == 'diff':
                        bvar = lhs.find(_M + 'bvar')[0].text.strip()
                        tkey = self.find((name, bvar))
                        if self.time_key not in (None, tkey):
                            raise LiftError('more than one variable of integration')
                        self.time_key = tkey
                        target = _args(lhs)[0].text.strip()
                        key = self.find((name, target))
                        self.state_of[key] = (name, rhs)
                        self.home[key] = (name, target)
                    elif _tag(lhs) == 'ci':
                        key = self.find((name, lhs.text.strip()))
                        if key in self.alg_of:
                            raise LiftError(f'{name}/{lhs.text.strip()} has two equations')
                        self.alg_of[key] = (name, rhs)
                        self.home[key] = (name, lhs.text.strip())
                    else:
                        raise LiftError(f'{name}: an equation whose left-hand side is not a variable or a rate')
        for key, (comp, _value) in self.init.items():
            self.home.setdefault(key, (comp, next(v for c, v in self.members[key] if c == comp)))
        if self.time_key is not None:
            self.home.setdefault(self.time_key, self.time_key)
        # names: the variable's own name if unique, else component__variable; never a Python keyword or a
        # name sympy parses as something else
        homes = {k: self.home.get(k, k) for k in self.members}
        counts = {}
        for c, v in homes.values():
            counts[v] = counts.get(v, 0) + 1
        self.name, used = {}, set()
        for key, (c, v) in homes.items():
            n = v if counts[v] == 1 else f'{c}__{v}'
            while keyword.iskeyword(n) or n in _RESERVED or n in used:
                n += '_'
            used.add(n)
            self.name[key] = n
        self.sym = {k: sp.Symbol(n, real=True) for k, n in self.name.items()}

    def find(self, a):
        while self.parent.get(a, a) != a:
            a = self.parent[a]
        return a

    def qname(self, key):
        c, v = self.home.get(key, key)
        return f'{c}.{v}'

    def expr(self, e, comp):
        t = _tag(e)
        if t == 'ci':
            return self.sym[self.find((comp, e.text.strip()))]
        if t == 'cn':
            if e.get('type') == 'e-notation':
                sep = e.find(_M + 'sep')
                return _number(f'{(e.text or "").strip()}e{sep.tail.strip()}')
            return _number(e.text)
        if t == 'pi':
            return sp.pi
        if t == 'exponentiale':
            return sp.E
        if t == 'true':
            return sp.true
        if t == 'false':
            return sp.false
        if t == 'piecewise':
            pieces = [(self.expr(p[0], comp), self.expr(p[1], comp)) for p in e if _tag(p) == 'piece']
            pieces += [(self.expr(o[0], comp), True) for o in e if _tag(o) == 'otherwise']
            return sp.Piecewise(*pieces)
        if t != 'apply':
            raise LiftError(f'unsupported MathML element <{t}>')
        op, args = _op(e), [self.expr(a, comp) for a in _args(e)]
        if op == 'plus':
            return sp.Add(*args)
        if op == 'minus':
            return -args[0] if len(args) == 1 else args[0] - args[1]
        if op == 'times':
            return sp.Mul(*args)
        if op == 'divide':
            return args[0] / args[1]
        if op == 'power':
            return args[0] ** args[1]
        if op == 'root':
            degree = e.find(_M + 'degree')
            n = self.expr(degree[0], comp) if degree is not None else sp.Integer(2)
            return args[0] ** (1 / n)
        if op == 'log':
            base = e.find(_M + 'logbase')
            b = self.expr(base[0], comp) if base is not None else sp.Integer(10)
            return sp.log(args[0]) / sp.log(b)
        if op in _FUNCS:
            return _FUNCS[op](*args)
        if op in ('max', 'min'):
            return (sp.Max if op == 'max' else sp.Min)(*args)
        if op in _REL:
            return _REL[op](*args)
        if op == 'and':
            return sp.And(*args)
        if op == 'or':
            return sp.Or(*args)
        if op == 'not':
            return sp.Not(args[0])
        if op == 'xor':
            return sp.Xor(*args)
        if op == 'rem':
            return sp.Mod(*args)
        raise LiftError(f'unsupported MathML operator <{op}>')


# ---------------------------------------------------------------------------------------------
# the model as sympy

class Model:
    """A flat model as sympy: states, parameters, inputs and right-hand sides in the states only.

    Attributes:
        states: state symbols, in myokit's order when myokit can import the model, then the held inputs.
        state_qnames: the original states' ``component.variable`` names.
        n_orig: the number of original states (the first entries of ``states``).
        held: the inputs held as states (``held_inputs``): du/dt = 0 within a chunk.
        rhs: {state: right-hand side in states, smoothing sites, parameters and input symbols}.
        sites: {symbol: {kind, k, x, eps, T, z}}: the smoothing pass's ``sm<k>_w = sqrt(x^2 + eps^2)``,
            ``sm<k>_z = 1/sm<k>_w`` and ``sm<k>_q = 1/(x^2 + eps^2)``, kept whole: lifted variables with closed-form
            derivatives (``x`` in the states and other sites, ``T`` in the states).
        params: {parameter symbol: value}; ``derived``: {symbol: expression in parameters}, in order.
        inputs: [{symbol, name, expr, kind, ...}]: a named variable (``variable``), a variable cut from the model
            (``cut``: its values come from data), a known function of time (``time``), a non-smooth site
            (``nonsmooth``) or the derivative of an input (``derivative``).
        init: {state: initial value}; ``all_init`` also has the cut states'.
        time: the time symbol.
    """

    def __init__(self, text, inputs=(), myokit_order=True, as_inputs=(), keep=(), held_inputs=False, sites=True,
                 outputs=()):
        import fnmatch
        import re as _re
        root = _read(text)
        r = _Reader(root)
        self.reader = r
        if r.time_key is None:
            raise LiftError('no state variables')
        self.time = r.sym[r.time_key]

        def matching(patterns):
            found, used = set(), set()
            for key, members in r.members.items():
                for c, v in members:
                    for pat in patterns:
                        if v == pat or r.name[key] == pat or fnmatch.fnmatchcase(f'{c}/{v}', pat):
                            found.add(key)
                            used.add(pat)
            return found, set(patterns) - used

        input_keys, missing = matching(inputs or ())
        if missing:
            raise LiftError(f'inputs not in the model: {sorted(missing)}')
        state_keys_all = list(r.state_of)
        cut_keys, missing = matching(as_inputs or ())
        if missing:
            raise LiftError(f'as_inputs not in the model: {sorted(missing)}')
        if keep:
            kept = {k for k in state_keys_all if any(fnmatch.fnmatchcase(r.home[k][0], pat) for pat in keep)}
            if not kept:
                raise LiftError(f'keep: no states in components {sorted(keep)}')
            cut_keys |= set(state_keys_all) - kept
        self.params, self.inputs = {}, []
        alg_raw = {k: r.expr(e, c) for k, (c, e) in r.alg_of.items()}
        consts = {}
        for key in r.members:
            if key in r.state_of or key in r.alg_of or key == r.time_key:
                continue
            comp, value = r.init.get(key, (None, None))
            try:
                consts[key] = float(value)
            except (TypeError, ValueError):
                raise LiftError(f'{r.qname(key)} has no equation and no numeric initial value') from None
        cut_keys -= set(consts)
        stop = input_keys | cut_keys
        self._sym_key = {s: k for k, s in r.sym.items()}
        # smoothing sites, kept whole when their equations have the smoothing pass's forms
        site_keys = {}
        if sites:
            for key in alg_raw:
                c, v = r.home[key]
                mm = _re.fullmatch(r'sm(\d+)_(w|z|q)', v)
                if not mm or key in stop:
                    continue
                k, kind = mm.group(1), mm.group(2)
                xk, ek = r.find((c, f'sm{k}_x')), r.find((c, f'eps_sm{k}'))
                if xk not in r.sym or ek not in consts:
                    continue
                xs, es = r.sym[xk], r.sym[ek]
                form = {'w': sp.sqrt(es ** 2 + xs ** 2), 'q': 1 / (es ** 2 + xs ** 2)}.get(kind)
                if kind == 'z':
                    wk = r.find((c, f'sm{k}_w'))
                    form = 1 / r.sym[wk] if wk in r.sym else None
                if form is not None and sp.simplify(alg_raw[key] - form) == 0:
                    site_keys[key] = (int(k), kind, xk, ek, c)
            # a z whose w is not kept whole is not either
            site_keys = {key: v for key, v in site_keys.items()
                         if v[1] != 'z' or r.find((v[4], f'sm{v[0]}_w')) in site_keys}
        resolved_full, resolved_atomic = {}, {}

        def resolve(key, full, stack=()):
            cache = resolved_full if full else resolved_atomic
            if key in cache:
                return cache[key]
            if key in stack:
                raise LiftError(f'algebraic loop through {r.qname(key)}')
            e = alg_raw[key]
            subs = {}
            for x in e.free_symbols:
                k2 = self._key_of(x)
                if k2 in alg_raw and k2 not in stop and (full or k2 not in site_keys):
                    subs[x] = resolve(k2, full, stack + (key,))
            cache[key] = e.xreplace(subs) if subs else e
            return cache[key]

        for key in alg_raw:
            resolve(key, True)
        # a site is kept whole only when its argument reaches a state the lift keeps; one of inputs alone (held or
        # not) is a function of the inputs
        dyn_syms = {r.sym[k] for k in set(state_keys_all) - cut_keys}
        for key in list(site_keys):
            k_, kind, xk, ek, c = site_keys[key]
            x_full = resolved_full[xk] if xk in resolved_full else r.sym[xk]
            if not (x_full.free_symbols & dyn_syms):
                del site_keys[key]
        site_keys = {key: v for key, v in site_keys.items()
                     if v[1] != 'z' or r.find((v[4], f'sm{v[0]}_w')) in site_keys}
        # each kept site's argument sm<k>_x is lifted too, so a site's rate is x z times x', not a product of sums
        # (not when the argument is a single term already: a state or a monomial)
        for key, (k_, kind, xk, ek, c) in list(site_keys.items()):
            if kind in ('w', 'q') and xk in alg_raw and xk not in stop:
                x_full = resolved_full[xk]
                if len(sp.Add.make_args(x_full)) > 1 or not _is_poly_in(x_full, list(x_full.free_symbols)):
                    site_keys[xk] = (k_, 'x', xk, ek, c)
        for key in alg_raw:
            resolve(key, False)
        for key, value in consts.items():
            if key not in input_keys:
                self.params[r.sym[key]] = value
        param_vals = {s: v for s, v in self.params.items()}
        self.all_init = {}
        for key in state_keys_all:
            comp, value = r.init.get(key, (None, None))
            if value is None:
                raise LiftError(f'state {r.qname(key)} has no initial value')
            try:
                self.all_init[r.sym[key]] = float(value)
            except ValueError:
                k2 = r.find((comp, value))
                e = resolved_full.get(k2, r.sym[k2])
                try:
                    self.all_init[r.sym[key]] = _value(e, param_vals)
                except TypeError:
                    v = e.xreplace(param_vals)
                    raise LiftError(f'initial value of {r.qname(key)} is not a constant: {v}') from None
        # named and cut inputs: their symbol stays; their value is their constant, their equation or data
        for key in sorted(stop, key=lambda k: r.name[k]):
            s_ = r.sym[key]
            if key in consts:
                self.inputs.append({'symbol': s_, 'name': r.name[key], 'expr': sp.Float(consts[key]),
                                    'kind': 'variable', 'held': True, 'qname': r.qname(key)})
            elif key in cut_keys:
                e = s_ if key in r.state_of else alg_raw[key].xreplace(
                    {x: resolved_full[self._key_of(x)] for x in alg_raw[key].free_symbols
                     if self._key_of(x) in resolved_full and self._key_of(x) not in stop})
                self.inputs.append({'symbol': s_, 'name': r.name[key], 'expr': e, 'kind': 'cut',
                                    'qname': r.qname(key), 'state': key in r.state_of})
            elif key in alg_raw:
                e = alg_raw[key].xreplace({x: resolved_full[self._key_of(x)] for x in alg_raw[key].free_symbols
                                           if self._key_of(x) in resolved_full and self._key_of(x) not in stop})
                self.inputs.append({'symbol': s_, 'name': r.name[key], 'expr': e, 'kind': 'variable',
                                    'held': not (e.free_symbols - set(self.params)), 'qname': r.qname(key)})
            else:
                raise LiftError(f'input {r.qname(key)} is a state: name it in as_inputs to cut it')
        state_keys = [k for k in state_keys_all if k not in cut_keys]
        rhs = {}
        for key in state_keys:
            comp, e = r.state_of[key]
            e = r.expr(e, comp)
            subs = {x: resolved_atomic[self._key_of(x)] for x in e.free_symbols
                    if self._key_of(x) in resolved_atomic and self._key_of(x) not in stop}
            rhs[r.sym[key]] = e.xreplace(subs) if subs else e
        self.states = [r.sym[k] for k in state_keys]
        self.state_qnames = [r.qname(k) for k in state_keys]
        if myokit_order:
            order = _myokit_state_order(text)
            if order and set(self.state_qnames) <= set(order):
                order = [q for q in order if q in self.state_qnames]
                idx = [self.state_qnames.index(q) for q in order]
                self.states = [self.states[i] for i in idx]
                self.state_qnames = order
        self.init = {s_: self.all_init[s_] for s_ in self.states}
        self.n_orig = len(self.states)
        self.sites = {}
        for key, (k, kind, xk, ek, c) in site_keys.items():
            if kind == 'x':
                x = alg_raw[xk].xreplace({v: resolved_atomic[self._key_of(v)] for v in alg_raw[xk].free_symbols
                                          if self._key_of(v) in resolved_atomic and self._key_of(v) not in stop
                                          and self._key_of(v) not in site_keys})
            elif xk in site_keys:
                x = r.sym[xk]
            else:
                x = resolve(xk, False) if xk in alg_raw else r.sym[xk]
            zk = r.find((c, f'sm{k}_z'))
            self.sites[r.sym[key]] = {'kind': kind, 'k': k, 'x': x, 'eps': r.sym[ek], 'T': resolved_full[key],
                                      'z': r.sym[zk] if kind == 'w' and zk in site_keys else None,
                                      'qname': r.qname(key)}
        self.derived = {}
        self._abstract_counter = itertools.count()
        self.rhs = {}
        self._input_by_expr = {}
        # held inputs: states with du/dt = 0, before abstraction, so a function of them is not an input
        self.held = []
        self.held_mode = 'inputs' if held_inputs == 'inputs' else ('states' if held_inputs else None)
        if self.held_mode == 'inputs':
            # held inputs stay inputs (B and N terms) with du/dt = 0
            for i in self.inputs:
                if i['kind'] == 'cut' or (i['kind'] == 'variable' and i.get('held')):
                    i['held'] = True
        elif held_inputs:
            for i in [i for i in self.inputs if i['kind'] in ('variable', 'cut')]:
                if i['kind'] == 'variable' and not i.get('held'):
                    continue
                self.inputs.remove(i)
                self.held.append(dict(i))
                self.init[i['symbol']] = _value(i['expr'], param_vals, self.all_init, {self.time: 0.0})
        held_syms = [h['symbol'] for h in self.held]
        self._held_syms = set(held_syms)
        dynamic = set(self.states) | set(self.sites) | set(held_syms)
        for s_ in self.states:
            self.rhs[s_] = self._abstract(rhs[s_], dynamic)
        # outputs: named variables lifted with the states (their non-polynomial parts become auxiliaries)
        out_keys, missing = matching(outputs or ())
        if missing:
            raise LiftError(f'outputs not in the model: {sorted(missing)}')
        self.outputs = []
        for key in sorted(out_keys, key=lambda k: r.qname(k)):
            e = resolve(key, False) if key in alg_raw else r.sym[key]
            self.outputs.append({'name': r.name[key], 'qname': r.qname(key), 'expr': self._abstract(e, dynamic)})
        for s_, info in self.sites.items():
            info['x'] = self._abstract(info['x'], dynamic)
        # only the sites and cut inputs the kept states reach
        reach, todo = set(), set().union(*[self.rhs[s_].free_symbols for s_ in self.states],
                                         *[o['expr'].free_symbols for o in self.outputs]) if self.states else set()
        while todo:
            x = todo.pop()
            if x in reach:
                continue
            reach.add(x)
            if x in self.sites:
                todo |= self.sites[x]['x'].free_symbols
                if self.sites[x]['z'] is not None:
                    todo.add(self.sites[x]['z'])
            for i in self.inputs:
                if i['symbol'] == x:
                    todo |= sp.sympify(i['expr']).free_symbols - {x}
        self.sites = {k: v for k, v in self.sites.items() if k in reach}
        self.inputs = [i for i in self.inputs if i['kind'] != 'cut' or i['symbol'] in reach]
        self.held_all = list(self.held)
        self.held_init = {h['symbol']: self.init[h['symbol']] for h in self.held_all}
        self.held = [h for h in self.held if h['symbol'] in reach]
        for h in self.held:
            self.states.append(h['symbol'])
            self.rhs[h['symbol']] = sp.Integer(0)
        self.init = {s_: self.init[s_] for s_ in self.states}
        self.n_inputs_named = len(self.inputs)

    def full(self, e):
        """An expression with the smoothing sites replaced by their expressions in the states."""
        return sp.sympify(e).xreplace({k: v['T'] for k, v in self.sites.items()}) if self.sites else e

    def _key_of(self, s):
        return self._sym_key.get(s)

    @property
    def input_symbols(self):
        return [i['symbol'] for i in self.inputs]

    def new_input(self, e, kind, **extra):
        key = (kind, e)
        if key in self._input_by_expr:
            return self._input_by_expr[key]
        names = {str(s) for s in self.reader.sym.values()} | {i['name'] for i in self.inputs}
        k = len(self.inputs) + 1
        while f'u_{k}' in names:
            k += 1
        s = sp.Symbol(f'u_{k}', real=True)
        held = {i['symbol'] for i in self.inputs if i.get('held')}
        fs = sp.sympify(e).free_symbols - set(self.params) - set(self.derived)
        if fs and fs <= held:
            extra = dict(extra, held=True)      # a function of held inputs alone is held too
        self.inputs.append(dict({'symbol': s, 'name': f'u_{k}', 'expr': e, 'kind': kind}, **extra))
        self._input_by_expr[key] = s
        return s

    def new_held(self, e):
        """A held variable for an expression of held inputs (constant within a chunk)."""
        for h in self.held:
            if h['kind'] == 'held_function' and h['expr'] == e:
                return h['symbol']
        names = {str(x) for x in self.reader.sym.values()} | {h['name'] for h in self.held}
        k = len([h for h in self.held if h['kind'] == 'held_function']) + 1
        while f'h_{k}' in names:
            k += 1
        sym = sp.Symbol(f'h_{k}', real=True)
        self.held.append({'symbol': sym, 'name': f'h_{k}', 'expr': e, 'kind': 'held_function'})
        self.init[sym] = _value(e, self.param_values(), {h['symbol']: self.init[h['symbol']] for h in self.held
                                                         if h['symbol'] in self.init})
        return sym

    def new_derived(self, e):
        for s, d in self.derived.items():
            if d == e:
                return s
        names = {str(s) for s in self.reader.sym.values()}
        k = len(self.derived) + 1
        while f'k_{k}' in names:
            k += 1
        s = sp.Symbol(f'k_{k}', real=True)
        self.derived[s] = e
        return s

    def _abstract(self, e, states):
        """Inputs and derived parameters out of an expression: what is left is elementary in the states."""
        fs = e.free_symbols
        params = set(self.params) | set(self.derived)
        if not fs or fs <= params:
            return self._param_part(e)
        held = getattr(self, '_held_syms', set())
        if held and fs & held and not (fs & (states - held)) and self.time not in fs \
                and not (fs & set(self.input_symbols)):
            # a function of held inputs alone is held too: one variable, du/dt = 0
            return e if e.is_Symbol else self.new_held(e)
        named = set(self.input_symbols[:len(self.inputs)])
        if not fs & states:
            if self.time in fs or not _is_poly_in(e, fs - params):
                if e == self.time:
                    return self.new_input(e, 'time')
                if isinstance(e, (sp.Add, sp.Mul)) and not e.has(*_NONSMOOTH):
                    # a sum or product of time and inputs: each non-polynomial part its own input
                    return e.func(*[self._abstract(a, states) for a in e.args])
                return self.new_input(e, 'time' if self.time in fs and not (fs & named) else 'input_function')
            return e
        if not e.args:
            return e
        if isinstance(e, _NONSMOOTH) or isinstance(e, (sp.logic.boolalg.BooleanFunction, sp.core.relational.Relational)):
            switch = sorted(str(s) for s in fs & states)
            return self.new_input(e, 'nonsmooth', switch_variables=switch)
        return e.func(*[self._abstract(a, states) for a in e.args])

    def _param_part(self, e):
        if e.is_Number or e.is_Symbol or e in (sp.pi, sp.E):
            return e
        if isinstance(e, (sp.Add, sp.Mul)):
            return e.func(*[self._param_part(a) for a in e.args])
        if isinstance(e, sp.Pow) and e.args[0].is_Symbol and e.args[1].is_Integer:
            return e
        return self.new_derived(e)

    def param_values(self):
        """{symbol: value} of the parameters and the derived parameters."""
        vals = dict(self.params)
        for s, e in self.derived.items():
            vals[s] = _value(e, vals)
        return vals


def _time_derivative(e, t):
    """d e/dt of an input's expression, with floor and ceiling (piecewise constant) held constant."""
    steps = {f: sp.Dummy() for f in e.atoms(sp.floor, sp.ceiling)}
    d = sp.diff(e.xreplace(steps), t)
    return d.xreplace({v: k for k, v in steps.items()})


def input_values(m, t, states):
    """{input symbol: value} of a Model's inputs at time t and the states {symbol: value}."""
    vals = {k: sp.Float(v) for k, v in m.param_values().items()}
    if t == 0:
        vals.update({k: sp.Float(v) for k, v in getattr(m, 'all_init', {}).items()})
    vals.update({k: sp.Float(v) for k, v in states.items()})
    vals[m.time] = sp.Float(t)
    out = {}
    for i in _input_order(m):
        e = sp.sympify(m.full(i['expr']) if hasattr(m, 'full') else i['expr'])
        v = vals[e] if (i['kind'] == 'cut' and e in vals) else e.xreplace(vals)
        try:
            out[i['symbol']] = float(v)
        except TypeError:
            out[i['symbol']] = float(v.evalf())
        vals[i['symbol']] = sp.Float(out[i['symbol']])
    return out


class _Evaluator:
    """Floats of parameter expressions at one point, caching every subexpression: the coefficients of a lifted
    system share most of their subexpressions."""
    _F = {sp.exp: math.exp, sp.log: math.log, sp.sin: math.sin, sp.cos: math.cos, sp.tan: math.tan,
          sp.tanh: math.tanh, sp.sinh: math.sinh, sp.cosh: math.cosh, sp.atan: math.atan, sp.Abs: abs}

    def __init__(self, values):
        self.memo = {k: float(v) for k, v in values.items()}

    def __call__(self, e):
        e = sp.sympify(e)
        got = self.memo.get(e)
        if got is not None:
            return got
        if e.is_Number:
            out = float(e)
        elif e is sp.pi or e is sp.E:
            out = float(e)
        elif isinstance(e, sp.Add):
            out = math.fsum(self(a) for a in e.args)
        elif isinstance(e, sp.Mul):
            out = 1.0
            for a in e.args:
                out *= self(a)
        elif isinstance(e, sp.Pow):
            out = self(e.args[0]) ** self(e.args[1])
        elif type(e) in self._F:
            out = self._F[type(e)](*[self(a) for a in e.args])
        else:
            out = float(e.xreplace({k: sp.Float(v) for k, v in self.memo.items() if isinstance(k, sp.Symbol)}))
        self.memo[e] = out
        return out


def _value(e, *values):
    """A float of an expression with the symbols of ``values`` (dicts {symbol: float}) replaced."""
    subs = {k: sp.Float(v) for d in values for k, v in d.items()}
    return float(sp.sympify(e).xreplace(subs))


def _is_poly_in(e, syms):
    try:
        return bool(sp.sympify(e).is_polynomial(*syms)) if syms else True
    except Exception:
        return False


def _myokit_state_order(text):
    try:
        import warnings
        import tempfile
        import myokit.formats
        with tempfile.NamedTemporaryFile('w', suffix='.cellml', delete=False) as f:
            f.write(text if text.lstrip().startswith('<') else open(text).read())
            path = f.name
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = myokit.formats.importer('cellml').model(path)
        os.unlink(path)
        return [s.qname() for s in m.states()]
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# QBee in a subprocess (it has no budget on polynomialization, and can recurse without end)

def _qbee_child(conn, stage, payload, budget):
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.dup2(devnull, 1)
        sys.setrecursionlimit(5000)
        import qbee
        from functools import partial
        fn = {n: qbee.functions(n, laurent=False) for n in payload['functions']}
        pn = {n: qbee.parameters(n) for n in payload['parameters']}
        loc = dict(fn, **pn)
        system = [(fn[n], sp.sympify(e, locals=loc)) for n, e in payload['equations']]
        if stage == 'polynomialize':
            res = qbee.polynomialize(system, upper_bound=payload.get('upper_bound', 50), new_vars_name='qw_')
            out = {'introduced': [(str(eq.lhs), str(eq.rhs)) for eq in res.introduced_variables],
                   'equations': [(str(eq.lhs), str(eq.rhs)) for eq in res.polynomial_equations]}
        else:
            prune = [partial(qbee.pruning_by_elapsed_time, start_t=time.time(), max_t=budget)]
            res = qbee.quadratize(system, input_free=True, pruning_functions=prune, new_vars_name='qm')
            out = None if res is None else {'introduced': [(str(eq.lhs), str(eq.rhs))
                                                           for eq in res.introduced_variables]}
        conn.send(('ok', out))
    except BaseException as e:  # noqa: BLE001 -- QBee errors of every kind end the attempt
        conn.send(('error', f'{type(e).__name__}: {str(e)[:300]}'))


def _run_qbee(stage, payload, budget):
    """('ok', result) | ('error', message) | ('timeout', message)."""
    ctx = multiprocessing.get_context('fork')
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_qbee_child, args=(child, stage, payload, budget))
    proc.start()
    child.close()
    if parent.poll(budget + 5.0):
        try:
            status, out = parent.recv()
        except EOFError:
            status, out = 'error', 'the QBee process ended without a result'
    else:
        status, out = 'timeout', f'over the time budget of {budget:g} s'
    if proc.is_alive():
        proc.kill()
    proc.join()
    return status, out


# ---------------------------------------------------------------------------------------------
# polynomialization by rules

class _Aux:
    def __init__(self, symbol, T, rule):
        self.symbol, self.T, self.rule = symbol, T, rule


class _Polynomializer:
    """Rewrites expressions in the states as polynomials in the states and auxiliaries."""

    def __init__(self, model, max_aux=DEFAULT_MAX_AUX, tanh_form='tanh'):
        self.m = model
        self.tanh_form = tanh_form
        self.states = list(model.states)
        self.sites = dict(model.sites)
        self.state_set = set(self.states) | set(self.sites)
        self._site_rate = {}
        self.aux, self.by_key = [], {}
        self.max_aux = max_aux
        self.notes = []
        self.input_free = True
        self._memo = {}

    def _new(self, key, T, rule):
        if key in self.by_key:
            return self.by_key[key].symbol
        if len(self.aux) >= self.max_aux:
            raise LiftError(f'more than {self.max_aux} auxiliaries from the rule table')
        s = sp.Symbol(f'pw_{len(self.aux)}', real=True)
        a = _Aux(s, T, rule)
        self.aux.append(a)
        self.by_key[key] = a
        return s

    def _pow_aux(self, g, e):
        """g^e for a rational e = a/b (b > 1) or a negative integer: in w = g^(1/b) or z = g^(-1/b)."""
        e = sp.Rational(e)
        a, b = e.p, e.q
        if b == 1:
            if ('pow', g, sp.Rational(-1, 2)) in self.by_key:
                return self.by_key[('pow', g, sp.Rational(-1, 2))].symbol ** (2 * -a)
            z = self._new(('pow', g, sp.Integer(-1)), 1 / g, 'reciprocal')
            return z ** (-a)
        if a > 0:
            w = self._new(('pow', g, sp.Rational(1, b)), g ** sp.Rational(1, b), 'root' if b == 2 else 'power')
            return w ** a
        z = self._new(('pow', g, sp.Rational(-1, b)), g ** sp.Rational(-1, b), 'reciprocal root' if b == 2 else 'power')
        return z ** (-a)

    def rewrite(self, e):
        if e in self._memo:
            return self._memo[e]
        out = self._rewrite(e)
        self._memo[e] = out
        return out

    def _rewrite(self, e):
        fs = e.free_symbols
        if not (fs & self.state_set):
            if fs & set(self.m.input_symbols) and not _is_poly_in(e, fs & set(self.m.input_symbols)):
                return self.m.new_input(e, 'input_function')
            return e
        if e.is_Symbol:
            return e
        if isinstance(e, (sp.Add, sp.Mul)):
            return e.func(*[self.rewrite(a) for a in e.args])
        if isinstance(e, sp.Pow):
            g, p = e.args
            if p.free_symbols & self.state_set:
                return self._aux_function(sp.exp(p * sp.log(g)), 'exp')
            if p.is_Integer and p >= 0:
                return self.rewrite(g) ** p
            if p.is_Float:
                r = sp.Rational(float(p)).limit_denominator(1000)
                if float(r) == float(p):
                    p = r
            if p.is_Rational:
                if (g.free_symbols & set(self.m.input_symbols)) and not self._mixed_ok(e):
                    return self._mixed_input(e)
                if p.is_Integer and p < 0 and not _is_poly_in(g, list(g.free_symbols & self.state_set)):
                    # a reciprocal of a rational expression in its natural form: g = n/d, g^-k = d^k (1/n)^k with n
                    # a polynomial (1/(1 + a C/(K + C)) = (K + C)/((1 + a) C + K)): a short rate, not a chain
                    num, den = sp.fraction(sp.together(g))
                    if den != 1 and _is_poly_in(num, list(num.free_symbols & self.state_set)):
                        return self.rewrite(den ** (-p)) * self._pow_aux(sp.expand(num), p)
                return self._pow_aux(g, p)
            if g.free_symbols & set(self.m.input_symbols):
                return self._mixed(e)
            # a symbolic or irrational exponent: g^(c + r) = g^c g^r with c rational; w = g^r, dw/dt = r w z dg/dt
            c, rest = p.as_coeff_Add()
            if rest == 0 or not c.is_Rational:
                c, rest = sp.Integer(0), p
            w = self._new(('pow', g, rest), g ** rest, 'power')
            self._pow_aux(g, -1)
            return w * (self.rewrite(g ** c) if c != 0 else 1)
        if isinstance(e, sp.exp):
            g = e.args[0]
            if isinstance(g, sp.Add):
                # exp(a + b) = exp(a) exp(b): the state-free part is a coefficient
                free = [a for a in g.args if not (a.free_symbols & self.state_set)]
                if free:
                    rest = sp.Add(*[a for a in g.args if a.free_symbols & self.state_set])
                    return self.rewrite(sp.exp(sp.Add(*free))) * self.rewrite(sp.exp(rest))
            return self._aux_function(e, 'exp')
        if isinstance(e, sp.tanh) and self.tanh_form == 'exp':
            # tanh(g) = 1 - 2 q, q = 1/(E + 1), E = exp(2 g): q has the identity q (E + 1) - 1 = 0
            return self.rewrite(1 - 2 / (sp.exp(2 * e.args[0]) + 1))
        if isinstance(e, (sp.sin, sp.cos)):
            g = e.args[0]
            s = self._aux_function(sp.sin(g), 'sin')
            c = self._aux_function(sp.cos(g), 'cos')
            return s if isinstance(e, sp.sin) else c
        if isinstance(e, sp.tan):
            g = e.args[0]
            return self.rewrite(sp.sin(g)) * self.rewrite(1 / sp.cos(g))
        if isinstance(e, (sp.sinh, sp.cosh)):
            return self.rewrite(e.rewrite(sp.exp))
        if isinstance(e, sp.Function) or isinstance(e, sp.log):
            return self._aux_function(e, type(e).__name__)
        raise LiftError(f'cannot polynomialize {e}')

    def _aux_function(self, e, rule):
        if e.free_symbols & set(self.m.input_symbols):
            return self._mixed(e)
        return self._new(('f', e), e, rule)

    def _mixed_ok(self, e):
        """Whether a non-polynomial term in both states and inputs can take the inputs' derivatives as inputs:
        each input in it is a known function of time or held. Records that the lift is not input-free."""
        ins = [i for i in self.m.inputs if i['symbol'] in e.free_symbols]
        ok = all(i.get('held') or (i['kind'] in ('time', 'variable', 'input_function')
                                   and not (i['expr'].free_symbols & self.state_set)) for i in ins)
        if ok and all(i.get('held') for i in ins):
            return True                 # held inputs: du/dt = 0, still input-free
        if ok:
            self.input_free = False
            note = (f'{", ".join(i["name"] for i in ins)} enter a non-polynomial term with the states: their '
                    f'derivatives are inputs')
            if note not in self.notes:
                self.notes.append(note)
        return ok

    def _mixed_input(self, e):
        """A non-polynomial term in the states and an input that itself depends on the states: an input."""
        self.notes.append(f'{e}: depends on states and an input that depends on the states; the term is an input')
        self.input_free = False
        return self.m.new_input(e, 'nonpolynomial_term', switch_variables=sorted(
            str(s) for s in e.free_symbols & self.state_set))

    def _mixed(self, e):
        """A non-polynomial function of states and inputs: an auxiliary (with the inputs' derivatives as inputs)
        when :meth:`_mixed_ok`, else an input."""
        if self._mixed_ok(e):
            return self._new(('f', e), e, 'mixed')
        return self._mixed_input(e)

    def ddt(self, T):
        """dT/dt along the model, as an expression in the states and inputs (and input derivatives)."""
        out = sp.Integer(0)
        for s in T.free_symbols & self.state_set:
            out += sp.diff(T, s) * (self.site_rate(s) if s in self.sites else self.m.rhs[s])
        for i in list(self.m.inputs):
            u = i['symbol']
            if u in T.free_symbols:
                du = self._input_derivative(i)
                out += sp.diff(T, u) * du
        return out

    def _input_derivative(self, i):
        if i.get('held'):
            return sp.Integer(0)        # constant within a chunk
        name = f'd{i["name"]}_dt'
        for j in self.m.inputs:
            if j['name'] == name:
                return j['symbol']
        expr = _time_derivative(i['expr'], self.m.time)
        s = sp.Symbol(name, real=True)
        self.m.inputs.append({'symbol': s, 'name': name, 'expr': expr, 'kind': 'derivative', 'of': i['name']})
        return s

    def site_rate(self, sym):
        """The closed-form rate of a smoothing site, in the states, sites and auxiliaries:
        w = sqrt(x^2 + eps^2): x x' z (z = 1/w, the site's or an auxiliary); z = 1/w: -x x' z^3;
        q = 1/(x^2 + eps^2): -2 x x' q^2."""
        if sym in self._site_rate:
            return self._site_rate[sym]
        info = self.sites[sym]
        x = info['x']
        if info['kind'] == 'x':
            out = self.ddt(x)
            self._site_rate[sym] = out
            return out
        # with x a lifted variable (the site's sm<k>_x), x z x' has as many terms as x', not their product
        xd = self.site_rate(x) if x in self.sites else self.ddt(x)
        if info['kind'] == 'w':
            z = info['z'] if info['z'] is not None else self._pow_aux(sym, -1)
            out = x * xd * z
        elif info['kind'] == 'z':
            out = -x * xd * sym ** 3
        else:
            out = -2 * x * xd * sym ** 2
        self._site_rate[sym] = out
        return out

    def run(self):
        """{state, site or auxiliary: polynomial right-hand side}, and the auxiliaries."""
        poly = {s: self.rewrite(self.m.rhs[s]) for s in self.states}
        self.output_poly = [self.rewrite(o['expr']) for o in self.m.outputs]
        for s in self.sites:
            poly[s] = self.rewrite(self.site_rate(s))
        done = 0
        while done < len(self.aux):
            a = self.aux[done]
            poly[a.symbol] = self.rewrite(self.ddt(a.T))
            done += 1
        return poly


# ---------------------------------------------------------------------------------------------
# polynomials as {exponent tuple over (V + U): coefficient}

class _TooLarge(Exception):
    pass


def _to_dict(e, gens, limit=None):
    """{exponent tuple over gens: coefficient} of an expression polynomial in gens. Only the generators are
    expanded: a subexpression free of them (in the parameters) stays one coefficient, so a model's
    parameter expressions do not multiply out. Coefficients that cancel are dropped later, numerically."""
    gset = set(gens)
    n = len(gens)
    zero = (0,) * n
    unit = {g: tuple(1 if j == i else 0 for j in range(n)) for i, g in enumerate(gens)}
    memo = {}

    def mul(a, b):
        out = {}
        for ma, ca in a.items():
            for mb, cb in b.items():
                _add_to(out, tuple(x + y for x, y in zip(ma, mb)), ca * cb)
            if limit is not None and len(out) > limit:
                raise _TooLarge(len(out))
        return out

    def rec(x):
        if x in memo:
            return memo[x]
        if not (x.free_symbols & gset):
            out = {zero: x} if x != 0 else {}
        elif x in unit:
            out = {unit[x]: sp.Integer(1)}
        elif isinstance(x, sp.Add):
            out = {}
            for a in x.args:
                for mono, c in rec(a).items():
                    _add_to(out, mono, c)
        elif isinstance(x, sp.Mul):
            out = {zero: sp.Integer(1)}
            for a in x.args:
                out = mul(out, rec(a))
        elif isinstance(x, sp.Pow) and x.args[1].is_Integer and x.args[1] >= 0:
            out, base = {zero: sp.Integer(1)}, rec(x.args[0])
            for _ in range(int(x.args[1])):
                out = mul(out, base)
        else:
            raise LiftError(f'not polynomial in the lifted variables: {x}')
        memo[x] = out
        return out

    return {k: v for k, v in rec(sp.sympify(e)).items() if v != 0}


def _drop_cancelled(d, points, rel=1e-12):
    """``d`` ({key: coefficient}) without the coefficients that cancel: |c| <= rel x the sum of |its terms| at every
    parameter point. A small coefficient that does not cancel stays, whatever its size against the others (the
    lifted variables' scales differ by many orders)."""
    evs = [_Evaluator(pt) for pt in points]
    out = {}
    for k, c in d.items():
        terms = sp.Add.make_args(sp.sympify(c))
        keep = False
        for ev in evs:
            vals = [ev(a) for a in terms]
            if abs(math.fsum(vals)) > rel * math.fsum(abs(v) for v in vals):
                keep = True
                break
        if keep:
            out[k] = c
    return out


def _parameter_points(values, seed=7):
    """The model's parameter values and a second point with each scaled by a random factor in [0.7, 1.4]."""
    rng = np.random.default_rng(seed)
    return [values, {k: v * float(rng.uniform(0.7, 1.4)) for k, v in values.items()}]


def _add_to(acc, mono, c):
    acc[mono] = acc.get(mono, 0) + c


class _Quadratic:
    """The search for monomial variables W over the polynomial variables V that make every right-hand
    side quadratic in W (a term with an input: at most one W factor).

    The search works on the coefficients' values at two parameter points (the model's, and one with each
    parameter scaled by a random factor), so a term is dropped only when it cancels at both; the symbolic
    right-hand sides are computed once, for the variables it keeps (:meth:`symbolic_rows`)."""

    def __init__(self, nV, nU, rhs_V, values=None):
        self.nV, self.nU = nV, nU
        self.rhs_V = rhs_V                       # list over V of {mono over V+U: coeff}
        self.points = _parameter_points(values or {})
        ev1, ev2 = (_Evaluator(pt) for pt in self.points)
        num = [{mono: (ev1(c), ev2(c)) for mono, c in _drop_cancelled(r, self.points).items()} for r in rhs_V]
        self.num_V = [{k: v for k, v in r.items() if v != (0.0, 0.0)} for r in num]
        self.rhs_V = rhs_V = [{k: c for k, c in r.items() if k in nr} for r, nr in zip(rhs_V, self.num_V)]
        self.W = [tuple(1 if i == j else 0 for i in range(nV)) for j in range(nV)]
        self.Wset = set(self.W)
        self.Widx = {w: i for i, w in enumerate(self.W)}
        self.rows = [set(r) for r in self.num_V]   # each W variable's monomials
        self.bad_set = set()
        for r in self.rows:
            self._check(r)

    def _numeric_row(self, m):
        out = {}
        for i, e in enumerate(m):
            if not e:
                continue
            base = list(m)
            base[i] -= 1
            for mono, (c1, c2) in self.num_V[i].items():
                shifted = tuple(a + b for a, b in zip(base, mono[:self.nV])) + mono[self.nV:]
                p = out.get(shifted, (0.0, 0.0, 0.0, 0.0))
                out[shifted] = (p[0] + e * c1, p[1] + e * c2, p[2] + abs(e * c1), p[3] + abs(e * c2))
        # a term is gone only when its own contributions cancel
        return {k for k, (a, b, ma, mb) in out.items() if abs(a) > 1e-12 * ma or abs(b) > 1e-12 * mb}

    def rhs_of_monomial(self, m):
        """The symbolic right-hand side of the monomial m."""
        out = {}
        for i, e in enumerate(m):
            if not e:
                continue
            base = list(m)
            base[i] -= 1
            for mono, c in self.rhs_V[i].items():
                shifted = tuple(a + b for a, b in zip(base, mono[:self.nV])) + mono[self.nV:]
                out.setdefault(shifted, []).append(e * c)
        return _drop_cancelled({k: sp.Add(*v) for k, v in out.items()}, self.points)

    def symbolic_rows(self):
        return list(self.rhs_V) + [self.rhs_of_monomial(m) for m in self.W[self.nV:]]

    def _check(self, monos):
        for mono in monos:
            if self.split(mono) is None:
                self.bad_set.add(mono)

    def add(self, m):
        if m in self.Wset:
            return
        self.Widx[m] = len(self.W)
        self.W.append(m)
        self.Wset.add(m)
        row = self._numeric_row(m)
        self.rows.append(row)
        # only a term whose state part the new monomial divides can become quadratic
        self.bad_set = {b for b in self.bad_set
                        if not all(x <= y for x, y in zip(m, b[:self.nV])) or self.split(b) is None}
        self._check(row)

    def split(self, mono):
        """(i,) | (i, j) | () -- indices into W whose product is the state part; None if not quadratic."""
        ms, mu = mono[:self.nV], mono[self.nV:]
        if not any(ms):
            return ()
        if ms in self.Wset:
            return (self.Widx[ms],)
        if any(mu):
            return None
        for i, a in enumerate(self.W):
            if all(x <= y for x, y in zip(a, ms)):
                rest = tuple(y - x for x, y in zip(a, ms))
                if rest in self.Wset:
                    return tuple(sorted((i, self.Widx[rest])))
        return None

    def _new_bad(self, d):
        """How many terms of the rate of the monomial d would not be quadratic, were d added."""
        self.W.append(d)
        self.Wset.add(d)
        self.Widx[d] = len(self.W) - 1
        try:
            return sum(1 for mono in self._numeric_row(d) if self.split(mono) is None)
        finally:
            self.W.pop()
            self.Wset.discard(d)
            del self.Widx[d]

    def bad(self):
        return {(mono[:self.nV], any(mono[self.nV:])) for mono in self.bad_set}

    def complete(self, max_new):
        """The quadratization every polynomial system without inputs has: all monomials of degree 2 to d - 1 in the
        variables (d the largest degree), then those not needed removed one at a time. False if over max_new."""
        base = self.rows[:self.nV]
        d = max((sum(k[:self.nV]) for r in base for k in r), default=0)
        used = sorted({i for r in base for k in r for i, e in enumerate(k[:self.nV]) if e})
        n_cands = sum(math.comb(len(used) + deg - 1, deg) for deg in range(2, d))
        if n_cands > max_new:
            return False        # too many: the greedy search's state is left as it is
        cands = []
        for deg in range(2, d):
            for combo in itertools.combinations_with_replacement(used, deg):
                mono = [0] * self.nV
                for i in combo:
                    mono[i] += 1
                cands.append(tuple(mono))
        # from the polynomial variables alone, not on top of a greedy search's monomials
        self._rebuild(self.W[:self.nV])
        for r in self.rows:
            self._check(r)
        for c in cands:
            self.add(c)
        if self.bad_set:
            return False
        # remove each added monomial, highest degree first, while the rest stays quadratic
        for c in sorted(cands, key=lambda m: -sum(m)):
            keep = [w for w in self.W if w != c]
            if self._quadratic_without(keep):
                self._rebuild(keep)
        return not self.bad_set

    def _quadratic_without(self, W):
        Wset = set(W)
        idx = [i for i, w in enumerate(self.W) if w in Wset]
        for i in idx:
            for mono in self.rows[i]:
                ms, mu = mono[:self.nV], mono[self.nV:]
                if not any(ms) or ms in Wset:
                    continue
                if any(mu):
                    return False
                if not any(all(x <= y for x, y in zip(a, ms)) and tuple(y - x for x, y in zip(a, ms)) in Wset
                           for a in W):
                    return False
        return True

    def _rebuild(self, W):
        rows = {w: r for w, r in zip(self.W, self.rows)}
        self.W = list(W)
        self.Wset = set(W)
        self.Widx = {w: i for i, w in enumerate(W)}
        self.rows = [rows[w] for w in W]
        self.bad_set = set()

    def reset(self):
        """Back to the polynomial variables alone."""
        self._rebuild(self.W[:self.nV])
        for r in self.rows:
            self._check(r)

    def greedy(self, max_new, deadline=None, lookahead=True):
        while True:
            bad = self.bad()
            if not bad:
                return True
            if len(self.W) - self.nV >= max_new or (deadline and time.perf_counter() > deadline):
                return False
            # every monomial d that would make a bad term quadratic on its own: the term's state part itself, or it
            # less one variable, or it less an existing monomial. A d fixes exactly the terms that propose it, so
            # its score is the number of terms proposing it.
            cands = {}
            for count, (ms, has_u) in enumerate(bad):
                if deadline and count % 500 == 0 and time.perf_counter() > deadline:
                    return False
                proposals = {ms}
                if not has_u:
                    for i in range(self.nV):
                        if ms[i]:
                            proposals.add(tuple(x - (1 if k == i else 0) for k, x in enumerate(ms)))
                    for a in self.W[self.nV:]:
                        if all(x <= y for x, y in zip(a, ms)):
                            proposals.add(tuple(y - x for x, y in zip(a, ms)))
                for d in proposals:
                    if sum(d) >= 2 or (has_u and d == ms):
                        cands[d] = cands.get(d, 0) + 1
            # look one step ahead: of the best-scoring candidates, the one whose own rate adds the fewest new terms
            # of degree > 2 (a square that fixes two terms can start a chain that never closes)
            if lookahead:
                top = sorted(cands, key=lambda d: (-cands[d], sum(d), d))[:10]
                best = max(top, key=lambda d: (cands[d] - self._new_bad(d), -sum(d), d))
            else:
                best = max(cands, key=lambda d: (cands[d], -sum(d), d))
            self.add(best)


# ---------------------------------------------------------------------------------------------
# the lifting

class Lifting:
    """A quadratic lifting of a model (see :func:`lift`).

    ``names``: the lifted variables, the original states first; ``T``: their expressions in the
    original states (and parameters); ``rhs``: {lifted index: [(coeff, (), (i,) or (i, j), input feature or None)]}.
    """


def lift(model_text, inputs=(), time_budget=DEFAULT_TIME_BUDGET, engine='auto', max_aux=DEFAULT_MAX_AUX,
         smoothing_report=None, max_terms=DEFAULT_MAX_TERMS, as_inputs=(), keep=(), held_inputs=False, sites=True,
         outputs=(), tanh_form='tanh', stabilize=None):
    """Lift a flat CellML model (text or path) to quadratic form.

    Args:
        inputs: the variables to keep as inputs u ("name", "component/name"): ports, open boundary conditions.
        time_budget: seconds for each QBee stage before the rule table or the greedy search take over.
        engine: ``"auto"`` (QBee, then the fallbacks), ``"qbee"`` (no fallback: fails when QBee does) or
            ``"rules"`` (the rule table and the greedy search only).
        smoothing_report: the smoothing pass's report (dict or path), for each auxiliary's epsilon and each
            site's smoothness class.
        max_terms: the largest polynomial system (terms over all right-hand sides) to quadratize; a larger one
            raises LiftError with its size.
        as_inputs: variables (or ``component/*`` globs) cut from the model: inputs whose values come from data.
        keep: component globs whose states are kept; every other state is cut (an input from data).
        held_inputs: True or ``"states"``: the named and cut inputs become states with du/dt = 0 (held within a
            chunk), so an input inside a non-polynomial function lifts with the states. ``"inputs"``: they stay
            inputs (B and N terms) with du/dt = 0, so the search adds no products with them.
        tanh_form: ``"tanh"`` (t = tanh(g), dt/dt = (1 - t^2) g') or ``"exp"`` (E = exp(2 g), q = 1/(E + 1),
            t = 1 - 2 q, with the identity q (E + 1) - 1 = 0 for ``stabilize``).
        stabilize: kappa [1/time]: -kappa g(w) is added to each auxiliary's equation whose identity g = 0 is at most
            quadratic (w z - 1, w^2 - x^2 - eps^2, q (d^2 + eps^2) - 1, w - x y, ...): zero on the manifold, it damps
            the directions off it. The operators then include these terms.
        sites: the smoothing sites (``sm<k>_w``, ``sm<k>_z``, ``sm<k>_q``) are lifted variables with closed-form
            rates, rather than their nested square roots polynomialized.
        outputs: variables lifted with the states: their non-polynomial parts become auxiliaries, and each is
            reported as a polynomial in the lifted variables (``L.outputs``).

    Returns:
        a :class:`Lifting`.
    """
    t_start = time.perf_counter()
    text = model_text
    if not text.lstrip().startswith('<'):
        with open(text, encoding='utf-8') as f:
            text = f.read()
    m = Model(text, inputs, as_inputs=as_inputs, keep=keep, held_inputs=held_inputs, sites=sites, outputs=outputs)
    states = list(m.states)
    params_all = lambda: list(m.params) + list(m.derived)  # noqa: E731
    engine_report = {'requested': engine, 'qbee_errors': []}
    timings = {}

    # 1. polynomialize
    t0 = time.perf_counter()
    aux_T, poly = None, None
    if engine == 'auto' and (m.sites or m.outputs or tanh_form != 'tanh' or m.held_mode == 'inputs'):
        engine_report['qbee_errors'].append(f'polynomialize: not tried, {len(m.sites)} smoothing sites lifted in '
                                            f'closed form, {len(m.outputs)} outputs')
    elif engine in ('auto', 'qbee'):
        res = _qbee_polynomialize(m, time_budget)
        if res[0] == 'ok':
            aux_T, poly = res[1]
            engine_report['polynomialize'] = 'qbee'
        else:
            engine_report['qbee_errors'].append(f'polynomialize: {res[1]}')
            if engine == 'qbee':
                raise LiftError(f'QBee polynomialize: {res[1]}')
    input_free, notes, output_poly = True, [], []
    if poly is None:
        p = _Polynomializer(m, max_aux, tanh_form)
        poly_sym = p.run()
        aux_T = [(a.symbol, a.T, a.rule) for a in p.aux]
        poly = poly_sym
        input_free, notes, output_poly = p.input_free, p.notes, p.output_poly
        engine_report['polynomialize'] = 'sites+rules' if m.sites else 'rules'
    timings['polynomialize'] = time.perf_counter() - t0
    site_syms = list(m.sites)
    V = states + site_syms + [s for s, _, _ in aux_T]
    U = m.input_symbols
    gens = V + U
    rhs_V, n_terms = [], 0
    for v in V:
        try:
            r = _to_dict(poly[v], gens, max_terms - n_terms)
        except _TooLarge:
            r = None
        if r is None or n_terms + len(r) > max_terms:
            raise LiftError(f'the polynomial system is too large to quadratize: over {max_terms} terms (max_terms) by '
                            f'{v}, {V.index(v) + 1} of {len(V)} right-hand sides, after {len(aux_T)} auxiliaries from '
                            f'{engine_report["polynomialize"]}')
        rhs_V.append(r)
        n_terms += len(r)
    degree = max((sum(k[:len(V)]) for r in rhs_V for k in r), default=0)

    # 2. quadratize
    t0 = time.perf_counter()
    q = _Quadratic(len(V), len(U), rhs_V, m.param_values())
    done = False
    if engine == 'auto' and n_terms > QBEE_MAX_TERMS:
        engine_report['qbee_errors'].append(f'quadratize: not tried, {n_terms} terms (over {QBEE_MAX_TERMS})')
    elif engine in ('auto', 'qbee'):
        res = _qbee_quadratize(V, U, params_all(), rhs_V, time_budget)
        if res[0] == 'ok':
            for mono in res[1]:
                q.add(mono)
            done = not q.bad()
            engine_report['quadratize'] = 'qbee' if done else 'qbee+greedy'
            if not done:
                engine_report['qbee_errors'].append('quadratize: QBee\'s variables leave terms of degree > 2')
        else:
            engine_report['qbee_errors'].append(f'quadratize: {res[1]}')
            if engine == 'qbee':
                raise LiftError(f'QBee quadratize: {res[1]}')
    if not done:
        half = max(time_budget, 30.0) * 5
        found = q.greedy(max_aux, time.perf_counter() + half)
        if not found:
            q.reset()
            found = q.greedy(max_aux, time.perf_counter() + half, lookahead=False)
            if found:
                engine_report['quadratize'] = 'greedy (no lookahead)'
        if not found and not U and q.complete(max_aux):
            engine_report['quadratize'] = 'complete'
        elif q.bad_set:
            raise LiftError(f'the greedy quadratization stopped at {len(q.W) - len(V)} monomials with {len(q.bad_set)} '
                            f'terms of degree > 2 left (max_aux {max_aux}; polynomial system: {len(V)} variables, '
                            f'{len(aux_T)} of them auxiliaries from {engine_report["polynomialize"]}, {n_terms} terms, '
                            f'degree {degree})')
        engine_report.setdefault('quadratize', 'greedy')
    timings['quadratize'] = time.perf_counter() - t0

    # 3. the lifted system
    site_full = {s: info['T'] for s, info in m.sites.items()}
    T_V = list(states) + [m.sites[s]['T'] for s in site_syms] + [sp.sympify(T).xreplace(site_full) for _, T, _ in aux_T]
    names = [str(s) for s in states] + [str(s) for s in site_syms]
    used = {str(s) for s in m.reader.sym.values()} | {i['name'] for i in m.inputs} | {str(s) for s in m.derived}
    aux_names = []
    for k in range(len(q.W) - len(states) - len(site_syms)):
        n = f'w_{k + 1}'
        while n in used:
            n = 'lift_' + n
        aux_names.append(n)
    names += aux_names
    T = []
    for mono in q.W:
        T.append(sp.Mul(*[T_V[i] ** e for i, e in enumerate(mono) if e]))
    rules = [None] * len(states) + [f'site_{m.sites[s]["kind"]}' for s in site_syms] + [r for _, _, r in aux_T] + \
        ['monomial'] * (len(q.W) - len(V))
    definitions = _definition_identities(V, [None] * len(states) + [None] * len(site_syms) + [T for _, T, _ in aux_T],
                                         m, q, rules)
    # the right-hand side in W: terms (coeff, W indices, input monomial)
    terms = []
    features, feature_index = [], {}
    rows_W = q.symbolic_rows()
    for r in rows_W:
        row = []
        for mono, c in r.items():
            idx = q.split(mono)
            if idx is None:
                raise LiftError(f'a term of degree > 2 left after the search: {mono} (coefficient {c})')
            mu = mono[len(V):]
            f = None
            if any(mu):
                if mu not in feature_index:
                    feature_index[mu] = len(features)
                    features.append(mu)
                f = feature_index[mu]
            row.append((c, idx, f))
        terms.append(row)

    pre_names = list(names)
    names, T, rules, terms, kept, sub = _reduce(m, states, names, T, rules, terms)

    L = Lifting()
    L.model, L.text = m, text
    n_named = len(states) + len([j for j in kept if len(states) <= j < len(states) + len(site_syms)])
    names = names[:n_named] + [aux_names[k] for k in range(len(names) - n_named)]
    L.invariant_definitions = _map_identities(definitions, pre_names, kept, sub, names)
    L.stabilization = None
    if stabilize:
        L.stabilization = _stabilize(terms, L.invariant_definitions, names, len(states), float(stabilize))
    L.outputs = _map_outputs(m, output_poly, V, U, q, pre_names, kept, sub, names)
    L.states, L.names, L.T, L.rules = states, names, T, rules
    L.n_orig = m.n_orig
    L.symbols = [sp.Symbol(n, real=True) for n in names]
    L.n, L.n_w = len(states), len(names)
    L.terms, L.features = terms, features
    L.input_symbols = U
    L.engine = dict(engine_report, input_free=input_free and not any(i['kind'] == 'derivative' for i in m.inputs),
                    notes=notes, n_polynomial_aux=len(aux_T), n_monomials=len(q.W) - len(V),
                    polynomial_terms=n_terms, polynomial_degree=degree)
    L.timings = timings
    L.smoothing_report = _load_json(smoothing_report)
    L.timings['lift'] = time.perf_counter() - t_start
    _symbolic_operators(L)
    return L


def _reduce(m, states, names, T, rules, terms):
    """Drops the auxiliaries whose T is constant (substituted as a coefficient) or equal to an earlier
    variable's T (replaced by it), e.g. QBee's w = sqrt(g) * (1/sqrt(g)) = 1. The right-hand sides stay
    quadratic and exact on the manifold. Equality is tested at random states near the initial state."""
    n = len(states)
    pv = m.param_values()
    rng = np.random.default_rng(3)
    s0 = np.array([m.init[s] for s in states], dtype=float)
    pts = s0[None, :] * (1 + 0.1 * rng.standard_normal((5, n))) + 0.1 * rng.standard_normal((5, n))
    # and widely spread points, where a saturated function no longer looks constant
    wide = s0[None, :] * rng.uniform(-2.0, 2.0, (5, n)) + rng.standard_normal((5, n))
    vals, wvals = [], []
    for e in T:
        try:
            f = sp.lambdify(states, sp.sympify(e).xreplace({k: sp.Float(v) for k, v in pv.items()}), 'numpy')
            with np.errstate(all='ignore'):
                v = np.broadcast_to(np.asarray(f(*pts.T), dtype=float), (len(pts),))
                vw = np.broadcast_to(np.asarray(f(*wide.T), dtype=float), (len(wide),))
        except Exception:
            v = vw = np.full(len(pts), np.nan)
        vals.append(v)
        wvals.append(vw)
    sub = {}
    for j in range(n, len(T)):
        v = vals[j]
        if not np.all(np.isfinite(v)):
            continue
        tol = 1e-10 * max(1.0, float(np.max(np.abs(v))))
        if not (sp.sympify(T[j]).free_symbols & set(states)) or np.ptp(v) <= tol and _const(T[j], states):
            sub[j] = ('const', sp.simplify(T[j]))
            continue
        for k in range(j):
            # numerically equal near the initial state (a saturated tanh makes distinct variables coincide there):
            # merged only when equal symbolically
            if k not in sub and np.all(np.abs(vals[k] - v) <= tol) and not _differ(wvals[k], wvals[j]) \
                    and _equal(T[j], T[k]):
                sub[j] = ('var', k)
                break
    if not sub:
        return names, T, rules, terms, list(range(len(T))), {}
    keep = [j for j in range(len(T)) if j not in sub]
    new_index = {j: i for i, j in enumerate(keep)}
    out_terms = []
    for j in keep:
        row = []
        for c, idx, f in terms[j]:
            new = []
            for k in idx:
                kind, what = sub.get(k, ('keep', k))
                if kind == 'const':
                    c = c * what
                else:
                    new.append(new_index[what])
            row.append((c, tuple(sorted(new)), f))
        out_terms.append(row)
    return [names[j] for j in keep], [T[j] for j in keep], [rules[j] for j in keep], out_terms, keep, sub


def _definition_identities(V, T_atomic, m, q, rules):
    """Each lifted variable's polynomial identity g = 0 in the variables before reduction (as sympy, in symbols
    ``_v<j>``), or the reason there is none: a monomial w - prod(v^e); a site's w^2 - x^2 - eps^2, z w - 1,
    q (x^2 + eps^2) - 1; an auxiliary g^(a/b) as w^b - g^a (w^b g^-a - 1 for a < 0); sin and cos as s^2 + c^2 - 1.
    An exp, log, tanh or symbolic power has none."""
    nV = len(V)
    vsym = [sp.Symbol(f'_v{j}') for j in range(len(q.W))]
    to_v = {V[j]: vsym[j] for j in range(nV)}
    known = {}
    for j in range(nV):
        if T_atomic[j] is not None:
            known[sp.sympify(T_atomic[j])] = vsym[j]

    def poly_in_v(g):
        g = sp.sympify(g).xreplace(known).xreplace(to_v)
        return g if _is_poly_in(g, vsym) and not (g.free_symbols - set(vsym) - set(m.params) - set(m.derived)
                                                   - set(m.input_symbols)) else None

    out = []
    pairs = {}
    for j, mono in enumerate(q.W):
        w = vsym[j]
        if j >= nV:
            # a monomial as the product of two other lifted variables where it can be (a quadratic identity, which
            # stabilize can use), else of the polynomial variables
            pair = None
            for a_i, a in enumerate(q.W):
                if a_i != j and all(x <= y for x, y in zip(a, mono)):
                    rest = tuple(y - x for x, y in zip(a, mono))
                    if rest in q.Widx and q.Widx[rest] != j:
                        pair = (a_i, q.Widx[rest])
                        break
            out.append((j, w - vsym[pair[0]] * vsym[pair[1]] if pair else
                        w - sp.Mul(*[vsym[i] ** e for i, e in enumerate(mono) if e]), None))
            continue
        v = V[j]
        if v in m.sites:
            info = m.sites[v]
            x = poly_in_v(info['x'])
            if x is None:
                out.append((j, None, 'the site argument is not polynomial in the lifted variables'))
            elif info['kind'] == 'x':
                out.append((j, w - x, None))
            elif info['kind'] == 'w':
                out.append((j, w ** 2 - x ** 2 - info['eps'] ** 2, None))
            elif info['kind'] == 'q':
                out.append((j, w * (x ** 2 + info['eps'] ** 2) - 1, None))
            else:
                wsym = [s for s, i in m.sites.items() if i['kind'] == 'w' and i['k'] == info['k']]
                out.append((j, w * to_v[wsym[0]] - 1, None) if wsym and wsym[0] in to_v else
                           (j, w ** 2 * (x ** 2 + info['eps'] ** 2) - 1, None))
            continue
        T = T_atomic[j]
        if T is None:
            continue
        T = sp.sympify(T)
        if isinstance(T, sp.Pow) and T.args[1].is_Rational:
            a, b = T.args[1].p, T.args[1].q
            g = poly_in_v(T.args[0])
            if g is None:
                out.append((j, None, 'its base is not polynomial in the lifted variables'))
            else:
                out.append((j, w ** b - g ** a if a > 0 else w ** b * g ** (-a) - 1, None))
        elif isinstance(T, (sp.sin, sp.cos)):
            pairs.setdefault(T.args[0], {})[type(T)] = w
        else:
            out.append((j, None, f'{rules[j]}: no polynomial identity ({type(T).__name__})'))
    for arg, d in pairs.items():
        if len(d) == 2:
            out.append((None, d[sp.sin] ** 2 + d[sp.cos] ** 2 - 1, None))
    return out


def _map_identities(defs, pre_names, kept, sub, names):
    """The identities in the final lifted names, as [{expr, degree, origin, variable}] and skipped records."""
    vsym = {f'_v{j}': j for j in range(len(pre_names))}
    final = {}
    for j in range(len(pre_names)):
        if j in kept:
            final[sp.Symbol(f'_v{j}')] = sp.Symbol(names[kept.index(j)], real=True)
        else:
            kind, what = sub[j]
            final[sp.Symbol(f'_v{j}')] = what if kind == 'const' else sp.Symbol(names[kept.index(what)], real=True)
    out = []
    for j, g, reason in defs:
        variable = None if j is None else (names[kept.index(j)] if j in kept else pre_names[j])
        if g is None:
            out.append({'variable': variable, 'skipped': reason, 'origin': 'definition'})
            continue
        e = sp.expand(sp.sympify(g).xreplace(final))
        if e == 0:
            continue
        lifted = [x for x in e.free_symbols if x.name in names]
        out.append({'expr': str(e), 'degree': int(sp.Poly(e, *lifted).total_degree()) if lifted else 0,
                    'origin': 'definition', 'variable': variable})
    return out


def _differ(a, b):
    """Whether two value arrays differ where both are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    return bool(np.any(np.abs(a[ok] - b[ok]) > 1e-10 * np.maximum(1.0, np.abs(a[ok]))))


def _equal(a, b):
    """Whether two expressions are equal symbolically."""
    try:
        return sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0
    except Exception:
        return False


def _stabilize(terms, identities, names, n, kappa):
    """Adds -kappa g(w) to the equation of each auxiliary whose identity g = 0 is at most quadratic in the lifted
    variables (and free of inputs); returns {kappa, identities: [{variable, expr}], skipped: [...]}."""
    syms = [sp.Symbol(x, real=True) for x in names]
    index = {x: i for i, x in enumerate(syms)}
    used, skipped = [], []
    for d in identities:
        v = d.get('variable')
        if 'expr' not in d or v not in names or names.index(v) < n:
            continue
        g = sp.sympify(d['expr'], locals={x.name: x for x in syms})
        lifted = [x for x in g.free_symbols if x in index]
        try:
            poly = sp.Poly(sp.expand(g), *lifted) if lifted else None
        except sp.PolynomialError:
            poly = None
        if poly is None or poly.total_degree() > 2:
            skipped.append({'variable': v, 'reason': 'identity of degree > 2' if poly is not None else 'not polynomial'})
            continue
        row = terms[names.index(v)]
        for mono, c in poly.terms():
            idx = tuple(sorted(i for x, e in zip(lifted, mono) for i in [index[x]] * e))
            row.append((-kappa * c, idx, None))
        used.append({'variable': v, 'expr': d['expr']})
    return {'kappa': kappa, 'identities': used, 'skipped': skipped,
            'note': 'the operators include these -kappa g(w) terms: OpInf and Galerkin must use the same ones'}


def _map_outputs(m, output_poly, V, U, q, pre_names, kept, sub, names):
    """Each output as a polynomial in the lifted names: each term through the lifted variables found (quadratic
    where they allow), else as a product of the polynomial-level variables."""
    gens = list(V) + list(U)
    final = {}
    for j in range(len(pre_names)):
        if j in kept:
            final[j] = sp.Symbol(names[kept.index(j)], real=True)
        else:
            kind, what = sub[j]
            final[j] = what if kind == 'const' else sp.Symbol(names[kept.index(what)], real=True)
    out = []
    for o, e in zip(m.outputs, output_poly):
        terms, degree = [], 0
        for mono, c in _to_dict(e, gens).items():
            idx = q.split(mono)
            if idx is None:
                factors = [final[i] ** k for i, k in enumerate(mono[:len(V)]) if k]
                d = sum(mono[:len(V)])
            else:
                factors = [final[i] for i in idx]
                d = len(idx)
            factors += [u ** k for u, k in zip(U, mono[len(V):]) if k]
            degree = max(degree, d)
            terms.append(c * sp.Mul(*factors))
        out.append({'name': o['name'], 'qname': o['qname'], 'expr': str(sp.Add(*terms)), 'degree': degree,
                    'original': str(m.full(o['expr']))})
    return out


def _const(e, states):
    """Whether e simplifies to an expression free of the states."""
    try:
        return not (sp.simplify(e).free_symbols & set(states))
    except Exception:
        return False


def _load_json(x):
    if x is None or isinstance(x, dict):
        return x
    with open(x) as f:
        return json.load(f)


def _qbee_names(m, V, U, params):
    vn = {s: f'v{i}' for i, s in enumerate(V)}
    un = {s: f'u{i}' for i, s in enumerate(U)}
    pn = {s: f'p{i}' for i, s in enumerate(params)}
    return vn, un, pn


def _qbee_polynomialize(m, budget):
    """('ok', (aux list, {var: poly rhs})) when QBee's polynomialization is input-free and checks out."""
    params = list(m.params) + list(m.derived)
    vn, un, pn = _qbee_names(m, m.states, m.input_symbols, params)
    ren = {**vn, **un, **pn}
    inv = {sp.Symbol(v): k for k, v in ren.items()}
    payload = {'functions': list(vn.values()) + list(un.values()), 'parameters': list(pn.values()),
               'equations': [(vn[s], str(m.rhs[s].xreplace({k: sp.Symbol(v) for k, v in ren.items()})))
                             for s in m.states], 'upper_bound': DEFAULT_MAX_AUX}
    status, out = _run_qbee('polynomialize', payload, budget)
    if status != 'ok':
        return status, out
    aux, aux_sym = [], {}
    for k, (lhs, rhs) in enumerate(out['introduced']):
        s = sp.Symbol(f'pw_{k}', real=True)
        aux_sym[sp.Symbol(lhs.replace('{', '').replace('}', ''))] = s
    loc = {str(k): k for k in inv}
    for k, (lhs, rhs) in enumerate(out['introduced']):
        T = sp.sympify(rhs, locals=loc).xreplace(inv)
        if T.free_symbols & set(m.input_symbols):
            return 'error', f'an auxiliary depends on an input: {T}'
        aux.append((aux_sym[sp.Symbol(lhs.replace('{', '').replace('}', ''))], T, 'qbee'))
    poly = {}
    for lhs, rhs in out['equations']:
        name = lhs.rstrip("'").replace('{', '').replace('}', '')
        e = sp.sympify(rhs.replace('{', '').replace('}', ''), locals=loc).xreplace(inv)
        e = e.xreplace({sp.Symbol(str(k)): v for k, v in aux_sym.items()})
        target = aux_sym.get(sp.Symbol(name)) or inv.get(sp.Symbol(name))
        poly[target] = e
    V = list(m.states) + [a for a, _, _ in aux]
    if set(poly) != set(V):
        return 'error', 'QBee\'s polynomial system does not match its variables'
    # check: each variable's polynomial right-hand side is d/dt of its definition
    subs = {a: T for a, T, _ in aux}
    rng = np.random.default_rng(0)
    pv = {k: sp.Float(v) for k, v in m.param_values().items()}
    for v, T in [(s, s) for s in m.states] + [(a, T) for a, T, _ in aux]:
        lhs = sp.sympify(sum(sp.diff(T, s) * m.rhs[s] for s in T.free_symbols & set(m.states))).xreplace(pv)
        diff = sp.sympify(poly[v].xreplace(subs)).xreplace(pv) - lhs
        for _ in range(3):
            point = {s: sp.Float(rng.uniform(0.5, 1.5)) for s in m.states}
            point.update({u: sp.Float(rng.uniform(0.5, 1.5)) for u in m.input_symbols})
            try:
                val = complex(diff.xreplace(point).evalf())
                scale = 1 + abs(complex(lhs.xreplace(point).evalf()))
            except (TypeError, ValueError, AttributeError):
                continue
            if abs(val) > 1e-8 * scale:
                return 'error', f'QBee\'s polynomial right-hand side of {v} fails the chain-rule check'
    return 'ok', (aux, poly)


def _qbee_quadratize(V, U, params, rhs_V, budget):
    """('ok', [monomial over V]) or an error."""
    vn = [f'v{i}' for i in range(len(V))]
    un = [f'u{i}' for i in range(len(U))]
    gens = [sp.Symbol(n) for n in vn + un]
    eqs = []
    # the search depends only on which monomials occur (each row's are distinct): coefficient 1 on every term, so
    # QBee's coefficient field needs no parameters, pi or floats
    for i, r in enumerate(rhs_V):
        e = sp.Add(*[sp.Mul(*[g ** k for g, k in zip(gens, mono)]) for mono in r])
        eqs.append((vn[i], str(e)))
    payload = {'functions': vn + un, 'parameters': [], 'equations': eqs}
    status, out = _run_qbee('quadratize', payload, budget)
    if status != 'ok':
        return status, out
    if out is None:
        return 'error', 'QBee found no quadratization'
    vsyms = [sp.Symbol(n) for n in vn]
    monos = []
    for _lhs, rhs in out['introduced']:
        e = sp.sympify(rhs.replace('{', '').replace('}', ''))
        try:
            p = sp.Poly(e, *vsyms)
        except sp.PolynomialError:
            return 'error', f'QBee introduced a non-monomial variable {rhs}'
        if len(p.terms()) != 1:
            return 'error', f'QBee introduced a non-monomial variable {rhs}'
        monos.append(tuple(int(k) for k in p.terms()[0][0]))
    return 'ok', monos


# ---------------------------------------------------------------------------------------------
# operators

def _symbolic_operators(L):
    """A, H (upper-triangular pairs), N (per input feature), B, C as {(i, j): coefficient expression}."""
    n_w = L.n_w
    A, H, B, C = {}, {}, {}, {}
    N = [dict() for _ in L.features]
    for i, row in enumerate(L.terms):
        for c, idx, f in row:
            if f is None:
                if len(idx) == 0:
                    _add_to(C, (i,), c)
                elif len(idx) == 1:
                    _add_to(A, (i, idx[0]), c)
                else:
                    _add_to(H, (i, idx[0], idx[1]), c)
            else:
                if len(idx) == 0:
                    _add_to(B, (i, f), c)
                else:
                    _add_to(N[f], (i, idx[0]), c)
    points = _parameter_points(L.model.param_values())
    clean = lambda d: _drop_cancelled(d, points)  # noqa: E731
    L.A, L.H, L.B, L.C, L.N = clean(A), clean(H), clean(B), clean(C), [clean(x) for x in N]
    L.feature_names = [_feature_name(L, f) for f in L.features]


def _pair_k(i, j, n):
    """The index of the pair (i, j), i <= j, in numpy.triu_indices(n) order."""
    return i * n - i * (i - 1) // 2 + (j - i)


def _feature_name(L, mu):
    return str(sp.Mul(*[u ** k for u, k in zip(L.input_symbols, mu) if k]))


def numeric_operators(L, values=None):
    """(A, H_coo, N list, B, C) as numpy arrays at the parameter values (default: the model's).
    H_coo is (rows, pair indices in numpy.triu_indices(n_w) order, values)."""
    pv = L.model.param_values()
    if values:
        pv.update({sp.Symbol(k, real=True) if isinstance(k, str) else k: v for k, v in values.items()})
        base = {s: pv[s] for s in L.model.params}
        for s, e in L.model.derived.items():
            base[s] = _value(e, base)
        pv.update(base)
    ev = lambda c: _value(c, pv)  # noqa: E731
    n_w, n_f = L.n_w, len(L.features)
    A = np.zeros((n_w, n_w))
    for (i, j), c in L.A.items():
        A[i, j] = ev(c)
    hr, hk, hv = [], [], []
    for (i, a, b), c in sorted(L.H.items()):
        hr.append(i)
        hk.append(_pair_k(a, b, n_w))
        hv.append(ev(c))
    N = [np.zeros((n_w, n_w)) for _ in range(n_f)]
    for f, Nf in enumerate(L.N):
        for (i, j), c in Nf.items():
            N[f][i, j] = ev(c)
    B = np.zeros((n_w, n_f))
    for (i, f), c in L.B.items():
        B[i, f] = ev(c)
    C = np.zeros(n_w)
    for (i,), c in L.C.items():
        C[i] = ev(c)
    return A, (np.array(hr, dtype=int), np.array(hk, dtype=int), np.array(hv)), N, B, C


def apply_operators(ops, W, F):
    """A w + H (w (x) w) + sum_k N_k w f_k + B f + C for snapshots W (n_snap, n_w) and input features
    F (n_snap, n_f)."""
    A, (hr, hk, hv), N, B, C = ops
    n_w = W.shape[1]
    rows, cols = np.triu_indices(n_w)
    out = W @ A.T + C[None, :]
    if len(hr):
        prod = W[:, rows[hk]] * W[:, cols[hk]]
        np.add.at(out.T, hr, (prod * hv[None, :]).T)
    for f, Nf in enumerate(N):
        out += (W @ Nf.T) * F[:, f:f + 1]
    if F.shape[1]:
        out += F @ B.T
    return out


# ---------------------------------------------------------------------------------------------
# CellML output

def _mml(e):
    """Content MathML of a sympy expression (CellML 2.0: every cn in dimensionless)."""
    def cn(v):
        if isinstance(v, sp.Rational) and not v.is_Integer:
            return f'<apply><divide/>{cn(sp.Integer(v.p))}{cn(sp.Integer(v.q))}</apply>'
        v = float(v)
        text = str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
        return f'<cn cellml:units="dimensionless">{text}</cn>'

    def ap(op, *args):
        return f'<apply><{op}/>' + ''.join(args) + '</apply>'

    if e.is_Symbol:
        return f'<ci>{e.name}</ci>'
    if e is sp.pi:
        return '<pi/>'
    if e is sp.E:
        return '<exponentiale/>'
    if e is sp.true:
        return '<true/>'
    if e is sp.false:
        return '<false/>'
    if e.is_Number:
        return cn(e)
    if isinstance(e, sp.Add):
        return ap('plus', *[_mml(a) for a in e.args])
    if isinstance(e, sp.Mul):
        num, den = [], []
        for a in e.args:
            if isinstance(a, sp.Pow) and a.args[1].is_Number and a.args[1] < 0:
                den.append(a.args[0] ** (-a.args[1]))
            elif a.is_Rational and not a.is_Integer:
                num.append(sp.Integer(a.p))
                den.append(sp.Integer(a.q))
            else:
                num.append(a)
        n = ap('times', *[_mml(a) for a in num]) if len(num) > 1 else (_mml(num[0]) if num else cn(1))
        if not den:
            return n
        d = ap('times', *[_mml(a) for a in den]) if len(den) > 1 else _mml(den[0])
        return ap('divide', n, d)
    if isinstance(e, sp.Pow):
        b, p = e.args
        if p == sp.Rational(1, 2):
            return ap('root', _mml(b))
        if p.is_Number and p < 0:
            return ap('divide', cn(1), _mml(b ** (-p)))
        return ap('power', _mml(b), _mml(p))
    fmap = {sp.exp: 'exp', sp.log: 'ln', sp.sin: 'sin', sp.cos: 'cos', sp.tan: 'tan', sp.tanh: 'tanh',
            sp.sinh: 'sinh', sp.cosh: 'cosh', sp.asin: 'arcsin', sp.acos: 'arccos', sp.atan: 'arctan',
            sp.Abs: 'abs', sp.floor: 'floor', sp.ceiling: 'ceiling', sp.Max: 'max', sp.Min: 'min', sp.Mod: 'rem',
            sp.And: 'and', sp.Or: 'or', sp.Not: 'not', sp.Xor: 'xor',
            sp.StrictLessThan: 'lt', sp.StrictGreaterThan: 'gt', sp.LessThan: 'leq', sp.GreaterThan: 'geq',
            sp.Eq: 'eq', sp.Ne: 'neq'}
    for cls, op in fmap.items():
        if isinstance(e, cls):
            return ap(op, *[_mml(a) for a in e.args])
    if isinstance(e, sp.Piecewise):
        out = '<piecewise>'
        for val, cond in e.args:
            if cond is sp.true:
                out += f'<otherwise>{_mml(val)}</otherwise>'
            else:
                out += f'<piece>{_mml(val)}{_mml(cond)}</piece>'
        return out + '</piecewise>'
    if isinstance(e, sp.sign):
        x = e.args[0]
        return _mml(sp.Piecewise((1, x > 0), (-1, x < 0), (0, True)))
    if isinstance(e, sp.Heaviside):
        x = e.args[0]
        return _mml(sp.Piecewise((1, x > 0), (0, True)))
    raise LiftError(f'cannot write {type(e).__name__} as MathML')


def lifted_expression(L, i):
    """The lifted right-hand side of lifted variable i, in the lifted variables and input features."""
    W = L.symbols
    F = [sp.Mul(*[u ** k for u, k in zip(L.input_symbols, mu) if k]) for mu in L.features]
    out = sp.Integer(0)
    for c, idx, f in L.terms[i]:
        out += c * sp.Mul(*[W[j] for j in idx]) * (F[f] if f is not None else 1)
    return out


def lifted_cellml(L, name=None):
    """The lifted model as CellML 2.0 text: one component, every variable dimensionless; the original
    states keep their names and initial values, each auxiliary starts at T(s(0))."""
    from libcellml import Parser, Printer
    m = L.model
    pv = m.param_values()
    s0 = dict(m.init)
    iv = input_values(m, 0.0, s0)
    w0 = [_value(T, s0, pv, iv) for T in L.T]
    # a held input is a constant of the lifted model; a cut input (data) is written at its value at t = 0
    constant = lambda i: (i['kind'] == 'variable' and i.get('held')) or i['kind'] == 'cut'  # noqa: E731
    tname = str(m.time)
    lines = [f'<variable name="{tname}" units="dimensionless"/>']
    for s, v in m.params.items():
        lines.append(f'<variable name="{s}" units="dimensionless" initial_value="{v!r}"/>')
    for s in m.derived:
        lines.append(f'<variable name="{s}" units="dimensionless"/>')
    for i in m.inputs:
        if not constant(i):
            lines.append(f'<variable name="{i["name"]}" units="dimensionless"/>')
    for n, v in zip(L.names, w0):
        lines.append(f'<variable name="{n}" units="dimensionless" initial_value="{v!r}"/>')
    eqs = []
    for s, e in m.derived.items():
        eqs.append(f'<apply><eq/><ci>{s}</ci>{_mml(e)}</apply>')
    for i in m.inputs:
        if constant(i):
            continue
        eqs.append(f'<apply><eq/><ci>{i["name"]}</ci>{_mml(m.full(i["expr"]))}</apply>')
    for k, n in enumerate(L.names):
        eqs.append(f'<apply><eq/><apply><diff/><bvar><ci>{tname}</ci></bvar><ci>{n}</ci></apply>'
                   f'{_mml(lifted_expression(L, k))}</apply>')
    for i in [i for i in m.inputs if constant(i)]:
        lines.append(f'<variable name="{i["name"]}" units="dimensionless" initial_value="{iv[i["symbol"]]!r}"/>')
    model_name = name or 'lifted'
    text = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model xmlns="{CELLML2_NS}" name="{model_name}">\n'
            f'<component name="lifted">\n' + '\n'.join(lines) +
            f'\n<math xmlns="{MATHML_NS}" xmlns:cellml="{CELLML2_NS}">\n' + '\n'.join(eqs) +
            '\n</math>\n</component>\n</model>\n')
    parser = Parser()
    model = parser.parseModel(text)
    if parser.errorCount():
        raise LiftError('libCellML could not parse the lifted model: ' +
                        '; '.join(parser.error(i).description() for i in range(parser.errorCount())))
    return Printer().printModel(model)


# ---------------------------------------------------------------------------------------------
# numpy functions

def python_module(L):
    """Source of a module with vectorised numpy functions of the lifting, at the model's parameters
    (``p`` overrides them by name): T(X), J(X), inputs(t, X), features(U), rhs_original(X, U),
    rhs_lifted(W, U), operators()."""
    from sympy.printing.numpy import NumPyPrinter
    pr = NumPyPrinter({'fully_qualified_modules': True, 'allow_unknown_functions': False})
    m = L.model
    code = lambda e: pr.doprint(sp.sympify(e))  # noqa: E731
    out = ['"""The lifting of a CellML model to quadratic form (written by libcuflynx.utilities.lift_cellml).',
           '',
           'X: (n_snap, n) original states; W: (n_snap, n_w) lifted; U: (n_snap, n_u) inputs, in INPUTS order;',
           'p: {parameter name: value} overriding PARAMS. features(U) are the input monomials the operators use."""',
           'import functools  # noqa: F401 (the printer writes functools.reduce for max and min)', 'import numpy', '', '']
    out.append(f'STATES = {[str(s) for s in L.states]!r}')
    out.append(f'LIFTED = {L.names!r}')
    out.append(f'INPUTS = {[i["name"] for i in m.inputs]!r}')
    out.append(f'FEATURES = {L.feature_names!r}')
    out.append('PARAMS = {' + ', '.join(f'{str(s)!r}: {v!r}' for s, v in m.params.items()) + '}')
    out.append('')
    out.append('')
    out.append('def _b(v, n):\n    return numpy.broadcast_to(numpy.asarray(v, dtype=float), (n,))')
    out.append('')
    out.append('')
    out.append('def _p(p=None):\n    q = dict(PARAMS)\n    q.update(p or {})')
    for s, e in m.derived.items():
        unpack = ''.join(f'{x} = q[{str(x)!r}]; ' for x in sorted(e.free_symbols, key=str))
        out.append(f'    {unpack}q[{str(s)!r}] = {code(e)}')
    out.append('    return q')

    def header(args, unpack_states=None, unpack_w=None, unpack_u=None):
        lines = [f'    q = _p(p)']
        names = sorted(set(map(str, m.params)) | set(map(str, m.derived)))
        for n in names:
            lines.append(f'    {n} = q[{n!r}]')
        if unpack_states:
            for k, s in enumerate(L.states):
                lines.append(f'    {s} = {unpack_states}[:, {k}]')
        if unpack_w:
            for k, n in enumerate(L.names):
                lines.append(f'    {n} = {unpack_w}[:, {k}]')
        if unpack_u:
            for k, i in enumerate(m.inputs):
                lines.append(f'    {i["name"]} = {unpack_u}[:, {k}]')
        return lines

    needs_u = any(sp.sympify(T).free_symbols & set(L.input_symbols) for T in L.T)
    unpack_u = []
    if needs_u:
        unpack_u = ['    if U is None:', '        U = inputs(numpy.zeros(X.shape[0]) if t is None else t, X, p)',
                    '    U = numpy.atleast_2d(U)'] + [f'    {i["name"]} = U[:, {k}]' for k, i in enumerate(m.inputs)]
    out.append(f'NEEDS_INPUTS = {needs_u!r}  # T and J depend on the inputs: pass t (or U) with X')
    out += ['', '', 'def T(X, p=None, t=None, U=None):',
            '    """(n_snap, n) -> (n_snap, n_w). t (n_snap,) or U (n_snap, n_u): the inputs, when NEEDS_INPUTS."""',
            '    X = numpy.atleast_2d(X)']
    out += header(None, 'X') + unpack_u
    out.append('    n = X.shape[0]')
    out.append('    return numpy.stack([' + ', '.join(f'_b({code(T)}, n)' for T in L.T) + '], axis=1)')

    out += ['', '', 'def J(X, p=None, t=None, U=None):', '    """(n_snap, n) -> (n_snap, n_w, n): dT/ds."""',
            '    X = numpy.atleast_2d(X)']
    out += header(None, 'X') + unpack_u
    out.append(f'    n = X.shape[0]\n    out = numpy.zeros((n, {L.n_w}, {L.n}))')
    for i, T in enumerate(L.T):
        if i < L.n:
            out.append(f'    out[:, {i}, {i}] = 1.0')
            continue
        for k, s in enumerate(L.states):
            if s in T.free_symbols:
                out.append(f'    out[:, {i}, {k}] = {code(sp.diff(T, s))}')
    out.append('    return out')

    n_u = len(m.inputs)
    out += ['', '', 'def JU(X, p=None, t=None, U=None):',
            '    """(n_snap, n) -> (n_snap, n_w, n_u): dT/du, nonzero only where an auxiliary depends on an input\n'
            '    (then dT/dt = J f + JU du/dt, with du/dt the input d<u>_dt)."""', '    X = numpy.atleast_2d(X)']
    out += header(None, 'X') + unpack_u
    out.append(f'    n = X.shape[0]\n    out = numpy.zeros((n, {L.n_w}, {n_u}))')
    for i, T in enumerate(L.T):
        for k, inp in enumerate(m.inputs):
            if inp['symbol'] in sp.sympify(T).free_symbols:
                out.append(f'    out[:, {i}, {k}] = {code(sp.diff(T, inp["symbol"]))}')
    out.append('    return out')

    cut = [i for i in m.inputs if i['kind'] == 'cut']
    out.append(f'CUT = {[i["name"] for i in cut]!r}  # inputs cut from the model: their values come from data')
    out += ['', '', 'def inputs(t, X, p=None, cut=None):',
            '    """(n_snap,) times and (n_snap, n) states -> (n_snap, n_u). cut: (n_snap, len(CUT)), the cut inputs\' '
            'values\n    (data; NaN without)."""', '    X = numpy.atleast_2d(X)']
    out += header(None, 'X')
    out.append(f'    {m.time} = numpy.asarray(t, dtype=float) * numpy.ones(X.shape[0])')
    out.append('    n = X.shape[0]')
    for k, i in enumerate(cut):
        out.append(f'    {i["name"]} = numpy.asarray(cut, dtype=float)[:, {k}] if cut is not None else numpy.full(n, numpy.nan)')
    order = [i for i in _input_order(m) if i['kind'] != 'cut']
    for i in order:
        out.append(f'    {i["name"]} = _b({code(m.full(i["expr"]))}, n)')
    out.append('    return numpy.stack([' + ', '.join(i['name'] for i in m.inputs) + '], axis=1) if '
               f'{len(m.inputs)} else numpy.zeros((n, 0))')

    out += ['', '', 'def features(U):', '    """(n_snap, n_u) inputs -> (n_snap, n_f) the input monomials B and N use."""',
            '    U = numpy.atleast_2d(U)']
    for k, i in enumerate(m.inputs):
        out.append(f'    {i["name"]} = U[:, {k}]')
    out.append('    n = U.shape[0]')
    out.append('    return numpy.stack([' + ', '.join(f'_b({code(sp.sympify(f))}, n)' for f in
                                                       [sp.Mul(*[u ** k for u, k in zip(L.input_symbols, mu) if k])
                                                        for mu in L.features]) + '], axis=1) if '
               f'{len(L.features)} else numpy.zeros((n, 0))')

    out += ['', '', 'def rhs_original(X, U, p=None):', '    """(n_snap, n), (n_snap, n_u) -> (n_snap, n): ds/dt."""',
            '    X = numpy.atleast_2d(X)']
    out += header(None, 'X', None, 'U')
    out.append('    n = X.shape[0]')
    out.append('    return numpy.stack([' + ', '.join(f'_b({code(m.full(m.rhs[s]))}, n)' for s in L.states) + '], axis=1)')

    out += ['', '', 'def rhs_lifted(W, U, p=None):', '    """(n_snap, n_w), (n_snap, n_u) -> (n_snap, n_w): dw/dt."""',
            '    W = numpy.atleast_2d(W)']
    out += header(None, None, 'W', 'U')
    out.append('    n = W.shape[0]')
    out.append('    return numpy.stack([' + ', '.join(f'_b({code(lifted_expression(L, i))}, n)'
                                                       for i in range(L.n_w)) + '], axis=1)')

    out += ['', '', 'def operators(p=None):',
            '    """(A, (H rows, H pair indices in numpy.triu_indices(n_w) order, H values), [N_f], B, C)."""']
    out += header(None)
    nw, nf = L.n_w, len(L.features)
    out.append(f'    A = numpy.zeros(({nw}, {nw}))')
    for (i, j), c in sorted(L.A.items()):
        out.append(f'    A[{i}, {j}] = {code(c)}')
    hs = sorted(L.H.items())
    out.append(f'    Hr = numpy.array({[k[0] for k, _ in hs]!r}, dtype=int)')
    out.append(f'    Hk = numpy.array({[_pair_k(k[1], k[2], nw) for k, _ in hs]!r}, dtype=int)')
    out.append('    Hv = numpy.array([' + ', '.join(code(c) for _, c in hs) + '], dtype=float)')
    out.append(f'    N = [numpy.zeros(({nw}, {nw})) for _ in range({nf})]')
    for f, Nf in enumerate(L.N):
        for (i, j), c in sorted(Nf.items()):
            out.append(f'    N[{f}][{i}, {j}] = {code(c)}')
    out.append(f'    B = numpy.zeros(({nw}, {nf}))')
    for (i, f), c in sorted(L.B.items()):
        out.append(f'    B[{i}, {f}] = {code(c)}')
    out.append(f'    C = numpy.zeros({nw})')
    for (i,), c in sorted(L.C.items()):
        out.append(f'    C[{i}] = {code(c)}')
    out.append('    return A, (Hr, Hk, Hv), N, B, C')
    return '\n'.join(out) + '\n'


def _input_order(m):
    """The inputs in an order where each comes after the inputs its expression uses (cut inputs first: data)."""
    done, out = set(), []
    pending = list(m.inputs)
    for i in [i for i in pending if i['kind'] == 'cut']:
        out.append(i)
        done.add(i['symbol'])
    pending = [i for i in pending if i['kind'] != 'cut']
    while pending:
        for i in pending:
            deps = ({j['symbol'] for j in m.inputs} & sp.sympify(i['expr']).free_symbols) - {i['symbol']}
            if deps <= done:
                out.append(i)
                done.add(i['symbol'])
                pending.remove(i)
                break
        else:
            raise LiftError('inputs that depend on each other in a loop')
    return out


def load_functions(path):
    """The generated module of :func:`python_module` at ``path``."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f'lifting_functions_{abs(hash(path))}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------
# structure (for the beat-map analysis)

def structure(L, ops=None, values=None):
    """Clocks, input kinds, block-triangular pattern and conserved linear functionals of a lifting."""
    m = L.model
    ops = ops or numeric_operators(L, values)
    A, (hr, hk, hv), N, B, C = ops
    n_w = L.n_w
    out = {'clocks': _clocks(L), 'inputs': [], 'blocks': None, 'conserved': []}
    # inputs
    for k, i in enumerate(m.inputs):
        fs = {f for f, mu in enumerate(L.features) if mu[k]}
        states_in = sorted(str(s) for s in m.full(i['expr']).free_symbols & set(m.states))
        kind = ('data (cut from the model)' if i['kind'] == 'cut' else
                'held' if i['kind'] == 'variable' and i.get('held') else
                'known function of time' if i['kind'] in ('time',) or (
                    i['kind'] in ('derivative', 'variable', 'input_function') and not states_in) else
                'state-dependent')
        out['inputs'].append({'name': i['name'], 'kind': kind, 'source': i['kind'],
                              'multiplies_state': any(np.any(N[f]) for f in fs),
                              'expr': str(i['expr']), 'switch_variables': i.get('switch_variables', states_in)})
    # blocks: strongly connected components of "w_j enters dw_i/dt", in topological order
    rows, cols = np.triu_indices(n_w)
    deps = [set() for _ in range(n_w)]
    for i, j in zip(*np.nonzero(A)):
        deps[i].add(int(j))
    for i, k in zip(hr, hk):
        deps[i].update((int(rows[k]), int(cols[k])))
    for Nf in N:
        for i, j in zip(*np.nonzero(Nf)):
            deps[i].add(int(j))
    comps = _scc(deps)
    comp_of = {v: c for c, vs in enumerate(comps) for v in vs}
    edges = sorted({(comp_of[j], comp_of[i]) for i in range(n_w) for j in deps[i] if comp_of[i] != comp_of[j]})
    out['blocks'] = {'blocks': [[L.names[v] for v in sorted(c)] for c in comps],
                     'feeds': [[a, b] for a, b in edges],
                     'triangular': len(comps) > 1,
                     'autonomous': [L.names[v] for c, vs in enumerate(comps)
                                    if not any(b == c for _, b in edges) for v in sorted(vs)]}
    # driven blocks: is the coupling from upstream only through y (x) y (H)?
    by_h = {}
    for i, k in zip(hr, hk):
        by_h.setdefault(int(i), set()).update((int(rows[k]), int(cols[k])))
    coupling = []
    for a, b in edges:
        linear = any(A[i, j] != 0 for i in comps[b] for j in comps[a])
        quad = any(j in comps[a] for i in comps[b] for j in by_h.get(i, ()))
        coupling.append({'from': a, 'to': b, 'linear': bool(linear), 'quadratic': bool(quad)})
    out['blocks']['coupling'] = coupling
    # conserved linear functionals: l^T A = 0, l^T H = 0, l^T N_f = 0
    M = [A]
    if len(hr):
        Hd = np.zeros((n_w, len(rows)))
        np.add.at(Hd, (hr, hk), hv)
        M.append(Hd[:, np.any(Hd != 0, axis=0)])
    M += [Nf for Nf in N]
    M = np.hstack(M)
    if n_w <= 2000:
        # rows normalised (the lifted variables' scales differ by many orders), the null space by SVD, and each
        # functional kept only when l^T M vanishes entry by entry against |l|^T |M|
        rn = np.max(np.abs(M), axis=1)
        rn[rn == 0] = 1.0
        Mn = M / rn[:, None]
        Uu, S, _ = np.linalg.svd(Mn, full_matrices=True)
        tol = 1e-12 * (float(S[0]) if S.size else 1.0) * max(Mn.shape)
        rank = int(np.sum(S > tol))
        Z = Uu[:, rank:] / rn[:, None]

        def holds(lv):
            return bool(np.all(np.abs(lv @ M) <= 1e-9 * np.maximum(np.abs(lv) @ np.abs(M), 1e-300)))
        rows_ = [row for row in _rref(Z.T) if holds(row)]
        held_names = {h['name'] for h in getattr(m, 'held', [])}
        for row in rows_:
            big = float(np.max(np.abs(row)))
            l = {L.names[i]: float(f'{v:.12g}') for i, v in enumerate(row) if abs(v) > 1e-12 * big}
            if set(l) <= held_names:
                continue  # a held input is constant within a chunk by construction (held_inputs), not conserved
            rate_C = float(row @ C)
            rate_B = {L.feature_names[f]: float(row @ B[:, f]) for f in range(B.shape[1]) if abs(row @ B[:, f]) > 1e-12}
            out['conserved'].append({'l': l, 'rate_constant': rate_C, 'rate_inputs': rate_B,
                                     'in_original_states_only': all(n in {str(s) for s in L.states[:m.n_orig]}
                                                                    for n in l)})
    return out


def _rref(Mx, tol=1e-9):
    M = np.array(Mx, dtype=float)
    if M.size == 0:
        return []
    r = 0
    for c in range(M.shape[1]):
        if r >= M.shape[0]:
            break
        p = r + int(np.argmax(np.abs(M[r:, c])))
        if abs(M[p, c]) < tol:
            continue
        M[[r, p]] = M[[p, r]]
        M[r] /= M[r, c]
        for k in range(M.shape[0]):
            if k != r:
                M[k] -= M[k, c] * M[r]
        r += 1
    M[np.abs(M) < tol] = 0.0
    return [row for row in M[:r]]


def _scc(deps):
    """Strongly connected components of the graph j -> i (i depends on j), in topological order."""
    n = len(deps)
    succ = [set() for _ in range(n)]
    for i, ds in enumerate(deps):
        for j in ds:
            succ[j].add(i)
    index, low, on, stack, comps, counter = {}, {}, set(), [], [], [0]
    sys.setrecursionlimit(max(10000, 4 * n))

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in succ[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in range(n):
        if v not in index:
            strong(v)
    return comps[::-1]


def _clocks(L):
    """State pairs (c, s) with c^2 + s^2 conserved (exactly, or on the unit circle), and their rate."""
    m = L.model
    st = L.states
    if len(st) > 80:
        return []
    pv = m.param_values()
    rng = np.random.default_rng(1)
    U = m.input_symbols
    fns = {}
    for s in st:
        fns[s] = sp.lambdify(st + U, m.full(m.rhs[s]).xreplace(pv), 'numpy')
    out = []
    pts = rng.uniform(-1.5, 1.5, size=(6, len(st) + len(U)))
    for a, b in itertools.combinations(range(len(st)), 2):
        def q(pt):
            fa, fb = fns[st[a]](*pt), fns[st[b]](*pt)
            return pt[a] * fa + pt[b] * fb, pt[a] * fb - pt[b] * fa
        try:
            vals = [q(pt) for pt in pts]
        except Exception:
            continue
        if not all(np.isfinite(v).all() for v in vals):
            continue
        rot = max(abs(v[1]) for v in vals)
        if rot < 1e-9:
            continue
        exact = max(abs(v[0]) for v in vals) < 1e-9 * max(1.0, rot)
        circle = False
        if not exact:
            cpts = pts.copy()
            th = rng.uniform(0, 2 * np.pi, size=len(cpts))
            cpts[:, a], cpts[:, b] = np.cos(th), np.sin(th)
            try:
                circle = max(abs(q(pt)[0]) for pt in cpts) < 1e-9 * max(1.0, rot)
            except Exception:
                circle = False
        if not (exact or circle):
            continue
        c, s = st[a], st[b]
        omega = sp.simplify((c * m.full(m.rhs[s]) - s * m.full(m.rhs[c])).subs({c: sp.cos(sp.Symbol('th')), s: sp.sin(sp.Symbol('th'))}))
        fs = omega.free_symbols - set(m.params) - set(m.derived) - {sp.Symbol('th')}
        time_inputs = {i['symbol'] for i in m.inputs if i['kind'] in ('time', 'derivative')}
        rate = ('constant' if not fs else 'known function of time' if fs <= time_inputs else
                'depends on ' + ', '.join(sorted(map(str, fs))))
        out.append({'c': str(c), 's': str(s), 'invariant': 'c^2 + s^2 conserved' if exact else
                    'c^2 + s^2 = 1 invariant (attracting limit cycle)' if circle else '',
                    'rate': str(omega), 'rate_kind': rate,
                    'section': f'{s} = 0 with {c} > 0 (one beat per crossing)'})
    return out


# ---------------------------------------------------------------------------------------------
# lifting.json

#: lifting.json's schema version. Fields are only ever added (and this number raised); none is renamed or removed.
SCHEMA_VERSION = 1
#: Every field of lifting.json: (path, type, required, meaning). ``tutorial/docs/lifting.md`` documents each, and
#: :func:`check_lifting_json` checks a document against the list.
LIFTING_JSON_FIELDS = [
    ('format', str, True, 'always "libcuflynx lift_cellml"'),
    ('schema_version', int, True, 'the schema version (1); later versions only add fields'),
    ('parse', str, True, 'how to parse the expression strings and the H pair indices'),
    ('states', list, True, 'the original states kept, as myokit qnames, in myokit\'s order: the first n entries of w'),
    ('state_symbols', list, True, 'their names in the expressions'),
    ('lifted', list, True, 'the names of w in order: the original states, the held inputs, the smoothing sites, the auxiliaries'),
    ('reverse', str, True, 'the reverse map, w[:n]'),
    ('held_inputs', list, True, 'inputs held as states (du/dt = 0 within a chunk): {name, qname, kind, index, expr (a held function), note}'),
    ('sites', list, True, 'smoothing sites lifted in closed form: {name, kind (w, z, q, or x: a site\'s argument when '
                          'it is a sum), k, qname, x, epsilon}'),
    ('size', dict, True, '{n, n_held, n_sites, n_w, n_aux, n_inputs, n_features}'),
    ('symbols', dict, True, '{expression name: component.variable}'),
    ('time', str, True, 'the name of time'),
    ('auxiliaries', list, True, 'each lifted variable after the states: {name, expr (T), jac (dT/d states and held inputs), '
                                'jac_inputs (dT/du, when it depends on an input), exact, epsilon, rule, scale}'),
    ('inputs', list, True, '{name, expr, kind (variable, cut, time, input_function, nonsmooth, derivative, '
                           'nonpolynomial_term), qname, of, switch_variables}'),
    ('input_features', list, True, 'the input monomials the B and N operators use, in order'),
    ('parameters', dict, True, '{name: value} at which the numeric operators are evaluated'),
    ('derived_parameters', dict, True, '{name: expression in the parameters}'),
    ('operators', dict, True, 'present (which of A, H, N, B, C are non-zero), A (n_w x n_w), H ({rows, pairs, values, shape}: '
                              'COO over the pairs i <= j in numpy.triu_indices(n_w) order), N (one n_w x n_w per input '
                              'feature), B (n_w x n_f), C (n_w), sparsity, symbolic (the same in the parameters)'),
    ('initial', dict, True, '{w0}: T at the initial state and the parameters'),
    ('engine', dict, True, 'which engine ran each stage, QBee errors, input_free, notes, sizes of the polynomial system'),
    ('timings', dict, True, 'seconds of each stage'),
    ('invariants', list, True, 'g(w) = 0 exactly: {expr, degree, origin (definition, conserved, oscillator), variable, '
                               'max_reference, relative_reference, max_lifted_run, relative_lifted_run}, or {variable, '
                               'skipped, origin} where a definition has no polynomial identity'),
    ('python', str, False, 'the generated module: T, J, JU, inputs, features, rhs_original, rhs_lifted, operators'),
    ('check', dict, False, 'the operator form against J f on the model\'s trajectory: {max_relative_residual, per_equation}'),
    ('verification', dict, False, 'simulated, states, manifold, max_state_error, max_manifold_defect, defect_growth '
                                  '(times past 1e-8, 1e-6, 1e-3 and the rate), jf'),
    ('structure', dict, False, 'clocks, inputs (kind, multiplies_state), blocks (block-triangular pattern), conserved, '
                               'spectrum (the lifted Jacobian at states of the run: leading real part, count above '
                               '1e3/time, the leading eigenvectors\' fraction in the tangent space)'),
    ('conditioning', dict, False, '{scales: each variable\'s min, max, max_abs over the run, ratio, note}'),
    ('smoothing_sites', list, False, 'each smoothing site: rule, smoothness class before and after'),
    ('stabilization', (dict, type(None)), True, 'null, or {kappa, identities: [{variable, expr}], skipped, note}: the '
                                                '-kappa g(w) terms added to the auxiliaries\' equations; the operators '
                                                'include them, so OpInf and Galerkin must use the same'),
    ('outputs', list, False, 'named variables lifted with the states: {name, qname, expr (a polynomial in the lifted '
                             'names and inputs: quadratic where the lifted variables allow), degree, original}'),
    ('model', str, False, 'the model lifted'),
    ('lifted_model', str, False, 'the lifted CellML model'),
    ('libcellml', dict, False, 'Validator and Analyser issues of the lifted CellML model'),
]


def check_lifting_json(doc):
    """The problems of a lifting.json document against :data:`LIFTING_JSON_FIELDS` (empty when it conforms)."""
    problems = []
    known = {f for f, *_ in LIFTING_JSON_FIELDS}
    for field, kind, required, _ in LIFTING_JSON_FIELDS:
        if field not in doc:
            if required:
                problems.append(f'missing {field}')
        elif not isinstance(doc[field], kind):
            problems.append(f'{field} is {type(doc[field]).__name__}, not {getattr(kind, "__name__", kind)}')
    problems += [f'unknown field {k}' for k in doc if k not in known]
    if doc.get('schema_version') != SCHEMA_VERSION:
        problems.append(f'schema_version {doc.get("schema_version")} is not {SCHEMA_VERSION}')
    for key in ('A', 'H', 'N', 'B', 'C', 'present', 'symbolic'):
        if key not in doc.get('operators', {}):
            problems.append(f'operators.{key} missing')
    if doc.get('lifted', [])[:len(doc.get('states', []))] != doc.get('state_symbols'):
        problems.append('lifted does not start with the states')
    return problems


def lifting_json(L, python_path=None, check=None, scales=None, verification=None, structure_info=None):
    m = L.model
    pv = m.param_values()
    state_set = list(L.states)
    eps_syms = {s for s in m.params if str(s).startswith('eps_sm')}
    ops = numeric_operators(L)
    A, (hr, hk, hv), N, B, C = ops
    ex = lambda e: str(e)  # noqa: E731
    aux = []
    for k in range(L.n, L.n_w):
        T = L.T[k]
        eps = sorted(eps_syms & T.free_symbols, key=str)
        dep = [i for i in m.inputs if i['symbol'] in sp.sympify(T).free_symbols]
        aux.append({'name': L.names[k], 'expr': ex(T), 'jac': [ex(sp.diff(T, s)) for s in state_set],
                    **({'jac_inputs': {i['name']: ex(sp.diff(T, i['symbol'])) for i in dep}} if dep else {}),
                    'exact': not eps, 'epsilon': {str(e): m.params[e] for e in eps},
                    'rule': L.rules[k], 'scale': (scales or {}).get(L.names[k])})
    sym_ops = {'A': [[i, j, ex(c)] for (i, j), c in sorted(L.A.items())],
               'H': [[i, a, b, _pair_k(a, b, L.n_w), ex(c)] for (i, a, b), c in sorted(L.H.items())],
               'N': [[[i, j, ex(c)] for (i, j), c in sorted(Nf.items())] for Nf in L.N],
               'B': [[i, f, ex(c)] for (i, f), c in sorted(L.B.items())],
               'C': [[i, ex(c)] for (i,), c in sorted(L.C.items())]}
    present = {'A': bool(L.A), 'H': bool(L.H), 'N': any(L.N), 'B': bool(L.B), 'C': bool(L.C)}
    doc = {
        'format': 'libcuflynx lift_cellml',
        'schema_version': SCHEMA_VERSION,
        'parse': 'expressions are sympy strings: sympy.sympify(expr, locals={name: sympy.Symbol(name) for name in '
                 'symbols}); H pairs (i, j), i <= j, are indexed in numpy.triu_indices(n_w) order',
        'states': list(m.state_qnames),
        'state_symbols': [str(s) for s in L.states[:m.n_orig]],
        'lifted': list(L.names),
        'reverse': f'w[:{m.n_orig}]',
        'held_inputs': [{'name': h['name'], 'qname': h.get('qname'), 'kind': h['kind'], 'index': m.n_orig + k,
                         **({'expr': ex(h['expr'])} if h['kind'] == 'held_function' else {}),
                         'note': 'an input held as a state: du/dt = 0 within a chunk (beat); set it from the data at '
                                 'each chunk start'} for k, h in enumerate(m.held)],
        'sites': [{'name': str(sym), 'kind': info['kind'], 'k': info['k'], 'qname': info['qname'],
                   'x': str(m.full(info['x']) if info['kind'] == 'x' else info['x']),
                   'epsilon': None if info['eps'] is None or info['kind'] == 'x' else str(info['eps'])}
                  for sym, info in m.sites.items()],
        'size': {'n': m.n_orig, 'n_held': len(m.held), 'n_sites': len(m.sites), 'n_w': L.n_w,
                 'n_aux': L.n_w - L.n, 'n_inputs': len(m.inputs), 'n_features': len(L.features)},
        'symbols': {str(s): L.model.reader.qname(k) for k, s in L.model.reader.sym.items()},
        'time': str(m.time),
        'auxiliaries': aux,
        'inputs': [{'name': i['name'], 'expr': ex(m.full(i['expr'])), 'kind': i['kind'],
                    **({'qname': i['qname']} if 'qname' in i else {}),
                    **({'of': i['of']} if 'of' in i else {}),
                    **({'switch_variables': i['switch_variables']} if 'switch_variables' in i else {})}
                   for i in m.inputs],
        'input_features': L.feature_names,
        'parameters': {str(s): v for s, v in m.params.items()},
        'derived_parameters': {str(s): ex(e) for s, e in m.derived.items()},
        'operators': {
            'present': present,
            'A': A.tolist(),
            'H': {'rows': hr.tolist(), 'pairs': hk.tolist(), 'values': hv.tolist(),
                  'shape': [L.n_w, L.n_w * (L.n_w + 1) // 2]},
            'N': [Nf.tolist() for Nf in N],
            'B': B.tolist(), 'C': C.tolist(),
            'sparsity': {'A': [[i, j] for (i, j) in sorted(L.A)], 'H': [[i, a, b] for (i, a, b) in sorted(L.H)],
                         'N': [[[i, j] for (i, j) in sorted(Nf)] for Nf in L.N],
                         'B': [[i, f] for (i, f) in sorted(L.B)], 'C': [i for (i,) in sorted(L.C)]},
            'symbolic': sym_ops,
        },
        'initial': {'w0': [_value(T, m.init, pv, input_values(m, 0.0, m.init)) for T in L.T]},
        'engine': L.engine,
        'timings': L.timings,
    }
    if python_path:
        doc['python'] = python_path
    if check is not None:
        doc['check'] = check
    if verification is not None:
        doc['verification'] = verification
    if structure_info is not None:
        doc['structure'] = structure_info
    if scales:
        vals = [v['max_abs'] for v in scales.values() if v.get('max_abs', 0) > 0]
        doc['conditioning'] = {
            'scales': scales,
            'ratio': (max(vals) / min(vals)) if vals else None,
            'note': 'max |w| of each lifted variable over the reference run; scale each before POD',
        }
    if L.smoothing_report:
        sites = []
        for s in L.smoothing_report.get('sites', []):
            sites.append({'k': s.get('k'), 'rule': s.get('rule', s.get('kind')), 'variable': s.get('variable'),
                          'component': s.get('component'), 'status': s.get('status'),
                          'original_class': SMOOTHNESS.get(s.get('rule', s.get('kind')), 'unknown'),
                          'smoothed_class': 'C-infinity (analytic: sqrt, products, sums and 1/D of positive D)'
                          if s.get('status') == 'smoothed' else 'as written',
                          'switch_variables': s.get('switch_variables')})
        doc['smoothing_sites'] = sites
    return doc


# ---------------------------------------------------------------------------------------------
# verification

def _simulate(path, sim_time, dt, solver_info, names):
    import contextlib
    import io
    from libcuflynx.solver_wrappers import get_simulation_helper
    with contextlib.redirect_stdout(io.StringIO()):
        helper = get_simulation_helper(model_path=path, solver='CVODE_myokit', model_type='cellml', dt=dt,
                                       sim_time=sim_time, solver_info=solver_info, pre_time=0.0)
        ok = helper.run()
    if not ok:
        raise LiftError(f'the simulation of {path} failed')
    res = helper.get_results([[n] for n in names], flatten=True)
    t = np.asarray(helper.get_time(), dtype=float).ravel() if hasattr(helper, 'get_time') else None
    if hasattr(helper, 'close_simulation'):
        helper.close_simulation()
    return t, [np.asarray(r, dtype=float).ravel() for r in res]


def verify(L, model_path, lifted_path, funcs, sim_time=10.0, dt=0.01, solver_info=None, n_samples=200, seed=0):
    """The lifted model against the model it lifts (both CVODE_myokit, from t = 0).

    Returns (verification dict, check dict, scales): ``states``: the largest error of each original
    state over its scale max(range, |mean|, atol); ``manifold``: max |w - T(s)| of each auxiliary over its
    scale along the lifted run; ``jf``: max |J f - rhs_lifted(T(s))| over |J f| at sampled states;
    ``check``: the operator form against J f on the reference trajectory, per equation. The J f and operator
    errors are relative to the size of the terms that cancel (|J| |f|, or the operators applied in absolute value
    to |w|) or max |w|, whichever is larger, so an equation whose right-hand side stays near 0 is not judged
    on rounding alone.
    """
    solver_info = dict({'rtol': 1e-10, 'atol': 1e-12, 'MaximumStep': dt}, **(solver_info or {}))
    m = L.model
    cut = [i for i in m.inputs if i['kind'] == 'cut']
    held_cut = [h for h in getattr(m, 'held_all', m.held) if h['kind'] == 'cut']
    qn = [q.replace('.', '/') for q in m.state_qnames]
    data = [i['qname'].replace('.', '/') for i in held_cut + cut]
    t, ref = _simulate(model_path, sim_time, dt, solver_info, qn + data)
    ref = [np.asarray(r, dtype=float) for r in ref]
    if t is None or len(t) != len(ref[0]):
        t = np.linspace(0.0, sim_time, len(ref[0]))
    data_vals = dict(zip([i['name'] for i in held_cut + cut], ref[len(qn):]))
    cols = list(ref[:len(qn)])
    held_vals = {}
    pv_ = {k: sp.Float(v) for k, v in m.param_values().items()}
    for h in getattr(m, 'held_all', m.held):
        if h['kind'] == 'cut':
            held_vals[h['symbol']] = data_vals[h['name']]
        elif h['kind'] == 'variable':
            held_vals[h['symbol']] = np.full(len(t), getattr(m, 'held_init', m.init)[h['symbol']])
        else:
            e = sp.sympify(h['expr']).xreplace(pv_)
            args = sorted(e.free_symbols, key=str)
            held_vals[h['symbol']] = np.broadcast_to(
                np.asarray(sp.lambdify(args, e, 'numpy')(*[held_vals[a] for a in args]), dtype=float), (len(t),))
    for h in m.held:
        cols.append(held_vals[h['symbol']])
    X = np.stack(cols, axis=1)
    C = np.stack([data_vals[i['name']] for i in cut], axis=1) if cut else None
    U = funcs.inputs(t, X, cut=C)
    Tref = funcs.T(X, U=U)
    atol = float(solver_info.get('atol', 1e-12))
    out = {'sim_time': sim_time, 'dt': dt, 'solver_info': solver_info, 'states': {}, 'manifold': {}}
    scales = {}
    for k, n in enumerate(L.names):
        col = Tref[:, k]
        scales[n] = {'min': float(np.min(col)), 'max': float(np.max(col)), 'max_abs': float(np.max(np.abs(col)))}
    Wl = None
    if cut or held_cut:
        # the inputs cut from the model come from data: the lifted model is checked on the model's trajectory
        out.update({'simulated': False, 'reason': 'inputs cut from the model come from data (the model run); the '
                    'lifted model is checked algebraically on that trajectory', 'max_state_error': None,
                    'max_manifold_defect': None, 'defect_growth': {}})
    else:
        _, lif = _simulate(lifted_path, sim_time, dt, solver_info, [f'lifted/{n}' for n in L.names])
        Wl = np.stack(lif, axis=1)
        # a state that stays near 0 (by symmetry) is judged against 1e-8 of the largest state, not against rounding
        floor = max(atol, 1e-8 * float(np.max(np.abs(X)))) if X.size else atol
        for k, s_ in enumerate(L.states):
            scale = max(float(np.ptp(X[:, k])), abs(float(np.mean(X[:, k]))), floor)
            out['states'][str(s_)] = float(np.max(np.abs(Wl[:, k] - X[:, k]))) / scale
        Tl = funcs.T(Wl[:, :L.n], t=t)
        for k, n in enumerate(L.names):
            if k >= L.n:
                col = Tref[:, k]
                sc = max(float(np.ptp(col)), float(np.max(np.abs(col))), atol)
                out['manifold'][n] = float(np.max(np.abs(Wl[:, k] - Tl[:, k]))) / sc
        out['simulated'] = True
        out['max_state_error'] = max(out['states'].values()) if out['states'] else 0.0
        out['max_manifold_defect'] = max(out['manifold'].values()) if out['manifold'] else 0.0
        out['defect_growth'] = _defect_growth(L, t, X, Wl, Tl, atol)
    # J f = rhs_lifted(T(s)) at sampled states: the run, and the run perturbed by 5 % of each state's range
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    Xs = X[idx]
    spread = np.maximum(np.ptp(X, axis=0), 1e-3 * np.maximum(np.abs(X).max(axis=0), 1e-12))
    Xs = np.vstack([Xs, Xs + 0.05 * spread * rng.standard_normal(Xs.shape)])
    ts = np.concatenate([t[idx], t[idx]])
    Us = funcs.inputs(ts, Xs, cut=None if C is None else np.vstack([C[idx], C[idx]]))
    Js, fs = funcs.J(Xs, U=Us), funcs.rhs_original(Xs, Us)
    Jf = np.einsum('kij,kj->ki', Js, fs) + _input_rate_term(L, funcs, Xs, Us)
    Ts = funcs.T(Xs, U=Us)
    R = funcs.rhs_lifted(Ts, Us)
    ok = np.all(np.isfinite(Jf), axis=1) & np.all(np.isfinite(R), axis=1)
    # each variable's scale: the larger of |J| |f| (the size of the terms that cancel in J f) and |w|
    size = np.einsum('kij,kj->ki', np.abs(Js), np.abs(fs))
    denom = np.maximum(np.maximum(np.max(size[ok], axis=0), np.max(np.abs(Ts[ok]), axis=0)), 1e-300) \
        if ok.any() else np.ones(L.n_w)
    jf = np.max(np.abs(Jf[ok] - R[ok]), axis=0) / denom if ok.any() else np.full(L.n_w, np.inf)
    out['jf'] = {'max': float(np.max(jf)), 'n_points': int(ok.sum()), 'per_variable': dict(zip(L.names, map(float, jf)))}
    # the operator form on the reference trajectory
    dW = np.einsum('kij,kj->ki', funcs.J(X, U=U), funcs.rhs_original(X, U)) + _input_rate_term(L, funcs, X, U)
    ops = funcs.operators()
    F = funcs.features(U)
    op = apply_operators(ops, Tref, F)
    A, (hr, hk, hv), N, B, C = ops
    size = apply_operators((np.abs(A), (hr, hk, np.abs(hv)), [np.abs(x) for x in N], np.abs(B), np.abs(C)),
                           np.abs(Tref), np.abs(F))
    den = np.maximum(np.maximum(np.max(size, axis=0), np.max(np.abs(Tref), axis=0)), 1e-300)
    res = np.max(np.abs(op - dW), axis=0) / den
    check = {'max_relative_residual': float(np.max(res)), 'per_equation': dict(zip(L.names, map(float, res))),
             'trajectory': 'the model run of the verification, dw/dt = J(s) f(s)'}
    return out, check, scales, (t, X, Wl, U, Tref)


def _input_rate_term(L, funcs, X, U):
    """JU du/dt: the part of dT/dt from auxiliaries that depend on inputs (0 for an input-free lift)."""
    names = [i['name'] for i in L.model.inputs]
    rates = np.zeros_like(U)
    for k, n in enumerate(names):
        if f'd{n}_dt' in names:
            rates[:, k] = U[:, names.index(f'd{n}_dt')]
    return np.einsum('kij,kj->ki', funcs.JU(X, U=U), rates) if U.shape[1] else 0.0


def _defect_growth(L, t, X, Wl, Tl, atol):
    """When the lifted run leaves the manifold and the original states: the first time the largest relative
    defect (and state error) passes 1e-8, 1e-6 and 1e-3, and the defect's exponential growth rate between
    1e-10 and 1e-3 (the manifold is invariant but need not attract: solver error grows along it at that rate)."""
    def scaled(Y, Z, cols):
        floor = max(atol, 1e-8 * float(np.max(np.abs(Z[:, cols])))) if Z[:, cols].size else atol
        sc = np.maximum(np.maximum(np.ptp(Z[:, cols], axis=0), np.max(np.abs(Z[:, cols]), axis=0)), floor)
        return np.max(np.abs(Y[:, cols] - Z[:, cols]) / sc, axis=1)
    out = {}
    series = {'state_error': scaled(Wl, X, list(range(L.n)))}
    if L.n_w > L.n:
        series['defect'] = scaled(Wl, Tl, list(range(L.n, L.n_w)))
    for name, d in series.items():
        out[name] = {f'{th:g}': (float(t[np.argmax(d > th)]) if np.any(d > th) else None) for th in (1e-8, 1e-6, 1e-3)}
    d = series.get('defect')
    if d is not None:
        ok = (d > 1e-10) & (d < 1e-3)
        if ok.sum() >= 5 and np.ptp(t[ok]) > 0:
            out['rate'] = float(np.polyfit(t[ok], np.log(d[ok]), 1)[0])
    return out


def lifted_jacobian(ops, w, f):
    """d(rhs_lifted)/dw at one lifted state w (n_w,) with input features f (n_f,), from the operators."""
    A, (hr, hk, hv), N, B, C = ops
    rows, cols = np.triu_indices(len(w))
    J = np.array(A, dtype=float)
    a, b = rows[hk], cols[hk]
    np.add.at(J, (hr, a), hv * w[b])
    np.add.at(J, (hr, b), hv * w[a])
    for k, Nk in enumerate(N):
        J = J + Nk * f[k]
    return J


def spectrum(funcs, X, U, n_points=40, threshold=1e3, n_vectors=5):
    """The lifted Jacobian's eigenvalues at ``n_points`` states of a trajectory on the manifold (w = T(x)): the
    leading real part, how many exceed ``threshold`` (1/time), and the fraction of each leading eigenvector's norm in
    the tangent space span(J), where 0 means transverse to the manifold."""
    ops = funcs.operators()
    idx = np.unique(np.linspace(0, len(X) - 1, min(n_points, len(X))).astype(int))
    W = funcs.T(X[idx], U=U[idx])
    F = funcs.features(U[idx])
    Js = funcs.J(X[idx], U=U[idx])
    lead, count, frac = [], [], []
    for k in range(len(idx)):
        ev, vec = np.linalg.eig(lifted_jacobian(ops, W[k], F[k]))
        order = np.argsort(-ev.real)
        lead.append(float(ev.real[order[0]]))
        count.append(int(np.sum(ev.real > threshold)))
        u_, s_, _ = np.linalg.svd(Js[k], full_matrices=False)
        Q = u_[:, s_ > 1e-12 * (s_[0] if s_.size else 1.0)]
        for j in order[:n_vectors]:
            v = vec[:, j]
            frac.append(float(np.linalg.norm(Q.T @ v) / max(np.linalg.norm(v), 1e-300)))
    return {'n_points': len(idx), 'threshold': threshold, 'leading_re_max': max(lead), 'leading_re_median': float(np.median(lead)),
            'count_above_max': max(count), 'count_above_median': float(np.median(count)),
            'leading_vectors_in_tangent_median': float(np.median(frac)) if frac else None,
            'leading_vectors_in_tangent_min': float(np.min(frac)) if frac else None}


def simulate_lifted(funcs, x0, u0, t_end, relift_every=None, rtol=1e-6, atol=1e-9, max_step=np.inf, max_wall=None):
    """Integrates the lifted system dw/dt = rhs_lifted(w, u0) (inputs held at ``u0``) from w = T(x0) with SciPy's BDF,
    optionally re-lifting w <- T(w[:n]) every ``relift_every``. Returns {success, t_reached, max_defect}: the
    largest |w - T(w[:n])| over max |w| at the segment ends. ``max_wall``: seconds of computing before it stops
    (``success`` False, with the time reached)."""
    wall_end = None if max_wall is None else time.perf_counter() + max_wall

    class _OutOfTime(Exception):
        pass

    def guard(y):
        if wall_end is not None and time.perf_counter() > wall_end:
            raise _OutOfTime()
        return y
    from scipy.integrate import solve_ivp
    U0 = np.atleast_2d(np.asarray(u0, dtype=float))
    F0 = funcs.features(U0)[0]
    ops = funcs.operators()
    n = len(funcs.STATES)
    w = funcs.T(np.atleast_2d(x0), U=U0)[0]
    t, defect = 0.0, 0.0
    step = relift_every or t_end
    while t < t_end - 1e-12:
        t1 = min(t + step, t_end)
        reached = [t]

        def rhs(_t, y):
            reached[0] = max(reached[0], _t)
            return guard(funcs.rhs_lifted(y[None, :], U0)[0])
        try:
            with np.errstate(all='ignore'):
                sol = solve_ivp(rhs, (t, t1), w, method='BDF', rtol=rtol, atol=atol, max_step=max_step,
                                jac=lambda _t, y: lifted_jacobian(ops, y, F0))
        except _OutOfTime:
            return {'success': False, 't_reached': float(reached[0]), 'max_defect': defect,
                    'message': f'over max_wall ({max_wall} s)'}
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            return {'success': False, 't_reached': float(sol.t[-1]), 'max_defect': defect, 'message': sol.message}
        w = sol.y[:, -1]
        Tw = funcs.T(w[None, :n], U=U0)[0]
        defect = max(defect, float(np.max(np.abs(w - Tw)) / max(float(np.max(np.abs(Tw))), 1e-300)))
        if relift_every:
            w = Tw
        t = t1
    return {'success': True, 't_reached': float(t), 'max_defect': defect}


def conserved_drift(L, info, run, funcs):
    """Each conserved functional's max |l.w(t) - l.w(0) - integral of l.(B f + C)| along the lifted run (along
    T of the model's run when the lifted model is not simulated)."""
    t, X, Wl, U, Tref = run
    W = Wl if Wl is not None else Tref
    F = funcs.features(U)
    _, _, _, B, C = funcs.operators()
    for c in info.get('conserved', []):
        lvec = np.array([c['l'].get(n, 0.0) for n in L.names])
        rate = F @ (B.T @ lvec) + float(C @ lvec)
        integral = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
        val = W @ lvec
        scale = max(float(np.max(np.abs(val))), 1e-12)
        c['drift'] = float(np.max(np.abs(val - val[0] - integral))) / scale


def invariants(L, info, run):
    """g(w) = 0 that hold exactly among the lifted variables: each auxiliary's definition (``definition``), the
    conserved linear functionals with no rate (``conserved``) and the clock norms (``oscillator``), each with
    max |g| on T of the model's run and on the lifted run, absolute and over the size of g's terms."""
    t, X, Wl, U, Tref = run
    m = L.model
    out = [dict(d) for d in L.invariant_definitions]
    w0 = Tref[0]
    for c in info.get('conserved', []):
        if abs(c.get('rate_constant', 0.0)) < 1e-14 and not c.get('rate_inputs'):
            e = sp.Add(*[sp.Float(v) * sp.Symbol(n, real=True) for n, v in c['l'].items()])
            const = sum(v * w0[L.names.index(n)] for n, v in c['l'].items())
            out.append({'expr': str(e - sp.Float(const)), 'degree': 1, 'origin': 'conserved'})
    for c in info.get('clocks', []):
        cs, ss = sp.Symbol(c['c'], real=True), sp.Symbol(c['s'], real=True)
        if c['invariant'].startswith('c^2 + s^2 conserved'):
            r2 = float(w0[L.names.index(c['c'])] ** 2 + w0[L.names.index(c['s'])] ** 2)
            out.append({'expr': str(cs ** 2 + ss ** 2 - sp.Float(r2)), 'degree': 2, 'origin': 'oscillator'})
        else:
            out.append({'expr': str(cs ** 2 + ss ** 2 - 1), 'degree': 2, 'origin': 'oscillator',
                        'note': 'holds on the limit cycle only'})
    pv = {k: sp.Float(v) for k, v in m.param_values().items()}
    syms = [sp.Symbol(n, real=True) for n in L.names]
    inp = list(m.input_symbols)
    for d in out:
        if 'expr' not in d:
            continue
        loc = {n: sp.Symbol(n, real=True) for n in L.names + [str(x) for x in inp] + [str(x) for x in pv]}
        e = sp.sympify(d['expr'], locals=loc)
        e = e.xreplace(pv)
        terms = sp.Add.make_args(sp.expand(e))
        try:
            f = sp.lambdify(syms + inp, e, 'numpy')
            fa = sp.lambdify(syms + inp, sp.Add(*[sp.Abs(a) for a in terms]), 'numpy')
            for key, W in (('reference', Tref), ('lifted_run', Wl)):
                if W is None:
                    continue
                args = list(W.T) + list(U.T)
                with np.errstate(all='ignore'):
                    g = np.abs(np.broadcast_to(np.asarray(f(*args), dtype=float), (len(t),)))
                    size = np.broadcast_to(np.asarray(fa(*args), dtype=float), (len(t),))
                d[f'max_{key}'] = float(np.nanmax(g))
                d[f'relative_{key}'] = float(np.nanmax(g / np.maximum(size, 1e-300)))
        except Exception as e_:  # noqa: BLE001 -- a check that cannot be evaluated is reported, not fatal
            d['check_error'] = f'{type(e_).__name__}: {e_}'
    return out


# ---------------------------------------------------------------------------------------------
# files

def write_outputs(L, output_dir, name='model'):
    """Writes ``<name>_lifted.cellml`` and ``lifting_functions.py``; returns their paths."""
    os.makedirs(output_dir, exist_ok=True)
    cellml_path = os.path.join(output_dir, f'{name}_lifted.cellml')
    with open(cellml_path, 'w', encoding='utf-8') as f:
        f.write(lifted_cellml(L, f'{name}_lifted' if name[:1].isalpha() else 'lifted'))
    py_path = os.path.join(output_dir, 'lifting_functions.py')
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write(python_module(L))
    return cellml_path, py_path


def lift_cellml_file(model_path, output_dir=None, inputs=(), smoothing_report=None, time_budget=DEFAULT_TIME_BUDGET,
                     engine='auto', max_aux=DEFAULT_MAX_AUX, sim_time=10.0, dt=0.01, solver_info=None,
                     do_verify=True, max_terms=DEFAULT_MAX_TERMS, as_inputs=(), keep=(), held_inputs=False,
                     sites=True, outputs=(), tanh_form='tanh', stabilize=None):
    """Lifts a (smoothed) flat CellML model and writes the lifted CellML model, ``lifting_functions.py`` and
    ``lifting.json`` (with the verification, the operator check, the structure and the invariants) into
    ``output_dir``. ``as_inputs``, ``keep``, ``held_inputs``, ``sites``: as :func:`lift`. Returns the lifting.json
    dict."""
    from libcuflynx.utilities.smooth_cellml import libcellml_issues
    t0 = time.perf_counter()
    name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(model_path)), 'lifting')
    L = lift(model_path, inputs, time_budget, engine, max_aux, smoothing_report, max_terms, as_inputs=as_inputs,
             keep=keep, held_inputs=held_inputs, sites=sites, outputs=outputs, tanh_form=tanh_form, stabilize=stabilize)
    cellml_path, py_path = write_outputs(L, output_dir, name)
    with open(cellml_path, encoding='utf-8') as f:
        issues = libcellml_issues(f.read())
    funcs = load_functions(py_path)
    verification = check = scales = None
    struct = structure(L)
    invs = [dict(d) for d in L.invariant_definitions]
    if do_verify:
        verification, check, scales, run = verify(L, model_path, cellml_path, funcs, sim_time, dt, solver_info)
        conserved_drift(L, struct, run, funcs)
        invs = invariants(L, struct, run)
        if L.n_w <= 400:
            try:
                struct['spectrum'] = spectrum(funcs, run[1], run[3])
            except Exception as e:  # noqa: BLE001 -- informational
                struct['spectrum'] = {'error': f'{type(e).__name__}: {e}'}
    L.timings['total'] = time.perf_counter() - t0
    doc = lifting_json(L, os.path.abspath(py_path), check, scales, verification, struct)
    doc['invariants'] = invs
    doc['stabilization'] = L.stabilization
    if L.outputs:
        doc['outputs'] = L.outputs
    doc['model'] = os.path.abspath(model_path)
    doc['lifted_model'] = os.path.abspath(cellml_path)
    doc['libcellml'] = issues
    with open(os.path.join(output_dir, 'lifting.json'), 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='cuflynx-lift',
        description='Lift a smooth flat CellML 2.0 model (e.g. from cuflynx-smooth) to quadratic form for Operator '
                    'Inference: writes <name>_lifted.cellml, lifting_functions.py and lifting.json. Does not read '
                    'user_inputs.yaml.')
    parser.add_argument('model', help='flat CellML 2.0 model, smoothed')
    parser.add_argument('--output-dir', help='where to write (default: lifting/ next to the model)')
    parser.add_argument('--inputs', default='', help='comma-separated variables kept as inputs u ([component/]name)')
    parser.add_argument('--smoothing-report', help="the smoothing pass's smoothing_report.json")
    parser.add_argument('--time-budget', type=float, default=DEFAULT_TIME_BUDGET, help='s for each QBee stage')
    parser.add_argument('--engine', choices=('auto', 'qbee', 'rules'), default='auto')
    parser.add_argument('--max-aux', type=int, default=DEFAULT_MAX_AUX)
    parser.add_argument('--max-terms', type=int, default=DEFAULT_MAX_TERMS,
                        help='largest polynomial system (terms) to quadratize')
    parser.add_argument('--sim-time', type=float, default=10.0, help='verification run length')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--no-verify', action='store_true')
    parser.add_argument('--as-inputs', default='', help='comma-separated variables (or component/* globs) cut from '
                                                        'the model as inputs whose values come from data')
    parser.add_argument('--keep', default='', help='comma-separated component globs whose states are kept; every '
                                                   'other state is cut as an input')
    parser.add_argument('--params-as-inputs', default='', help='comma-separated parameters driven as inputs (as --inputs)')
    parser.add_argument('--held-inputs', nargs='?', const='states', choices=('states', 'inputs'),
                        help='named and cut inputs are held (du/dt = 0 within a chunk): as states (default) or as '
                             'inputs (B and N terms)')
    parser.add_argument('--tanh-form', choices=('tanh', 'exp'), default='tanh')
    parser.add_argument('--stabilize', type=float, help='kappa: add -kappa g(w) for each quadratic identity')
    parser.add_argument('--outputs', default='', help='comma-separated variables lifted with the states')
    parser.add_argument('--no-sites', action='store_true',
                        help='polynomialize the smoothing sites rather than lifting them in closed form')
    args = parser.parse_args(argv)
    doc = lift_cellml_file(args.model, args.output_dir,
                           [s for s in (args.inputs + ',' + args.params_as_inputs).split(',') if s],
                           args.smoothing_report, args.time_budget, args.engine, args.max_aux, args.sim_time,
                           args.dt, do_verify=not args.no_verify, max_terms=args.max_terms,
                           as_inputs=[s for s in args.as_inputs.split(',') if s], keep=[s for s in args.keep.split(',') if s],
                           held_inputs=args.held_inputs or False, sites=not args.no_sites,
                           tanh_form=args.tanh_form, stabilize=args.stabilize,
                           outputs=[s for s in args.outputs.split(',') if s])
    e = doc['engine']
    print(f'{doc["size"]["n"]} -> {doc["size"]["n_w"]} lifted variables ({e.get("polynomialize")}, '
          f'{e.get("quadratize")}); operators: {", ".join(k for k, v in doc["operators"]["present"].items() if v)}')
    if 'verification' in doc:
        v = doc['verification']
        g = lambda x: 'not simulated' if x is None else f'{x:.3g}'  # noqa: E731
        print(f'state error {g(v["max_state_error"])}, manifold defect {g(v["max_manifold_defect"])}, '
              f'J f check {v["jf"]["max"]:.3g}, operator residual {doc["check"]["max_relative_residual"]:.3g}')
    print(doc['lifted_model'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
