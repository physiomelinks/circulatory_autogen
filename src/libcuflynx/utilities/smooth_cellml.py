"""Smooth approximations of the non-smooth operations in a flat CellML 2.0 model.

Every replacement uses only square roots, reciprocals, products and sums, so the smoothed model
lifts exactly to quadratic form. Each site ``k`` gets named algebraic variables in its component:

- ``sm<k>_x``: the site's argument
- ``eps_sm<k>``: a constant in the units of x
- ``sm<k>_w = sqrt(x^2 + eps^2)`` and ``sm<k>_z = 1/sm<k>_w``
- ``sm<k>_H = (1 + x*z)/2`` for a comparison in a piecewise condition

========================================  ===========================================
original                                  replacement
========================================  ===========================================
``abs(x)``                                ``sm_w``
``max(a, b)``                             ``(a + b + sm_w)/2``, x = a - b
``min(a, b)``                             ``(a + b - sm_w)/2``, x = a - b
``f if a >= b (or >) else g``             ``g + sm_H*(f - g)``, x = a - b
``f if a <= b (or <) else g``             as above with x = b - a
``c1 and c2`` / ``c1 or c2`` / ``not c``  ``H1*H2`` / ``1 - (1-H1)(1-H2)`` / ``1 - H``
several pieces                            blended from the last piece back to the first
========================================  ===========================================

``floor`` and piecewise conditions using ``eq``, ``neq``, ``xor`` or ``floor`` are left as written
and reported as not smoothed. Rewrites go bottom-up, so nested operations become separate sites.

The report flags a site downstream of a floor (``floor_upstream``, through algebraic equations
and connections), a piecewise branch that may be undefined where it is not active
(``branch_domain``, e.g. ln), and an x that reaches zero only at one end of its range
(``one_sided``).

Use in two stages: :func:`probe` adds only the ``sm<k>_x`` variables, so reference runs of the
probed model give each argument's range; :func:`smooth` then sets ``eps_sm<k> = rel * range`` and
rewrites the sites. :func:`smooth_cellml_file` does both, running the reference runs with
CVODE_myokit.
"""

import argparse
import copy
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

CELLML_NS = 'http://www.cellml.org/cellml/2.0#'
MATHML_NS = 'http://www.w3.org/1998/Math/MathML'
_C = '{' + CELLML_NS + '}'
_M = '{' + MATHML_NS + '}'

DEFAULT_REL = {'abs': 1e-3, 'maxmin': 1e-3, 'piecewise': 1e-3}
#: eps of a site whose argument is zero throughout the reference runs, in the units of x
DEFAULT_ABS_EPS = 1e-6

_DIMENSIONLESS_FUNCTIONS = {'exp', 'ln', 'log', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh',
                            'tanh', 'sech', 'csch', 'coth', 'arcsin', 'arccos', 'arctan', 'arcsec',
                            'arccsc', 'arccot', 'arcsinh', 'arccosh', 'arctanh', 'arcsech', 'arccsch',
                            'arccoth'}
_COMPARISONS = {'gt', 'geq', 'lt', 'leq'}
_UNSMOOTHED_CONDITIONS = {'eq', 'neq', 'xor', 'floor', 'ceiling'}
#: operations that may be undefined where a piecewise branch is not active
_DOMAIN_LIMITED = {'ln', 'log', 'root', 'arcsin', 'arccos', 'arccosh', 'arctanh'}
_INFIX = {'plus': '+', 'minus': '-', 'times': '*', 'divide': '/', 'power': '^', 'gt': '>', 'geq': '>=',
          'lt': '<', 'leq': '<=', 'eq': '==', 'neq': '!=', 'and': ' and ', 'or': ' or '}


def _tag(e):
    return e.tag.split('}')[-1]


def _apply(op, *args):
    e = ET.Element(_M + 'apply')
    ET.SubElement(e, _M + op)
    e.extend(args)
    return e


def _ci(name):
    e = ET.Element(_M + 'ci')
    e.text = name
    return e


def _cn(value, units='dimensionless'):
    e = ET.Element(_M + 'cn', {_C + 'units': units})
    value = float(value)
    e.text = str(int(value)) if value.is_integer() else repr(value)
    return e


def _op(e):
    return _tag(e[0]) if _tag(e) == 'apply' and len(e) else None


def _args(e):
    return [c for c in list(e)[1:] if _tag(c) not in ('bvar', 'degree', 'logbase')]


def infix(e):
    """A readable one-line form of a MathML expression, for the report."""
    t = _tag(e)
    if t in ('ci', 'cn'):
        return (e.text or '').strip()
    if t == 'piecewise':
        parts = [f'{infix(p[0])} if {infix(p[1])}' for p in e if _tag(p) == 'piece']
        parts += [f'otherwise {infix(o[0])}' for o in e if _tag(o) == 'otherwise']
        return '{' + '; '.join(parts) + '}'
    if t != 'apply':
        return t
    op, args = _op(e), _args(e)
    if op == 'diff':
        return f'd{infix(args[0])}/d{infix(e.find(_M + "bvar")[0])}'
    if op == 'minus' and len(args) == 1:
        return f'-({infix(args[0])})'
    if op == 'not':
        return f'not({infix(args[0])})'
    if op in _INFIX:
        return '(' + _INFIX[op].join(infix(a) for a in args) + ')'
    if op == 'root':
        return f'sqrt({infix(args[0])})'
    return f'{op}(' + ', '.join(infix(a) for a in args) + ')'


# ---------------------------------------------------------------------------------------------
# units: a dict {units name: exponent}; {} is dimensionless

def _units_of(e, var_units):
    """The units of a MathML expression as {units name: exponent}, or None if not inferable."""
    t = _tag(e)
    if t == 'ci':
        u = var_units.get((e.text or '').strip())
        return None if u is None else dict(u)
    if t == 'cn':
        u = e.get(_C + 'units')
        return None if u is None else _named(u)
    if t in ('pi', 'exponentiale'):
        return {}
    if t == 'piecewise':
        for p in e:
            if _tag(p) in ('piece', 'otherwise'):
                u = _units_of(p[0], var_units)
                if u is not None:
                    return u
        return None
    if t != 'apply':
        return None
    op, args = _op(e), _args(e)
    if op in ('plus', 'minus', 'abs', 'max', 'min', 'floor', 'ceiling', 'rem'):
        for a in sorted(args, key=lambda a: _tag(a) != 'ci'):
            u = _units_of(a, var_units)
            if u is not None:
                return u
        return None
    if op in ('times', 'divide'):
        total = {}
        for i, a in enumerate(args):
            u = _units_of(a, var_units)
            if u is None:
                return None
            total = _combine(total, u, -1 if (op == 'divide' and i > 0) else 1)
        return total
    if op == 'power':
        base, exponent = _units_of(args[0], var_units), args[1]
        if base is None:
            return None
        if base == {}:
            return {}
        if _tag(exponent) == 'cn':
            return _combine({}, base, float(exponent.text))
        return None
    if op == 'root':
        base = _units_of(args[0], var_units)
        degree = e.find(_M + 'degree')
        n = float(degree[0].text) if degree is not None and _tag(degree[0]) == 'cn' else 2.0
        if degree is not None and _tag(degree[0]) != 'cn':
            return None
        return None if base is None else _combine({}, base, 1.0 / n)
    if op in _DIMENSIONLESS_FUNCTIONS:
        return {}
    if op == 'diff':
        u = _units_of(args[0], var_units)
        bv = _units_of(e.find(_M + 'bvar')[0], var_units)
        return None if u is None or bv is None else _combine(u, bv, -1)
    return None


def _named(name):
    return {} if name == 'dimensionless' else {name: 1.0}


def _combine(a, b, factor):
    out = dict(a)
    for name, exponent in b.items():
        out[name] = out.get(name, 0.0) + factor * exponent
        if abs(out[name]) < 1e-12:
            del out[name]
    return out


# ---------------------------------------------------------------------------------------------

class _Site:
    def __init__(self, k, kind, component, variable, original):
        self.k, self.kind, self.component, self.variable = k, kind, component, variable
        self.record = {'k': k, 'kind': kind, 'component': component, 'variable': variable,
                       'original': original, 'status': 'smoothed', 'flags': []}


class _Pass:
    def __init__(self, root, mode, ranges=None, rel=None, collapse=None, abs_eps=DEFAULT_ABS_EPS):
        self.root, self.mode = root, mode
        self.ranges = ranges or {}
        self.rel = dict(DEFAULT_REL)
        if isinstance(rel, (int, float)):
            self.rel = {kind: float(rel) for kind in DEFAULT_REL}
        elif rel:
            self.rel.update(rel)
        self.collapse = collapse
        self.abs_eps = abs_eps
        self.k = 0
        self.sites, self.not_smoothed = [], []
        self.units_defs = {}  # name -> units dict, for composite units added to the model
        self.existing_units = {u.get('name') for u in root.findall(_C + 'units')}
        self.floor_tainted = _floor_tainted(root)

    # -- model edits ----------------------------------------------------------------------
    def _units_name(self, u):
        if not u:
            return 'dimensionless'
        if len(u) == 1:
            (name, exponent), = u.items()
            if exponent == 1.0:
                return name
        key = tuple(sorted(u.items()))
        for name, value in self.units_defs.items():
            if tuple(sorted(value.items())) == key:
                return name
        name = 'smu_' + '_'.join(f'{n}_e{_fmt(e)}' for n, e in key).replace('-', 'm').replace('.', 'p')
        while name in self.existing_units:
            name += '_'
        self.units_defs[name] = dict(u)
        return name

    def _add_variable(self, name, u, initial_value=None):
        if name in self.var_units:
            raise ValueError(f'component {self.component.get("name")} already has a variable {name}')
        attrs = {'name': name, 'units': self._units_name(u)}
        if initial_value is not None:
            attrs['initial_value'] = repr(float(initial_value))
        variables = self.component.findall(_C + 'variable')
        index = list(self.component).index(variables[-1]) + 1 if variables else 0
        self.component.insert(index, ET.Element(_C + 'variable', attrs))
        self.var_units[name] = dict(u)

    def _add_equation(self, name, rhs):
        if self.new_math is None:
            self.new_math = ET.SubElement(self.component, _M + 'math')
        self.new_math.append(_apply('eq', _ci(name), rhs))

    def _write_units(self):
        first = self.root.find(_C + 'component')
        index = list(self.root).index(first) if first is not None else len(self.root)
        for name, u in self.units_defs.items():
            el = ET.Element(_C + 'units', {'name': name})
            for unit, exponent in sorted(u.items()):
                ET.SubElement(el, _C + 'unit', {'units': unit, 'exponent': _fmt(exponent)})
            self.root.insert(index, el)
            index += 1

    # -- traversal --------------------------------------------------------------------------
    def run(self):
        for component in self.root.findall(_C + 'component'):
            self.component, self.new_math = component, None
            self.var_units = {v.get('name'): _named(v.get('units')) for v in component.findall(_C + 'variable')}
            for m in list(component.findall(_M + 'math')):
                for eq in list(m):
                    if _op(eq) != 'eq' or len(_args(eq)) != 2:
                        continue
                    lhs, rhs = _args(eq)
                    if _op(lhs) == 'diff':
                        lhs = _args(lhs)[0]
                    self.variable = (lhs.text or '').strip() if _tag(lhs) == 'ci' else infix(lhs)
                    new = self._visit(rhs)
                    if new is not rhs:
                        eq[list(eq).index(rhs)] = new
        self._write_units()

    def _visit(self, e):
        if _tag(e) not in ('apply', 'piecewise', 'piece', 'otherwise'):
            return e
        original = infix(e) if _op(e) in ('abs', 'max', 'min') or _tag(e) == 'piecewise' else None
        if _tag(e) == 'piecewise':
            snapshot = copy.deepcopy(e)
        for i, child in enumerate(list(e)):
            new = self._visit(child)
            if new is not child:
                e[i] = new
        op = _op(e)
        if op == 'abs':
            return self._abs(e, original)
        if op in ('max', 'min'):
            return self._maxmin(e, op, original)
        if op in ('floor', 'ceiling'):
            self._not_smoothed(op, infix(e), f'{op} is left as written')
        if _tag(e) == 'piecewise':
            return self._piecewise(e, snapshot)
        return e

    def _not_smoothed(self, kind, original, reason):
        self.not_smoothed.append({'kind': kind, 'component': self.component.get('name'),
                                  'variable': self.variable, 'original': original, 'reason': reason})

    # -- sites ------------------------------------------------------------------------------
    def _new_site(self, kind, x, original, strict=None):
        """Create site k for argument x. Returns the site, or None when it is skipped.
        ``strict`` is True for a > or < comparison, False for >= or <=, None otherwise."""
        self.k += 1
        k = self.k
        site = _Site(k, kind, self.component.get('name'), self.variable, original)
        site.strict = strict
        names = {s: f'sm{k}_{s}' for s in ('x', 'w', 'z')}
        if kind == 'piecewise':
            names['H'] = f'sm{k}_H'
        site.names, site.eps_name = names, f'eps_sm{k}'
        if self._floor_upstream(x):
            site.record['flags'].append('floor_upstream')
        self.sites.append(site)
        u = _units_of(x, self.var_units)
        if u is None:
            site.record['status'] = 'skipped'
            site.record['reason'] = 'units of x could not be inferred'
            site.units = None
            return None
        site.units = u
        site.record['units_x'] = self._units_name(u)
        if 'floor_upstream' in site.record['flags']:
            self.floor_tainted.update((site.component, n) for n in names.values())
        if self.mode == 'smooth':
            self._set_eps(site)
            if self._collapses(site):
                return site
        self._add_variable(names['x'], u)
        self._add_equation(names['x'], copy.deepcopy(x))
        if self.mode == 'probe':
            return site
        self._add_variable(site.eps_name, u, initial_value=site.record['eps'])
        self._add_variable(names['w'], u)
        self._add_variable(names['z'], _combine({}, u, -1))
        self._add_equation(names['w'], _apply('root', _apply(
            'plus', _apply('power', _ci(names['x']), _cn(2)), _apply('power', _ci(site.eps_name), _cn(2)))))
        self._add_equation(names['z'], _apply('divide', _cn(1), _ci(names['w'])))
        if kind == 'piecewise':
            self._add_variable(names['H'], {})
            self._add_equation(names['H'], _apply('times', _cn(0.5), _apply(
                'plus', _cn(1), _apply('times', _ci(names['x']), _ci(names['z'])))))
        site.record['aux'] = [names['x'], site.eps_name, names['w'], names['z']] + (
            [names['H']] if kind == 'piecewise' else [])
        return site

    def _set_eps(self, site):
        r = self.ranges.get(site.names['x'])
        rel = float(self.rel[site.kind])
        record = site.record
        record['rel'] = rel
        if r is None:
            record['flags'].append('no_range')
            record['range'] = None
            record['eps'] = self.abs_eps
            site.lo = site.hi = site.min_abs = None
            return
        if isinstance(r, dict):
            lo, hi, min_abs = float(r['min']), float(r['max']), r.get('min_abs')
        else:
            lo, hi, min_abs = float(r[0]), float(r[1]), None
        if min_abs is None:
            min_abs = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
        site.lo, site.hi, site.min_abs = lo, hi, float(min_abs)
        record['range'] = [lo, hi]
        record['margin'] = float(min_abs)
        if (lo == 0.0 < hi) or (lo < hi == 0.0):
            # x reaches the switch only at one end of its range, e.g. where a floor wraps
            record['flags'].append('one_sided')
        span = hi - lo
        if span > 0.0:
            record['eps'] = rel * span
        else:
            record['flags'].append('zero_range')
            record['eps'] = rel * abs(lo) if lo != 0.0 else self.abs_eps
        eps, m = record['eps'], float(min_abs)
        w_dev = math.hypot(m, eps) - m
        record['max_deviation'] = {'abs': w_dev, 'maxmin': 0.5 * w_dev,
                                   'piecewise': 0.5 * (1.0 - m / math.hypot(m, eps))}[site.kind]
        record['deviation_of'] = 'sm_H' if site.kind == 'piecewise' else 'value'

    def _collapses(self, site):
        """Whether the site never switches in the reference runs and collapse is on for it."""
        record = site.record
        record['collapsed'] = False
        if not self.collapse or site.lo is None:
            return False
        if self.collapse is True:
            if site.min_abs == 0.0 and site.lo != site.hi:
                return False  # x reaches the switch: only a site named in collapse is collapsed
        elif site.k not in self.collapse:
            return False
        if site.strict:
            positive, negative = site.lo > 0.0, site.hi <= 0.0
        else:
            positive, negative = site.lo >= 0.0, site.hi < 0.0
        if not (positive or negative):
            return False
        record['collapsed'] = True
        # +1: x >= 0 throughout (the condition holds); -1: it never does
        record['sign'] = 1 if positive else -1
        record['max_deviation'] = 0.0
        return True

    def _abs(self, e, original):
        x = _args(e)[0]
        site = self._new_site('abs', x, original)
        if site is None or self.mode == 'probe':
            return e
        if site.record['collapsed']:
            new = copy.deepcopy(x) if site.record['sign'] > 0 else _apply('minus', copy.deepcopy(x))
        else:
            new = _ci(site.names['w'])
        site.record['replacement'] = infix(new)
        return new

    def _maxmin(self, e, op, original):
        args = _args(e)
        result = args[0]
        for b in args[1:]:
            a = result
            x = _apply('minus', copy.deepcopy(a), copy.deepcopy(b))
            site = self._new_site('maxmin', x, original)
            if site is None or self.mode == 'probe':
                result = _apply(op, a, b)
                continue
            if site.record['collapsed']:
                result = a if (site.record['sign'] > 0) == (op == 'max') else b
            else:
                result = _apply('times', _cn(0.5), _apply(
                    'plus' if op == 'max' else 'minus', _apply('plus', a, b), _ci(site.names['w'])))
            site.record['replacement'] = infix(result)
        return e if self.mode == 'probe' else result

    def _piecewise(self, e, snapshot):
        pieces = [p for p in e if _tag(p) == 'piece']
        otherwise = [o for o in e if _tag(o) == 'otherwise']
        original = infix(snapshot)
        if not otherwise:
            self._not_smoothed('piecewise', original, 'no otherwise branch')
            return e
        bad = [c for p in pieces for c in p[1].iter() if _tag(c) in _UNSMOOTHED_CONDITIONS]
        if bad:
            self._not_smoothed('piecewise', original,
                               f'condition uses {sorted({_tag(c) for c in bad})}, left as written')
            return e
        unsupported = [p for p in pieces if not self._condition_supported(p[1])]
        if unsupported:
            self._not_smoothed('piecewise', original, 'condition is not a comparison, and, or or not')
            return e
        branch_domain = any(self._domain_limited(b) for p in pieces for b in (p[0],)) or \
            self._domain_limited(otherwise[0][0])
        first_site = self.k + 1
        conditions = [self._condition(p[1], original) for p in pieces]
        for site in self.sites[first_site - 1:]:
            site.record['piecewise'] = original
            if branch_domain:
                site.record['flags'].append('branch_domain')
        if self.mode == 'probe' or any(c is None for c in conditions):
            return e
        result = otherwise[0][0]
        for piece, h in reversed(list(zip(pieces, conditions))):
            f = piece[0]
            if h is True:
                result = f
            elif h is False:
                continue
            else:
                result = _apply('plus', result, _apply('times', h, _apply('minus', f, copy.deepcopy(result))))
        for site in self.sites[first_site - 1:]:
            site.record['replacement'] = infix(result)
        return result

    def _condition_supported(self, c):
        op = _op(c)
        if _tag(c) in ('true', 'false'):
            return True
        if op in _COMPARISONS:
            return len(_args(c)) == 2
        if op in ('and', 'or', 'not'):
            return all(self._condition_supported(a) for a in _args(c))
        return False

    def _condition(self, c, original):
        """The smooth indicator of a condition: an expression, True or False (a collapsed
        or constant condition), or None in probe mode or when a site is skipped."""
        if _tag(c) in ('true', 'false'):
            return _tag(c) == 'true'
        op, args = _op(c), _args(c)
        if op in _COMPARISONS:
            a, b = args
            x = _apply('minus', copy.deepcopy(a), copy.deepcopy(b)) if op in ('gt', 'geq') else \
                _apply('minus', copy.deepcopy(b), copy.deepcopy(a))
            site = self._new_site('piecewise', x, infix(c), strict=op in ('gt', 'lt'))
            if site is None or self.mode == 'probe':
                return None
            if site.record['collapsed']:
                return site.record['sign'] > 0
            return _ci(site.names['H'])
        parts = [self._condition(a, original) for a in args]
        if any(p is None for p in parts):
            return None
        if op == 'not':
            p = parts[0]
            return (not p) if isinstance(p, bool) else _apply('minus', _cn(1), p)
        if op == 'and':
            if any(p is False for p in parts):
                return False
            parts = [p for p in parts if p is not True]
            return True if not parts else (parts[0] if len(parts) == 1 else _apply('times', *parts))
        # or
        if any(p is True for p in parts):
            return True
        parts = [p for p in parts if p is not False]
        if not parts:
            return False
        if len(parts) == 1:
            return parts[0]
        return _apply('minus', _cn(1), _apply('times', *[_apply('minus', _cn(1), p) for p in parts]))

    def _domain_limited(self, e):
        for node in e.iter():
            op = _op(node)
            if op in _DOMAIN_LIMITED:
                return True
            if op == 'power':
                exponent = _args(node)[1]
                if _tag(exponent) != 'cn' or float(exponent.text) != int(float(exponent.text)):
                    return True
        return False

    def _floor_upstream(self, x):
        for node in x.iter():
            if _tag(node) in ('floor', 'ceiling'):
                return True
            if _tag(node) == 'ci' and (self.component.get('name'), (node.text or '').strip()) in self.floor_tainted:
                return True
        return False


def _fmt(value):
    value = float(value)
    return str(int(value)) if value == int(value) else repr(value)


def _floor_tainted(root):
    """(component, variable) pairs whose value depends on a floor or ceiling through algebraic
    equations and connections (not through integration)."""
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    for conn in root.findall(_C + 'connection'):
        c1, c2 = conn.get('component_1'), conn.get('component_2')
        for mv in conn.findall(_C + 'map_variables'):
            a, b = find((c1, mv.get('variable_1'))), find((c2, mv.get('variable_2')))
            if a != b:
                parent[a] = b
    deps, direct = {}, set()
    for comp in root.findall(_C + 'component'):
        name = comp.get('name')
        for m in comp.findall(_M + 'math'):
            for eq in m:
                if _op(eq) != 'eq' or len(_args(eq)) != 2:
                    continue
                lhs, rhs = _args(eq)
                if _tag(lhs) != 'ci':
                    continue
                target = find((name, lhs.text.strip()))
                deps.setdefault(target, set()).update(
                    find((name, n.text.strip())) for n in rhs.iter() if _tag(n) == 'ci')
                if any(_tag(n) in ('floor', 'ceiling') for n in rhs.iter()):
                    direct.add(target)
    tainted, changed = set(direct), True
    while changed:
        changed = False
        for target, ds in deps.items():
            if target not in tainted and ds & tainted:
                tainted.add(target)
                changed = True
    out = set()
    for comp in root.findall(_C + 'component'):
        for v in comp.findall(_C + 'variable'):
            key = (comp.get('name'), v.get('name'))
            if find(key) in tainted:
                out.add(key)
    return out


# ---------------------------------------------------------------------------------------------

def _read(model):
    if isinstance(model, str) and not model.lstrip().startswith('<') and os.path.exists(model):
        with open(model, encoding='utf-8') as f:
            model = f.read()
    root = ET.fromstring(model.encode('utf-8') if isinstance(model, str) else model)
    if root.tag != _C + 'model':
        raise ValueError('smooth_cellml needs a CellML 2.0 model')
    if root.find(_C + 'import') is not None:
        raise ValueError('smooth_cellml needs a flat model (no imports): flatten it first')
    return root


def _write(root):
    """The model as CellML text, normalised by libCellML's printer."""
    from libcellml import Parser, Printer
    ET.register_namespace('cellml', CELLML_NS)
    ET.register_namespace('mml', MATHML_NS)
    text = ET.tostring(root, encoding='unicode')
    # CellML elements in the default namespace, each math element declaring MathML as its default
    text = text.replace(f' xmlns:mml="{MATHML_NS}"', '', 1)
    text = text.replace('<cellml:model ', f'<model xmlns="{CELLML_NS}" ', 1)
    text = text.replace('<mml:math', f'<math xmlns="{MATHML_NS}"')
    text = re.sub(r'<(/?)(?:cellml|mml):', r'<\1', text)
    parser = Parser()
    model = parser.parseModel(text)
    if parser.errorCount():
        raise RuntimeError('libCellML could not parse the smoothed model: '
                           + '; '.join(parser.error(i).description() for i in range(parser.errorCount())))
    return Printer().printModel(model)


def probe(model):
    """``model`` (CellML text or a path) with each site's argument as an ``sm<k>_x`` variable,
    and nothing else changed. Run it to get the ranges :func:`smooth` needs."""
    root = _read(model)
    _Pass(root, 'probe').run()
    return _write(root)


def smooth(model, ranges, rel=None, collapse=None, abs_eps=DEFAULT_ABS_EPS):
    """The smoothed model and its report.

    Args:
        model: CellML text or a path: the original flat model (not the probed one).
        ranges: ``{"sm<k>_x": {"min": .., "max": .., "min_abs": ..}}`` over the reference runs
            (``[min, max]`` also accepted). ``min_abs``, the smallest |x| seen, sets the largest
            deviation; without it the deviation is the worst case for the range.
        rel: ``{"abs", "maxmin", "piecewise"}`` -> eps relative to each argument's range, or one
            float for all. Defaults to 1e-3.
        collapse: replace a site that never switches in the reference runs by its active branch.
            None or False: off. True: each such site whose |x| stays above zero. A list of site
            numbers k: those sites, also when x reaches zero without changing sign (flagged
            ``one_sided``, e.g. a phase just after a floor wraps).
        abs_eps: eps of a site whose argument is zero throughout, or that has no range.

    Returns:
        (CellML text, report dict)
    """
    root = _read(model)
    p = _Pass(root, 'smooth', ranges, rel, collapse, abs_eps)
    p.run()
    report = {'settings': {'rel': p.rel, 'collapse': collapse if collapse in (None, True, False)
                           else sorted(collapse), 'abs_eps': abs_eps},
              'counts': _counts(p),
              'sites': [s.record for s in p.sites],
              'not_smoothed': p.not_smoothed}
    return _write(root), report


def _counts(p):
    counts = {}
    for s in p.sites:
        key = s.kind if s.record['status'] == 'smoothed' else f'{s.kind}_{s.record["status"]}'
        if s.record.get('collapsed'):
            key += '_collapsed'
        counts[key] = counts.get(key, 0) + 1
    for n in p.not_smoothed:
        key = f'{n["kind"]}_not_smoothed'
        counts[key] = counts.get(key, 0) + 1
    return counts


def probe_variables(probed):
    """``[(component, "sm<k>_x")]`` of a probed model."""
    root = _read(probed)
    return [(c.get('name'), v.get('name')) for c in root.findall(_C + 'component')
            for v in c.findall(_C + 'variable') if re.fullmatch(r'sm\d+_x', v.get('name'))]


def libcellml_issues(model_text):
    """Validator and Analyser issue descriptions of a model."""
    from libcellml import Analyser, Parser, Validator
    model = Parser().parseModel(model_text)
    validator, analyser = Validator(), Analyser()
    validator.validateModel(model)
    analyser.analyseModel(model)
    return {'validator': [validator.issue(i).description() for i in range(validator.issueCount())],
            'analyser': [analyser.issue(i).description() for i in range(analyser.issueCount())]}


def reference_ranges(probed_path, runs):
    """Each ``sm<k>_x``'s min, max and smallest |x| over the reference runs of a probed model.

    Each run is a dict with ``sim_time`` and optionally ``pre_time``, ``dt``, ``solver_info`` and
    ``param_vals`` (``{"component/variable": value}``), simulated with CVODE_myokit.
    """
    from libcuflynx.solver_wrappers import get_simulation_helper
    names = [f'{c}/{v}' for c, v in probe_variables(probed_path)]
    ranges = {}
    for run in runs:
        helper = get_simulation_helper(model_path=probed_path, solver='CVODE_myokit', model_type='cellml',
                                       dt=run.get('dt', 0.01), sim_time=run['sim_time'],
                                       solver_info=run.get('solver_info'), pre_time=run.get('pre_time', 0.0))
        if run.get('param_vals'):
            helper.set_param_vals(list(run['param_vals']), list(run['param_vals'].values()))
        if not helper.run():
            raise RuntimeError(f'reference run {run} failed')
        for name, values in zip(names, helper.get_results([[n] for n in names], flatten=True)):
            values = np.asarray(values, dtype=float).ravel()
            key = name.split('/')[-1]
            lo, hi, m = float(values.min()), float(values.max()), float(np.abs(values).min())
            if key in ranges:
                r = ranges[key]
                lo, hi, m = min(lo, r['min']), max(hi, r['max']), min(m, r['min_abs'])
            ranges[key] = {'min': lo, 'max': hi, 'min_abs': m}
        helper.close_simulation()
    return ranges


def smooth_cellml_file(model_path, runs=None, output_dir=None, ranges=None, rel=None, collapse=None,
                       abs_eps=DEFAULT_ABS_EPS):
    """Probe, run the reference runs (unless ``ranges`` is given), smooth, and write
    ``<name>_smooth.cellml`` and ``smoothing_report.json`` into ``output_dir``.

    Returns the paths of the smoothed model and of the report.
    """
    if (runs is None) == (ranges is None):
        raise ValueError('give either runs or ranges')
    name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = output_dir or os.path.dirname(os.path.abspath(model_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(model_path, encoding='utf-8') as f:
        original = f.read()
    if ranges is None:
        probed_path = os.path.join(output_dir, f'{name}_probe.cellml')
        with open(probed_path, 'w', encoding='utf-8') as f:
            f.write(probe(original))
        ranges = reference_ranges(probed_path, runs)
    smoothed, report = smooth(original, ranges, rel=rel, collapse=collapse, abs_eps=abs_eps)
    before, after = libcellml_issues(original), libcellml_issues(smoothed)
    report['model'] = os.path.abspath(model_path)
    report['runs'] = runs
    report['new_issues'] = {k: [i for i in after[k] if i not in before[k]] for k in after}
    smooth_path = os.path.join(output_dir, f'{name}_smooth.cellml')
    report_path = os.path.join(output_dir, 'smoothing_report.json')
    with open(smooth_path, 'w', encoding='utf-8') as f:
        f.write(smoothed)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return smooth_path, report_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='cuflynx-smooth',
        description='Write a smooth copy of a flat CellML 2.0 model, <name>_smooth.cellml, and '
                    'smoothing_report.json. Does not read user_inputs.yaml: the ranges come from '
                    '--ranges, or from reference runs of the probed model given by --runs.')
    parser.add_argument('model', help='flat CellML 2.0 model')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--ranges', help='JSON {"sm<k>_x": {"min", "max", "min_abs"}}')
    source.add_argument('--runs', help='JSON list of runs: {"sim_time", "pre_time", "dt", "solver_info", '
                                       '"param_vals"}, simulated with CVODE_myokit')
    parser.add_argument('--output-dir', help='where to write (default: next to the model)')
    parser.add_argument('--rel', type=float, help='eps relative to each range, for every kind (default 1e-3)')
    parser.add_argument('--collapse', action='store_true',
                        help='replace sites that never switch in the runs by their active branch')
    args = parser.parse_args(argv)

    def load(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    smooth_path, report_path = smooth_cellml_file(
        args.model, runs=load(args.runs) if args.runs else None,
        ranges=load(args.ranges) if args.ranges else None, output_dir=args.output_dir,
        rel=args.rel, collapse=True if args.collapse else None)
    print(smooth_path)
    print(report_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
