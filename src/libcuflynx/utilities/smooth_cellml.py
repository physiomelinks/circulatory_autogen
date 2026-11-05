"""Smooth approximations of the non-smooth operations in a flat CellML 2.0 model.

Every replacement uses only square roots, reciprocals, products and sums, so the smoothed model
lifts exactly to quadratic form. Each site ``k`` gets named algebraic variables in its component:

- ``sm<k>_x``: the site's argument (the denominator of a guarded division)
- ``eps_sm<k>``: a constant in the units of x
- ``sm<k>_w = sqrt(x^2 + eps^2)`` and ``sm<k>_z = 1/sm<k>_w``; ``sm<k>_H = (1 + x*z)/2`` where a rule
  needs a step; ``sm<k>_q = 1/(x^2 + eps^2)`` for a guarded division

The rules, one per kind of operation (the report's ``rule``). "Lifted" counts the variables a site adds
to the quadratic lift of the smoothed model (the report sums them as ``lifted_estimate``):

===============  ==========================  ==========================================  ======================  ===============================================
rule             pattern (example)           smooth form                                 lifted (per use)        eps (rel kind)
===============  ==========================  ==========================================  ======================  ===============================================
abs              abs(x), x changes sign      w                                           w, z: 2                 rel x range (abs)
maxmin           max/min(a, b), a - b        (a + b +- w)/2  (form "w", the default)     w, z: 2                 rel x range (maxmin)
                 changes sign (max(v, 0))    or b + (a - b) H, a - (a - b) H (form "xH")  z, x z, z^2, w: ~4
rectifier        max(x, 0), min(x, 0)        as maxmin, with b = 0: (x + w)/2, x H        as maxmin              as maxmin
piecewise        f if a >= b else g; and,    g + H (f - g), split on one comparison at  per comparison z, x z,  rel x range (piecewise)
                 or, not; several pieces     a time; weights sum to 1; a comparison     plus products with the
                                             repeated either way round is one H         branches
division         num/den, den reaches or     num den q, q = 1/(den^2 + eps_d^2)          q, den q, q^2: ~3       rel x range of den (division)
                 crosses 0 in a run
exempt           named in ``exempt``         left as written (switch_variables, phase)   0                       -
left as written  floor, eq, neq, xor         left as written                             0                       -
===============  ==========================  ==========================================  ======================  ===============================================

eps: ``rel`` (per kind, ``rel["division"]`` is rel_div, ``rel["variables"]`` per equation) times the
range of x in the nominal reference run. An abs or max/min argument that keeps one sign in every
run (an ``abs(tau)`` guard) takes ``rel x min(range, smallest |x|)`` instead, so eps stays below its
smallest value. A comparison keeps the range rule: one never seen to switch in the sampled runs may
still switch between samples. A division is guarded only when its denominator reaches or crosses 0 in some reference run;
denominators of constants alone are not probed. No site is collapsed unless ``collapse`` says so.

``floor`` and piecewise conditions using ``eq``, ``neq``, ``xor`` or ``floor`` are left as written
and reported as not smoothed. Rewrites go bottom-up, so nested operations become separate sites.

The report flags a site downstream of a floor (``floor_upstream``, through algebraic equations
and connections), a piecewise branch that may be undefined where it is not active
(``branch_domain``, e.g. ln), an x that reaches zero only at one end of its range
(``one_sided``), an x that is NaN or Inf in a reference run (``nan_in_runs``; the runs are
read from t = 0, through ``pre_time``), and a site NaN or Inf at the initial state in the probed or
smoothed model (``nan_at_init``, :func:`flag_nan_at_init`). Each range is the nominal (first) reference run's.

``exempt`` leaves the sites of named equations as written (a switch meant to stay, such as a phase
gate). ``jump_candidates`` in the report suggest such sites, and exempt sites and candidates carry
``switch_variables``, the states or floor phases their comparisons switch on.

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

DEFAULT_REL = {'abs': 1e-3, 'maxmin': 1e-3, 'piecewise': 1e-3, 'division': 1e-3}
#: lifted variables per site, by rule and form (estimates for the size of the quadratic lift)
LIFTED = {('abs', 'w'): 2, ('maxmin', 'w'): 2, ('maxmin', 'xH'): 4, ('rectifier', 'w'): 2, ('rectifier', 'xH'): 4,
          ('piecewise', 'H'): 2, ('division', 'q'): 3}
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
    def __init__(self, root, mode, ranges=None, rel=None, collapse=None, abs_eps=DEFAULT_ABS_EPS, only=None,
                 exempt=None, maxmin_form='w'):
        self.root, self.mode = root, mode
        if maxmin_form not in ('w', 'xH'):
            raise ValueError(f"maxmin_form: 'w' or 'xH', not {maxmin_form!r}")
        self.maxmin_form = maxmin_form
        self.exempt = [(e['variable'], e.get('reason', '')) for e in exempt or []]
        self.graph = _ModelGraph(root)
        self.jump_candidates = []
        self.only = None if only is None else set(only)
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
                    self.is_rate = _op(lhs) == 'diff'
                    if self.is_rate:
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
        if op == 'divide' and len(_args(e)) == 2 and self._varies(_args(e)[1]):
            return self._division(e)
        if op in ('floor', 'ceiling'):
            self._not_smoothed(op, infix(e), f'{op} is left as written')
        if _tag(e) == 'piecewise':
            return self._piecewise(e, snapshot)
        return e

    def _exempt_reason(self):
        import fnmatch
        for pattern, reason in self.exempt:
            name = f'{self.component.get("name")}/{self.variable}' if '/' in pattern else self.variable
            if fnmatch.fnmatchcase(name, pattern):
                return reason or 'exempt'
        return None

    def _jump_candidate(self, pieces, otherwise):
        """Why a piecewise looks meant to jump (reported, never applied), or None: a constant branch that
        differs from another branch (a jump at the switch), and a result that reaches only a state
        derivative, or only arguments of cos/sin (check that function is continuous across the jump)."""
        branches = [p[0] for p in pieces] + [otherwise]
        constants = [b for b in branches if _tag(b) == 'cn']
        if not constants or len({infix(b) for b in branches}) < 2:
            return None
        if self.is_rate:
            return 'an on/off rate: its result is a state derivative only'
        uses = self.graph.uses(self.component.get('name'), self.variable)
        if uses and uses <= {'trig'}:
            return 'used only inside cos/sin: check that it is continuous across the jump'
        return None

    def _not_smoothed(self, kind, original, reason):
        self.not_smoothed.append({'kind': kind, 'rule': 'left_as_written', 'component': self.component.get('name'),
                                  'variable': self.variable, 'original': original, 'reason': reason})

    # -- sites ------------------------------------------------------------------------------
    def _varies(self, e):
        """Whether an expression can change during a run: it names a state, an algebraic variable, the
        time or a variable this pass added (not only constants)."""
        comp = self.component.get('name')
        return any(_tag(n) == 'ci' and not self.graph.is_constant(comp, (n.text or '').strip())
                   for n in e.iter())

    def _new_site(self, kind, x, original, strict=None, rule=None):
        """Create site k for argument x. Returns the site, or None when it is skipped.
        ``strict`` is True for a > or < comparison, False for >= or <=, None otherwise."""
        self.k += 1
        k = self.k
        site = _Site(k, kind, self.component.get('name'), self.variable, original)
        site.strict = strict
        site.record['rule'] = rule or kind
        site.x = copy.deepcopy(x)
        site.form = {'maxmin': self.maxmin_form, 'piecewise': 'H', 'division': 'q'}.get(kind, 'w')
        site.needs_H = kind == 'piecewise' or (kind == 'maxmin' and self.maxmin_form == 'xH')
        names = {s: f'sm{k}_{s}' for s in (('x', 'q') if kind == 'division' else ('x', 'w', 'z'))}
        if site.needs_H:
            names['H'] = f'sm{k}_H'
        site.names, site.eps_name = names, f'eps_sm{k}'
        if self._floor_upstream(x):
            site.record['flags'].append('floor_upstream')
        self.sites.append(site)
        reason = self._exempt_reason()
        if reason is not None:
            site.record['status'] = 'exempt'
            site.record['rule'] = 'exempt'
            site.record['reason'] = reason
            site.record['switch_variables'] = self.graph.switch_variables(site.component, x)
            return None
        if self.mode == 'smooth' and self.only is not None and k not in self.only:
            site.record['status'] = 'left'
            site.record['reason'] = 'not in only'
            return None
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
            if kind == 'division' and not self._reaches_zero(site):
                site.record['status'] = 'not_needed'
                site.record['reason'] = 'the denominator does not reach 0 in any reference run'
                return None
            if self._collapses(site):
                return site
        self._add_variable(names['x'], u)
        self._add_equation(names['x'], copy.deepcopy(x))
        if self.mode == 'probe':
            return site
        self._add_variable(site.eps_name, u, initial_value=site.record['eps'])
        site.record['lifted'] = LIFTED[(site.record['rule'], site.form)]
        if kind == 'division':
            self._add_variable(names['q'], _combine({}, u, -2))
            self._add_equation(names['q'], _apply('divide', _cn(1), _apply(
                'plus', _apply('power', _ci(names['x']), _cn(2)), _apply('power', _ci(site.eps_name), _cn(2)))))
            site.record['aux'] = [names['x'], site.eps_name, names['q']]
            return site
        self._add_variable(names['w'], u)
        self._add_variable(names['z'], _combine({}, u, -1))
        self._add_equation(names['w'], _apply('root', _apply(
            'plus', _apply('power', _ci(names['x']), _cn(2)), _apply('power', _ci(site.eps_name), _cn(2)))))
        self._add_equation(names['z'], _apply('divide', _cn(1), _ci(names['w'])))
        if site.needs_H:
            self._add_variable(names['H'], {})
            self._add_equation(names['H'], _apply('times', _cn(0.5), _apply(
                'plus', _cn(1), _apply('times', _ci(names['x']), _ci(names['z'])))))
        site.record['aux'] = [names['x'], site.eps_name, names['w'], names['z']] + (
            [names['H']] if site.needs_H else [])
        return site

    def _set_eps(self, site):
        r = self.ranges.get(site.names['x'])
        by_variable = self.rel.get('variables') or {}
        rel = float(by_variable.get(f'{site.component}/{site.variable}', by_variable.get(site.variable, self.rel[site.kind])))
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
        if isinstance(r, dict) and r.get('nan'):
            record['flags'].append('nan_in_runs')
        record['range'] = [lo, hi]
        record['margin'] = float(min_abs)
        if (lo == 0.0 < hi) or (lo < hi == 0.0):
            # x reaches the switch only at one end of its range, e.g. where a floor wraps
            record['flags'].append('one_sided')
        span = hi - lo
        one_sign = lo * hi > 0.0 and not (isinstance(r, dict) and r.get('any_sign_change'))
        if span > 0.0 and one_sign and site.kind in ('abs', 'maxmin'):
            # keeps one sign: eps below its smallest |x| (an abs(tau) guard over three decades). Not for a
            # comparison: sampled on one side, it may still be crossed between samples (a spike's upstroke
            # crosses |V| = 1 within 0.2 ms), and a tiny eps there is a near-step.
            record['eps'] = rel * min(span, float(min_abs))
            record['eps_rule'] = 'rel x min(range, smallest |x|): x keeps one sign'
        elif span > 0.0:
            record['eps'] = rel * span
            record['eps_rule'] = 'rel x range'
        else:
            record['flags'].append('zero_range')
            record['eps'] = rel * abs(lo) if lo != 0.0 else self.abs_eps
            record['eps_rule'] = 'rel x |x| (x constant)'
        eps, m = record['eps'], float(min_abs)
        w_dev = math.hypot(m, eps) - m
        record['max_deviation'] = {'abs': w_dev, 'maxmin': 0.5 * w_dev, 'division': None,
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

    def _reaches_zero(self, site):
        """Whether the denominator reaches 0 (is 0 at a sample) or crosses it in some reference run. A
        denominator that only gets small (an inertance over a closing valve's area) is left alone."""
        r = self.ranges.get(site.names['x'])
        if not isinstance(r, dict):
            return False
        min_abs = float(r.get('any_min_abs', r.get('min_abs', 1.0)))
        return bool(r.get('any_sign_change')) or float(r['min']) * float(r['max']) <= 0.0 or min_abs == 0.0

    def _division(self, e):
        num, den = _args(e)
        site = self._new_site('division', den, infix(e))
        if site is None or self.mode == 'probe':
            return e
        new = _apply('times', num, _ci(site.names['x']), _ci(site.names['q']))
        site.record['replacement'] = infix(new)
        return new

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
            zero = any(_tag(v) == 'cn' and float(v.text) == 0.0 for v in (a, b))
            site = self._new_site('maxmin', x, original, rule='rectifier' if zero else 'maxmin')
            if site is None or self.mode == 'probe':
                result = _apply(op, a, b)
                continue
            if site.record['collapsed']:
                result = a if (site.record['sign'] > 0) == (op == 'max') else b
            elif site.form == 'xH':
                # max = b + (a - b) H, min = a - (a - b) H
                xh = _apply('times', _ci(site.names['x']), _ci(site.names['H']))
                result = _apply('plus', b, xh) if op == 'max' else _apply('minus', a, xh)
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
        atoms = {}      # (A, B) -> atom index: the comparison "A over B", sm<k>_x = A - B
        values = []     # per atom: the sm<k>_H element, True / False (collapsed), or None (skipped)
        conditions = [self._condition(p[1], atoms, values) for p in pieces]
        jump = self._jump_candidate(pieces, otherwise[0][0])
        for site in self.sites[first_site - 1:]:
            site.record['piecewise'] = original
            site.record['piecewise_k'] = first_site
            if branch_domain:
                site.record['flags'].append('branch_domain')
            if jump:
                site.record['flags'].append('jump_candidate')
                site.record['jump_reason'] = jump
                site.record.setdefault('switch_variables', self.graph.switch_variables(site.component, site.x))
        if jump:
            self.jump_candidates.append({'component': self.component.get('name'), 'variable': self.variable,
                                         'k': [s.k for s in self.sites[first_site - 1:]], 'reason': jump,
                                         'exempt': self._exempt_reason() is not None,
                                         'switch_variables': _merge_switch(self.sites[first_site - 1:])})
        if self.mode == 'probe' or any(v is None for v in values):
            return e
        fixed = {i: v for i, v in enumerate(values) if isinstance(v, bool)}
        result = self._blend(pieces, conditions, otherwise[0][0], values, fixed)
        for site in self.sites[first_site - 1:]:
            site.record['replacement'] = infix(result)
        return result

    def _blend(self, pieces, conditions, otherwise, values, assigned):
        """The pieces blended so the weights sum to 1: split on one comparison at a time (an atom shared
        by several conditions is split on once), ``g + H*(f - g)`` with f the value where it holds and g
        where it does not, down to where the first piece whose condition holds is known."""
        for piece, condition in zip(pieces, conditions):
            known = _evaluate(condition, assigned)
            if known is False:
                continue
            if known is True:
                return copy.deepcopy(piece[0])
            atom = next(i for i in _atoms_of(condition) if i not in assigned)
            f = self._blend(pieces, conditions, otherwise, values, {**assigned, atom: True})
            g = self._blend(pieces, conditions, otherwise, values, {**assigned, atom: False})
            if infix(f) == infix(g):
                return f
            return _apply('plus', g, _apply('times', copy.deepcopy(values[atom]), _apply('minus', f, copy.deepcopy(g))))
        return copy.deepcopy(otherwise)

    def _condition_supported(self, c):
        op = _op(c)
        if _tag(c) in ('true', 'false'):
            return True
        if op in _COMPARISONS:
            return len(_args(c)) == 2
        if op in ('and', 'or', 'not'):
            return all(self._condition_supported(a) for a in _args(c))
        return False

    def _condition(self, c, atoms, values):
        """A condition as a boolean tree over the piecewise's comparisons: ('atom', i), ('not', t),
        ('and', [t, ...]), ('or', [t, ...]) or ('const', bool). A comparison of the same two operands
        as an earlier one (either way round) is that atom, or its negation: one site, H and 1 - H."""
        if _tag(c) in ('true', 'false'):
            return ('const', _tag(c) == 'true')
        op, args = _op(c), _args(c)
        if op in _COMPARISONS:
            a, b = args if op in ('gt', 'geq') else (args[1], args[0])
            key, flipped = (infix(a), infix(b)), (infix(b), infix(a))
            if key in atoms:
                return ('atom', atoms[key])
            if flipped in atoms:
                return ('not', ('atom', atoms[flipped]))
            atoms[key] = len(values)
            site = self._new_site('piecewise', _apply('minus', copy.deepcopy(a), copy.deepcopy(b)), infix(c),
                                  strict=op in ('gt', 'lt'))
            if site is None or self.mode == 'probe':
                values.append(None if site is None else _ci(site.names['H']))
            elif site.record['collapsed']:
                values.append(site.record['sign'] > 0)
            else:
                values.append(_ci(site.names['H']))
            return ('atom', atoms[key])
        parts = [self._condition(x, atoms, values) for x in args]
        return ('not', parts[0]) if op == 'not' else (op, parts)

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


def _evaluate(tree, assigned):
    """A condition tree's value under ``assigned`` {atom: bool}: True, False, or None (not yet known)."""
    kind = tree[0]
    if kind == 'const':
        return tree[1]
    if kind == 'atom':
        return assigned.get(tree[1])
    if kind == 'not':
        v = _evaluate(tree[1], assigned)
        return None if v is None else not v
    vs = [_evaluate(t, assigned) for t in tree[1]]
    if kind == 'and':
        return False if False in vs else (None if None in vs else True)
    return True if True in vs else (None if None in vs else False)


def _atoms_of(tree):
    """The atoms of a condition tree, in order of appearance."""
    if tree[0] == 'atom':
        return [tree[1]]
    if tree[0] == 'const':
        return []
    if tree[0] == 'not':
        return _atoms_of(tree[1])
    return [i for t in tree[1] for i in _atoms_of(t)]


def _fmt(value):
    value = float(value)
    return str(int(value)) if value == int(value) else repr(value)


def _merge_switch(sites):
    out = []
    for site in sites:
        for v in site.record.get('switch_variables') or []:
            if v not in out:
                out.append(v)
    return out


class _ModelGraph:
    """Variables across connections, each with its defining equation: for switch_variables (the
    states or floor phases a comparison switches on) and for where a variable is used."""

    def __init__(self, root):
        self.parent = {}
        for conn in root.findall(_C + 'connection'):
            c1, c2 = conn.get('component_1'), conn.get('component_2')
            for mv in conn.findall(_C + 'map_variables'):
                a, b = self.find((c1, mv.get('variable_1'))), self.find((c2, mv.get('variable_2')))
                if a != b:
                    self.parent[a] = b
        self.defs, self._uses, self.initialised = {}, {}, set()
        for comp in root.findall(_C + 'component'):
            for v in comp.findall(_C + 'variable'):
                try:
                    float(v.get('initial_value'))
                    self.initialised.add(self.find((comp.get('name'), v.get('name'))))
                except (TypeError, ValueError):
                    pass
        for comp in root.findall(_C + 'component'):
            name = comp.get('name')
            for m in comp.findall(_M + 'math'):
                for eq in m:
                    if _op(eq) != 'eq' or len(_args(eq)) != 2:
                        continue
                    lhs, rhs = _args(eq)
                    rate = _op(lhs) == 'diff'
                    target = _args(lhs)[0] if rate else lhs
                    if _tag(target) != 'ci':
                        continue
                    key = self.find((name, target.text.strip()))
                    self.defs[key] = ('state', name, target.text.strip()) if rate else \
                        self._algebraic(name, target.text.strip(), rhs)
                    self._collect_uses(name, rhs, False)

    def is_constant(self, comp, var):
        """Whether a variable is a constant: no equation defines it and it has a numeric initial value
        (a variable this pass added, the time or an unset input is not)."""
        key = self.find((comp, var))
        return key not in self.defs and key in self.initialised

    def find(self, a):
        while self.parent.get(a, a) != a:
            a = self.parent[a]
        return a

    def _algebraic(self, comp, var, rhs):
        args = _args(rhs)
        if _op(rhs) == 'minus' and len(args) == 2 and _tag(args[0]) == 'ci' and _op(args[1]) == 'floor' \
                and _tag(_args(args[1])[0]) == 'ci' and args[0].text.strip() == _args(args[1])[0].text.strip():
            return ('phase', comp, var, self.find((comp, args[0].text.strip())))
        return ('algebraic', comp, var, {self.find((comp, n.text.strip())) for n in rhs.iter() if _tag(n) == 'ci'})

    def _collect_uses(self, comp, e, trig):
        if _tag(e) == 'ci':
            self._uses.setdefault(self.find((comp, e.text.strip())), set()).add('trig' if trig else 'other')
            return
        trig = trig or _op(e) in ('sin', 'cos')
        for child in e:
            self._collect_uses(comp, child, trig)

    def uses(self, comp, var):
        """How a variable is used on right-hand sides: {'trig'} (only inside cos/sin), 'other', or empty."""
        return self._uses.get(self.find((comp, var)), set())

    def switch_variables(self, comp, x):
        """The states and floor phases x depends on through algebraic equations, as
        [{"variable", "phase", "state"}]: a floor phase (v = s - floor(s)) is {v, true, s}."""
        out, seen = [], set()
        todo = [self.find((comp, n.text.strip())) for n in x.iter() if _tag(n) == 'ci']
        while todo:
            key = todo.pop(0)
            if key in seen:
                continue
            seen.add(key)
            d = self.defs.get(key)
            if d is None:
                continue
            if d[0] == 'state':
                entry = {'variable': d[2], 'phase': False, 'state': d[2]}
            elif d[0] == 'phase':
                state = self.defs.get(d[3])
                entry = {'variable': d[2], 'phase': True, 'state': state[2] if state else d[3][1]}
            else:
                todo.extend(d[3])
                continue
            if entry not in out:
                out.append(entry)
        return out


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


def smooth(model, ranges, rel=None, collapse=None, abs_eps=DEFAULT_ABS_EPS, only=None, exempt=None, maxmin_form='w'):
    """The smoothed model and its report.

    Args:
        model: CellML text or a path: the original flat model (not the probed one).
        ranges: ``{"sm<k>_x": {"min": .., "max": .., "min_abs": ..}}`` over the reference runs
            (``[min, max]`` also accepted). ``min_abs``, the smallest |x| seen, sets the largest
            deviation; without it the deviation is the worst case for the range.
        maxmin_form: ``"w"``, max/min as (a + b +- w)/2, or ``"xH"``, b + (a - b) H.
        rel: ``{"abs", "maxmin", "piecewise", "division"}`` -> eps relative to each argument's range, or one
            float for all. Defaults to 1e-3. ``rel["variables"]``: ``{variable or "component/variable":
            rel}`` for the sites in the equation of that variable, over their kind's.
        collapse: replace a site that never switches in the reference runs by its active branch.
            None or False: off. True: each such site whose |x| stays above zero. A list of site
            numbers k: those sites, also when x reaches zero without changing sign (flagged
            ``one_sided``, e.g. a phase just after a floor wraps).
        abs_eps: eps of a site whose argument is zero throughout, or that has no range.
        only: the site numbers k to smooth; the others are left as written (status ``left``). A
            piecewise is smoothed only when all its comparisons are (``piecewise_k`` groups them).
        exempt: ``[{"variable": "[component/]glob", "reason": ...}]``: the sites in the equations of
            those variables are left as written (status ``exempt``, with the reason and the
            ``switch_variables`` they switch on), for switches meant to stay (a phase gate).

    Returns:
        (CellML text, report dict)
    """
    root = _read(model)
    p = _Pass(root, 'smooth', ranges, rel, collapse, abs_eps, only, exempt, maxmin_form)
    p.run()
    report = {'settings': {'rel': p.rel, 'maxmin_form': maxmin_form, 'collapse': collapse if collapse in (None, True, False)
                           else sorted(collapse), 'abs_eps': abs_eps},
              'counts': _counts(p),
              'sites': [s.record for s in p.sites],
              'not_smoothed': p.not_smoothed, 'exempt': exempt or [],
              'jump_candidates': p.jump_candidates,
              'lifted_estimate': sum(s.record.get('lifted', 0) for s in p.sites
                                     if s.record['status'] == 'smoothed' and not s.record.get('collapsed'))}
    return _write(root), report


def _counts(p):
    counts = {}
    for s in p.sites:
        key = s.kind if s.record['status'] == 'smoothed' else f'{s.kind}_{s.record["status"]}'
        if s.record['status'] in ('left', 'not_needed'):
            continue
        if s.record['status'] == 'exempt':
            counts[f'{s.kind}_exempt'] = counts.get(f'{s.kind}_exempt', 0) + 1
            continue
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


def reference_ranges(probed_path, runs, per_run=False, over='nominal'):
    """Each ``sm<k>_x``'s min, max and smallest |x| in the reference runs of a probed model, from
    t = 0: the run is logged through ``pre_time`` too, where a value such as 0/0 at the initial state
    shows. ``over='nominal'`` (default) takes them from the first run, the nominal point, so one
    outlying run cannot set eps for every point; ``over='union'`` takes them over all the runs.

    ``nan`` records a NaN or Inf in x, which the original operation may have hidden (``max(NaN, 0)``
    is 0) but the smooth form passes on. Each run is a dict with ``sim_time`` and optionally
    ``pre_time``, ``dt``, ``solver_info`` and ``param_vals`` (``{"component/variable": value}``),
    simulated with CVODE_myokit; runs with the same times and solver settings share one compiled
    simulation. With ``per_run``, also returns the ranges of each run, in order.
    """
    from libcuflynx.solver_wrappers import get_simulation_helper
    names = [f'{c}/{v}' for c, v in probe_variables(probed_path)]
    param_names = sorted({n for run in runs for n in (run.get('param_vals') or {})})
    helpers, merged, each = {}, {}, []
    for run in runs:
        dt, total = run.get('dt', 0.01), run.get('pre_time', 0.0) + run['sim_time']
        key = json.dumps([dt, total, run.get('solver_info')], sort_keys=True, default=str)
        if key not in helpers:
            helper = get_simulation_helper(model_path=probed_path, solver='CVODE_myokit', model_type='cellml',
                                           dt=dt, sim_time=total, solver_info=run.get('solver_info'), pre_time=0.0)
            defaults = helper.get_init_param_vals(param_names) if param_names else []
            helpers[key] = (helper, [d[0] if isinstance(d, list) else d for d in defaults])
        helper, defaults = helpers[key]
        if hasattr(helper, 'reset_states'):
            helper.reset_states()
        if param_names:
            values = dict(zip(param_names, defaults))
            values.update(run.get('param_vals') or {})
            helper.set_param_vals(param_names, [values[n] for n in param_names])
        if not helper.run():
            raise RuntimeError(f'reference run {run} failed')
        ranges = {}
        for name, values in zip(names, helper.get_results([[n] for n in names], flatten=True)):
            values = np.asarray(values, dtype=float).ravel()
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                finite = np.zeros(1)
            ranges[name.split('/')[-1]] = {'min': float(finite.min()), 'max': float(finite.max()),
                                           'min_abs': float(np.abs(finite).min()),
                                           'nan': bool(finite.size < values.size)}
        each.append(ranges)
        for key_x, r in ranges.items():
            if key_x in merged:
                m = merged[key_x]
                r = {'min': min(r['min'], m['min']), 'max': max(r['max'], m['max']),
                     'min_abs': min(r['min_abs'], m['min_abs']), 'nan': r['nan'] or m['nan']}
            merged[key_x] = dict(r)
    for helper, _ in helpers.values():
        helper.close_simulation()
    for k, m in merged.items():
        # over all runs, whatever ``over``: where x reaches or crosses 0 (a division's guard)
        m['any_min_abs'] = min(r[k]['min_abs'] for r in each if k in r)
        m['any_sign_change'] = any(r[k]['min'] < 0.0 < r[k]['max'] for r in each if k in r)
    if over == 'nominal':
        # a NaN or Inf in any run is still recorded
        merged = {k: dict(r, nan=merged[k]['nan'], any_min_abs=merged[k]['any_min_abs'],
                          any_sign_change=merged[k]['any_sign_change']) for k, r in each[0].items()}
    elif over != 'union':
        raise ValueError(f"over: 'nominal' or 'union', not {over!r}")
    return (merged, each) if per_run else merged


def nan_at_init(model_path, report, probed=False):
    """``{k: [variable, ...]}``: the variables of each site that are NaN or Inf at the model's initial
    state (t = 0, default parameters). For a probed model (``probed``) that is ``sm<k>_x``; for a
    smoothed one, also ``sm<k>_w``, ``sm<k>_z``, ``sm<k>_H`` and the variable the site's equation
    defines. myokit evaluates in Python, where a division by zero is an error even inside a max, so
    the probed model's own variables are not checked."""
    import warnings
    import myokit.formats
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = myokit.formats.importer('cellml').model(model_path)
    bad = {}
    for site in report['sites']:
        if probed:
            names = [f'sm{site["k"]}_x']
        else:
            names = [n for n in site.get('aux') or [] if not n.startswith('eps_')] + [site['variable']]
        for name in names:
            qname = f'{site["component"]}.{name}'
            if not model.has_variable(qname):
                continue
            try:
                with np.errstate(all='ignore'):
                    value = float(model.get(qname).eval())
            except Exception:      # myokit raises its NumericalError for 0/0, log(-1), ...
                value = math.nan
            if not math.isfinite(value):
                bad.setdefault(site['k'], []).append(name)
    return bad


def flag_nan_at_init(report, probed_path, smooth_path):
    """Flags ``nan_at_init`` on each site with a NaN or Inf at t = 0 in the probed or smoothed model,
    and lists them in ``report["nan_at_init"]``."""
    found = {}
    for which, path in (('probed', probed_path), ('smoothed', smooth_path)):
        if path:
            for k, names in nan_at_init(path, report, probed=which == 'probed').items():
                found.setdefault(k, {})[which] = names
    for site in report['sites']:
        if site['k'] in found and 'nan_at_init' not in site['flags']:
            site['flags'].append('nan_at_init')
    report['nan_at_init'] = [{'k': k, **v} for k, v in sorted(found.items())]
    return report


def smooth_cellml_file(model_path, runs=None, output_dir=None, ranges=None, rel=None, collapse=None,
                       abs_eps=DEFAULT_ABS_EPS, exempt=None, over='nominal', maxmin_form='w'):
    """Probe, run the reference runs (unless ``ranges`` is given), smooth, and write
    ``<name>_smooth.cellml`` and ``smoothing_report.json`` into ``output_dir``, with the ranges from
    the runs in ``smoothing_ranges.json`` (for ``--ranges``).

    Returns the paths of the smoothed model and of the report.
    """
    if (runs is None) == (ranges is None):
        raise ValueError('give either runs or ranges')
    name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = output_dir or os.path.dirname(os.path.abspath(model_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(model_path, encoding='utf-8') as f:
        original = f.read()
    probed_path = None
    if ranges is None:
        probed_path = os.path.join(output_dir, f'{name}_probe.cellml')
        with open(probed_path, 'w', encoding='utf-8') as f:
            f.write(probe(original))
        ranges = reference_ranges(probed_path, runs, over=over)
        with open(os.path.join(output_dir, 'smoothing_ranges.json'), 'w', encoding='utf-8') as f:
            json.dump(ranges, f, indent=2)
    smoothed, report = smooth(original, ranges, rel=rel, collapse=collapse, abs_eps=abs_eps, exempt=exempt,
                              maxmin_form=maxmin_form)
    before, after = libcellml_issues(original), libcellml_issues(smoothed)
    report['model'] = os.path.abspath(model_path)
    report['runs'] = runs
    report['new_issues'] = {k: [i for i in after[k] if i not in before[k]] for k in after}
    smooth_path = os.path.join(output_dir, f'{name}_smooth.cellml')
    report_path = os.path.join(output_dir, 'smoothing_report.json')
    with open(smooth_path, 'w', encoding='utf-8') as f:
        f.write(smoothed)
    flag_nan_at_init(report, probed_path, smooth_path)
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
    parser.add_argument('--exempt', help='JSON [{"variable": "[component/]glob", "reason": ...}]: sites left as written')
    parser.add_argument('--ranges-over', choices=('nominal', 'union'), default='nominal',
                        help='each range from the first run (nominal, default) or over all the runs')
    parser.add_argument('--maxmin-form', choices=('w', 'xH'), default='w',
                        help='max/min as (a + b +- w)/2 (w, default) or b + (a - b) H (xH)')
    parser.add_argument('--collapse', action='store_true',
                        help='replace sites that never switch in the runs by their active branch')
    args = parser.parse_args(argv)

    def load(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    smooth_path, report_path = smooth_cellml_file(
        args.model, runs=load(args.runs) if args.runs else None,
        ranges=load(args.ranges) if args.ranges else None, output_dir=args.output_dir,
        rel=args.rel, collapse=True if args.collapse else None, exempt=load(args.exempt) if args.exempt else None,
        over=args.ranges_over, maxmin_form=args.maxmin_form)
    print(smooth_path)
    print(report_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
