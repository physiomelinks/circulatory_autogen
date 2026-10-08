'''
A model's equations as LaTeX, one section per module, in the symbols of a variable mapping.

Each module (a component of the generated model that defines variables) gets:

* its ODEs (``\\frac{dx}{dt} = ...``) and algebraic equations, in an ``align*``;
* a table of the parameters its equations use: symbol, value, units, and the name in the
  model.

Expressions are written by Myokit's ``LatexExpressionWriter``; numbers are written without
units, to four significant figures.
'''

import re

import myokit
from myokit.formats.latex import LatexExpressionWriter

from libcuflynx.reporting.latex_names import escape
from libcuflynx.reporting.variable_mapping import (PARAMETER_COMPONENTS, canonical_name,
                                                   default_latex, is_time, load_model,
                                                   module_components, module_display_names,
                                                   strip_module, variable_kind,
                                                   variable_mapping)


def format_number(value, digits=4):
    '''``value`` to ``digits`` significant figures, as LaTeX (``1.2\\times10^{-3}``).'''
    value = float(value)
    if value == 0:
        return '0'
    text = f'{value:.{digits}g}'
    if 'e' in text:
        mantissa, exponent = text.split('e')
        return f'{mantissa}\\times10^{{{int(exponent)}}}'
    return text


class MappedLatexWriter(LatexExpressionWriter):
    '''Myokit's LaTeX writer with names from a ``{variable_name: latex}`` mapping and
    numbers without units.'''

    def __init__(self, symbols, digits=4):
        super().__init__()
        self._symbols = symbols
        self._digits = digits

        def lhs(expression):
            variable = expression.var()
            name = self.symbol(variable)
            if isinstance(expression, myokit.Derivative):
                return r'\frac{d' + _group(name) + r'}{dt}'
            if isinstance(expression, myokit.InitialValue):
                return name + r'(t=0)'
            return name

        self.set_lhs_function(lhs)

    def symbol(self, variable):
        if is_time(variable):
            return self._symbols.get(canonical_name(variable)) or 't'
        return self._symbols.get(canonical_name(variable)) or default_latex(variable)

    def _ex_number(self, e, b):
        b.append(format_number(e.eval(), self._digits))

    # Myokit writes command operators flush against the next operand (``\cdotv``), which is
    # an undefined control sequence as soon as that operand starts with a letter.
    def _ex_infix(self, e, b, op):
        super()._ex_infix(e, b, _pad(op))

    def _ex_infix_condition(self, e, b, op):
        super()._ex_infix_condition(e, b, _pad(op))

    # Myokit leaves conditionals as a "piecewise(...)" function call; write them as cases.
    def _ex_piecewise(self, e, b):
        conditions, pieces = list(e.conditions()), list(e.pieces())
        rows = [self.ex(p) + r' & \text{if } ' + self.ex(c) for c, p in zip(conditions, pieces)]
        rows.append(self.ex(pieces[-1]) + r' & \text{otherwise}')
        b.append(r'\begin{cases}' + r' \\ '.join(rows) + r'\end{cases}')

    def _ex_if(self, e, b):
        b.append(r'\begin{cases}' + self.ex(e.value(True)) + r' & \text{if } '
                 + self.ex(e.condition()) + r' \\ ' + self.ex(e.value(False))
                 + r' & \text{otherwise}\end{cases}')


def _pad(op):
    return op + ' ' if op.startswith('\\') and not op.endswith(' ') else op


_UNIT_TOKEN = re.compile(r'([A-Za-z]+)(?:\^(-?[0-9.]+))?')


def format_units(text):
    '''Myokit's unit string (``m^4*s^2/g (0.001)``) as LaTeX math
    (``\mathrm{m}^{4}\,\mathrm{s}^{2}/\mathrm{g}\ (\times 0.001)``); plain text when it
    does not parse.'''
    text = (text or '').strip()
    if not text or text == 'dimensionless':
        return text
    scale = ''
    match = re.fullmatch(r'(.*?)\s*\(([^)]*)\)', text)
    if match:
        text, scale = match.group(1).strip(), match.group(2).strip()
    parts = re.split(r'([*/])', text)
    out = []
    for part in parts:
        if part == '*':
            out.append(r'\,')
        elif part == '/':
            out.append('/')
        else:
            token = _UNIT_TOKEN.fullmatch(part.strip())
            if not token:
                return escape(text + (f' ({scale})' if scale else ''))
            unit = r'\mathrm{' + token.group(1) + '}'
            out.append(unit + (f'^{{{token.group(2)}}}' if token.group(2) else ''))
    if scale:
        try:
            out.append(r'\ (\times ' + format_number(float(scale)) + ')')
        except ValueError:
            return escape(text + f' ({scale})')
    return '$' + ''.join(out) + '$'


def _group(latex):
    '''``latex`` wrapped in braces when it has a subscript, so ``d`` applies to all of it.'''
    return '{' + latex + '}' if ('_' in latex or '^' in latex) else latex


def _parameters_used(variables):
    used = {}
    for variable in variables:
        for ref in variable.rhs().references() if variable.rhs() is not None else ():
            target = ref.var()
            if target.parent().name() in PARAMETER_COMPONENTS:
                used[canonical_name(target)] = target
    return used


def _module_heading(component, display, module_info):
    stripped = strip_module(component.name())
    info = (module_info or {}).get(stripped) or {}
    shown = display.get(stripped, stripped) or stripped
    title = info.get('title') or f'Module {escape(shown)}'
    detail = []
    if info.get('module_type'):
        detail.append(r'\texttt{' + escape(info['module_type']) + '}')
    if info.get('version'):
        detail.append('version ' + r'\texttt{' + escape(info['version']) + '}')
    if info.get('instance'):
        detail.append('instance ' + r'\texttt{' + escape(info['instance']) + '}')
    return title, detail


def module_sections(model, mapping=None, module_info=None, section='subsection', digits=4,
                    parameter_tables=True):
    '''LaTeX for every module of ``model`` (a path or Myokit model).

    ``mapping`` is a path to a variable mapping CSV, a ``{variable_name: row}`` dict, or None
    for the default symbols. ``module_info`` optionally maps a module name (the component
    without ``_module``) to ``{'title', 'module_type', 'version', 'instance'}`` for the
    headings.
    '''
    if not hasattr(model, 'components'):
        model = load_model(model)
    rows = variable_mapping(model, mapping)
    symbols = {r['variable_name']: r['latex'] for r in rows}
    writer = MappedLatexWriter(symbols, digits=digits)
    display = module_display_names(model)
    out = []
    for component in module_components(model):
        variables = [v for v in component.variables(deep=True)
                     if v.rhs() is not None and not is_time(v)]
        states = [v for v in variables if v.is_state()]
        others = [v for v in variables if not v.is_state()]
        if not variables:
            continue
        title, detail = _module_heading(component, display, module_info)
        out.append(f'\\{section}{{{title}}}')
        out.append(r'\label{sec:module:' + escape(strip_module(component.name())).replace('\\', '')
                   + '}')
        if detail:
            out.append(', '.join(detail) + '.')
        lines = []
        for variable in states:
            lines.append(writer.ex(myokit.Derivative(myokit.Name(variable))) + ' &= '
                         + writer.ex(variable.rhs()))
        for variable in others:
            if variable_kind(variable) == 'constant':
                continue      # literal constants go in the table below
            lines.append(writer.symbol(variable) + ' &= ' + writer.ex(variable.rhs()))
        if lines:
            out.append(r'\begin{align*}')
            out.append(' \\\\\n'.join(lines))
            out.append(r'\end{align*}')
        if states:
            # one formula per value, so a long list can break between them
            initial = ', '.join(f'${writer.symbol(v)}(0) = {format_number(_initial(v), digits)}$'
                                for v in states if _initial(v) is not None)
            if initial:
                out.append('with initial values ' + initial + '.')
        if parameter_tables:
            constants = {canonical_name(v): v for v in others
                         if variable_kind(v) == 'constant'}
            constants.update(_parameters_used(variables))
            if constants:
                out.append(_parameter_table(constants, writer, rows, digits))
        out.append('')
    return '\n'.join(out)


def _initial(variable):
    try:
        return variable.initial_value(True)
    except Exception:
        return None


def _parameter_table(variables, writer, rows, digits):
    units = {r['variable_name']: r.get('units', '') for r in rows}
    lines = [r'\begin{center}', r'\small', r'\begin{tabular}{llll}', r'\toprule',
             r'Symbol & Value & Units & Name in the model \\', r'\midrule']
    for name, variable in sorted(variables.items(), key=lambda kv: kv[0]):
        try:
            value = format_number(variable.eval(), digits)
        except Exception:
            value = '--'
        lines.append(f'${writer.symbol(variable)}$ & ${value}$ & '
                     f'{format_units(units.get(name, ""))} & \\texttt{{{escape(name)}}} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    return '\n'.join(lines)
