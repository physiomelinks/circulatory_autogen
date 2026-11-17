"""The calibrated model: a model's CellML with a calibration's best fit written in.

A finished calibration writes ``<prefix>_calibrated.cellml`` at the top of the study's working
dir (its ``param_id_output_dir``), where the study manifest finds it as the study's ``model``
(``cuflynx_study._calibrated_model``). ``CVS0DParamID.write_calibrated_model`` writes it for a
calibration engine; :func:`write_calibrated_model` takes the parts, for a caller holding them
without one. :func:`write_cellml_values` writes values named ``vessel/param`` or
``component/variable`` into any CellML file, and :func:`substitute_parameter_values` into CellML
text; each reports where every value went.
"""
import os
import re
import warnings

import numpy as np

from libcuflynx.parsers.PrimitiveParsers import (expand_modifier_param_vals,
                                                 model_qname_candidates)
from libcuflynx.solver_wrappers.param_grouping import pair_names_with_values

__all__ = ['CALIBRATED_MODEL_SUFFIX', 'PARAMETER_COMPONENTS', 'calibrated_model_name',
           'constant_named',
           'calibrated_parameter_values', 'flattened_cellml_text', 'substitute_parameter_values',
           'write_calibrated_model', 'write_cellml_values']

CALIBRATED_MODEL_SUFFIX = '_calibrated.cellml'

#: the components of a generated model that hold its parameters
PARAMETER_COMPONENTS = ('parameters', 'parameters_global')

_COMPONENT_OPEN = re.compile(r'<component\b[^>]*>')
_COMPONENT_CLOSE = re.compile(r'</component\s*>')
_VARIABLE_TAG = re.compile(r'<variable\b[^>]*?/?>')
_NAME = re.compile(r'\bname\s*=\s*"([^"]*)"')
_INITIAL_VALUE = re.compile(r'(\binitial_value\s*=\s*")([^"]*)(")')
_IMPORT = re.compile(r'<import\b')


def calibrated_model_name(file_prefix):
    """The calibrated model's file name: ``<file_prefix>_calibrated.cellml``."""
    return f'{file_prefix}{CALIBRATED_MODEL_SUFFIX}'


def calibrated_parameter_values(param_id_info, best_param_vals):
    """``(names, values)``: each model parameter (``component/variable``) the best fit sets.

    A row naming several vessels gives each of them its one value, and a modifier row gives
    each target the value its modifier function makes of theta (``expand_modifier_param_vals``,
    which needs the row's baselines resolved, as a built calibration engine has them).
    """
    values = np.asarray(best_param_vals, dtype=float).ravel().tolist()
    expanded = expand_modifier_param_vals(param_id_info, values)
    names, out = [], []
    for row_names, value in zip(param_id_info['param_names'], expanded):
        for name, member_value in pair_names_with_values(row_names, value):
            names.append(str(name))
            out.append(float(member_value))
    return names, out


def _component_blocks(text):
    """``(name, start, end)`` of each component defined in ``text``: those with a body, not the
    empty ``<component .../>`` of an import."""
    blocks, pos = [], 0
    while True:
        match = _COMPONENT_OPEN.search(text, pos)
        if match is None:
            return blocks
        pos = match.end()
        if match.group(0).endswith('/>'):
            continue
        close = _COMPONENT_CLOSE.search(text, pos)
        if close is None:
            return blocks
        name = _NAME.search(match.group(0))
        if name:
            blocks.append((name.group(1), match.start(), close.end()))
        pos = close.end()


def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def _constants(text):
    """The ``component/variable`` names of the variables in ``text`` with a numeric initial
    value."""
    found = set()
    for component, start, end in _component_blocks(text):
        for tag in _VARIABLE_TAG.findall(text, start, end):
            name, value = _NAME.search(tag), _INITIAL_VALUE.search(tag)
            if name and value and _is_number(value.group(2)):
                found.add(f'{component}/{name.group(1)}')
    return found


def _resolve(name, constants, by_variable):
    """The constant of ``constants`` that ``name`` is: the first of ``model_qname_candidates``
    the text has. A bare candidate (the generated name) counts only when one constant has that
    variable name, or one in ``PARAMETER_COMPONENTS`` does."""
    for candidate in model_qname_candidates(name):
        if '/' in candidate:
            if candidate in constants:
                return candidate
            continue
        hits = by_variable.get(candidate, [])
        if len(hits) > 1:
            hits = [h for h in hits if h.split('/', 1)[0] in PARAMETER_COMPONENTS]
        if len(hits) == 1:
            return hits[0]
    return None


def constant_named(name, constants):
    """The name in ``constants`` (``component/variable`` names) of the constant ``name`` is, by
    the rule ``substitute_parameter_values`` writes values with: itself, or the first of
    ``model_qname_candidates`` the model has (``global/q_init`` is ``parameters_global/q_init``
    in a model that has that component). None when it is none of them."""
    by_variable = {}
    for qname in sorted(constants):
        by_variable.setdefault(qname.split('/', 1)[1], []).append(qname)
    return _resolve(str(name), constants, by_variable)


def substitute_parameter_values(cellml_text, values):
    """``(new_text, report)``: ``cellml_text`` with each value of ``values`` as the initial value
    of the constant it names. Nothing else in the text changes.

    ``values`` maps a name to a number. A name is ``vessel/param`` (``aortic_root/C``,
    ``global/q_init``), resolved by ``model_qname_candidates`` (``parameters/C_aortic_root`` in
    a generated model), or ``component/variable``. Only a variable with a numeric initial value
    in a component defined in this text (not one it imports) is written.

    ``report`` is ``{'updated': [names written], 'unresolved': [names not found],
    'resolved': {name: 'component/variable' written}}``, names in the order given. ``resolved``
    tells a tool which component each value went into: PhLynx reads back the values in
    ``parameters`` and ``parameters_global`` only.
    """
    constants = _constants(cellml_text)
    by_variable = {}
    for qname in sorted(constants):
        by_variable.setdefault(qname.split('/', 1)[1], []).append(qname)
    wanted, report = {}, {'updated': [], 'unresolved': [], 'resolved': {}}
    for name, value in values.items():
        qname = _resolve(str(name), constants, by_variable)
        if qname is None:
            report['unresolved'].append(name)
            continue
        component, variable = qname.split('/', 1)
        wanted.setdefault(component, {})[variable] = repr(float(value))
        report['updated'].append(name)
        report['resolved'][name] = qname
    if not wanted:
        return cellml_text, report

    def set_value(tag, component):
        name = _NAME.search(tag)
        if not name or name.group(1) not in wanted[component]:
            return tag
        value = wanted[component][name.group(1)]
        return _INITIAL_VALUE.sub(lambda m: m.group(1) + value + m.group(3), tag, count=1)

    pieces, pos = [], 0
    for component, start, end in _component_blocks(cellml_text):
        if component not in wanted:
            continue
        pieces += [cellml_text[pos:start], _VARIABLE_TAG.sub(
            lambda m, c=component: set_value(m.group(0), c), cellml_text[start:end])]
        pos = end
    pieces.append(cellml_text[pos:])
    return ''.join(pieces), report


def flattened_cellml_text(cellml_path):
    """The CellML model at ``cellml_path`` with its imports resolved and flattened, as text
    (libCellML's printing of it)."""
    import libcellml
    import libcuflynx.utilities.libcellml_helper_funcs as cellml
    model = cellml.parse_model(cellml_path, False)
    if model.hasImports():
        importer = cellml.resolve_imports(model, os.path.dirname(os.path.abspath(cellml_path)),
                                          False)
        model = cellml.flatten_model(model, importer)
    return libcellml.Printer().printModel(model)


def write_cellml_values(cellml_path, values, output_path, *, flatten=None):
    """Write ``cellml_path`` with ``values`` (``{name: value}``, named as for
    :func:`substitute_parameter_values`) to ``output_path``.

    ``flatten``: True writes the model flattened, its imports resolved (libCellML prints the
    whole file anew); False writes the file itself with only the initial values changed, so a
    constant of a file it imports is unresolved; None (the default) flattens a file that has
    imports and no other. A generated model imports its parameters
    (``<prefix>_parameters.cellml``) and modules, so its calibrated model is one flattened file
    that runs by itself; a single-file model (one a GUI loads, or one PhLynx sends) keeps its
    form, with only its values changed.

    Returns the report of :func:`substitute_parameter_values` with ``'path': output_path`` and
    ``'flattened'``: whether the file written was flattened.
    """
    with open(cellml_path, encoding='utf-8') as f:
        text = f.read()
    flattened = bool(_IMPORT.search(text)) if flatten is None else bool(flatten)
    if flattened:
        text = flattened_cellml_text(cellml_path)
    new_text, report = substitute_parameter_values(text, values)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    report.update(path=output_path, flattened=flattened)
    return report


def write_calibrated_model(cellml_path, param_id_info, best_param_vals, output_path, *,
                           flatten=None, return_report=False):
    """Write ``cellml_path`` with ``best_param_vals`` set to ``output_path``.

    Each model parameter a row sets (:func:`calibrated_parameter_values`) is written by
    :func:`write_cellml_values`, which ``flatten`` is passed to: by default a model with
    imports, as a generated one has, is flattened, and a single-file model keeps its form. A
    parameter with no constant in the model is named in a warning.

    Returns ``output_path``, or with ``return_report`` the report of
    :func:`write_cellml_values`, keyed by model parameter (``component/variable``).
    """
    names, values = calibrated_parameter_values(param_id_info, best_param_vals)
    report = write_cellml_values(cellml_path, dict(zip(names, values)), output_path,
                                 flatten=flatten)
    if report['unresolved']:
        warnings.warn(f'{output_path}: {cellml_path} has no constant named '
                      f'{report["unresolved"]}, so their values are not written.', UserWarning,
                      stacklevel=2)
    return report if return_report else output_path
