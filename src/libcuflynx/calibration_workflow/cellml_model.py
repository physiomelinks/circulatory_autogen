'''
A CellML file as a workflow model (``{"cellml_model_path": ..., "parameters": ...}``): its
parameters are its constants, named ``component/variable`` in the flattened model, and a step
runs a flattened copy with its values written in.
'''

import os
import re

from libcuflynx.calibration_workflow.spec import WorkflowError

#: an import's href in a CellML file
_HREF = re.compile(r'''xlink:href\s*=\s*["']([^"']+)["']''')


def _flat_model(path, where):
    import libcellml
    if not os.path.isfile(path):
        raise WorkflowError(f'{where}: its CellML model file {path} does not exist.')
    with open(path, encoding='utf-8') as f:
        model = libcellml.Parser(False).parseModel(f.read())
    if model is None:
        raise WorkflowError(f'{where}: {path} is not a CellML model.')
    importer = libcellml.Importer(False)
    importer.resolveImports(model, os.path.dirname(os.path.abspath(path)))
    if model.hasUnresolvedImports():
        raise WorkflowError(f'{where}: {path} has imports that cannot be resolved; keep the files '
                            f'it imports where it names them.')
    return importer.flattenModel(model) if model.hasImports() else model


def read_cellml_constants(path, where=''):
    '''``{'component/variable': {'value': float, 'units': str}}`` of the constants of the CellML
    model at ``path`` (its imports resolved and flattened), in model order: the variables the
    libCellML analyser finds constant, with a numeric initial value.'''
    import libcellml
    flat = _flat_model(path, where)
    analyser = libcellml.Analyser()
    analyser.analyseModel(flat)
    analysed = analyser.analyserModel()
    if not analysed.isValid():
        issues = '; '.join(analyser.issue(i).description()
                           for i in range(min(analyser.issueCount(), 5)))
        raise WorkflowError(f'{where}: the CellML model {path} cannot be analysed: {issues}')
    constants = {}
    for variable in analysed.constants():
        var = variable.variable()
        try:
            value = float(var.initialValue())
        except ValueError:
            continue
        component = var.parent().name()
        units = var.units().name() if var.units() is not None else ''
        constants[f'{component}/{var.name()}'] = {'value': value, 'units': units}
    return constants


def write_cellml_with_values(source, values, output_path, where=''):
    '''Write ``source`` flattened, with each ``{'component/variable': value}`` of ``values`` as
    its initial value, to ``output_path`` (``utility_funcs.change_parameter_values_and_save``),
    and check each value is in the file written. Returns ``output_path``.'''
    from libcuflynx.utilities.utility_funcs import change_parameter_values_and_save
    names = list(values)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    change_parameter_values_and_save(source, names, [float(values[n]) for n in names],
                                     output_path)
    written = read_cellml_constants(output_path, where)
    wrong = [n for n in names if n not in written or
             abs(written[n]['value'] - float(values[n])) > 1e-12 * max(1.0, abs(float(values[n])))]
    if wrong:
        raise WorkflowError(f'{where}: writing the values of {wrong} into a copy of {source} '
                            f'failed; they are not constants of the model it writes.')
    return output_path


def cellml_imports(path):
    '''The files the CellML file at ``path`` imports, and those they import, as real paths.'''
    found, todo = [], [os.path.realpath(path)]
    seen = set(todo)
    while todo:
        current = todo.pop()
        with open(current, encoding='utf-8') as f:
            text = f.read()
        for href in _HREF.findall(text):
            if '://' in href:
                continue
            imported = os.path.realpath(os.path.join(os.path.dirname(current), href))
            if imported not in seen and os.path.isfile(imported):
                seen.add(imported)
                found.append(imported)
                todo.append(imported)
    return found
