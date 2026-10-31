'''
The ``cellml`` method of a derivation: a CellML 2.0 derivation model evaluated once with
libCellML.

The model is parsed, its imports resolved and flattened, validated and analysed; any issue is an
error, so a unit inconsistency in an equation (which the analyser reports) refuses the model.
Only an algebraic model whose outputs are computed constants is accepted: an implicit equation
needs the ``python`` method. libCellML's Python code is generated, the input constants are set
(converted from their source's units to the model variable's), the computed constants are
computed once, and the outputs are read and converted to the output parameters' units.
'''

import os

import libcellml as lc

from libcuflynx.utilities.libcellml_helper_funcs import (generate_implementation_code,
                                                         get_analysed_model)


class DerivationModelError(ValueError):
    '''A derivation model that cannot be evaluated: it does not parse, validate or analyse, its
    units do not match its inputs or outputs, or it does not name a symbol.'''


def _issues(logger):
    return [logger.issue(i).description() for i in range(logger.issueCount())]


def _refuse(path, step, issues):
    raise DerivationModelError(f'{path}: {step} found {len(issues)} issue(s):\n'
                               + '\n'.join(f'  - {d}' for d in issues))


def load_derivation_model(path):
    '''The flattened, validated and analysed CellML model at ``path``:
    ``(model, analysed_model)``. DerivationModelError on any parser, importer, validator or
    analyser issue, or for a model that is not algebraic.'''
    with open(path, encoding='utf-8') as f:
        text = f.read()
    parser = lc.Parser(False)
    model = parser.parseModel(text)
    if parser.errorCount():
        _refuse(path, 'parsing', _issues(parser))
    if model.hasUnresolvedImports():
        importer = lc.Importer(False)
        importer.resolveImports(model, os.path.dirname(os.path.abspath(path)))
        if importer.errorCount() or model.hasUnresolvedImports():
            _refuse(path, 'resolving imports', _issues(importer) or ['unresolved imports'])
        model = importer.flattenModel(model)
    validator = lc.Validator()
    validator.validateModel(model)
    if validator.issueCount():
        _refuse(path, 'validation', _issues(validator))
    analyser = lc.Analyser()
    analyser.analyseModel(model)
    # warnings too: the analyser reports units that do not match in an equation as a warning
    if analyser.issueCount():
        _refuse(path, 'analysis', _issues(analyser))
    analysed = get_analysed_model(analyser)
    if analysed.type() != lc.AnalyserModel.Type.ALGEBRAIC:
        raise DerivationModelError(
            f'{path}: a derivation model is algebraic, its outputs computed from constants; '
            f'libCellML analyses this one as {lc.AnalyserModel.typeAsString(analysed.type())}. '
            f'An implicit equation needs the python method.')
    return model, analysed


def _units(model, name):
    '''libCellML Units for ``name``: the model's, or a standard unit.'''
    units = model.units(name)
    if units is None:
        units = lc.Units(f'derivation_{name}')
        units.addUnit(name)
    return units


def units_factor(model, to_units, from_units):
    '''The factor converting a value in ``from_units`` to ``to_units`` (units the model defines,
    or standard units). DerivationModelError when they are not of one dimension.'''
    if to_units == from_units:
        return 1.0
    a, b = _units(model, to_units), _units(model, from_units)
    if not lc.Units.compatible(a, b):
        raise DerivationModelError(
            f"'{from_units}' and '{to_units}' are not units of one dimension (or the "
            f"derivation model does not define one of them).")
    return lc.Units.scalingFactor(a, b)


def _index(info, symbol):
    found = [i for i, entry in enumerate(info) if entry['name'] == symbol]
    if len(found) > 1:
        components = sorted(info[i]['component'] for i in found)
        raise DerivationModelError(f"'{symbol}' is a variable of several components "
                                   f"({', '.join(components)}); name each symbol once.")
    return found[0] if found else None


def evaluate_cellml(path, inputs, output_units):
    '''Evaluate the derivation model at ``path``.

    ``inputs``: ``{symbol: (value, units)}``, each a constant of the model; the value is
    converted to the variable's units. ``output_units``: ``{symbol: units}`` of the outputs,
    each a computed constant of the model; ``None`` takes the variable's units.
    Returns ``{symbol: value}`` in ``output_units``.
    '''
    model, analysed = load_derivation_model(path)
    profile = lc.GeneratorProfile(lc.GeneratorProfile.Profile.PYTHON)
    code = generate_implementation_code(lc.Generator(), analysed, profile)
    namespace = {}
    exec(compile(code, f'{path} (generated)', 'exec'), namespace)
    constant_info = namespace['CONSTANT_INFO']
    computed_info = namespace['COMPUTED_CONSTANT_INFO']
    constants = namespace['create_constants_array']()
    computed = namespace['create_computed_constants_array']()
    algebraic = namespace['create_algebraic_variables_array']()
    namespace['initialise_arrays'](constants, computed, algebraic)

    for symbol, (value, units) in inputs.items():
        i = _index(constant_info, symbol)
        if i is None:
            where = ('a computed constant' if _index(computed_info, symbol) is not None
                     else 'not a variable')
            raise DerivationModelError(f"{path}: input '{symbol}' is {where} of the model; an "
                                       f"input is a constant (with an initial value).")
        to_units = constant_info[i]['units']
        try:
            factor = units_factor(model, to_units, units)
        except DerivationModelError as e:
            raise DerivationModelError(f"{path}: input '{symbol}' is in '{units}' and the "
                                       f"model's '{symbol}' in '{to_units}': {e}") from None
        constants[i] = float(value) * factor

    namespace['compute_computed_constants'](constants, computed, algebraic)

    results = {}
    for symbol, units in output_units.items():
        i = _index(computed_info, symbol)
        if i is None:
            where = ('a constant' if _index(constant_info, symbol) is not None
                     else 'not a variable')
            raise DerivationModelError(f"{path}: output '{symbol}' is {where} of the model; an "
                                       f"output is computed from the inputs.")
        from_units = computed_info[i]['units']
        try:
            factor = 1.0 if units is None else units_factor(model, units, from_units)
        except DerivationModelError as e:
            raise DerivationModelError(f"{path}: output '{symbol}' is in '{from_units}' and its "
                                       f"parameter in '{units}': {e}") from None
        results[symbol] = float(computed[i]) * factor
    return results
