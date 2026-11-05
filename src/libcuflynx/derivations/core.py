'''
Loading, ordering, evaluating and checking parameter derivations. See the package docstring.
'''

import json
import math
import os
from dataclasses import dataclass, field

from libcuflynx.schemas import DERIVATIONS_SCHEMA, OBS_DATA_SCHEMA
from libcuflynx.schemas.validation import validate
from libcuflynx.utilities import parameters_file as pf
from libcuflynx.utilities.obs_data_helpers import parse_input_items

DERIVATIONS_SUFFIX = '_derivations.json'
PARAMETERS_SUFFIX = '_parameters.json'
#: the relative tolerance within which a parameter's cached value matches its derivation
CACHE_RTOL = 1e-9
#: where an input's value comes from: one of these keys
INPUT_KINDS = ('parameter', 'input_item', 'data_item')
METHODS = ('cellml', 'python')


class DerivationError(ValueError):
    '''A derivations file, or a derivation's inputs or outputs, that cannot be used.'''


def derivations_file_for(parameters_path):
    '''``<name>_derivations.json`` beside ``<name>_parameters.json``.'''
    directory, base = os.path.split(str(parameters_path))
    stem = base[:-len(PARAMETERS_SUFFIX)] if base.endswith(PARAMETERS_SUFFIX) \
        else os.path.splitext(base)[0]
    return os.path.join(directory, stem + DERIVATIONS_SUFFIX)


def parameters_file_for(derivations_path):
    '''``<name>_parameters.json`` beside ``<name>_derivations.json``.'''
    directory, base = os.path.split(str(derivations_path))
    stem = base[:-len(DERIVATIONS_SUFFIX)] if base.endswith(DERIVATIONS_SUFFIX) \
        else os.path.splitext(base)[0]
    return os.path.join(directory, stem + PARAMETERS_SUFFIX)


@dataclass
class Derivation:
    '''One derivation of a derivations file.'''
    name: str
    inputs: dict            # symbol -> (kind, reference): kind in INPUT_KINDS
    outputs: dict           # symbol -> parameter path as written
    method: str             # 'cellml' or 'python'
    spec: str               # the cellml path or python 'file.py:function', as written
    base_dir: str           # the derivations file's directory
    comment: str = ''

    def files(self):
        '''The files the method reads, absolute: the CellML model, or the Python file (none for
        a shipped function).'''
        if self.method == 'cellml':
            return [os.path.normpath(os.path.join(self.base_dir, self.spec))]
        from libcuflynx.derivations.python_method import split_python_spec
        file, _ = split_python_spec(self.spec, self.base_dir)
        return [file] if file else []


def load_derivations(source, base_dir=None):
    '''The derivations of ``source`` (a derivations file path, or its JSON object with
    ``base_dir`` the directory its paths are relative to), as a list of :class:`Derivation`.
    Checked against the schema; a name given twice is a DerivationError.'''
    if isinstance(source, dict):
        doc, where = source, 'derivations'
        base_dir = os.path.abspath(base_dir or os.getcwd())
    else:
        where = str(source)
        with open(source, encoding='utf-8-sig') as f:
            doc = json.load(f)
        base_dir = os.path.dirname(os.path.abspath(source))
    validate(DERIVATIONS_SCHEMA, doc, source=where, what='derivations file')
    derivations, names = [], set()
    for entry in doc['derivations']:
        if entry['name'] in names:
            raise DerivationError(f"{where}: derivation '{entry['name']}' is given twice.")
        names.add(entry['name'])
        method = next(m for m in METHODS if m in entry)
        inputs = {symbol: next(iter(ref.items())) for symbol, ref in entry['inputs'].items()}
        derivations.append(Derivation(entry['name'], inputs, dict(entry['outputs']), method,
                                      entry[method], base_dir, entry.get('comment', '')))
    return derivations


def read_parameters(path):
    '''A parameters file as its JSON object, numbers kept as text (``read_parameters_file``,
    accepting a model's file or a parameterisation).'''
    return pf.read_parameters_file(path, kind=pf.MODULE)


def parameter_path(doc, name):
    '''The path in ``doc`` of the parameter written ``name``: as ``name_path`` reads it, and a
    bare name not at the top of ``doc`` is a global.'''
    path = pf.name_path(name, pf.MODULE)
    if len(path) == 1 and pf.get_leaf(doc, path) is None:
        path = [pf.GLOBAL, path[0]]
    return tuple(path)


def _parameter_leaf(doc, name, what):
    leaf = pf.get_leaf(doc, list(parameter_path(doc, name)))
    if leaf is None:
        raise DerivationError(f"{what}: the parameters file has no parameter '{name}'; add it, "
                              f"with its units.")
    return leaf


def _number(value, what):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise DerivationError(f'{what} is {value!r}, not a number.') from None
    if not math.isfinite(number):
        raise DerivationError(f'{what} is {number}.')
    return number


def order_derivations(derivations, doc):
    '''``derivations`` in an order that evaluates each after those whose outputs it reads as
    parameter inputs. DerivationError for a parameter two derivations set, or a loop.'''
    setter = {}
    for d in derivations:
        for symbol, name in d.outputs.items():
            path = parameter_path(doc, name)
            if path in setter:
                raise DerivationError(f"parameter '{name}' is set by derivations "
                                      f"'{setter[path].name}' and '{d.name}'; set it in one.")
            setter[path] = d
    needs = {d.name: [setter[parameter_path(doc, ref)] for kind, ref in d.inputs.values()
                      if kind == 'parameter' and parameter_path(doc, ref) in setter]
             for d in derivations}
    ordered, state = [], {}

    def visit(d, chain):
        if state.get(d.name) == 'done':
            return
        if state.get(d.name) == 'visiting':
            loop = chain[chain.index(d.name):] + [d.name]
            raise DerivationError(f"derivations loop: {' -> '.join(loop)} (each reads a "
                                  f"parameter the next sets).")
        state[d.name] = 'visiting'
        for before in needs[d.name]:
            visit(before, chain + [d.name])
        state[d.name] = 'done'
        ordered.append(d)

    for d in derivations:
        visit(d, [])
    return ordered


def load_obs_items(obs_data):
    '''``{'input_item': {name: item}, 'data_item': {name: item}}`` of one obs_data or several
    (paths or JSON objects), each checked against the obs_data schema. A name in two files is a
    DerivationError.'''
    if obs_data is None:
        obs_data = []
    if isinstance(obs_data, (str, os.PathLike, dict)):
        obs_data = [obs_data]
    items = {'input_item': {}, 'data_item': {}}
    for source in obs_data:
        if isinstance(source, dict):
            doc, where = source, 'obs_data'
        else:
            where = str(source)
            with open(source, encoding='utf-8-sig') as f:
                doc = json.load(f)
        validate(OBS_DATA_SCHEMA, doc, source=where, what='obs_data file')
        found = {'input_item': parse_input_items(doc),
                 'data_item': {str(i['data_item_name']): i for i in doc.get('data_items') or []}}
        for kind, named in found.items():
            for name, item in named.items():
                if name in items[kind]:
                    raise DerivationError(f"{kind} '{name}' is in two obs_data files.")
                items[kind][name] = dict(item, obs_data=where)
    return items


def resolve_inputs(derivation, doc, obs_items, derived=None):
    '''``{symbol: {'kind', 'reference', 'value', 'units', 'std', 'source', 'data_reference'}}``
    of ``derivation``'s inputs. A parameter's value is ``derived[path]`` when an earlier
    derivation set it, otherwise its value in ``doc``.'''
    derived = derived or {}
    resolved = {}
    for symbol, (kind, ref) in derivation.inputs.items():
        what = f"derivation '{derivation.name}' input '{symbol}' ({kind} '{ref}')"
        if kind == 'parameter':
            leaf = _parameter_leaf(doc, ref, what)
            path = parameter_path(doc, ref)
            value = derived[path] if path in derived else _number(leaf.get('value'), what)
            units = str(leaf.get('units') or '').strip()
            record = {'value': value, 'units': units, 'std': None, 'source': None,
                      'data_reference': leaf.get('data_reference') or None}
        else:
            item = obs_items[kind].get(ref)
            if item is None:
                raise DerivationError(f'{what}: no obs_data has a {kind} named {ref!r}.')
            if kind == 'data_item' and item.get('data_type') not in (None, 'constant'):
                raise DerivationError(f"{what}: a {item.get('data_type')} data_item is not one "
                                      f"number; an input is a constant item.")
            value = _number(item.get('value'), what)
            units = str(item.get('unit') or '').strip()
            std = item.get('std')
            record = {'value': value, 'units': units,
                      'std': None if std is None else float(std),
                      'source': item.get('source'), 'data_reference': item.get('data_reference')}
        if not units:
            raise DerivationError(f'{what} has no units.')
        resolved[symbol] = dict(record, kind=kind, reference=ref)
    return resolved


def unresolved_inputs(derivations, doc, obs_data=None):
    '''Every input of ``derivations`` that does not resolve, as ``[(derivation name, symbol,
    reason)]`` in file order; empty when all do. Unlike ``resolve_inputs`` it does not stop at
    the first: each input is resolved on its own. A parameter input another derivation sets
    resolves, whatever its cached value.'''
    obs_items = load_obs_items(obs_data)
    set_by_derivation = {parameter_path(doc, name): 0.0
                         for d in derivations for name in d.outputs.values()}
    found = []
    for d in derivations:
        for symbol, source in d.inputs.items():
            one = Derivation(d.name, {symbol: source}, d.outputs, d.method, d.spec, d.base_dir)
            try:
                resolve_inputs(one, doc, obs_items, set_by_derivation)
            except ValueError as e:
                found.append((d.name, symbol, str(e)))
    return found


@dataclass
class DerivationResult:
    '''One evaluated derivation: its inputs (``resolve_inputs``) and outputs
    (``{symbol: {'parameter', 'path', 'value', 'units', 'cached'}}``, ``cached`` the value text in
    the parameters file).'''
    derivation: Derivation
    inputs: dict
    outputs: dict = field(default_factory=dict)


def evaluate_derivations(derivations, doc, obs_data=None):
    '''Evaluate ``derivations`` (``load_derivations``) against the parameters document ``doc``
    (``read_parameters``) and ``obs_data`` (paths or JSON objects; ``load_obs_items``), in
    ``order_derivations`` order. Returns a list of :class:`DerivationResult`.'''
    from libcuflynx.derivations.cellml_method import evaluate_cellml
    from libcuflynx.derivations.python_method import evaluate_python
    obs_items = load_obs_items(obs_data)
    derived, results = {}, []
    for d in order_derivations(derivations, doc):
        inputs = resolve_inputs(d, doc, obs_items, derived)
        leaves = {symbol: _parameter_leaf(doc, name, f"derivation '{d.name}' output '{symbol}'")
                  for symbol, name in d.outputs.items()}
        output_units = {s: (str(leaf.get('units') or '').strip() or None)
                        for s, leaf in leaves.items()}
        if d.method == 'cellml':
            values = evaluate_cellml(d.files()[0],
                                     {s: (i['value'], i['units']) for s, i in inputs.items()},
                                     output_units)
        else:
            units = {s: i['units'] for s, i in inputs.items()}
            units.update(output_units)
            values = evaluate_python(d.spec, d.base_dir,
                                     {s: i['value'] for s, i in inputs.items()}, units,
                                     list(d.outputs))
        result = DerivationResult(d, inputs)
        for symbol, name in d.outputs.items():
            path = parameter_path(doc, name)
            derived[path] = values[symbol]
            result.outputs[symbol] = {'parameter': name, 'path': path, 'value': values[symbol],
                                      'units': output_units[symbol],
                                      'cached': leaves[symbol].get('value')}
        results.append(result)
    return results


def check_derivations(results, doc, rtol=CACHE_RTOL):
    '''The problems with the cached values in ``doc`` of the parameters ``results`` set, as
    messages (none when the file is up to date): a cached value that is not the derived one
    within ``rtol``, a parameter naming another derivation, and a parameter naming a derivation
    no result has.'''
    problems, by_path = [], {}
    for result in results:
        name = result.derivation.name
        for symbol, out in result.outputs.items():
            by_path[out['path']] = name
            leaf = pf.get_leaf(doc, list(out['path']))
            try:
                cached = float(out['cached'])
            except (TypeError, ValueError):
                cached = None
            if cached is None or not math.isclose(cached, out['value'], rel_tol=rtol,
                                                  abs_tol=0.0):
                problems.append(f"{out['parameter']}: cached {out['cached']}, derivation "
                                f"'{name}' gives {out['value']!r}.")
            if leaf.get('derivation') not in (None, name):
                problems.append(f"{out['parameter']}: names derivation '{leaf['derivation']}', "
                                f"but '{name}' sets it.")
    for path, leaf in pf.leaves(doc):
        if leaf.get('derivation') and tuple(path) not in by_path:
            problems.append(f"{'/'.join(path)}: names derivation '{leaf['derivation']}', which "
                            f"does not set it.")
    return problems


def write_derived_values(parameters_path, results):
    '''Write each derived value into the parameters file ``parameters_path`` as its cached value,
    with ``derivation`` naming the derivation; every other key and parameter is kept. Returns
    the number of values that changed.'''
    doc = read_parameters(parameters_path)
    changed = 0
    for result in results:
        for out in result.outputs.values():
            leaf = dict(pf.get_leaf(doc, list(out['path'])))
            try:
                same = float(leaf.get('value')) == out['value']
            except (TypeError, ValueError):
                same = False
            if not same:
                leaf['value'] = out['value']
                changed += 1
            leaf['derivation'] = result.derivation.name
            pf.set_leaf(doc, list(out['path']), leaf)
    pf.write_parameters_file(parameters_path, doc)
    return changed


def derivation_files(derivations_path):
    '''Every file a derivations file reads, absolute: the CellML models and Python files of its
    derivations (shipped functions have none).'''
    files = []
    for d in load_derivations(derivations_path):
        files += [f for f in d.files() if f not in files]
    return files
