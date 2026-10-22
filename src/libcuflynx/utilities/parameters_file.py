'''
Parameter values: a model's parameters file and a module's parameterisations, in nested JSON.

A model's parameters file, ``<prefix>_parameters.json`` beside its module array, nests instance,
sub-instance and variable::

    {"schema_version": 1,
     "global": {"R": {"value": 8.314, "units": "J_per_mol_K", "data_reference": "..."}},
     "aortic_root": {"C": {"value": 1.2e-8, "units": "m3_per_Pa", "data_reference": "Liang2009"}},
     "heart": {"lv": {"E_A": {"value": 366575000, "units": "J_per_m6", "data_reference": "..."}}}}

A module's parameterisation (``parameterisations/<name>/<name>_parameters.json``) has the same
form rooted at the module: a component module's variables at the top (``{"C": {...}}``), a
supermodule's sub-instances (``{"lv": {"E_A": {...}}}``), and the globals the module sets under
``global``. Calibrated results (``<name>_calibrated_parameters.json``) are written in the same
form.

The rules (``schemas/parameters.schema.json``):

* A *leaf* is one parameter: an object with ``value`` (a number, or a string such as
  ``EMPTY_MUST_BE_FILLED``), ``units`` and ``data_reference``; any other key (a library's
  ``sourced``) is a string, kept and not read. Every other object is a *node*: an instance or
  sub-instance, whose keys are its sub-instances and variables.
* ``global`` at the top holds the global constants, as leaves. ``schema_version`` is 1.
* So a node never has a key ``value``, and nothing below the top is named ``global`` or
  ``schema_version``: no instance, sub-instance or variable may have these names
  (``RESERVED_NAMES``; module arrays refuse such a record name).
* In a model's file every top-level key other than ``global`` is an instance (a node).

A path maps one to one onto the names CA's users write (``utilities/parameter_names.py``):
``aortic_root/C``, ``heart/lv/E_A``, and ``global/R`` or ``R``. Values keep the text they are
written with: ``1.2e-8`` is read as the string ``"1.2e-8"``, as a CSV cell was.
'''

import json
import os
import re

from libcuflynx.upgrade import COMMAND as UPGRADE_COMMAND

SCHEMA_VERSION = 1
GLOBAL = 'global'
VALUE = 'value'
SCHEMA_VERSION_KEY = 'schema_version'
#: names no instance, sub-instance or variable may have
RESERVED_NAMES = (GLOBAL, VALUE, SCHEMA_VERSION_KEY)
#: a leaf's keys, in the order they are written; any other key follows them
LEAF_KEYS = ('value', 'units', 'data_reference')
#: the row keys every reader gives (a leaf's other keys are added after them)
ROW_KEYS = ('variable_name',) + LEAF_KEYS
SEPARATOR = '/'
LEVEL_SEPARATOR = '__'
EXTENSION = '.json'
#: converts a parameters CSV (and the older supermodule parameters files) to this form
CONVERT_COMMAND = f'{UPGRADE_COMMAND} parameters'

#: a model's parameters file (instances at the top) or a parameterisation (rooted at a module)
MODEL, MODULE = 'model', 'module'

_JSON_NUMBER = re.compile(r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$')


def parameters_file_name(prefix):
    '''``<prefix>_parameters.json``'''
    return f'{prefix}_parameters{EXTENSION}'


def is_leaf(node):
    '''Whether ``node`` is one parameter (an object with ``value``).'''
    return isinstance(node, dict) and VALUE in node


def refuse_csv(path, where=''):
    '''ValueError for a parameters file that is not JSON, naming the converter.'''
    stem, ext = os.path.splitext(str(path))
    if ext.lower() == EXTENSION:
        return
    raise ValueError(
        f'{where}{path}: libcuflynx reads parameter values from nested JSON '
        f'({os.path.basename(stem)}{EXTENSION}: instance, sub-instance, variable), not from '
        f'{ext or "a file without an extension"}. Convert it once with: {CONVERT_COMMAND} {path} '
        f'(and name the .json file wherever the .csv was named, such as input_param_file).')


def _load(path):
    with open(path, encoding='utf-8-sig') as f:
        text = f.read()
    # validated as JSON values; read again keeping each number's text, as a CSV cell kept it
    return json.loads(text), json.loads(text, parse_float=str, parse_int=str)


def read_parameters_file(path, where='', kind=MODEL):
    '''The parameters file at ``path`` (a model's, or a parameterisation when ``kind`` is
    MODULE) as its JSON object, numbers kept as their text. Refuses a CSV (naming the converter),
    a file its schema refuses, and the rules the schema cannot express (see the module
    docstring).'''
    refuse_csv(path, where)
    from libcuflynx.schemas import PARAMETERS_SCHEMA
    from libcuflynx.schemas.validation import validate
    doc, raw = _load(path)
    validate(PARAMETERS_SCHEMA, doc, source=str(path), what='parameters file')
    check_parameters(raw, f'{where}{path}', kind)
    raw[SCHEMA_VERSION_KEY] = doc[SCHEMA_VERSION_KEY]
    return raw


def check_parameters(doc, source, kind=MODEL):
    '''ValueError for what the schema cannot see: a reserved name below the top, and in a
    model's file a top-level leaf (a global goes under ``global``).'''
    for key, node in doc.items():
        if key in (SCHEMA_VERSION_KEY, GLOBAL):
            continue
        if kind == MODEL and is_leaf(node):
            raise ValueError(f'{source}: "{key}" is a parameter at the top of a model\'s '
                             f'parameters file, where only instances are; a global constant '
                             f'goes under "{GLOBAL}" ({{"{GLOBAL}": {{"{key}": ...}}}}).')
        _check_node(node, [key], source)
    return doc


def _check_node(node, path, source):
    if is_leaf(node):
        return
    for key, child in node.items():
        if key in RESERVED_NAMES:
            raise ValueError(f'{source}: "{SEPARATOR.join(path + [key])}": "{key}" is reserved '
                             f'(an instance, sub-instance or variable may not be named '
                             f'{", ".join(RESERVED_NAMES)}).')
        _check_node(child, path + [key], source)


def _text(value):
    '''A leaf's value as the text a parameters CSV cell holds.'''
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


#: what a parameter without a value is read as: generation reports it, and lists it in the
#: unfinished parameters file it writes
UNSET_VALUE = 'EMPTY_MUST_BE_FILLED'


def leaf_row(name, leaf):
    '''The row of ``leaf`` named ``name``: ``variable_name``, ``value`` (``UNSET_VALUE`` when it is
    empty), ``units``, ``data_reference`` (empty when absent) and any other key, all text.'''
    row = {'variable_name': name, 'value': _text(leaf.get(VALUE, '')).strip() or UNSET_VALUE,
           'units': str(leaf.get('units', '')), 'data_reference': str(leaf.get('data_reference', ''))}
    for key, value in leaf.items():
        if key not in row:
            row[key] = str(value)
    return {k: row[k] for k in ROW_KEYS + tuple(k for k in row if k not in ROW_KEYS)}


def parameter_rows(doc, kind=MODEL):
    '''The parameters of ``doc`` as rows (``leaf_row``) in file order, each named as a user
    writes it: ``aortic_root/C``, ``heart/lv/E_A``; a global ``R`` in a model's file and
    ``global/R`` in a parameterisation (``kind`` MODULE), whose own variables are plain
    (``C``).'''
    rows = []
    for name, leaf in (doc.get(GLOBAL) or {}).items():
        rows.append(leaf_row(name if kind == MODEL else f'{GLOBAL}{SEPARATOR}{name}', leaf))

    def walk(node, path):
        for key, child in node.items():
            if is_leaf(child):
                rows.append(leaf_row(SEPARATOR.join(path + [key]), child))
            else:
                walk(child, path + [key])
    walk({k: v for k, v in doc.items() if k not in (SCHEMA_VERSION_KEY, GLOBAL)}, [])
    return rows


def read_parameter_rows(path, where='', kind=MODEL):
    '''``parameter_rows`` of the parameters file at ``path``.'''
    return parameter_rows(read_parameters_file(path, where, kind), kind)


def name_path(name, kind=MODEL, where=''):
    '''The path in a parameters file of the parameter written ``name``: ``heart/lv/E_A`` and its
    flat form ``heart__lv/E_A`` -> ``['heart', 'lv', 'E_A']``; a global (``global/R``, and in a
    model's file a plain ``R``) -> ``['global', 'R']``; a parameterisation's own ``C`` ->
    ``['C']``.'''
    name = str(name).strip()
    parts = [p.strip() for p in name.split(SEPARATOR)]
    if any(not p for p in parts):
        raise ValueError(f'{where}parameter "{name}": nothing may be empty on either side of a '
                         f'"{SEPARATOR}".')
    if parts[0] == GLOBAL:
        if len(parts) != 2:
            raise ValueError(f'{where}parameter "{name}": a global is {GLOBAL}/<variable>.')
        return parts
    if len(parts) == 1:
        return [GLOBAL, parts[0]] if kind == MODEL else parts
    path = [level for part in parts[:-1] for level in part.split(LEVEL_SEPARATOR)]
    if any(not p for p in path):
        raise ValueError(f'{where}parameter "{name}": a level of its path is empty.')
    return path + [parts[-1]]


def leaf_from_row(row):
    '''A row (``variable_name``, ``value``, ``units``, ``data_reference`` and any other column)
    as a leaf. Empty columns other than those three are left out.'''
    leaf = {VALUE: row.get('value', ''), 'units': str(row.get('units') or '').strip(),
            'data_reference': str(row.get('data_reference') or '').strip()}
    if isinstance(leaf[VALUE], str):
        leaf[VALUE] = leaf[VALUE].strip()
    for key, value in row.items():
        if key in ROW_KEYS or key is None or value is None:
            continue
        value = str(value).strip()
        if key.strip() and value:
            leaf[key.strip()] = value
    return leaf


def new_document():
    return {SCHEMA_VERSION_KEY: SCHEMA_VERSION}


def get_leaf(doc, path):
    '''The leaf at ``path`` in ``doc``, None when there is none.'''
    node = doc
    for key in path:
        if not isinstance(node, dict) or is_leaf(node) or key not in node:
            return None
        node = node[key]
    return node if is_leaf(node) else None


def set_leaf(doc, path, leaf, where=''):
    '''Put ``leaf`` at ``path`` in ``doc``, making the nodes on the way. ValueError when the
    path runs through a parameter, or ends on an instance.'''
    node = doc
    for i, key in enumerate(path[:-1]):
        child = node.setdefault(key, {})
        if is_leaf(child):
            raise ValueError(f'{where}"{SEPARATOR.join(path)}": '
                             f'"{SEPARATOR.join(path[:i + 1])}" is a parameter, not an instance.')
        node = child
    if path[-1] in node and not is_leaf(node[path[-1]]):
        raise ValueError(f'{where}"{SEPARATOR.join(path)}" is an instance, not a parameter.')
    node[path[-1]] = leaf


def rows_to_document(rows, kind=MODEL, where='', duplicates=None):
    '''A parameters document from rows (``variable_name`` as ``name_path`` reads it). Of two
    rows for one parameter the first is kept, as generation used it; their names are added to
    ``duplicates`` when a list is given, and are a ValueError otherwise.'''
    doc = new_document()
    for row in rows:
        name = str(row.get('variable_name') or '').strip()
        if not name:
            continue
        path = name_path(name, kind, where)
        if get_leaf(doc, path) is not None:
            if duplicates is None:
                raise ValueError(f'{where}parameter "{name}" is set twice.')
            duplicates.append(name)
            continue
        set_leaf(doc, path, leaf_from_row(row), where)
    return _globals_first(doc)


def _globals_first(doc):
    head = {SCHEMA_VERSION_KEY: doc.get(SCHEMA_VERSION_KEY, SCHEMA_VERSION)}
    if GLOBAL in doc:
        head[GLOBAL] = doc[GLOBAL]
    head.update((k, v) for k, v in doc.items() if k not in head)
    return head


def _value_json(value):
    '''A value as JSON text: a number keeps the text it was given; other text is a string.'''
    if isinstance(value, bool) or value is None:
        return json.dumps(str(value))
    if isinstance(value, (int, float)):
        text = _text(value)
        return text if _JSON_NUMBER.match(text) else json.dumps(text)
    text = str(value)
    return text if _JSON_NUMBER.match(text) else json.dumps(text, ensure_ascii=False)


def _leaf_json(leaf):
    keys = [k for k in LEAF_KEYS if k in leaf] + [k for k in leaf if k not in LEAF_KEYS]
    return '{' + ', '.join(
        f'{json.dumps(k)}: ' + (_value_json(leaf[k]) if k == VALUE else
                                json.dumps(str(leaf[k]), ensure_ascii=False))
        for k in keys) + '}'


def dump_parameters(doc):
    '''JSON text of a parameters document: one line per parameter, one level of indent per
    level of nesting, numbers as their text.'''
    lines = []

    def node(key, value, depth, last):
        pad = ' ' * depth
        comma = '' if last else ','
        if key == SCHEMA_VERSION_KEY:
            lines.append(f'{pad}{json.dumps(key)}: {int(value)}{comma}')
        elif is_leaf(value):
            lines.append(f'{pad}{json.dumps(key, ensure_ascii=False)}: {_leaf_json(value)}{comma}')
        elif not value:
            lines.append(f'{pad}{json.dumps(key, ensure_ascii=False)}: {{}}{comma}')
        else:
            lines.append(f'{pad}{json.dumps(key, ensure_ascii=False)}: {{')
            items = list(value.items())
            for i, (k, v) in enumerate(items):
                node(k, v, depth + 1, i == len(items) - 1)
            lines.append(f'{pad}}}{comma}')

    doc = _globals_first(doc)
    items = list(doc.items())
    lines.append('{')
    for i, (k, v) in enumerate(items):
        node(k, v, 1, i == len(items) - 1)
    lines.append('}')
    return '\n'.join(lines) + '\n'


def write_parameters_file(path, doc):
    '''Write the parameters document ``doc`` to ``path`` (``dump_parameters``), making its
    directory. Returns ``path``.'''
    refuse_csv(path)
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(dump_parameters(doc))
    return path


def write_parameter_rows(path, rows, kind=MODEL):
    '''Write ``rows`` as the parameters file ``path`` (``rows_to_document``).'''
    return write_parameters_file(path, rows_to_document(rows, kind, f'{path}: '))


def parameters_array(rows, width=80):
    '''``rows`` as the structured array of strings a parameters table was read into: the fields
    ``variable_name``, ``units``, ``value``, ``data_reference`` and the other keys rows have, in
    that order, each ``<U{width}``.'''
    import numpy as np
    fields = ['variable_name', 'units', 'value', 'data_reference']
    for row in rows:
        fields += [k for k in row if k not in fields]
    dtype = [(f, f'<U{width}') for f in fields]
    return np.array([tuple(str(row.get(f, '')) for f in fields) for row in rows], dtype=dtype)
