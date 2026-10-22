'''
params_for_id files: the parameters a calibration identifies, in flat JSON.

``<prefix>_params_for_id.json`` is a list of entries, one per calibrated value
(``schemas/params_for_id.schema.json``)::

    {"version": 1,
     "params": [
      {"targets": ["aortic_root/C"], "min": 1e-9, "max": 1e-7, "name_for_plotting": "C_{ao}"},
      {"targets": ["heart/lv/E_A", "heart/rv/E_A"], "min": 1e8, "max": 1e9},
      {"targets": ["global/T"], "min": 0.6, "max": 1.2, "prior": "normal",
       "prior_params": {"prior_mean": 0.9}},
      {"name": "R_scale", "modifies": ["a/R", "b/R"], "modifier": "scale", "min": 0.5, "max": 2}]}

An entry names model parameters as users write them (``utilities/parameter_names.py``):
``<vessel>/<variable>``, a nested vessel by its path (``heart/lv/E_A``, read as the model's
``heart__lv``) and a global constant as ``global/<variable>``. A ``targets`` entry sets every
parameter it lists to one calibrated value; a ``modifies`` entry acts on them through a modifier
function. The file is flat, not nested by instance, because an entry is not a parameter: it
groups parameters, or modifies them.

A params_for_id CSV (``vessel_name,param_name,min,max,...``) is refused, naming
``cuflynx-upgrade params-for-id``, which writes this file beside it (``csv_to_document``).
'''

import csv
import io
import json
import os
import re

from libcuflynx.upgrade import COMMAND as UPGRADE_COMMAND

VERSION = 1
EXTENSION = '.json'
#: converts a params_for_id CSV to this form
CONVERT_COMMAND = f'{UPGRADE_COMMAND} params-for-id'
#: the columns that make a CSV a params_for_id CSV
CSV_COLUMNS = ('vessel_name', 'param_name')
#: the order an entry's keys are written in; any other key follows them
ENTRY_KEYS = ('name', 'targets', 'modifies', 'modifier', 'inputs', 'min', 'max', 'param_type',
              'name_for_plotting', 'prior', 'prior_params', 'unbounded', 'comment')
_JSON_NUMBER = re.compile(r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$')


def params_for_id_file_name(prefix):
    '''``<prefix>_params_for_id.json``'''
    return f'{prefix}_params_for_id{EXTENSION}'


def refuse_csv(path, where=''):
    '''ValueError for a params_for_id file that is not JSON, naming the converter.'''
    stem, ext = os.path.splitext(str(path))
    if ext.lower() == EXTENSION:
        return
    raise ValueError(
        f'{where}{path}: libcuflynx reads params_for_id from JSON '
        f'({os.path.basename(stem)}{EXTENSION}: a list of entries with "targets" or '
        f'"modifies"), not from {ext or "a file without an extension"}. Convert it once with: '
        f'{CONVERT_COMMAND} {path} (and name the .json file wherever the .csv was named, such '
        f'as params_for_id_file).')


def read_params_for_id(path, where='', keep_text=False):
    '''The params_for_id document at ``path``, as JSON. A CSV is refused, naming the converter;
    the document is checked by ``ObsAndParamDataParser.resolve_params_for_id_doc``. With
    ``keep_text`` each number is kept as its text, so ``write_params_for_id`` writes it back as
    it was written (for a file that is rewritten, not read).'''
    refuse_csv(path, where)
    with open(path, encoding='utf-8-sig') as f:
        if keep_text:
            return json.load(f, parse_float=_Number, parse_int=_Number)
        return json.load(f)


def is_params_for_id_csv(path):
    '''Whether ``path`` is a ``.csv`` with the vessel_name and param_name columns.'''
    if not str(path).lower().endswith('.csv'):
        return False
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            header = next(csv.reader(f), [])
    except (OSError, UnicodeDecodeError):
        return False
    return all(c in [h.strip() for h in header] for c in CSV_COLUMNS)


def _number(text):
    '''``text`` as written, marked to be written as a JSON number when it is one.'''
    text = str(text).strip()
    return _Number(text) if _JSON_NUMBER.match(text) else text


class _Number(str):
    '''A number's text, written unquoted by ``dump_params_for_id``.'''


def target_name(vessel, variable):
    '''The user-side name of ``variable`` of ``vessel``: a nested vessel's ``__`` levels as its
    path (``heart__lv`` -> ``heart/lv/E``).'''
    from libcuflynx.utilities.parameter_names import LEVEL_SEPARATOR, SEPARATOR
    return SEPARATOR.join(vessel.split(LEVEL_SEPARATOR) + [variable])


def csv_to_document(csv_text_or_path):
    '''The params_for_id document of a params_for_id CSV (its path, or its text).

    Per row: ``vessel_name`` "a b" with ``param_name`` "C" -> ``targets`` ["a/C", "b/C"] (a
    nested vessel ``heart__lv`` as its path, ``heart/lv/C``); ``min``, ``max``, ``param_type``,
    ``name_for_plotting``, ``prior``, ``comment`` -> the same keys; the ``prior_*`` columns ->
    ``prior_params``; ``unbounded`` -> true or false. Blank cells are left out, and numbers keep
    their text.
    '''
    from libcuflynx.parsers.PrimitiveParsers import (PARAMS_FOR_ID_CSV_PRIOR_COLUMNS,
                                                     PARAM_UNBOUNDED_COLUMN, _truthy_flag)
    text = str(csv_text_or_path)
    if '\n' not in text and os.path.exists(text):
        with open(text, newline='', encoding='utf-8-sig') as f:
            text = f.read()
    reader = csv.DictReader(io.StringIO(text.strip() + '\n'))
    params = []
    for idx, raw in enumerate(reader):
        row = {str(k).strip(): (v or '').strip() for k, v in raw.items() if k is not None}
        if not any(row.values()):
            continue
        vessels = row.get('vessel_name', '').split()
        variable = row.get('param_name', '')
        if not vessels or not variable:
            raise ValueError(f'params_for_id CSV row {idx}: both vessel_name and param_name are '
                             f'required (got vessel_name={row.get("vessel_name")!r}, '
                             f'param_name={variable!r}).')
        entry = {'targets': [target_name(v, variable) for v in vessels]}
        for key in ('min', 'max'):
            if row.get(key):
                entry[key] = _number(row[key])
        for key in ('param_type', 'name_for_plotting', 'prior', 'comment'):
            if row.get(key):
                entry[key] = row[key]
        prior_params = {k: _number(row[k]) for k in PARAMS_FOR_ID_CSV_PRIOR_COLUMNS
                        if row.get(k)}
        if prior_params:
            entry['prior_params'] = prior_params
        if row.get(PARAM_UNBOUNDED_COLUMN):
            entry[PARAM_UNBOUNDED_COLUMN] = _truthy_flag(row[PARAM_UNBOUNDED_COLUMN])
        params.append(entry)
    return {'version': VERSION, 'params': params}


def _value_json(value):
    if isinstance(value, _Number):
        return str(value)
    if isinstance(value, dict):
        return '{' + ', '.join(f'{json.dumps(k, ensure_ascii=False)}: {_value_json(v)}'
                               for k, v in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(_value_json(v) for v in value) + ']'
    return json.dumps(value, ensure_ascii=False)


def _ordered(entry):
    keys = [k for k in ENTRY_KEYS if k in entry] + [k for k in entry if k not in ENTRY_KEYS]
    return {k: entry[k] for k in keys}


def dump_params_for_id(doc):
    '''JSON text of a params_for_id document: one line per entry, numbers as their text.'''
    head = {k: v for k, v in doc.items() if k != 'params'}
    head.setdefault('version', VERSION)
    lines = ['{']
    for key, value in head.items():
        lines.append(f' {json.dumps(key)}: {_value_json(value)},')
    entries = list(doc.get('params') or [])
    lines.append(' "params": [')
    for i, entry in enumerate(entries):
        comma = ',' if i < len(entries) - 1 else ''
        lines.append(f'  {_value_json(_ordered(entry))}{comma}')
    lines.append(' ]')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def write_params_for_id(path, doc):
    '''Write the params_for_id document ``doc`` (or a list of its entries) to ``path``, making
    its directory. Returns ``path``.'''
    refuse_csv(path)
    if isinstance(doc, list):
        doc = {'version': VERSION, 'params': doc}
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(dump_params_for_id(doc))
    return path


def convert_csv(path, dry_run=False):
    '''Write the ``.json`` beside the params_for_id CSV ``path`` and remove the CSV. Returns the
    JSON path.'''
    out = os.path.splitext(str(path))[0] + EXTENSION
    doc = csv_to_document(path)
    if os.path.exists(out):
        raise ValueError(f'{path}: {out} exists already; remove one of the two.')
    if not dry_run:
        write_params_for_id(out, doc)
        os.remove(path)
    return out
