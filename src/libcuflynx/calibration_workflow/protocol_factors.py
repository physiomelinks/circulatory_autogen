'''
A protocol's ``params_to_change_factors``: ``{qname: rows}`` in the shape of ``params_to_change``
(one row per experiment of one number per sub-experiment), each number a factor on the value the
step's model gives that parameter.

A calibration workflow resolves them once per step, against the step model's values after
``fixed_from`` / ``result_from``, into absolute ``params_to_change`` in a copy of the obs_data the
step reads (:func:`resolve_factors`). Nothing downstream learns the key, as protocol_shapes
become protocol_traces; a plain run (``parse_obs_data_json``) refuses it.
'''

import json

from libcuflynx.calibration_workflow import naming
from libcuflynx.calibration_workflow.spec import WorkflowError

FACTORS_KEY = 'params_to_change_factors'


def _read(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def has_factors(obs_data_path):
    '''Whether the obs_data at ``obs_data_path`` (None: no obs_data) has factors to resolve.'''
    if not obs_data_path:
        return False
    return bool((_read(obs_data_path).get('protocol_info') or {}).get(FACTORS_KEY))


def resolve_factors(obs_path, out_path, model_rows, fixed, refused, where):
    '''Write to ``out_path`` the obs_data at ``obs_path`` with its factors resolved into
    ``params_to_change``. ``model_rows`` is ``{model name: row}`` of the step's model, ``fixed``
    ``{model name: row}`` of the values it was given (each with ``from_step``), and ``refused``
    ``{model name: why}`` of the parameters a factor may not name (those the step calibrates or
    sweeps). Returns the records of what each factor multiplied: ``parameter``, ``model_name``,
    ``base``, ``units`` and, for a value given by an earlier step, ``from_step``.'''
    from libcuflynx.parsers.PrimitiveParsers import validate_params_to_change

    obs = _read(obs_path)
    info = obs['protocol_info']
    factors = info.pop(FACTORS_KEY)
    try:
        validate_params_to_change({**info, FACTORS_KEY: factors}, entry=FACTORS_KEY)
    except ValueError as exc:
        raise WorkflowError(f'{where}: {exc}') from exc
    absolute = info.get('params_to_change') or {}
    records = []
    for qname, rows in factors.items():
        name = naming.host_model_name(qname)
        if qname in absolute:
            raise WorkflowError(f'{where}: "{qname}" is in both params_to_change and '
                                f'{FACTORS_KEY}; give it in one of them.')
        if name in refused:
            raise WorkflowError(f'{where}: {FACTORS_KEY} names "{qname}", which the step '
                                f'{refused[name]}; a factor multiplies a value the step is '
                                f'given.')
        if name not in model_rows:
            raise WorkflowError(f'{where}: {FACTORS_KEY} names "{qname}", which its model has '
                                f'no parameter "{name}" for.')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               for row in rows for v in row):
            raise WorkflowError(f'{where}: every {FACTORS_KEY} entry of "{qname}" is a number.')
        base = float(model_rows[name]['value'])
        absolute[qname] = [[factor * base for factor in row] for row in rows]
        given = fixed.get(name)
        records.append({'parameter': qname, 'model_name': name, 'base': base,
                        'units': model_rows[name].get('units', ''),
                        **({'from_step': given['from_step']} if given else {})})
    info['params_to_change'] = absolute
    with open(out_path, 'w') as f:
        json.dump(obs, f, indent=1)
    return records
