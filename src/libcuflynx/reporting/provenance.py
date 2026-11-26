'''
How a calibration workflow run's values were set: a graph of the run, and that graph as SVG.

:func:`provenance_graph` reads a run's output directory alone, so it needs no module library.
Its nodes are the model source (the target's model, and one more for each other model a step is
generated from, each with an arrow to every step it generates), one node for each step (naming
the values it calibrated, its priors and the values it was given, with what set them), a node
each for the measured values, the derived values, the fixed constants of the derivations (when a
derived value has any), the values set by hand in the parameters files and the values calibrated
before the run (when there are any), and the merged target the simulation runs. A sweep's node has no
arrow onward: nothing takes its values. The nodes of values given in the parameters files point
to the target's model source, whose generated step models hold them (to the target when no step
comes from that source). Each value the target uses is marked with what set it:

* **calibrated**: a step fitted it (the step's results and the run's merged parameters say so),
  or its entry in the parameters has origin ``calibrated`` (fitted before this run; its
  data_reference says by what).
* **set_by_hand**: a set step declared it, with its reason, or the ``origin`` of its entry in the
  parameters says so. The box of a reduction chosen by hand (``"method": "manual"``) has this
  mark too: the choice of what to calibrate was made by hand, though it sets no value.
* **measured**, **derived**, **inherited**: the ``origin`` of its entry in the parameters
  (``derived``: a parameter derivation calculated it from data).
* **unclassified**: nothing says. A value is never counted as inherited by default.

:func:`write_provenance` writes ``provenance_graph.json``, checked against
``schemas/provenance_graph.schema.json``, and ``provenance.svg``. A graph's text is plain, with
code between backticks (:mod:`libcuflynx.reporting.flowchart`), so CUFLynx can draw the same file.
'''

import json
import os
import textwrap
from xml.sax.saxutils import escape, quoteattr

from libcuflynx.calibration_workflow.spec import INHERITED_BASE, raw_sweep, scale_sources
from libcuflynx.reporting import flowchart
from libcuflynx.utilities import parameters_file
from libcuflynx.utilities.parameters_file import (CALIBRATED, DERIVED, INHERITED, MEASURED,
                                                  SET_BY_HAND)

#: The graph's schema is closed (no unknown keys; enums for a node's kind and an edge's style), so
#: SCHEMA_VERSION goes up whenever the schema gains or changes a kind, style or key. A reader
#: refuses a version it does not know rather than misdraw it. 2 added the figure and external
#: node kinds and the writes and reads edge styles; 3 the derived origin and mark, and the box of
#: values calibrated before the run (origin calibrated with no from_step).
SCHEMA_VERSION = 3
#: the schema versions this module reads
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION,)
GRAPH_FILE = 'provenance_graph.json'
SVG_FILE = 'provenance.svg'
SOURCE_NODE = '__source__'
MEASURED_NODE = '__measured__'
HAND_NODE = '__set_by_hand__'
DERIVED_NODE = '__derived__'
FIXED_NODE = '__fixed_constants__'
EARLIER_NODE = '__calibrated_earlier__'
TARGET_NODE = '__target__'
#: the kind of the one box naming the files a sweep, simulate or figure step wrote
FIGURE = 'figure'
#: a script outside the workflow, in a graph written before figures were steps; drawn, never
#: produced
EXTERNAL = 'external'
UNCLASSIFIED = 'unclassified'
MIXED = 'mixed'
MARKS = (CALIBRATED, MEASURED, DERIVED, INHERITED, SET_BY_HAND, UNCLASSIFIED)
CURRENT, STALE, MISSING = 'current', 'stale', 'missing'
#: how many values the drawn box of hand-set or measured values, the model source or the target
#: lists; the graph has them all
DRAWN_VALUES = 6
#: how many values a step's box names of those it calibrated, and of those it was given
DRAWN_STEP_VALUES = 4

_SOURCE_KEYS = ('fixed_from', 'priors_from', 'emulator_from', 'result_from', 'sensitivity_from',
                'params_for_id_from', 'start_from', 'combine_from')
_MARK_LABELS = {CALIBRATED: 'calibrated', MEASURED: 'measured', DERIVED: 'derived',
                INHERITED: 'inherited', SET_BY_HAND: 'set by hand', UNCLASSIFIED: 'unclassified',
                MIXED: 'mixed'}
# mid-tones that read on a light and on a dark page
_MARK_COLOURS = {CALIBRATED: '#3b7dd8', MEASURED: '#2e9e6b', DERIVED: '#1a9aa8',
                 INHERITED: '#8a8f98', SET_BY_HAND: '#d98a1f', UNCLASSIFIED: '#b45cc4',
                 MIXED: '#8a8f98'}
_STALE_COLOUR = '#d9483b'
_EDGE_DASHES = {'prior': '7 4', 'merge': '2 4', 'given': '7 4', 'writes': '2 3',
                'reads': '2 3'}


def _sources(raw_step):
    '''``{source key: [step ids]}`` of a step as the workflow file writes it.'''
    out = {}
    for key in _SOURCE_KEYS:
        value = raw_step.get(key)
        if value:
            items = value if isinstance(value, list) else [value]
            out[key] = [item['step'] if isinstance(item, dict) else item for item in items]
    scaled = scale_sources(raw_step.get('set') or [])
    if scaled:
        out['scale_from'] = scaled
    if raw_step.get('kind') == 'figure':
        out['inputs_from'] = list(dict.fromkeys(
            source['step'] for source in _figure(raw_step)['inputs'].values()))
    return out


def _figure(raw_step):
    '''A figure step of the workflow file as the flow chart takes it.'''
    inputs = {}
    for name, source in (raw_step.get('inputs') or {}).items():
        (kind, step_id), = source.items()
        inputs[name] = {'kind': kind, 'step': step_id}
    script, _, function = (raw_step.get('function') or '').rpartition(':')
    return {'function': f'{os.path.basename(script)}:{function}', 'inputs': inputs}


def _is_helper_item(item):
    '''Whether a data_item scores nothing: its weight is 0 (a list counts only when every entry
    is). Such items feed ratios or differences. The std never makes an item a helper.'''
    weights = item.get('weight', 1.0)
    weights = weights if isinstance(weights, list) else [weights]
    return bool(weights) and all(isinstance(w, (int, float)) and w == 0 for w in weights)


def _data_item_counts(result):
    '''(observables, helper items) of a step's obs_data: helper items are not observables.'''
    path = ((result or {}).get('inputs', {}).get('obs_data') or {}).get('path')
    if not path or not os.path.isfile(path):
        return 0, 0
    with open(path, encoding='utf-8') as f:
        items = json.load(f).get('data_items') or []
    helpers = sum(1 for item in items if _is_helper_item(item))
    return len(items) - helpers, helpers


def _n_parameters(result):
    '''The parameters a step calibrates: the entries of the params_for_id it ran with
    (``params_for_id_used``, which holds those ``priors_from`` adds), else of its own.'''
    path = (result or {}).get('params_for_id_used') or \
        ((result or {}).get('inputs', {}).get('params_for_id') or {}).get('path')
    if path and os.path.isfile(path):
        from libcuflynx.utilities.params_for_id_file import read_params_for_id
        document = read_params_for_id(path)
        return len(document.get('params', []) if isinstance(document, dict) else document)
    return len((result or {}).get('calibrated', []))


def _done(result):
    return bool(result) and result.get('status') == 'done'


def _chosen_by_hand(raw_step):
    '''Whether a step is a reduction whose parameters were chosen by hand.'''
    options = (raw_step.get('settings') or {}).get('param_reduction_options') or {}
    return raw_step.get('kind') == 'reduction' and options.get('method') == 'manual'


def _graph_steps(run):
    '''The steps of a run as :func:`flowchart.workflow_graph` takes them.'''
    workflow = run['workflow']
    steps = []
    for raw in workflow.get('steps', []):
        result = run['steps'].get(raw.get('id'))
        sources = _sources(raw)
        if raw.get('on_target'):
            # given every calibration's, set step's and combine step's values, as the target is
            sources['fixed_from'] = [s['id'] for s in workflow.get('steps', [])
                                     if s.get('kind', 'calibration') in flowchart.MERGED_KINDS
                                     and not s.get('point')]
        depends = list(dict.fromkeys(i for ids in sources.values() for i in ids))
        observables, helpers = _data_item_counts(result) if _done(result) else (0, 0)
        steps.append({
            **{key: (ids if key in ('fixed_from', 'priors_from', 'scale_from', 'combine_from',
                                    'inputs_from')
                     else ids[0])
               for key, ids in sources.items()},
            'id': raw['id'], 'kind': raw.get('kind', 'calibration'),
            'emulator_dir': raw.get('emulator_dir'),
            'model': result.get('model') if _done(result) else None,
            'n_parameters': _n_parameters(result) if _done(result) else 0,
            'n_data_items': observables, 'n_helper_items': helpers,
            'settings': {**(workflow.get('settings') or {}), **(raw.get('settings') or {})},
            'depends_on': depends, 'fixed_from': sources.get('fixed_from', []),
            'priors_from': sources.get('priors_from', []),
            'store_distribution': bool(raw.get('store_distribution')),
            'reduction': (raw.get('settings') or {}).get('param_reduction_options'),
            'by_hand': _chosen_by_hand(raw),
            'set': raw.get('set'), 'sweep': raw_sweep(raw), 'combine': raw.get('combine'),
            'figure': _figure(raw) if raw.get('kind') == 'figure' else None,
            'on_target': bool(raw.get('on_target')), 'point': bool(raw.get('point'))})
    return steps


FILES_SUFFIX = '/files'


def _figure_node(parent_id, paths, output_dir, stale):
    '''The files a sweep, simulate or figure step wrote, named in one box below it: their common
    folder once, then each file. Done when every file exists; a missing one is marked.'''
    shown = [os.path.relpath(p, output_dir) if os.path.isabs(p) else p for p in paths]
    folder = os.path.commonpath(shown) if len(shown) > 1 else ''
    if folder in shown:                          # one file, or a file named as the folder
        folder = os.path.dirname(folder)
    lines = [f'in `{folder}/`'] if folder else []
    for path, name in zip(paths, shown):
        name = os.path.relpath(name, folder) if folder else name
        lines.append(f'`{name}`' + ('' if os.path.isfile(path) else ' (missing)'))
    count = len(paths)
    return {'id': parent_id + FILES_SUFFIX, 'kind': FIGURE,
            'title': f'{count} file{"s" if count != 1 else ""} written', 'lines': lines,
            'mark': None, 'status': 'done' if all(os.path.isfile(p) for p in paths) else 'not_run',
            'stale': stale, 'counts': {}, 'parameters': [], 'inputs': []}


def _target_model(run):
    target = (run['result'] or {}).get('target') or run['workflow'].get('target')
    return {'module_array': target} if isinstance(target, str) else target


def _generated_model(raw_step, result):
    '''The model a step generates from a model source, as its result records it or else as the
    workflow names it. None for a step with no model of its own: a set or reduction step, a
    sweep on the target, or a step on an emulator (it runs the emulation step's model).'''
    if raw_step.get('on_target') or raw_step.get('emulator_from') or not raw_step.get('model'):
        return None
    model = ((result or {}).get('model') if _done(result) else None) or raw_step['model']
    return {'module_array': model} if isinstance(model, str) else model


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _rows(path):
    if not path or not os.path.isfile(path):
        return []
    return parameters_file.model_parameter_rows(path)


def _target_rows(run):
    '''The rows of the parameters the target uses, or of the last step that ran when the target
    was not generated.'''
    target = (run['result'] or {}).get('target_model') or {}
    if target.get('parameters_path'):
        return _rows(target['parameters_path'])
    for step in reversed(run['workflow'].get('steps', [])):
        result = run['steps'].get(step.get('id'))
        path = ((result or {}).get('generated') or {}).get('parameters_path')
        if _done(result) and path:
            return _rows(path)
    return []


def _value(row, origin, from_step=None):
    return {'name': row['variable_name'], 'value': _number(row.get('value')),
            'units': row.get('units', ''), 'origin': origin, 'from_step': from_step,
            'data_reference': row.get('data_reference', ''), 'reason': row.get('reason', '')}


def _classified(row):
    origin = row.get('origin') or ''
    return origin if origin in parameters_file.ORIGINS else UNCLASSIFIED


def _counts(values):
    return {mark: sum(1 for v in values if v['origin'] == mark) for mark in MARKS}


def _mark(counts):
    '''The mark of a box of values: the only one they have, ``mixed`` when they have several.'''
    present = [mark for mark in MARKS if counts.get(mark)]
    if not present:
        return None
    return present[0] if len(present) == 1 else MIXED


def _settings_status(step_id, path, recorded, workflows):
    '''Whether a step's settings in the workflow file at ``path`` still have the digest its
    result recorded, computed as the runner computes it (so a re-formatted file, or a change to
    another step's settings, leaves it current). ``workflows`` caches each file's parse.'''
    from libcuflynx.calibration_workflow.runner import step_settings_digest
    from libcuflynx.calibration_workflow.spec import WorkflowError, load_workflow
    if not path or not os.path.isfile(path):
        return MISSING
    if path not in workflows:
        try:
            workflows[path] = load_workflow(path)
        except (WorkflowError, OSError):
            workflows[path] = None
    workflow = workflows[path]
    step = next((s for s in workflow.steps if s.id == step_id), None) if workflow else None
    if step is None:
        return STALE
    return CURRENT if step_settings_digest(workflow, step) == recorded else STALE


def _input_entries(step_id, result, workflows):
    from libcuflynx.calibration_workflow.runner import (LIBRARY_INPUT, SETTINGS_INPUT,
                                                        input_digest)
    entries = []
    for key, info in (result or {}).get('inputs', {}).items():
        path, recorded = info.get('path'), info.get('sha256')
        if key == SETTINGS_INPUT:
            status = _settings_status(step_id, path, recorded, workflows)
        elif key != LIBRARY_INPUT and (not path or not os.path.exists(path)):
            status = MISSING
        else:
            status = CURRENT if input_digest(key, path) == recorded else STALE
        entries.append({'key': key, 'path': path, 'sha256': recorded, 'status': status})
    return entries


def _workflow_file_status(result):
    from libcuflynx.calibration_workflow.runner import sha256, workflow_digest
    path = (result or {}).get('workflow_file')
    if not path:
        return CURRENT
    if not os.path.isfile(path):
        return MISSING
    recorded = result.get('workflow_sha256')
    # a run recorded either the digest that leaves figure steps out or the file's own sha256
    try:
        same = recorded in (workflow_digest(path), sha256(path))
    except (OSError, ValueError):
        same = recorded == sha256(path)
    return CURRENT if same else STALE


def _stale_steps(steps, inputs, results):
    '''``{step id: whether its result is out of date}``: an input changed, or a step it took
    something from is stale, has no result, or ran again after it. A changed module library is
    shown on the step's input and does not make it stale, as in ``workflow_status``.'''
    from libcuflynx.calibration_workflow.runner import LIBRARY_INPUT
    depends = {s['id']: s['depends_on'] for s in steps}
    stale = {}

    def is_stale(step_id):
        if step_id not in stale:
            result = results.get(step_id)
            changed = any(entry['status'] != CURRENT and entry['key'] != LIBRARY_INPUT
                          for entry in inputs[step_id])
            stale[step_id] = False
            for source in depends[step_id]:
                upstream = results.get(source)
                if is_stale(source) or not _done(upstream) or \
                        (upstream.get('finished') or '') > ((result or {}).get('started') or ''):
                    changed = True
            stale[step_id] = _done(result) and changed
        return stale[step_id]

    for step in steps:
        is_stale(step['id'])
    return stale


def _record_value(record, from_step):
    '''A step result's record as a graph value: set by hand if it says so, else calibrated.'''
    return _value({'variable_name': record.get('model_name'), 'value': record.get('value'),
                   'units': record.get('units', ''), 'reason': record.get('reason', '')},
                  record.get('origin', CALIBRATED), from_step)


def _step_values(step_id, result):
    '''The values a step fitted or set by hand, and those it was given by earlier steps, as graph
    values.'''
    result = result or {}
    return ([_record_value(r, step_id) for r in result.get('calibrated', [])]
            + [_record_value(r, step_id) for r in result.get('set', [])]
            # a set step's modifier thetas: not model values, named by their modifier entry
            + [_value({'variable_name': flowchart.theta_name(t), 'value': t['theta'],
                       'reason': t.get('reason', '')}, SET_BY_HAND, step_id)
               for t in result.get('thetas', [])]
            + [_record_value(r, r.get('from_step')) for r in result.get('fixed', [])])


def _step_counts(result):
    '''How the values of a step's own generated model were set, with ``fixed`` for those it was
    given by earlier steps.'''
    calibrated = {r.get('model_name') for r in (result or {}).get('calibrated', [])}
    fixed = {r.get('model_name') for r in (result or {}).get('fixed', [])}
    rows = _rows(((result or {}).get('generated') or {}).get('parameters_path'))
    others = [_value(r, _classified(r)) for r in rows
              if r['variable_name'] not in calibrated | fixed]
    counts = dict(_counts(others), calibrated=len(calibrated), fixed=len(fixed))
    counts[SET_BY_HAND] += len((result or {}).get('set', [])) + \
        len((result or {}).get('thetas', []))
    return counts


def _described(value):
    units = f' {value["units"]}' if value['units'] else ''
    note = value['reason'] if value['origin'] == SET_BY_HAND else value['data_reference']
    return f'`{value["name"]}` = {value["value"]}{units}: {note}'


def _values_node(node_id, title, origin, values):
    return {'id': node_id, 'title': title, 'kind': 'values', 'mark': origin,
            'lines': [_described(v) for v in values], 'status': 'done', 'stale': False,
            'counts': _counts(values), 'parameters': values, 'inputs': []}


def _fixed_constants(rows, values):
    '''The fixed constants of the derivations that calculate the target's derived values, as
    graph values named ``<derivation>/<constant>``, each once. A constant's origin is the one its
    derivation declares, ``unclassified`` when it declares none. The ``fixed_constants`` key of
    a derived row holds them as JSON text.'''
    found = {}
    for row, value in zip(rows, values):
        if value['origin'] != DERIVED or not row.get(parameters_file.FIXED_CONSTANTS):
            continue
        for entry in json.loads(row[parameters_file.FIXED_CONSTANTS]):
            origin = entry.get('origin')
            origin = origin if origin in parameters_file.ORIGINS else UNCLASSIFIED
            name = f'{entry["derivation"]}/{entry["name"]}'
            found.setdefault(name, _value(
                {'variable_name': name, 'value': entry['value'], 'units': entry.get('units', ''),
                 'data_reference': entry.get('data_reference', ''),
                 'reason': entry.get('reason', '')}, origin))
    return list(found.values())


def _fixed_line(value):
    units = f' {value["units"]}' if value['units'] else ''
    note = value['reason'] if value['origin'] == SET_BY_HAND else value['data_reference']
    return (f'`{value["name"]}` = {value["value"]}{units}, {_MARK_LABELS[value["origin"]]}'
            + (f': {note}' if note else ''))


def _quantity(value, units=''):
    text = f'{value:.4g}' if isinstance(value, (int, float)) else str(value)
    return f'{text} {units}' if units else text


def _record_name(record):
    '''A step result's record named as a params_for_id names it: ``module/param``.'''
    if record.get('module') and record.get('param'):
        return f'{record["module"]}/{record["param"]}'
    return record.get('model_name')


def _first(lines, limit, what):
    '''The first ``limit`` of ``lines``, then how many more ``what`` the graph file has.'''
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f'and {len(lines) - limit} more {what} in {GRAPH_FILE}']


def _set_by(origin, from_step, reason, data):
    '''What set a value a step passed on: by hand, with the reason, or a calibration on its
    data.'''
    if origin == SET_BY_HAND:
        return f'set by hand in {from_step}: {reason}'
    if origin == CALIBRATED:
        return f'calibrated by {from_step}' + (f' on `{data}`' if data else '')
    return f'{_MARK_LABELS.get(origin, origin)}, from {from_step}'


def _set_record_line(record):
    '''One value of a set step: its value, how a scaled one was made, and the reason.'''
    how = (f' ({record["scale"]:g} × {_quantity(record["base"])} from `{record["of"]}`)'
           if 'scale' in record else '')
    return (f'`{_record_name(record)}` = {_quantity(record["value"], record.get("units"))}'
            f'{how}: {record.get("reason", "")}')


def _step_value_lines(result, data_of):
    '''A done step's priors, the values it calibrated and those earlier steps fixed, by name,
    value and what set them; ``data_of(step id)`` names a step's obs_data.'''
    from libcuflynx.calibration_workflow.runner import POINT_POSTERIOR_MEDIAN
    calibrated = result.get('calibrated') or []
    names = {r.get('model_name'): _record_name(r) for r in calibrated}
    lines = []
    for prior in result.get('priors') or []:
        shown = ', '.join(f'`{names.get(n, n)}`' for n in prior.get('parameters') or [])
        inflated = f', widened {prior["inflate"]:g} times' if prior.get('inflate', 1) != 1 else ''
        lines.append(f'prior of {shown}: {prior.get("kind")} fit to the posterior of '
                     f'{prior["step"]}{inflated}')
    point = (' (posterior median)' if result.get('point_estimate') == POINT_POSTERIOR_MEDIAN
             else '')
    lines += _first([f'`{_record_name(r)}` = {_quantity(r["value"], r.get("units"))}, '
                     f'calibrated here{point}' for r in calibrated],
                    DRAWN_STEP_VALUES, 'calibrated values')
    lines += _first([f'`{_record_name(r)}` = {_quantity(r["value"], r.get("units"))}, fixed: '
                     + _set_by(r.get('origin', CALIBRATED), r.get('from_step'), r.get('reason'),
                               data_of(r.get('from_step')))
                     for r in result.get('fixed') or []],
                    DRAWN_STEP_VALUES, 'fixed values')
    return lines


def provenance_graph(output_dir):
    '''The provenance graph of the workflow run in ``output_dir``, as a JSON-ready dict:

    ``{schema_version, workflow: {name, output_dir, file, file_status, order, complete, finished,
    stale}, producer: {name, version, git}, code: {libcuflynx_version, libcuflynx_git,
    project_git, libcellml_version}, counts,
    nodes, edges}``. ``counts`` says how the values of the target were set (``MARKS``). A node is
    ``{id, kind, title, lines, mark, status, stale, counts, parameters, inputs}``, with ``kind``
    source, step, analysis, values or target; an edge is ``{from, to, label, style}``.
    '''
    from libcuflynx.calibration_workflow.runner import load_workflow_run
    run = load_workflow_run(output_dir)
    result = run['result']
    steps = _graph_steps(run)
    merged = (result or {}).get('merged_parameters', [])
    nodes, edges = flowchart.workflow_graph(steps + [{
        'id': None, 'model': _target_model(run), 'n_merged': len(merged),
        'n_set': sum(1 for r in merged if r.get('origin')),
        'merged_from': {r['from_step'] for r in merged} if result else None}])
    file_status = _workflow_file_status(result)
    workflows = {}
    inputs = {s['id']: _input_entries(s['id'], run['steps'].get(s['id']), workflows)
              for s in steps}
    stale = _stale_steps(steps, inputs, run['steps'])
    by_id = {s['id']: s for s in steps}
    figures = []

    by_name = {r['model_name']: r for r in merged}

    def target_value(row):
        record = by_name.get(row['variable_name'])
        if record is None:
            return _value(row, _classified(row))
        return _value(dict(row, reason=record.get('reason', '')),
                      record.get('origin', CALIBRATED), record['from_step'])

    rows = _target_rows(run)
    values = [target_value(row) for row in rows]
    target_counts = _counts(values)
    data_names = {sid: os.path.basename(e['path']) for sid, entries in inputs.items()
                  for e in entries if e['key'] == 'obs_data' and e['path']}

    for node in nodes:
        step_id = node['id']
        if step_id == TARGET_NODE:
            node.update(mark=None, status='done' if (result or {}).get('target_model')
                        else 'not_run', stale=any(stale.values()), counts=target_counts,
                        parameters=[v for v in values if v['origin'] == CALIBRATED
                                    and v['from_step'] is not None], inputs=[])
            node['lines'] += _first(
                [f'`{by_name[v["name"]].get("parameterisation_name") or v["name"]}` = '
                 f'{_quantity(v["value"], v["units"])}, '
                 + _set_by(v['origin'], v['from_step'], v['reason'],
                           data_names.get(v['from_step']))
                 for v in values if v['from_step']], DRAWN_VALUES, 'merged values')
            continue
        step_result = run['steps'].get(step_id)
        done = _done(step_result)
        mark = {'step': CALIBRATED, 'values': SET_BY_HAND}.get(node['kind'])
        if by_id[step_id].get('by_hand'):
            mark = SET_BY_HAND
        node.update(mark=mark if done else None,
                    status='done' if done else 'not_run', stale=stale[step_id],
                    counts=_step_counts(step_result) if done else {},
                    parameters=_step_values(step_id, step_result) if done else [],
                    inputs=inputs[step_id])
        if done and node['kind'] == 'values':
            node['lines'] = [_set_record_line(r) for r in step_result.get('set', [])] + \
                [flowchart.set_line(t) for t in step_result.get('thetas', [])]
            if by_id[step_id].get('point'):
                node['lines'].append(flowchart.POINT_ONLY)
        if done and 'n_points' in step_result:
            points = step_result['n_points']
            node['lines'].append(f'{points} point{"" if points == 1 else "s"}, '
                                 f'{step_result["n_failed"]} failed')
        written = []
        if done:
            written = [p for p in [step_result.get('sweep_results')]
                       + list(step_result.get('plots') or [])
                       + list(step_result.get('figures') or []) if p]
        if written:
            figures.append(_figure_node(step_id, written, run['output_dir'], node['stale']))
            edges.append({'from': step_id, 'to': figures[-1]['id'], 'label': 'writes',
                          'style': 'writes'})
        observables = by_id[step_id].get('n_data_items', 0)
        helpers = by_id[step_id].get('n_helper_items', 0)
        if done and observables + helpers:
            node['counts'].update(observables=observables, helper_items=helpers)
        obs_data = next((e for e in inputs[step_id] if e['key'] == 'obs_data'), None)
        if obs_data:
            node['lines'].append(f'data `{os.path.basename(obs_data["path"])}`')
        if done and node['kind'] != 'values':
            node['lines'] += _step_value_lines(step_result, data_names.get)
        if any(e['key'] == 'module_library' and e['status'] != CURRENT for e in inputs[step_id]):
            node['lines'].append('module library changed since it ran')

    source_values = [v for v in values if v['origin'] in (INHERITED, UNCLASSIFIED)]
    source_counts = _counts(source_values)
    target_line = flowchart.model_line(_target_model(run))
    source = {'id': SOURCE_NODE, 'title': 'Model source', 'kind': 'source',
              'mark': _mark(source_counts),
              'lines': [target_line] +
                       [f'{source_counts[m]} {_MARK_LABELS[m]}' for m in (INHERITED, UNCLASSIFIED)
                        if source_counts[m]] +
                       _first([_fixed_line(v) for v in source_values], DRAWN_VALUES,
                              'values of the model source'),
              'status': 'done', 'stale': False, 'counts': source_counts,
              'parameters': source_values, 'inputs': []}
    extra = [source]
    generates = []
    for raw in run['workflow'].get('steps', []):
        model = _generated_model(raw, run['steps'].get(raw.get('id')))
        if model is None:
            continue
        line = flowchart.model_line(model)
        node_id = SOURCE_NODE if 'select' in model or line == target_line else next(
            (n['id'] for n in extra if n['lines'][0] == line), None)
        if node_id is None:
            # a step model that is not the target's has a source box of its own
            node_id = f'{SOURCE_NODE}{len(extra)}'
            extra.append({'id': node_id, 'title': 'Model source', 'kind': 'source',
                          'mark': None, 'lines': [line], 'status': 'done', 'stale': False,
                          'counts': {}, 'parameters': [], 'inputs': []})
        generates.append({'from': node_id, 'to': raw['id'], 'label': 'generates',
                          'style': 'order'})
    edges = generates + edges
    edges += [{'from': SOURCE_NODE, 'to': s['id'], 'label': 'scaled from', 'style': 'order'}
              for s in steps if any(entry.get('of') == INHERITED_BASE
                                    for entry in s.get('set') or [])]
    given_to = _given_to(edges)
    for node_id, title, origin in ((MEASURED_NODE, 'Measured values', MEASURED),
                                   (DERIVED_NODE, 'Derived values', DERIVED),
                                   (HAND_NODE, 'Values set by hand', SET_BY_HAND),
                                   (EARLIER_NODE, 'Values calibrated before this run',
                                    CALIBRATED)):
        # a step's values (calibrated, or declared by a set step) are in its own box
        given = [v for v in values if v['origin'] == origin and not v['from_step']]
        if given:
            extra.append(_values_node(node_id, title, origin, given))
            edges.append({'from': node_id, 'to': given_to, 'label': _MARK_LABELS[origin],
                          'style': 'given'})
    constants = _fixed_constants(rows, values)
    if constants:
        counts = _counts(constants)
        # first in its row, so its arrow to the derived values (first in the next) crosses none
        extra.insert(0, {'id': FIXED_NODE, 'title': 'Fixed constants of derivations',
                         'kind': 'values', 'mark': _mark(counts),
                         'lines': [_fixed_line(v) for v in constants], 'status': 'done',
                         'stale': False, 'counts': counts, 'parameters': constants, 'inputs': []})
        # a constant comes from a derived value the target uses, so the derived box exists
        edges.append({'from': FIXED_NODE, 'to': DERIVED_NODE, 'label': 'constants of',
                      'style': 'given'})
    return {
        'schema_version': SCHEMA_VERSION,
        'workflow': {'name': run['workflow'].get('workflow_name'), 'output_dir': run['output_dir'],
                     'file': (result or {}).get('workflow_file'), 'file_status': file_status,
                     'order': (result or {}).get('order') or [s['id'] for s in steps],
                     'complete': bool((result or {}).get('complete')),
                     'finished': (result or {}).get('finished'),
                     'stale': any(stale.values())},
        'producer': {'name': 'libcuflynx',
                     'version': (result or {}).get('libcuflynx_version'),
                     'git': (result or {}).get('libcuflynx_git') or {'commit': None,
                                                                     'dirty': None}},
        'code': {key: (result or {}).get(key) for key in
                 ('libcuflynx_version', 'libcuflynx_git', 'project_git', 'libcellml_version')},
        'counts': target_counts, 'nodes': extra + nodes + figures, 'edges': edges}


def _given_to(edges):
    '''The node the boxes of values given in the parameters files (measured, derived, set by
    hand, calibrated before the run) point to: the model source, since those values are in the
    model of every step generated from it; the target when no step comes from the source.'''
    return SOURCE_NODE if any(e['from'] == SOURCE_NODE for e in edges) else TARGET_NODE


# -- drawing -----------------------------------------------------------------------------------

_BOX_WIDTH = 270
_GAP_X = 36
_GAP_Y = 58
_LINE = 17
_PADDING = 12
_WRAP = 40
_MARGIN = 16
# an arrow past a row runs down a lane to the right of the boxes, each with its own lane
_LANE_GAP = 14
_LANE_LABEL_WIDTH = 96
_MIN_WIDTH = 640
# characters of footer text per unit of drawing width
_FOOTER_CHARS_PER_UNIT = 0.15


def _text(x, y, text, size=12, weight='normal', fill='currentColor', anchor='start'):
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{escape(text)}</text>')


def _box_lines(node):
    lines = list(node['lines'])
    if node['kind'] == 'values' and len(lines) > DRAWN_VALUES:
        lines = lines[:DRAWN_VALUES] + [f'and {len(lines) - DRAWN_VALUES} more in the graph file']
    return [piece for line in lines
            for piece in textwrap.wrap(flowchart.plain_text(line), _WRAP) or ['']]


def _box_height(node):
    return 2 * _PADDING + 2 * _LINE + (len(_box_lines(node)) + 1) * _LINE


def _feeds_target(nodes, edges):
    '''The ids of the target and of every box with a path to it.'''
    incoming = {}
    for edge in edges:
        incoming.setdefault(edge['to'], []).append(edge['from'])
    feeds, todo = set(), [TARGET_NODE] if any(n['id'] == TARGET_NODE for n in nodes) else []
    while todo:
        node_id = todo.pop()
        if node_id not in feeds:
            feeds.add(node_id)
            todo.extend(incoming.get(node_id, []))
    return feeds


def _output_levels(nodes, edges, level):
    '''``level`` with every output (a box with no path to the target: a sweep, a figure step,
    the files they wrote) at the target's row or below, so it is drawn beside or after the
    target rather than among the steps the target is built from.'''
    feeds = _feeds_target(nodes, edges)
    if not feeds:
        return level
    floor = level[TARGET_NODE]
    incoming = {}
    for edge in edges:
        incoming.setdefault(edge['to'], []).append(edge['from'])
    placed = {}
    for node_id in sorted(level, key=level.get):
        below = 1 + max((placed[p] for p in incoming.get(node_id, [])), default=-1)
        placed[node_id] = level[node_id] if node_id in feeds else max(floor, below)
    return placed


def _layout(graph):
    '''``{node id: (x, y, height, row)}``, ``{(from, to): lane x}`` for the arrows that pass a
    row, and the drawing's width and height.'''
    nodes, edges = graph['nodes'], graph['edges']
    level = _output_levels(nodes, edges, flowchart.levels(nodes, edges))
    rows = {}
    for node in nodes:
        rows.setdefault(level[node['id']], []).append(node)
    # in a row, the boxes the target is built from first, then the outputs, right of the target
    # and each under what it reads
    feeds = _feeds_target(nodes, edges)
    incoming, column = {}, {}
    for edge in edges:
        incoming.setdefault(edge['to'], []).append(edge['from'])
    for depth in sorted(rows):
        outputs = [n for n in rows[depth] if n['id'] not in feeds]
        outputs.sort(key=lambda n: max((column[p] for p in incoming.get(n['id'], [])
                                        if p in column), default=-1))
        rows[depth] = [n for n in rows[depth] if n['id'] in feeds] + outputs
        column.update({n['id']: i for i, n in enumerate(rows[depth])})
    widest = max(len(row) for row in rows.values())
    boxes_width = 2 * _MARGIN + widest * _BOX_WIDTH + (widest - 1) * _GAP_X
    long_edges = [(e['from'], e['to']) for e in edges if level[e['to']] - level[e['from']] > 1]
    lanes = {edge: boxes_width + i * _LANE_GAP for i, edge in enumerate(long_edges)}
    width = max(_MIN_WIDTH, boxes_width + len(lanes) * _LANE_GAP
                + (_LANE_LABEL_WIDTH if lanes else 0))
    places, y = {}, _MARGIN
    for depth in sorted(rows):
        row = rows[depth]
        x = (boxes_width - len(row) * _BOX_WIDTH - (len(row) - 1) * _GAP_X) / 2
        height = max(_box_height(node) for node in row)
        for node in row:
            # a box of files has nothing below it, so it takes its own height, not the row's
            own = _box_height(node) if node['kind'] == FIGURE else height
            places[node['id']] = (x, y, own, depth)
            x += _BOX_WIDTH + _GAP_X
        y += height + _GAP_Y
    return places, lanes, width, y - _GAP_Y


def _draw_node(node, x, y, height, _row):
    colour = _STALE_COLOUR if node['stale'] else _MARK_COLOURS.get(node['mark'], 'currentColor')
    external = node['kind'] == EXTERNAL
    dash = ' stroke-dasharray="6 4"' if node['stale'] else \
        ' stroke-dasharray="2 4"' if external else ''
    parts = [f'<rect x="{x:.0f}" y="{y:.0f}" width="{_BOX_WIDTH}" height="{height:.0f}" rx="5" '
             f'fill="currentColor" fill-opacity="0.04" stroke="{colour}" stroke-width="2"{dash}/>']
    label = 'outside the workflow' if external else _MARK_LABELS.get(node['mark'], '')
    if external and node['status'] == 'not_run':
        label = f'{label}, files missing'
    elif node['stale']:
        label = f'{label}, stale' if label else 'stale'
    elif node['status'] == 'not_run':
        label = 'not run'
    text_x, line_y = x + _PADDING, y + _PADDING + 11
    if label:
        parts.append(_text(text_x, line_y, label.upper(), size=11, weight='bold', fill=colour))
    line_y += _LINE
    parts.append(_text(text_x, line_y, flowchart.plain_text(node['title']), size=14,
                       weight='bold'))
    for line in _box_lines(node):
        line_y += _LINE
        parts.append(_text(text_x, line_y, line))
    return parts


def _draw_edge(edge, places, lanes):
    x_1, y_1, height, _ = places[edge['from']]
    x_2, y_2, _, _ = places[edge['to']]
    start, end = (x_1 + _BOX_WIDTH / 2, y_1 + height), (x_2 + _BOX_WIDTH / 2, y_2 - 2)
    lane = lanes.get((edge['from'], edge['to']))
    if lane is None:
        bend = (end[1] - start[1]) / 2
        path = (f'M{start[0]:.0f} {start[1]:.0f} C{start[0]:.0f} {start[1] + bend:.0f} '
                f'{end[0]:.0f} {end[1] - bend:.0f} {end[0]:.0f} {end[1]:.0f}')
        # below the lanes' run through this gap
        label_x, label_y = (start[0] + end[0]) / 2 + 6, start[1] + 0.8 * (end[1] - start[1])
    else:
        # through the gaps between rows and down its lane, so it crosses no box
        below = start[1] + _GAP_Y / 3
        above = end[1] - _GAP_Y / 3
        path = (f'M{start[0]:.0f} {start[1]:.0f} V{below:.0f} H{lane:.0f} V{above:.0f} '
                f'H{end[0]:.0f} V{end[1]:.0f}')
        label_x, label_y = lane + 5, (below + above) / 2
    dash = _EDGE_DASHES.get(edge.get('style'))
    dashes = f' stroke-dasharray="{dash}"' if dash else ''
    width = 2.2 if edge.get('style') == 'fixed' else 1.4
    parts = [f'<path d="{path}" fill="none" stroke="currentColor" stroke-width="{width}"{dashes} '
             f'marker-end="url(#provenance-arrow)"/>']
    if edge.get('label'):
        parts.append(_text(label_x, label_y, flowchart.plain_text(edge['label']), size=11))
    return parts


def _git_text(name, state):
    state = state or {}
    if not state.get('commit'):
        return f'{name}: no commit recorded'
    dirty = ' with uncommitted changes' if state.get('dirty') else ''
    return f'{name}: {state["commit"][:10]}{dirty}'


def _summary(graph):
    '''The target's values by what set them, the unclassified always among them, and, when the
    derivations have fixed constants, how many of those are unclassified.'''
    counts = graph['counts']
    parts = [f'{counts.get(mark, 0)} {_MARK_LABELS[mark]}' for mark in MARKS
             if counts.get(mark) or mark == UNCLASSIFIED]
    fixed = next((n for n in graph['nodes'] if n['id'] == FIXED_NODE), None)
    if fixed:
        n = fixed['counts'].get(UNCLASSIFIED, 0)
        parts.append(f'{n} unclassified constant{"" if n == 1 else "s"}')
    return ' · '.join(parts)


class ProvenanceVersionError(ValueError):
    '''A provenance graph has a ``schema_version`` this libcuflynx does not read.'''


def check_schema_version(graph, source='provenance graph'):
    '''Refuse a graph whose ``schema_version`` is not one of ``SUPPORTED_SCHEMA_VERSIONS``.'''
    version = graph.get('schema_version') if isinstance(graph, dict) else None
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ', '.join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)
        raise ProvenanceVersionError(
            f'{source} has schema_version {version!r}; this libcuflynx reads provenance graph '
            f'schema_version {supported} only. Write it again with `cuflynx-provenance '
            f'<output_dir>`, or read it with the libcuflynx that wrote it.')


def read_provenance_graph(path):
    '''The provenance graph in the file at ``path``, after checking its version and schema.'''
    from libcuflynx.schemas import PROVENANCE_GRAPH_SCHEMA
    from libcuflynx.schemas.validation import validate
    with open(path, encoding='utf-8') as f:
        graph = json.load(f)
    check_schema_version(graph, source=path)
    validate(PROVENANCE_GRAPH_SCHEMA, graph, source=path, what='provenance graph')
    return graph


def provenance_svg(graph):
    '''``graph`` (:func:`provenance_graph`) as a self-contained ``<svg>`` to place inline in a
    page: no scripts or ``<style>`` element, text and lines in the page's ``currentColor``.
    Refuses a graph of a ``schema_version`` it does not read.'''
    check_schema_version(graph)
    places, lanes, width, height = _layout(graph)
    footer = [f'Values of the target: {_summary(graph)}',
              f'Output directory: {graph["workflow"]["output_dir"]}',
              _git_text('libcuflynx', graph['code'].get('libcuflynx_git')),
              _git_text('project', graph['code'].get('project_git')),
              f'libCellML {graph["code"].get("libcellml_version")}']
    producer = graph.get('producer') or {}
    if producer.get('name') and producer['name'] != 'libcuflynx':
        footer.insert(2, _git_text(f'{producer["name"]} {producer.get("version") or ""}'.strip(),
                                   producer.get('git')))
    characters = int((width - 2 * _MARGIN) * _FOOTER_CHARS_PER_UNIT)
    footer = [piece for line in footer
              for piece in textwrap.wrap(line, characters, break_long_words=True,
                                         break_on_hyphens=False)]
    total = height + _LINE * (len(footer) + 1)
    claim = (f'Provenance of calibration workflow {graph["workflow"]["name"]}: '
             f'{_summary(graph)}.')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {total:.0f}" '
             f'role="img" aria-label={quoteattr(claim)} font-family="sans-serif" '
             f'style="max-width:100%;height:auto">',
             '<defs><marker id="provenance-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto">'
             '<path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>']
    for edge in graph['edges']:
        parts += _draw_edge(edge, places, lanes)
    for node in graph['nodes']:
        parts += _draw_node(node, *places[node['id']])
    y = height + _LINE
    for line in footer:
        y += _LINE
        parts.append(_text(_MARGIN, y - _LINE / 2, line, size=12))
    parts.append('</svg>')
    return '\n'.join(parts)


def write_provenance(output_dir):
    '''Write ``provenance_graph.json`` and ``provenance.svg`` into the run in ``output_dir``;
    returns their paths.'''
    from libcuflynx.schemas import PROVENANCE_GRAPH_SCHEMA
    from libcuflynx.schemas.validation import validate
    graph = provenance_graph(output_dir)
    graph_path = os.path.join(graph['workflow']['output_dir'], GRAPH_FILE)
    validate(PROVENANCE_GRAPH_SCHEMA, graph, source=graph_path, what='provenance graph')
    svg_path = os.path.join(graph['workflow']['output_dir'], SVG_FILE)
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=1)
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(provenance_svg(graph) + '\n')
    return graph_path, svg_path
