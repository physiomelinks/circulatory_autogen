'''
Flow charts of a calibration: the steps of a calibration workflow, or the stages of a single
calibration.

Each node is a dict ``{'id', 'title', 'lines', 'kind'}``; each edge ``{'from', 'to',
'label', 'style'}``. Their text is plain, with code between backticks, so any renderer can draw
them. :func:`tikz` lays them out top to bottom by dependency depth, side by side within a depth,
and needs only ``\\usetikzlibrary{arrows.meta}`` in the preamble;
:mod:`libcuflynx.reporting.provenance` draws the same graph as SVG.
'''

from libcuflynx.reporting.latex_names import escape

NODE_STYLES = {
    'step': 'draw, rounded corners, fill=blue!6',
    'analysis': 'draw, rounded corners, fill=yellow!10',
    'target': 'draw, rounded corners, very thick, fill=green!8',
    'stage': 'draw, rounded corners, fill=gray!8',
    'data': 'draw, fill=orange!8',
}
EDGE_STYLES = {
    'order': '-{Stealth}',
    'fixed': '-{Stealth}, thick',
    'prior': '-{Stealth}, dashed, thick',
    'merge': '-{Stealth}, dotted',
}


def latex(text):
    '''A graph's plain ``text`` as LaTeX: escaped, with what is between backticks in ``\\texttt``.'''
    parts = str(text).split('`')
    return ''.join(r'\texttt{' + escape(part) + '}' if i % 2 else escape(part)
                   for i, part in enumerate(parts))


def plain_text(text):
    '''A graph's ``text`` without its code marks.'''
    return str(text).replace('`', '')


def levels(nodes, edges):
    '''Depth of each node: 0 for a node nothing points to, else one more than its deepest
    predecessor.'''
    incoming = {n['id']: [e['from'] for e in edges if e['to'] == n['id']] for n in nodes}
    level = {}

    def depth(node_id, seen=()):
        if node_id in level:
            return level[node_id]
        if node_id in seen:
            raise ValueError(f'the flow chart has a cycle through {node_id}')
        level[node_id] = 1 + max((depth(p, seen + (node_id,)) for p in incoming[node_id]),
                                 default=-1)
        return level[node_id]

    for n in nodes:
        depth(n['id'])
    return level


def _tikz_id(node_id):
    return 'n' + ''.join(ch if ch.isalnum() else 'x' for ch in str(node_id))


def tikz(nodes, edges, x_gap=5.2, y_gap=2.6, text_width='4.2cm'):
    '''The ``tikzpicture`` for ``nodes`` and ``edges``.'''
    level = levels(nodes, edges)
    rows = {}
    for n in nodes:
        rows.setdefault(level[n['id']], []).append(n)
    width = max(len(r) for r in rows.values())
    lines = [r'\begin{tikzpicture}[>=Stealth, every node/.style={font=\small},',
             f'  box/.style={{align=center, text width={text_width}, inner sep=4pt}}]']
    for depth in sorted(rows):
        row = rows[depth]
        offset = (width - len(row)) / 2.0
        for i, n in enumerate(row):
            x = (offset + i) * x_gap
            y = -depth * y_gap
            body = r'\textbf{' + latex(n['title']) + '}'
            if n.get('lines'):
                body += r'\\ ' + r'\\ '.join(latex(line) for line in n['lines'])
            style = NODE_STYLES.get(n.get('kind', 'step'), NODE_STYLES['step'])
            lines.append(f'  \\node[box, {style}] ({_tikz_id(n["id"])}) at ({x:.2f},{y:.2f}) '
                         f'{{{body}}};')
    for e in edges:
        style = EDGE_STYLES.get(e.get('style', 'order'), EDGE_STYLES['order'])
        label = (f' node[midway, fill=white, inner sep=1pt, font=\\scriptsize] '
                 f'{{{latex(e["label"])}}}') if e.get('label') else ''
        lines.append(f'  \\draw[{style}] ({_tikz_id(e["from"])}) -- '
                     f'({_tikz_id(e["to"])}){label};')
    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)


def width_cm(nodes, edges, x_gap=5.2, node_cm=4.6):
    '''Roughly how wide :func:`tikz` draws ``nodes``: its widest row.'''
    level = levels(nodes, edges)
    widest =max(list(level.values()).count(d) for d in set(level.values()))
    return (widest - 1) * x_gap + node_cm


def figure(picture, caption, label, natural_width_cm=None, max_width_cm=15.0):
    '''A floating figure around ``picture``, shrunk to the text width only when it is wider
    (stretching a one-column chart to the full width makes it huge).'''
    if natural_width_cm is not None and natural_width_cm <= max_width_cm:
        body = [picture]
    else:
        body = [r'\resizebox{\linewidth}{!}{%', picture, '}']
    return '\n'.join([r'\begin{figure}[htbp]', r'\centering', *body,
                      r'\caption{' + caption + '}', r'\label{' + label + '}',
                      r'\end{figure}'])


def _method_label(settings):
    method = settings.get('param_id_method') or 'genetic_algorithm'
    try:
        from libcuflynx.parsers.PrimitiveParsers import PARAM_ID_METHODS
        return PARAM_ID_METHODS.get(method, {}).get('label') or method
    except ImportError:
        return method


def _count(n, word):
    return f'{n} {word}' + ('' if n == 1 else 's')


def _model_name(model):
    if 'module_array' in model:
        from libcuflynx.calibration_workflow.spec import array_prefix
        return array_prefix(model['module_array'])
    return model['module_type']


def _model_detail(model):
    if 'module_array' in model:
        return 'module array'
    return f'parameterisation `{model["parameterisation"]}`'


def model_line(model):
    '''A workflow model in one line.'''
    if 'select' in model:
        return f'selection `{", ".join(model["select"])}` of the target'
    if 'module_array' in model:
        return f'model `{_model_name(model)}`'
    return f'`{model["module_type"]}`, parameterisation `{model["parameterisation"]}`'


#: what an edge from each source key is labelled
_SOURCE_LABELS = {'fixed_from': 'fixed', 'priors_from': 'prior', 'emulator_from': 'emulator',
                  'result_from': 'at its fit', 'sensitivity_from': 'indices',
                  'params_for_id_from': 'parameters', 'scale_from': 'scaled from'}
#: the kinds whose values are merged into the target
MERGED_KINDS = ('calibration', 'set')


def _kind_line(s):
    '''What a step that is not a calibration does, in one line.'''
    settings, kind = s.get('settings') or {}, s.get('kind')
    if kind == 'sensitivity':
        method = (settings.get('sa_options') or {}).get('method', 'sobol')
        return f'{method.capitalize()} sensitivity analysis'
    if kind == 'emulation':
        return 'emulator trained'
    if kind == 'identifiability':
        method = (settings.get('ia_options') or {}).get('method', 'Laplace')
        return f'{method} identifiability analysis'
    return _method_label(settings) + (' on the emulator' if s.get('emulator_from') else '')


def set_line(entry):
    '''One entry of a set step, with its reason.'''
    how = (f'= {entry["value"]}' if 'value' in entry
           else f'scaled {entry["scale"]} times `{entry["of"]}`')
    return f'`{entry["parameter"]}` {how}: {entry["reason"]}'


def _sweep_lines(s):
    sweep = s.get('sweep') or {}
    design = 'one at a time' if sweep.get('design') == 'one_at_a_time' else 'grid'
    return [model_line(s['model']),
            f'{_count(len(sweep.get("parameters") or {}), "parameter")} swept, {design}',
            'feeds nothing forward']


def workflow_graph(steps):
    '''Nodes and edges for a workflow. ``steps`` are dicts ``{'id', 'kind', 'model',
    'n_parameters', 'n_data_items', 'settings', 'depends_on', 'fixed_from', 'priors_from' (step
    ids), 'store_distribution'}``, with ``emulator_from``, ``result_from``,
    ``sensitivity_from``, ``params_for_id_from``, ``scale_from`` and (a reduction, set or sweep
    step) ``reduction``, ``set`` or ``sweep`` where a step has them, plus ``model`` (the
    workflow's target), ``n_merged``, optionally ``n_set`` (the merged values set by hand, the
    rest being calibrated) and optionally ``merged_from`` (the steps whose values reached the
    target, by default every calibration and set step) in a final dict with ``'id': None``. A
    model is a library module ``{module_type, version, parameterisation}``, ``{module_array}`` or
    ``{select}``; a reduction or set step has none, and any other step given ``'model': None``
    has not run.'''
    *step_list, target = steps
    nodes, edges = [], []
    for s in step_list:
        kind = s.get('kind', 'calibration')
        if kind == 'reduction':
            options = s.get('reduction') or {}
            lines = [f'keeps the top {options.get("top_n")} by Sobol '
                     f'{options.get("index_type", "ST")}']
        elif kind == 'set':
            lines = [set_line(entry) for entry in s.get('set') or []]
        elif s.get('model') is None:
            lines = ['not run']
        elif kind == 'sweep':
            lines = _sweep_lines(s)
        else:
            lines = [model_line(s['model']),
                     f'{_count(s["n_parameters"], "parameter")}, '
                     f'{_count(s["n_data_items"], "observable")}',
                     _kind_line(s)]
        if s.get('store_distribution'):
            lines.append('then MCMC: posterior stored')
        nodes.append({'id': s['id'], 'title': s['id'], 'lines': lines,
                      'kind': {'calibration': 'step', 'set': 'values'}.get(kind, 'analysis')})
        for dep in s['depends_on']:
            kinds = [label for key, label in _SOURCE_LABELS.items()
                     if dep in (s.get(key) or []) or dep == s.get(key)]
            edges.append({'from': dep, 'to': s['id'], 'label': ' + '.join(kinds),
                          'style': 'prior' if 'prior' in kinds else
                          ('fixed' if 'fixed' in kinds else 'order')})
    t = target['model']
    n_set = target.get('n_set', 0)
    nodes.append({'id': '__target__', 'title': 'Merged: ' + _model_name(t),
                  'lines': [_model_detail(t),
                            f'{_count(target["n_merged"] - n_set, "calibrated value")}']
                           + ([f'{_count(n_set, "value")} set by hand'] if n_set else []),
                  'kind': 'target'})
    # every calibration and set step is merged into the target; a later one wins a parameter
    # they share
    merged_from = target.get('merged_from')
    for s in step_list:
        if s.get('kind', 'calibration') in MERGED_KINDS and \
                (merged_from is None or s['id'] in merged_from):
            edges.append({'from': s['id'], 'to': '__target__', 'label': '',
                          'style': 'merge'})
    return nodes, edges


def single_calibration_graph(n_parameters, n_data_items, n_experiments, settings,
                             do_uq=False, do_ia=False, use_emulator=False, model_label=''):
    '''Nodes and edges for one calibration (the cuflynx-param-id stages).'''
    nodes = [
        {'id': 'model', 'title': 'Model', 'kind': 'stage',
         'lines': [model_label or 'generated CellML']},
        {'id': 'data', 'title': 'Observables', 'kind': 'data',
         'lines': [f'{_count(n_data_items, "data item")}, '
                   f'{_count(n_experiments, "experiment")}']},
        {'id': 'params', 'title': 'Parameters', 'kind': 'data',
         'lines': [_count(n_parameters, 'calibrated parameter')]},
        {'id': 'fit', 'title': 'Optimisation', 'kind': 'step',
         'lines': [_method_label(settings)]},
    ]
    edges = [{'from': 'model', 'to': 'fit'}, {'from': 'data', 'to': 'fit'},
             {'from': 'params', 'to': 'fit'}]
    if use_emulator:
        nodes.append({'id': 'emulator', 'title': 'Emulator', 'kind': 'stage',
                      'lines': ['surrogate of the observables']})
        edges = [e for e in edges if e['from'] != 'model'] + [
            {'from': 'model', 'to': 'emulator'}, {'from': 'emulator', 'to': 'fit'}]
    last = 'fit'
    if do_uq:
        nodes.append({'id': 'uq', 'title': 'MCMC', 'kind': 'step',
                      'lines': ['posterior from the best fit']})
        edges.append({'from': last, 'to': 'uq'})
        last = 'uq'
    if do_ia:
        nodes.append({'id': 'ia', 'title': 'Identifiability', 'kind': 'step',
                      'lines': ['about the best fit']})
        edges.append({'from': 'fit', 'to': 'ia'})
    nodes.append({'id': 'out', 'title': 'Calibrated model', 'kind': 'target', 'lines': []})
    edges.append({'from': last, 'to': 'out'})
    return nodes, edges
