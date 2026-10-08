'''
TikZ flow charts of a calibration: the steps of a calibration workflow, or the stages of a
single calibration.

Each node is a dict ``{'id', 'title', 'lines', 'kind'}``; each edge ``{'from', 'to',
'label', 'style'}``. :func:`tikz` lays them out top to bottom by dependency depth, side by
side within a depth, and needs only ``\\usetikzlibrary{arrows.meta}`` in the preamble.
'''

from libcuflynx.reporting.latex_names import escape

NODE_STYLES = {
    'step': 'draw, rounded corners, fill=blue!6',
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


def _levels(nodes, edges):
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
    level = _levels(nodes, edges)
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
            body = r'\textbf{' + n['title'] + '}'
            if n.get('lines'):
                body += r'\\ ' + r'\\ '.join(n['lines'])
            style = NODE_STYLES.get(n.get('kind', 'step'), NODE_STYLES['step'])
            lines.append(f'  \\node[box, {style}] ({_tikz_id(n["id"])}) at ({x:.2f},{y:.2f}) '
                         f'{{{body}}};')
    for e in edges:
        style = EDGE_STYLES.get(e.get('style', 'order'), EDGE_STYLES['order'])
        label = (f' node[midway, fill=white, inner sep=1pt, font=\\scriptsize] '
                 f'{{{e["label"]}}}') if e.get('label') else ''
        lines.append(f'  \\draw[{style}] ({_tikz_id(e["from"])}) -- '
                     f'({_tikz_id(e["to"])}){label};')
    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)


def width_cm(nodes, edges, x_gap=5.2, node_cm=4.6):
    '''Roughly how wide :func:`tikz` draws ``nodes``: its widest row.'''
    level = _levels(nodes, edges)
    widest = max(list(level.values()).count(d) for d in set(level.values()))
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


def workflow_graph(steps):
    '''Nodes and edges for a workflow. ``steps`` are dicts ``{'id', 'target',
    'submodule_path', 'n_parameters', 'n_data_items', 'settings', 'depends_on',
    'fixed_from', 'priors_from' (step ids), 'store_distribution'}`` plus ``target`` (the
    workflow's target dict) and ``n_merged`` in a final dict with ``'id': None``.'''
    *step_list, target = steps
    nodes, edges = [], []
    for s in step_list:
        t = s['target']
        lines = [r'\texttt{' + escape(t['module_type']) + '}, instance \\texttt{'
                 + escape(t['instance']) + '}',
                 f'{_count(s["n_parameters"], "parameter")}, '
                 f'{_count(s["n_data_items"], "observable")}',
                 escape(_method_label(s['settings']))]
        if s.get('store_distribution'):
            lines.append('then MCMC: posterior stored')
        nodes.append({'id': s['id'], 'title': escape(s['id']), 'lines': lines,
                      'kind': 'step'})
        for dep in s['depends_on']:
            kinds = []
            if dep in s['fixed_from']:
                kinds.append('fixed')
            if dep in s['priors_from']:
                kinds.append('prior')
            edges.append({'from': dep, 'to': s['id'], 'label': ' + '.join(kinds),
                          'style': 'prior' if 'prior' in kinds else
                          ('fixed' if kinds else 'order')})
    t = target['target']
    nodes.append({'id': '__target__', 'title': 'Merged: ' + escape(t['module_type']),
                  'lines': [r'instance \texttt{' + escape(t['instance']) + '}',
                            f'{_count(target["n_merged"], "calibrated value")}'],
                  'kind': 'target'})
    referenced = {d for s in step_list for d in s['depends_on']}
    for s in step_list:
        if s['id'] not in referenced:       # the last step(s) of each branch
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
         'lines': [escape(_method_label(settings))]},
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
