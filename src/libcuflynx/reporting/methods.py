'''
A methods section in LaTeX: the model's equations module by module, a flow chart of the
calibration, and what was done in each step of it.

Two sources:

* :func:`workflow_methods_latex` -- a calibration workflow (``calibration_workflow.json``):
  the model is the workflow's target supermodule, and every step of the workflow is a node
  of the flow chart and a subsection;
* :func:`calibration_methods_latex` -- one calibration (a model, its obs_data and
  params_for_id, and the param_id settings).

Either reads results when given a run (a workflow's output directory, or a param_id run
directory) and reports them. Symbols come from a variable mapping
(``<prefix>_variable_mapping.csv``, see ``variable_mapping.py``).
'''

import csv
import json
import os
import tempfile

from libcuflynx.reporting import flowchart
from libcuflynx.reporting.latex_names import escape
from libcuflynx.reporting.model_latex import format_number, format_units, module_sections
from libcuflynx.reporting.variable_mapping import (default_mapping_path, load_model,
                                                   module_components, strip_module,
                                                   variable_mapping)

PREAMBLE = r'''\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\allowdisplaybreaks
% long \texttt names (option keys, file names) cannot hyphenate; let lines stretch instead
\setlength{\emergencystretch}{3em}
'''
PACKAGES_NOTE = ('% Needs: amsmath, amssymb, booktabs, graphicx, tikz '
                 '(\\usetikzlibrary{arrows.meta}); \\allowdisplaybreaks and '
                 '\\setlength{\\emergencystretch}{3em} recommended.')

#: what each cost function scores, for an item with prediction y-hat, data y, std sigma and
#: weight w (series are averaged over their points). Matches funcs/cost_funcs_user.py.
COST_DESCRIPTIONS = {
    'gaussian_MLE': (r'the Gaussian negative log-likelihood (up to a constant), '
                     r'$\frac{1}{2}\,w\left(\frac{\hat{y}-y}{\sigma}\right)^2$'),
    'MSE': (r'the weighted squared error in units of the standard deviation, '
            r'$w\left(\frac{\hat{y}-y}{\sigma}\right)^2$'),
    'AE': (r'the weighted absolute error in units of the standard deviation, '
           r'$w\left|\frac{\hat{y}-y}{\sigma}\right|$'),
    'gaussian_MLE_robust': (r'the negative log-likelihood of a Gaussian mixed with a uniform '
                            r'outlier component, $-\log\left[(1-\epsilon)\,\mathcal{N}(y;\hat{y},'
                            r'\sigma^2) + \epsilon/W\right]$ (weighted by $w$)'),
    'poisson_MLE': (r'the Poisson negative log-likelihood of the observed count $k$, '
                    r'$w\,(\lambda - k\log\lambda)$ with rate $\lambda=\hat{y}$'),
}

PRIOR_DESCRIPTIONS = {
    'mvnormal': ('a multivariate normal fitted to the stored posterior samples (means and '
                 'full covariance), on a scale on which each parameter\'s bounds are '
                 'unbounded (logit for a bounded parameter, log for one bounded below)'),
    'normal': ('independent normals fitted to each parameter of the stored posterior, on a '
               'bounds-respecting (logit or log) scale'),
    'kde': ('a Gaussian kernel density estimate of the stored posterior samples, on a '
            'bounds-respecting (logit or log) scale'),
}


def _math(latex):
    return f'${latex}$'


def _sentence(text):
    text = escape(str(text).strip())
    return text if text.endswith(('.', '!', '?')) else text + '.'


def _tt(text):
    return r'\texttt{' + escape(text) + '}'


def _number(value):
    try:
        return _math(format_number(float(value)))
    except (TypeError, ValueError):
        return escape(str(value))


class Symbols:
    '''Names as they appear in a step -- ``vessel/variable`` operands and params_for_id
    rows -- written with the target model's symbols.'''

    def __init__(self, rows, path=''):
        self.latex = {r['variable_name']: r['latex'] for r in rows}
        self.path = path or ''

    def _vessel(self, vessel):
        from libcuflynx.calibration_workflow.naming import map_vessel
        try:
            return map_vessel(vessel, self.path) if self.path else vessel
        except ValueError:
            return vessel

    def variable(self, name):
        '''An obs_data operand ``vessel/variable``.'''
        vessel, sep, variable = str(name).partition('/')
        if not sep:
            return _tt(name)
        canonical = f'{strip_module(self._vessel(vessel))}/{variable}'
        latex = self.latex.get(canonical) or self.latex.get(f'parameters/{variable}_{vessel}')
        return _math(latex) if latex else _tt(name)

    def parameter(self, vessel, param):
        from libcuflynx.parsers.PrimitiveParsers import param_name_for_gen
        model_name = param_name_for_gen(self._vessel(vessel), param)
        for key in (f'parameters/{model_name}', f'parameters_global/{model_name}'):
            if key in self.latex:
                return _math(self.latex[key])
        return _tt(f'{vessel}/{param}')

    def model_name(self, name):
        '''A generated-model parameter name (``p_mod_A``).'''
        for key in (f'parameters/{name}', f'parameters_global/{name}'):
            if key in self.latex:
                return _math(self.latex[key])
        return _tt(name)


# --------------------------------------------------------------------------------------------
# reading a step's inputs
# --------------------------------------------------------------------------------------------

def read_params_for_id(path):
    '''params_for_id rows as dicts: ``targets`` [(vessel, param)], ``min``, ``max``, ``prior``
    and its settings.'''
    rows = []
    if str(path).lower().endswith('.json'):
        with open(path) as f:
            doc = json.load(f)
        for entry in doc.get('params', doc if isinstance(doc, list) else []):
            targets = [tuple(t.split('/', 1)) for t in entry.get('targets', [])]
            prior = entry.get('prior') or 'uniform'
            rows.append({'targets': targets, 'min': entry.get('min'), 'max': entry.get('max'),
                         'prior': prior, 'prior_params': entry.get('prior_params') or {}})
        return rows
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            vessels = (row.get('vessel_name') or '').split()
            param = (row.get('param_name') or '').strip()
            if not vessels or not param:
                continue
            prior_params = {k: row[k] for k in row
                            if k and k.startswith('prior_') and (row[k] or '').strip()}
            rows.append({'targets': [(v, param) for v in vessels], 'min': row.get('min'),
                         'max': row.get('max'), 'prior': (row.get('prior') or 'uniform').strip()
                         or 'uniform', 'prior_params': prior_params})
    return rows


def _load_json(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------
# the parts of a step's subsection
# --------------------------------------------------------------------------------------------

def protocol_text(obs):
    info = obs.get('protocol_info') or {}
    pre = info.get('pre_times') or []
    sims = info.get('sim_times') or []
    if not sims:
        return 'The obs\\_data sets no protocol of its own; the run settings give the time window.'
    labels = info.get('experiment_labels') or []
    changes = info.get('params_to_change') or {}
    lines = [r'\begin{center}', r'\small', r'\begin{tabular}{llll}', r'\toprule',
             r'Experiment & Pre-time & Segment durations & Inputs set per segment \\',
             r'\midrule']
    for e, segments in enumerate(sims):
        label = labels[e] if e < len(labels) else ''
        set_here = []
        for name, values in changes.items():
            per_exp = values[e] if e < len(values) else None
            if per_exp is None:
                continue
            shown = ', '.join(escape(v) if isinstance(v, str) else format_number(v)
                              for v in (per_exp if isinstance(per_exp, list) else [per_exp]))
            set_here.append(f'{_tt(name)}: {shown}')
        lines.append(f'{e}{(" (" + escape(label) + ")") if label else ""} & '
                     f'{_number(pre[e]) if e < len(pre) else "--"} & '
                     f'{", ".join(format_number(s) for s in segments)} & '
                     f'{"; ".join(set_here) or "--"} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    n = len(sims)
    return (f'The protocol has {n} experiment{"s" if n != 1 else ""}; each starts with an '
            'unrecorded pre-time and then runs its segments in turn.\n' + '\n'.join(lines))


def _feature(item, symbols):
    operands = item.get('operands') or ([item['variable']] if item.get('variable') else [])
    shown = ', '.join(symbols.variable(o) for o in operands)
    op = item.get('operation')
    if op:
        kwargs = item.get('operation_kwargs') or {}
        extra = ', '.join(f'{escape(k)}={escape(v)}' for k, v in kwargs.items())
        return f'{_tt(op)}({shown}{", " + extra if extra else ""})'
    return shown or '--'


def _observed(item):
    if item.get('prob_dist_params'):
        return 'distribution: ' + escape(json.dumps(item['prob_dist_params']))
    value, std = item.get('value'), item.get('std')
    if isinstance(value, list):
        return f'series ({len(value)} points)'
    if value is None:
        return 'series from file' if item.get('value_path') or item.get('t_path') else '--'
    text = format_number(value)
    if isinstance(std, (int, float)):
        text += r' \pm ' + format_number(std)
    return _math(text)


def observables_text(obs, symbols, default_cost):
    items = obs.get('data_items') or obs.get('data_item') or (obs if isinstance(obs, list) else [])
    if not items:
        return 'No data items.'
    lines = [r'\begin{center}', r'\small', r'\begin{tabular}{lllll}', r'\toprule',
             r'Observable & Feature & Observed & Weight & Exp./seg. \\', r'\midrule']
    costs = set()
    for item in items:
        costs.add(item.get('cost_type') or default_cost)
        unit = item.get('unit')
        observed = _observed(item)
        if unit and unit != 'dimensionless':
            observed += ' ' + format_units(unit)
        where = f'{item.get("experiment_idx", 0)}/{item.get("subexperiment_idx", "--")}'
        lines.append(f'{escape(item.get("data_item_name") or item.get("variable") or "")} & '
                     f'{_feature(item, symbols)} & {observed} & '
                     f'{format_number(item.get("weight", 1.0))} & {where} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    sources = sorted({str(i['source']) for i in items if i.get('source')})
    text = '\n'.join(lines)
    if obs.get('reference_key') if isinstance(obs, dict) else None:
        text = f'Data: {escape(obs["reference_key"])}.\n' + text
    if sources:
        text += '\nSources: ' + '; '.join(escape(s) for s in sources) + '.'
    predictions = obs.get('prediction_items') if isinstance(obs, dict) else None
    if predictions:
        text += (f'\n{len(predictions)} further item'
                 f'{"s are" if len(predictions) != 1 else " is"} predicted but not fitted.')
    return text, costs


def parameters_text(entries, symbols):
    lines = [r'\begin{center}', r'\small', r'\begin{tabular}{lll}', r'\toprule',
             r'Parameter & Range & Prior \\', r'\midrule']
    for entry in entries:
        names = ', '.join(symbols.parameter(v, p) for v, p in entry['targets'])
        prior = escape(entry['prior'])
        if entry['prior_params']:
            prior += ' (' + ', '.join(f'{escape(k[len("prior_"):])}={escape(v)}'
                                      for k, v in entry['prior_params'].items()) + ')'
        lines.append(f'{names} & $[{_fmt(entry["min"])},\\,{_fmt(entry["max"])}]$ & '
                     f'{prior} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    return '\n'.join(lines)


def _fmt(value):
    try:
        return format_number(float(value))
    except (TypeError, ValueError):
        return r'\infty' if value in (None, '', 'inf') else escape(str(value))


def method_text(settings):
    from libcuflynx.parsers.PrimitiveParsers import PARAM_ID_METHODS
    method = settings.get('param_id_method') or 'genetic_algorithm'
    meta = PARAM_ID_METHODS.get(method, {})
    label = meta.get('label') or method
    text = f'Parameters were estimated with {escape(label)} ({_tt(method)})'
    if meta.get('description'):
        text += ': ' + escape(meta['description'].rstrip('.')) + '.'
    else:
        text += '.'
    options = dict(settings.get('optimiser_options') or {})
    if settings.get('DEBUG') and settings.get('debug_optimiser_options'):
        options.update(settings['debug_optimiser_options'])
    options.pop('cost_type', None)
    if options:
        text += ' Settings: ' + ', '.join(f'{_tt(k)}~$=$~{escape(v)}'
                                          for k, v in sorted(options.items())) + '.'
    solver = (settings.get('solver_info') or {}).get('solver') or settings.get('solver') \
        or 'CVODE_myokit'
    info = {k: v for k, v in (settings.get('solver_info') or {}).items() if k != 'solver'}
    text += f' The model was solved with {_tt(solver)}'
    if info:
        text += ' (' + ', '.join(f'{_tt(k)}~$=$~{escape(v)}' for k, v in sorted(info.items())) + ')'
    if settings.get('dt'):
        text += f', recording every {_number(settings["dt"])}'
    return text + '.'


def cost_text(costs, settings):
    default = (settings.get('optimiser_options') or {}).get('cost_type') or 'gaussian_MLE'
    costs = sorted({c or default for c in costs}) or [default]
    parts = []
    for cost in costs:
        described = COST_DESCRIPTIONS.get(cost)
        parts.append(f'{_tt(cost)}, ' + described if described else _tt(cost))
    return ('Each observable $i$ contributes ' + '; or '.join(parts)
            + ', and the cost is the mean of these contributions over the weighted observables.')


def uq_text(settings):
    uq = (settings.get('debug_UQ_options') if settings.get('DEBUG') else None) \
        or settings.get('UQ_options') or {}
    library = uq.get('library') or uq.get('sampler') or 'emcee'
    details = ', '.join(f'{_tt(k)}~$=$~{escape(v)}' for k, v in sorted(uq.items())
                        if k not in ('library', 'sampler'))
    return (f'After the fit, the posterior was sampled by MCMC ({_tt(library)}; {details}), '
            'starting from the best fit; the samples after burn-in are stored and summarised.')


def results_text(result, symbols):
    if not result:
        return ''
    estimate = ('posterior median' if result.get('point_estimate') == 'posterior_median'
                else 'best fit')
    lines = [r'\begin{center}', r'\small', r'\begin{tabular}{ll}', r'\toprule',
             f'Parameter & Calibrated value ({estimate}) \\\\', r'\midrule']
    for record in result.get('calibrated', []):
        lines.append(f'{symbols.parameter(record["vessel"], record["param"])} & '
                     f'{_number(record["value"])} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    cost = result.get('best_cost')
    head = f'The best cost was {_number(cost)}.' if cost is not None else ''
    return head + '\n' + '\n'.join(lines)


# --------------------------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------------------------

def _document(body, title, standalone):
    if not standalone:
        return PACKAGES_NOTE + '\n' + body + '\n'
    return (PREAMBLE + r'\title{' + escape(title) + '}\n\\date{}\n\\begin{document}\n'
            '\\maketitle\n' + body + '\n\\end{document}\n')


def _model_section(model, mapping_rows, module_info, mapping_path):
    modules = [c for c in module_components(model)
               if any(v.rhs() is not None for v in c.variables(deep=True))]
    intro = (f'The model has {len(modules)} module{"s" if len(modules) != 1 else ""}. '
             'Each subsection gives a module\'s equations and the parameters they use; '
             'a symbol\'s subscript names the module it belongs to.')
    if mapping_path:
        intro += (f' Symbols are those of \\texttt{{{escape(os.path.basename(mapping_path))}}}.')
    by_name = {r['variable_name']: r for r in mapping_rows}
    return ('\\section{Model}\n' + intro + '\n\n'
            + module_sections(model, by_name, module_info=module_info))


def _step_section(step, symbols):
    out = [f'\\subsection{{Step {escape(step["id"])}}}', r'\label{sec:step:'
           + step['id'].replace('_', '-') + '}']
    t = step['target']
    where = (f', the submodule {_tt(step["submodule_path"])} of the target'
             if step.get('submodule_path') else '')
    out.append(f'This step calibrates instance {_tt(t["instance"])} of module '
               f'{_tt(t["module_type"])} (version {_tt(t["version"])}){where}, against that '
               f'instance\'s own observations ({_tt(os.path.basename(step["obs_path"]))}).')
    if step.get('description'):
        out.append(_sentence(step['description']))
    if step['fixed_from']:
        fixed = (step.get('result') or {}).get('fixed') or []
        if fixed:
            values = ', '.join(f'{symbols["target"].model_name(_target_name(f, step))}'
                               f'$\\,=\\,${_number(f["value"])} (from {escape(f["from_step"])})'
                               for f in fixed)
            out.append(f'The values calibrated in earlier steps were fixed: {values}.')
        else:
            out.append('The values calibrated in step' + ('s ' if len(step['fixed_from']) > 1
                                                           else ' ')
                       + ', '.join(escape(s) for s in step['fixed_from'])
                       + ' were fixed in this step\'s model.')
    for prior in step['priors_from']:
        out.append(f'The parameters calibrated in step {escape(prior["step"])} were '
                   f'calibrated again, with a joint prior from that step\'s stored posterior: '
                   + PRIOR_DESCRIPTIONS.get(prior.get('kind', 'mvnormal'), '')
                   + (f', with its variance multiplied by {_number(prior["inflate"])}'
                      if prior.get('inflate', 1.0) != 1.0 else '') + '.')
    out.append(r'\paragraph{Protocol.} ' + protocol_text(step['obs']))
    observables, costs = observables_text(step['obs'], symbols['step'],
                                          (step['settings'].get('optimiser_options') or {})
                                          .get('cost_type') or 'gaussian_MLE')
    out.append(r'\paragraph{Observables.} ' + observables)
    out.append(r'\paragraph{Parameters.} The calibrated parameters, their ranges and priors:'
               + '\n' + parameters_text(step['params'], symbols['step']))
    out.append(r'\paragraph{Method.} ' + method_text(step['settings']) + ' '
               + cost_text(costs, step['settings']))
    if step.get('store_distribution'):
        out.append(r'\paragraph{Uncertainty.} ' + uq_text(step['settings']))
    if step.get('result'):
        out.append(r'\paragraph{Result.} ' + results_text(step['result'], symbols['step']))
    return '\n\n'.join(out)


def _target_name(fixed_record, step):
    return fixed_record['model_name'] if not step.get('submodule_path') else \
        _remap_model_name(fixed_record, step['submodule_path'])


def _remap_model_name(record, path):
    from libcuflynx.calibration_workflow.naming import map_vessel, model_name
    return model_name(map_vessel(record['vessel'], path), record['param'])


def workflow_methods_latex(workflow_path, run_dir=None, mapping=None, module_library_dirs=None,
                           standalone=True, title=None, flowchart_only=False):
    '''The methods for a calibration workflow (see the module docstring).

    ``run_dir`` is the workflow's output directory: with it, the target model carries the
    calibrated values and each step reports its result. ``mapping`` is a variable mapping
    CSV for the *target* model; by default ``<target instance>_variable_mapping.csv`` in the
    target instance's directory, if it exists, else the default symbols.
    '''
    from libcuflynx.calibration_workflow import load_workflow, resolve_workflow, workflow_model

    workflow = load_workflow(workflow_path, module_library_dirs)
    resolved = resolve_workflow(workflow)
    with tempfile.TemporaryDirectory(prefix='cuflynx-methods-') as work:
        view = workflow_model(workflow, 'target', output_dir=run_dir or os.path.join(work, 'run'),
                              work_dir=work)
        model = load_model(view['flat_model_path'] or view['model_path'])
    if mapping is None:
        candidate = resolved.target.file('variable_mapping.csv')
        mapping = candidate if os.path.isfile(candidate) else None
    rows = variable_mapping(model, mapping)
    module_info = target_module_info(resolved)

    steps, results = [], {}
    for step in workflow.topological_order():
        rs = resolved.steps[step.id]
        result = None
        if run_dir:
            path = os.path.join(run_dir, step.id, 'step_result.json')
            if os.path.isfile(path):
                result = _load_json(path)
                if result.get('status') != 'done':
                    result = None
        results[step.id] = result
        steps.append({
            'id': step.id, 'target': step.target.as_dict(), 'submodule_path': rs.path,
            'description': step.description,
            'obs_path': rs.instance.obs_data_path, 'obs': _load_json(rs.instance.obs_data_path),
            'params': read_params_for_id(rs.instance.params_for_id_path),
            'settings': workflow.step_settings(step), 'depends_on': list(step.depends_on),
            'fixed_from': list(step.fixed_from),
            'priors_from': [p.as_dict() for p in step.priors_from],
            'store_distribution': step.store_distribution, 'result': result,
        })
    graph_steps = [dict(s, n_parameters=len(s['params']),
                        n_data_items=len(s['obs'].get('data_items') or []),
                        priors_from=[p['step'] for p in s['priors_from']]) for s in steps]
    n_merged = sum(len(r['calibrated']) for r in results.values() if r) or \
        sum(len(s['params']) for s in steps)
    nodes, edges = flowchart.workflow_graph(
        graph_steps + [{'id': None, 'target': workflow.target.as_dict(), 'n_merged': n_merged}])
    figure = flowchart.figure(
        flowchart.tikz(nodes, edges),
        f'Calibration workflow {escape(workflow.name)}: each step calibrates one module '
        'instance against its own observations; solid arrows pass calibrated values on as '
        'fixed values, dashed arrows as priors, and the results are merged into the target.',
        'fig:calibration-flowchart', natural_width_cm=flowchart.width_cm(nodes, edges))
    if flowchart_only:
        return _document(figure, f'Calibration of {workflow.name}', standalone)

    body = [_model_section(model, rows, module_info, mapping)]
    body.append('\\section{Calibration}\n'
                + (_sentence(workflow.description) + ' ' if workflow.description else '')
                + f'The parameters were calibrated in {len(steps)} step'
                + ('s' if len(steps) != 1 else '')
                + ' (Figure~\\ref{fig:calibration-flowchart}), each against the observations '
                'of one module instance, in the order shown; the values of every step were '
                'then merged into the target, '
                + _tt(workflow.target.label()) + '.\n\n' + figure)
    for step in steps:
        symbols = {'step': Symbols(rows, step['submodule_path']), 'target': Symbols(rows)}
        body.append(_step_section(step, symbols))
    return _document('\n\n'.join(body), title or f'Methods: {workflow.name}', standalone)


def workflow_mapping_path(workflow_path, module_library_dirs=None):
    '''``<target instance dir>/<instance>_variable_mapping.csv`` for a workflow.'''
    from libcuflynx.calibration_workflow import load_workflow, resolve_workflow
    resolved = resolve_workflow(load_workflow(workflow_path, module_library_dirs))
    return resolved.target.file('variable_mapping.csv')


def update_workflow_variable_mapping(workflow_path, out_path=None, module_library_dirs=None,
                                     run_dir=None):
    '''Create or refresh the variable mapping of a workflow's target model (by default in the
    target instance's directory). Returns ``(path, rows)``.'''
    from libcuflynx.calibration_workflow import load_workflow, workflow_model
    from libcuflynx.reporting.variable_mapping import update_variable_mapping_file
    workflow = load_workflow(workflow_path, module_library_dirs)
    out_path = out_path or workflow_mapping_path(workflow_path, module_library_dirs)
    with tempfile.TemporaryDirectory(prefix='cuflynx-mapping-') as work:
        view = workflow_model(workflow, 'target', output_dir=run_dir or os.path.join(work, 'run'),
                              work_dir=work)
        rows = update_variable_mapping_file(view['flat_model_path'] or view['model_path'],
                                            out_path)
    return out_path, rows


def target_module_info(resolved):
    '''``{module name in the target model: {module_type, version, instance}}``.'''
    from libcuflynx.calibration_workflow.naming import STEP_VESSEL
    entry = resolved.target.entry
    info = {}

    def walk(container, prefix):
        for sub in container.get('submodules', []):
            path = prefix + sub['name']
            info[f'{STEP_VESSEL}_{path}'] = {'module_type': sub['vessel_type'],
                                             'version': sub['BC_type'],
                                             'instance': sub.get('instance')}
            nested = _supermodule(resolved, sub)
            if nested is not None:
                walk(nested, path + '_')

    if entry.get('submodules'):
        walk(entry, '')
    else:
        t = resolved.workflow.target
        info[STEP_VESSEL] = {'module_type': t.module_type, 'version': t.version,
                             'instance': t.instance}
    return info


def _supermodule(resolved, sub):
    from libcuflynx.utilities.module_library import ModuleSources
    registry = getattr(resolved, '_supermodules', None)
    if registry is None:
        # the sources' registry, shadowed types left out, as the workflow resolved them
        registry = ModuleSources(resolved.library_inputs).supermodule_registry()
        resolved._supermodules = registry
    return registry.get((sub['vessel_type'], sub['BC_type']))


def calibration_methods_latex(model_path, obs_data_path, params_for_id_path, settings=None,
                              run_dir=None, mapping=None, standalone=True, title=None,
                              do_uq=False, do_ia=False, use_emulator=False):
    '''The methods for one calibration of the CellML model at ``model_path``.

    ``settings`` are the param_id user_inputs (``param_id_method``, ``optimiser_options``,
    ``solver_info``, ``dt``, ``UQ_options`` ...). ``run_dir`` is the param_id run directory
    (``<method>_<prefix>_<obs>``), whose best fit is then reported. ``mapping`` defaults to
    ``<prefix>_variable_mapping.csv`` beside the model, if it exists.
    '''
    settings = dict(settings or {})
    model = load_model(model_path)
    if mapping is None:
        candidate = default_mapping_path(model_path)
        mapping = candidate if os.path.isfile(candidate) else None
    rows = variable_mapping(model, mapping)
    obs = _load_json(obs_data_path)
    params = read_params_for_id(params_for_id_path)
    result = _param_id_result(run_dir, params) if run_dir else None
    step = {'id': 'calibration', 'target': {'module_type': os.path.splitext(
        os.path.basename(model_path))[0], 'version': '--', 'instance': '--'},
        'submodule_path': '', 'description': '', 'obs_path': obs_data_path, 'obs': obs,
        'params': params, 'settings': settings, 'depends_on': [], 'fixed_from': [],
        'priors_from': [], 'store_distribution': do_uq, 'result': result}
    info = obs.get('protocol_info') or {} if isinstance(obs, dict) else {}
    nodes, edges = flowchart.single_calibration_graph(
        len(params), len(obs.get('data_items') or []) if isinstance(obs, dict) else len(obs),
        len(info.get('sim_times') or [1]), settings, do_uq=do_uq, do_ia=do_ia,
        use_emulator=use_emulator, model_label=_tt(os.path.basename(model_path)))
    figure = flowchart.figure(flowchart.tikz(nodes, edges),
                              'The calibration: the model is fitted to the observations by '
                              'varying the calibrated parameters.', 'fig:calibration-flowchart',
                              natural_width_cm=flowchart.width_cm(nodes, edges))
    symbols = {'step': Symbols(rows), 'target': Symbols(rows)}
    section = _step_section(step, symbols).replace(
        '\\subsection{Step calibration}', '\\subsection{Calibration details}').replace(
        'This step calibrates instance \\texttt{--} of module ', 'The model ').replace(
        ' (version \\texttt{--})', '')
    body = [_model_section(model, rows, None, mapping),
            '\\section{Calibration}\nFigure~\\ref{fig:calibration-flowchart} shows the '
            'calibration.\n\n' + figure, section]
    return _document('\n\n'.join(body), title or 'Methods', standalone)


def _param_id_result(run_dir, params):
    import numpy as np
    best_path = os.path.join(run_dir, 'best_param_vals.npy')
    if not os.path.isfile(best_path):
        return None
    best = np.atleast_1d(np.load(best_path))
    calibrated = []
    for entry, value in zip(params, best):
        for vessel, param in entry['targets']:
            calibrated.append({'vessel': vessel, 'param': param, 'value': float(value)})
    cost_path = os.path.join(run_dir, 'best_cost.npy')
    cost = float(np.load(cost_path)) if os.path.isfile(cost_path) else None
    return {'calibrated': calibrated, 'best_cost': cost, 'point_estimate': 'best_fit'}


def write_latex(text, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path
