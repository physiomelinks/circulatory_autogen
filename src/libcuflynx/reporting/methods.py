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
from libcuflynx.reporting.bibliography import cite, cited_keys, module_bib_files, read_bib, write_bib
from libcuflynx.reporting.latex_names import escape
from libcuflynx.utilities.parameter_names import LEVEL_SEPARATOR, PARAMETER_COMPONENTS
from libcuflynx.reporting.model_latex import (format_number, format_units, module_sections,
                                              parameter_sources_section)
from libcuflynx.reporting.variable_mapping import (default_mapping_path, load_model,
                                                   module_components, strip_module,
                                                   variable_mapping)

PREAMBLE = r'''\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\allowdisplaybreaks
% long \texttt names (option keys, file names) cannot hyphenate; let lines stretch instead
\setlength{\emergencystretch}{3em}
'''
PACKAGES_NOTE = ('% Needs: amsmath, amssymb, booktabs, longtable, graphicx, tikz '
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

    def __init__(self, rows, to_target=None):
        '''``to_target`` names a vessel of the step's model as the target model does (None:
        the step's model is the model the rows are for).'''
        self.latex = {r['variable_name']: r['latex'] for r in rows}
        self.to_target = to_target

    def _vessel(self, vessel):
        try:
            return self.to_target(vessel) if self.to_target else vessel
        except ValueError:
            return vessel

    def variable(self, name):
        '''An obs_data operand ``vessel/variable``.'''
        vessel, sep, variable = str(name).partition('/')
        if not sep:
            return _tt(name)
        from libcuflynx.parsers.PrimitiveParsers import model_qname_candidates
        latex = self._first([f'{strip_module(self._vessel(vessel))}/{variable}']
                            + model_qname_candidates(f'{vessel}/{variable}'))
        return _math(latex) if latex else _tt(name)

    def parameter(self, vessel, param):
        from libcuflynx.parsers.PrimitiveParsers import model_qname_candidates
        latex = self._first(model_qname_candidates(f'{self._vessel(vessel)}/{param}'))
        return _math(latex) if latex else _tt(f'{vessel}/{param}')

    def model_name(self, name):
        '''A generated-model parameter name (``p_mod_A``).'''
        latex = self._first([f'{component}/{name}' for component in PARAMETER_COMPONENTS])
        return _math(latex) if latex else _tt(name)

    def _first(self, keys):
        return next((self.latex[key] for key in keys if key in self.latex), None)


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
    operands = item.get('operands') or []
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
    items = obs if isinstance(obs, list) else (obs.get('data_items') or [])
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
        lines.append(f'{escape(item.get("data_item_name") or "")} & '
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

def _document(body, title, standalone, bibliography=None):
    '''``bibliography`` is what ``\\bibliography{}`` names, for a document that cites.'''
    if bibliography and standalone:
        body += '\n\n\\bibliographystyle{plain}\n\\bibliography{' + bibliography + '}'
    if not standalone:
        return PACKAGES_NOTE + '\n' + body + '\n'
    return (PREAMBLE + r'\title{' + escape(title) + '}\n\\date{}\n\\begin{document}\n'
            '\\maketitle\n' + body + '\n\\end{document}\n')


def _read_csv(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _parameter_sources(parameter_rows, entries, calibrated):
    '''``{model name: source in LaTeX}``: a calibrated parameter names the step section that
    calibrated it (``calibrated`` maps a model name to a step id); any other parameter's
    data_reference, with its BibTeX keys cited. Returns the sources and the keys cited.'''
    sources, keys = {}, set()
    for row in parameter_rows:
        name = row.get('variable_name')
        if name in calibrated:
            step = calibrated[name]
            sources[name] = (f'Calibrated in step {_tt(step)} '
                             f'(Section~\\ref{{sec:step:{step.replace("_", "-")}}})')
        else:
            reference = row.get('data_reference') or ''
            sources[name] = cite(reference, entries) if reference else ''
            keys.update(cited_keys(reference, entries))
    return sources, keys


def _bibliography(entries, keys, bib_files, bib_out):
    '''The ``\\bibliography{}`` argument for the ``keys`` cited: ``bib_out`` (a .bib written
    with the cited entries) when given, else the source .bib files themselves.'''
    if not keys:
        return None
    if bib_out:
        write_bib({k: entries[k] for k in keys}, bib_out)
        return os.path.splitext(os.path.basename(bib_out))[0]
    return ','.join(os.path.splitext(os.path.abspath(p))[0] for p in bib_files)


def workflow_bib_files(resolved):
    '''The references (``*_references*.bib``) of every module in the workflow target, from
    the module versions they live in.'''
    return module_bib_files(resolved.module_configs())


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


#: how the overview counts each kind of step: (singular, plural)
KIND_NAMES = {'calibration': ('calibration', 'calibrations'),
              'sensitivity': ('sensitivity analysis', 'sensitivity analyses'),
              'reduction': ('reduction', 'reductions'),
              'emulation': ('emulator training', 'emulator trainings'),
              'identifiability': ('identifiability analysis', 'identifiability analyses')}
_INDEX_NAMES = {'ST': 'total', 'S1': 'first-order'}


def _count_of(kind, n):
    singular, plural = KIND_NAMES[kind]
    return f'{n} {singular if n == 1 else plural}'


def _join(items):
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def _kind(step):
    return step.get('kind', 'calibration')


def _step_section(step, symbols):
    out = [f'\\subsection{{Step {escape(step["id"])}}}', r'\label{sec:step:'
           + step['id'].replace('_', '-') + '}']
    out.append(step.get('intro') or _step_intro(step))
    if step.get('description'):
        out.append(_sentence(step['description']))
    if _kind(step) == 'reduction':
        return '\n\n'.join(out)
    if step.get('emulator_from'):
        out.append(f'It runs on the emulator trained in step {escape(step["emulator_from"])} '
                   f'(Section~\\ref{{sec:step:{step["emulator_from"].replace("_", "-")}}}), '
                   'not on the model.')
    if step.get('params_for_id_from'):
        out.append(f'It calibrates the parameters kept by step '
                   f'{escape(step["params_for_id_from"])}.')
    if step['fixed_from'] or step.get('result_from'):
        fixed = (step.get('result') or {}).get('fixed') or []
        given = step['fixed_from'] + ([step['result_from']] if step.get('result_from') else [])
        if fixed:
            values = ', '.join(f'{symbols["target"].model_name(_target_name(f, step))}'
                               f'$\\,=\\,${_number(f["value"])} (from {escape(f["from_step"])})'
                               for f in fixed)
            out.append(f'The values calibrated in earlier steps were fixed: {values}.')
        else:
            out.append('The values calibrated in step' + ('s ' if len(given) > 1 else ' ')
                       + ', '.join(escape(s) for s in given)
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
    if _kind(step) == 'calibration':
        out.append(r'\paragraph{Parameters.} The calibrated parameters, their ranges and '
                   r'priors:' + '\n' + parameters_text(step['params'], symbols['step']))
        out.append(r'\paragraph{Method.} ' + method_text(step['settings']) + ' '
                   + cost_text(costs, step['settings']))
    else:
        out.append(r'\paragraph{Parameters.} The parameters ' + ANALYSED[_kind(step)]
                   + ' and their ranges:\n' + parameters_text(step['params'], symbols['step']))
        out.append(r'\paragraph{Method.} ' + _analysis_method(step))
    if step.get('store_distribution'):
        out.append(r'\paragraph{Uncertainty.} ' + uq_text(step['settings']))
    if step.get('result') and _kind(step) == 'calibration':
        out.append(r'\paragraph{Result.} ' + results_text(step['result'], symbols['step']))
    elif _kind(step) == 'identifiability' and (step.get('result') or {}).get('laplace'):
        result = step['result']
        stds = ', '.join(f'{escape(name)}: {_number(value)}' for name, value in
                         zip(result['parameters'], result['laplace']['std']))
        out.append(r'\paragraph{Result.} The standard deviations of the Laplace '
                   f'approximation: {stds}.')
    return '\n\n'.join(out)


#: what the Parameters paragraph says a non-calibration step does with its parameters
ANALYSED = {'sensitivity': 'whose influence was analysed',
            'emulation': 'the emulator was trained over',
            'identifiability': 'whose identifiability was analysed'}


def _analysis_method(step):
    settings, kind = step['settings'], _kind(step)
    if kind == 'sensitivity':
        sa = settings.get('sa_options') or {}
        if sa.get('method', 'sobol') == 'sobol':
            return (f'Sobol indices were estimated from {sa.get("num_samples", 32)} base '
                    f'samples ({escape(str(sa.get("sample_type", "saltelli")))} sampling).')
        return ('Local sensitivities, the derivatives of each observable with respect to each '
                'parameter, were computed at the model\'s values.')
    if kind == 'emulation':
        em = settings.get('emulator_settings') or {}
        return (f'An emulator of the observables was trained on '
                f'{em.get("num_train_samples", 128)} model runs '
                f'({escape(str(em.get("sample_type", "sobol")))} sampling).')
    ia = settings.get('ia_options') or {}
    return (f'The {escape(str(ia.get("method", "Laplace")))} approximation of the posterior '
            f'at the calibrated values, with gradients by '
            f'{escape(str(ia.get("gradient_source", "FD")))}.')


def _step_intro(step):
    '''What the step does, on which model, against which observations.'''
    if _kind(step) == 'reduction':
        options = step['reduction']
        text = (f'This step ranks the parameters of step {escape(step["sensitivity_from"])} by '
                f'the {options["criterion"]} over the observables of their '
                f'{_INDEX_NAMES[options["index_type"]]} Sobol index, and keeps the top '
                f'{options["top_n"]}'
                + (f' scoring at least {_number(options["threshold"])}'
                   if options.get('threshold') is not None else '') + '.')
        selected = (step.get('result') or {}).get('selected')
        if selected:
            text += ' It kept ' + _join([f'{_tt(s["parameter"])} ({_number(s["score"])})'
                                         for s in selected]) + '.'
        return text
    m, obs = step['model'], _tt(os.path.basename(step['obs_path']))
    if 'module_array' in m:
        what = f'the model {_tt(_array_label(m))} ({_tt(os.path.basename(m["module_array"]))})'
        where = ''
    else:
        what = (f'parameterisation {_tt(m["parameterisation"])} of module '
                f'{_tt(m["module_type"])} (version {_tt(m["version"])})')
        where = f', {_tt(step["place"])} in the target' if step.get('place') else ''
    verb = {'calibration': 'calibrates', 'sensitivity': 'analyses the sensitivity of',
            'emulation': 'trains an emulator of',
            'identifiability': 'analyses the identifiability of'}[_kind(step)]
    at = (f', at the values calibrated in step {escape(step["result_from"])}'
          if step.get('result_from') else '')
    return f'This step {verb} {what}{where}{at}, against the observations in {obs}.'


def _array_label(model):
    from libcuflynx.calibration_workflow.spec import array_prefix
    return array_prefix(model['module_array'])


def _target_name(fixed_record, step):
    '''A value the step was given, named as the target model names it.'''
    from libcuflynx.calibration_workflow.naming import model_name
    to_target = step.get('to_target')
    if to_target is None:
        return fixed_record['model_name']
    try:
        return model_name(to_target(fixed_record['vessel']), fixed_record['param'])
    except ValueError:
        return fixed_record['model_name']


def workflow_methods_latex(workflow_path, run_dir=None, mapping=None, module_library_dirs=None,
                           standalone=True, title=None, flowchart_only=False, bib_files=None,
                           bib_out=None):
    '''The methods for a calibration workflow (see the module docstring).

    ``run_dir`` is the workflow's output directory: with it, the target model carries the
    calibrated values and each step reports its result. ``mapping`` is a variable mapping
    CSV for the *target* model; by default ``<target parameterisation>_variable_mapping.csv``
    in the target parameterisation's directory, if it exists, else the default symbols.

    The parameters table cites the BibTeX keys in each parameter's data_reference, from
    ``bib_files`` (default: the ``*_references*.bib`` of every module version in the target).
    ``bib_out`` is a .bib file to write the cited entries to, for ``\\bibliography``.
    '''
    from libcuflynx.calibration_workflow import load_workflow, resolve_workflow, workflow_model

    workflow = load_workflow(workflow_path, module_library_dirs)
    resolved = resolve_workflow(workflow)
    with tempfile.TemporaryDirectory(prefix='cuflynx-methods-') as work:
        view = workflow_model(workflow, 'target', output_dir=run_dir or os.path.join(work, 'run'),
                              work_dir=work)
        model = load_model(view['flat_model_path'] or view['model_path'])
        parameter_rows = _read_csv(view.get('parameters_path'))
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
        entry = {
            'id': step.id, 'kind': step.kind, 'description': step.description,
            'settings': workflow.step_settings(step), 'depends_on': list(step.depends_on),
            'fixed_from': list(step.fixed_from),
            'priors_from': [p.as_dict() for p in step.priors_from],
            'store_distribution': step.store_distribution, 'result': result,
            'emulator_from': step.emulator_from, 'result_from': step.result_from,
            'sensitivity_from': step.sensitivity_from,
            'params_for_id_from': step.params_for_id_from, 'reduction': step.reduction,
        }
        if step.model is not None:
            # a calibration of a reduction's choice: what it kept, once it has run; until
            # then every parameter it chooses from
            params_path = ((result or {}).get('params_for_id_used') or step.params_for_id
                           or workflow.step(workflow.step(step.params_for_id_from)
                                            .sensitivity_from).params_for_id)
            entry.update({
                'model': step.model.as_dict(), 'place': rs.place, 'to_target': rs.to_target,
                'obs_path': rs.obs_data_path, 'obs': _load_json(rs.obs_data_path),
                'params': read_params_for_id(params_path)})
        else:
            entry.update({'model': None, 'place': '', 'to_target': None, 'obs_path': None,
                          'obs': {}, 'params': []})
        steps.append(entry)
    graph_steps = [dict(s, n_parameters=len(s['params']),
                        n_data_items=len(s['obs'].get('data_items') or []),
                        priors_from=[p['step'] for p in s['priors_from']]) for s in steps]
    calibrations = [s for s in steps if s['kind'] == 'calibration']
    n_merged = sum(len(results[s['id']]['calibrated']) for s in calibrations
                   if results[s['id']]) or sum(len(s['params']) for s in calibrations)
    nodes, edges = flowchart.workflow_graph(
        graph_steps + [{'id': None, 'model': workflow.target.as_dict(), 'n_merged': n_merged}])
    figure = flowchart.figure(
        flowchart.tikz(nodes, edges),
        f'Calibration workflow {escape(workflow.name)}: each step calibrates one model against '
        'its own observations; solid arrows pass calibrated values on as '
        'fixed values, dashed arrows as priors, and the results are merged into the target.',
        'fig:calibration-flowchart', natural_width_cm=flowchart.width_cm(nodes, edges))
    if flowchart_only:
        return _document(figure, f'Calibration of {workflow.name}', standalone)

    calibrated = {}
    for step in workflow.topological_order():
        if step.kind != 'calibration':
            continue
        result = results.get(step.id)
        if result:
            names = [r['target_model_name'] for r in result['calibrated']]
        else:
            rs = resolved.steps[step.id]
            names = [rs.target_model_name(p) for p in rs.parameters]
        calibrated.update({name: step.id for name in names})
    bib_files = workflow_bib_files(resolved) if bib_files is None else list(bib_files)
    entries = read_bib(bib_files)
    sources, keys = _parameter_sources(parameter_rows, entries, calibrated)

    body = [_model_section(model, rows, module_info, mapping),
            parameter_sources_section(model, {r['variable_name']: r for r in rows}, sources)]
    if len(calibrations) == len(steps):
        overview = (f'The parameters were calibrated in {len(steps)} step'
                    + ('s' if len(steps) != 1 else '')
                    + ' (Figure~\\ref{fig:calibration-flowchart}), each a calibration of one '
                    'model against its own observations, in the order shown; the values of '
                    'every step were then merged into the target, ')
    else:
        counts = [_count_of(kind, sum(s['kind'] == kind for s in steps))
                  for kind in KIND_NAMES if any(s['kind'] == kind for s in steps)]
        overview = (f'The workflow has {len(steps)} steps '
                    '(Figure~\\ref{fig:calibration-flowchart}), run in the order shown: '
                    + _join(counts) + '. Each calibration fits one model against its own '
                    'observations, and the values of every calibration were then merged into '
                    'the target, ')
    body.append('\\section{Calibration}\n'
                + (_sentence(workflow.description) + ' ' if workflow.description else '')
                + overview + _tt(workflow.target.label()) + '.\n\n' + figure)
    for step in steps:
        symbols = {'step': Symbols(rows, step['to_target']), 'target': Symbols(rows)}
        body.append(_step_section(step, symbols))
    return _document('\n\n'.join(body), title or f'Methods: {workflow.name}', standalone,
                     _bibliography(entries, keys, bib_files, bib_out))


def workflow_mapping_path(workflow_path, module_library_dirs=None):
    '''``<target parameterisation dir>/<parameterisation>_variable_mapping.csv`` for a workflow.'''
    from libcuflynx.calibration_workflow import load_workflow, resolve_workflow
    resolved = resolve_workflow(load_workflow(workflow_path, module_library_dirs))
    return resolved.target.file('variable_mapping.csv')


def update_workflow_variable_mapping(workflow_path, out_path=None, module_library_dirs=None,
                                     run_dir=None):
    '''Create or refresh the variable mapping of a workflow's target model (by default in the
    target parameterisation's directory). Returns ``(path, rows)``.'''
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
    '''``{module name in the target model: {module_type, version, parameterisation}}``.'''
    from libcuflynx.calibration_workflow.naming import STEP_VESSEL
    supermodules = resolved.supermodules
    info = {}

    def walk(container, prefix):
        for sub in container.get('submodules', []):
            path = prefix + sub['name']
            info[path] = {'module_type': sub['module_type'], 'version': sub['module_subtype'],
                          'parameterisation': sub.get('parameterisation')}
            nested = supermodules.get((sub['module_type'], sub['module_subtype']))
            if nested is not None:
                walk(nested, path + LEVEL_SEPARATOR)

    target = resolved.target
    if target.kind == 'array':
        for record in target.records:
            info[record['name']] = {'module_type': record['module_type'],
                                    'version': record['module_subtype'],
                                    'parameterisation': record.get('parameterisation')}
            nested = supermodules.get((record['module_type'], record['module_subtype']))
            if nested is not None:
                walk(nested, record['name'] + LEVEL_SEPARATOR)
    elif target.entry.get('submodules'):
        walk(target.entry, STEP_VESSEL + LEVEL_SEPARATOR)
    else:
        t = resolved.workflow.target
        info[STEP_VESSEL] = {'module_type': t.module_type, 'version': t.version,
                             'parameterisation': t.parameterisation}
    return info


def calibration_methods_latex(model_path, obs_data_path, params_for_id_path, settings=None,
                              run_dir=None, mapping=None, standalone=True, title=None,
                              do_uq=False, do_ia=False, use_emulator=False, parameters_path=None,
                              bib_files=(), bib_out=None):
    '''The methods for one calibration of the CellML model at ``model_path``.

    ``settings`` are the param_id user_inputs (``param_id_method``, ``optimiser_options``,
    ``solver_info``, ``dt``, ``UQ_options`` ...). ``run_dir`` is the param_id run directory
    (``<method>_<prefix>_<obs>``), whose best fit is then reported. ``mapping`` defaults to
    ``<prefix>_variable_mapping.csv`` beside the model, if it exists. ``parameters_path`` is the
    model's parameters CSV, whose data_reference column gives each parameter's source (default:
    ``<prefix>_parameters.csv`` beside the model); its BibTeX keys are cited from
    ``bib_files``, and the cited entries written to ``bib_out`` when given.
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
    step = {'id': 'calibration', 'model': {'module_array': model_path}, 'place': '',
        'intro': f'The model {_tt(os.path.basename(model_path))} is calibrated against the '
                 f'observations in {_tt(os.path.basename(obs_data_path))}.',
        'description': '', 'obs_path': obs_data_path, 'obs': obs,
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
        '\\subsection{Step calibration}', '\\subsection{Calibration details}')
    if parameters_path is None:
        prefix = os.path.splitext(os.path.basename(model_path))[0]
        prefix = prefix[:-len('_flat')] if prefix.endswith('_flat') else prefix
        parameters_path = os.path.join(os.path.dirname(model_path), f'{prefix}_parameters.csv')
    entries = read_bib(list(bib_files or ()))
    from libcuflynx.parsers.PrimitiveParsers import param_name_for_gen
    calibrated = {param_name_for_gen(v, p): 'calibration' for row in params
                  for v, p in row['targets']}
    sources, keys = _parameter_sources(_read_csv(parameters_path), entries, calibrated)
    body = [_model_section(model, rows, None, mapping),
            parameter_sources_section(model, {r['variable_name']: r for r in rows}, sources),
            '\\section{Calibration}\nFigure~\\ref{fig:calibration-flowchart} shows the '
            'calibration.\n\n' + figure, section]
    return _document('\n\n'.join(body), title or 'Methods', standalone,
                     _bibliography(entries, keys, list(bib_files or ()), bib_out))


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
