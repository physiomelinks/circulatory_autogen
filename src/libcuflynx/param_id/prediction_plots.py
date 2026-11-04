"""
``prediction_plots``: figures across an obs_data's experiments (#568).

An obs_data may list figures at its top level::

    "prediction_plots": [
      {"name": "Peak I_Na vs step potential", "kind": "feature_vs_input",
       "x": {"params_to_change": "clamp/V_cmd", "subexperiment_idx": 1},
       "y": "I_peak", "series": null},
      {"name": "Clamp and current", "kind": "input_and_output",
       "input": {"params_to_change": "clamp/V_cmd"}, "output": "cell/membrane/I_Na",
       "experiments": [0, 2]},
      {"name": "I-V loop", "kind": "output_vs_input",
       "input": {"params_to_change": "clamp/V_cmd"}, "output": "cell/membrane/I_Na"}]

Feature kinds: one point per experiment
---------------------------------------
A *group* is the items (data_items and prediction_items) that share an
``item_name_for_plotting`` -- the explicit label or its default. A point is one experiment: its y
is the y group's item for that experiment, its x the x group's item for the same experiment
(``feature_vs_feature``) or the value ``params_to_change[key]`` takes in that sub-experiment
(``feature_vs_input``). ``series`` (an input reference too) draws one line per distinct value of
that input. Points are sorted by x within a series; an item with a ``value``/``std`` adds the
data as error bars.

An input reference is ``{"params_to_change", "subexperiment_idx", "reduce"?}``. A number is used
as it is. A trace (a string naming a ``protocol_traces`` entry, which is how the solvers resolve
it) is made one number by ``reduce`` (``REDUCE_MODES``): ``"final"``, ``"mean"``, ``"max"``,
``"min"`` or ``{"at": t}``, t seconds from the start of that sub-experiment. It is taken over the
trace as the sub-experiment applies it: linear between its points, held at its ends, over
``[0, sim_time]`` (``reduce_trace``). An experiment whose input is a trace with no ``reduce`` has
no point; it is skipped with a ``PredictionPlotWarning``.

Time-course kinds: from a simulation
------------------------------------
``input_and_output`` draws, per experiment, the input (``params_to_change[key]`` over the
experiment: a number as a step, a trace as its waveform) on the left axis and the model output
on the right, against the time since the end of the warm-up. Sub-experiment boundaries are
dashed, and a data_item that has the output as a series (no operation) in that experiment is
overlaid. ``output_vs_input`` draws the output against the input sampled at the same times, one
line per experiment. ``experiments`` defaults to all of them.

``output`` is written as an obs_data operand is: ``<vessel>/<variable>``, a nested vessel by its
path (``SN/soma/membrane/V``, ``parameter_names.output_model_name``), or a bare global (tried as
written, then as ``global/<name>``). ``record_time_courses`` records the outputs as prediction
items are recorded: they are the ``result_variables`` of a ``ProtocolExecutor.run_protocol``
pass, so any variable the solver returns (a state, an algebraic variable or a constant) can be
drawn. An output the model does not have is refused before anything runs.

``check_prediction_plots`` runs the structural rules when the obs_data is parsed.
``plot_prediction_plots`` checks that every item is a scalar feature (which needs the operation
funcs) and draws the figures from the model's feature values and recorded time courses.
"""

import os
import re
import warnings

import numpy as np

from libcuflynx.utilities.obs_data_helpers import data_reference_line

#: kinds with one point per experiment, from scalar features
FEATURE_KINDS = ('feature_vs_feature', 'feature_vs_input')
#: kinds drawn from an input and a model output over time (record_time_courses)
TIME_COURSE_KINDS = ('input_and_output', 'output_vs_input')
KINDS = FEATURE_KINDS + TIME_COURSE_KINDS
#: the named ``reduce`` modes of an input reference; ``{"at": <seconds>}`` is the other form
REDUCE_MODES = ('final', 'mean', 'max', 'min')


class PredictionPlotError(ValueError):
    '''A prediction_plots entry that cannot be drawn.'''


class PredictionPlotWarning(UserWarning):
    '''A point a prediction plot leaves out.'''


def item_groups(gt_df, prediction_info):
    """``{label: [item]}``, an item being a dict with ``source`` (``data_items`` /
    ``prediction_items``), ``index``, ``name``, ``experiment_idx``, ``value``, ``std``,
    ``data_type`` and ``data_reference`` (None when the item has none)."""
    groups = {}
    if gt_df is not None and len(gt_df):
        for i, row in enumerate(gt_df.to_dict('records')):
            groups.setdefault(str(row['item_name_for_plotting']), []).append({
                'source': 'data_items', 'index': i, 'name': str(row['data_item_name']),
                'experiment_idx': int(row.get('experiment_idx', 0) or 0),
                'value': row.get('value'), 'std': row.get('std'),
                'data_type': row.get('data_type'),
                'data_reference': _reference(row.get('data_reference'))})
    info = prediction_info or {}
    names = info.get('data_item_names') or []
    for i, label in enumerate(info.get('item_names_for_plotting') or []):
        groups.setdefault(str(label), []).append({
            'source': 'prediction_items', 'index': i, 'name': str(names[i]),
            'experiment_idx': int(info['experiment_idxs'][i]),
            'value': (info.get('values') or [None] * len(names))[i],
            'std': (info.get('stds') or [None] * len(names))[i],
            'data_type': (info.get('data_types') or [None] * len(names))[i],
            'data_reference': _reference(
                (info.get('data_references') or [None] * len(names))[i])})
    return groups


def _reference(value):
    return value if isinstance(value, str) else None


def _by_experiment(where, label, groups):
    if label not in groups:
        raise PredictionPlotError(
            f'{where}: no item has item_name_for_plotting {label!r}. Known groups: '
            f'{sorted(groups)}.')
    by_exp = {}
    for item in groups[label]:
        exp = item['experiment_idx']
        if exp in by_exp:
            raise PredictionPlotError(
                f'{where}: group {label!r} has two items for experiment {exp} '
                f'({by_exp[exp]["name"]!r} and {item["name"]!r}); a group gives one point '
                f'per experiment.')
        by_exp[exp] = item
    return by_exp


def _where(plot, index):
    return f'prediction_plots[{index}] ({plot["name"]!r})'


def _is_number(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)))


def _rows(where, key, protocol_info):
    """``params_to_change[key]``; raises for a key the protocol does not set."""
    params = (protocol_info or {}).get('params_to_change') or {}
    if key not in params:
        raise PredictionPlotError(
            f'{where}: {key!r} is not a params_to_change key ({sorted(params)}).')
    return params[key]


def _row(protocol_info, key, exp):
    """``params_to_change[key]`` in experiment ``exp``, one value per sub-experiment."""
    row = protocol_info['params_to_change'][key][exp]
    return list(row) if isinstance(row, (list, tuple, np.ndarray)) else [row]


def _trace(where, protocol_info, name):
    """``(t, values)`` of the ``protocol_traces`` entry ``name`` (``protocol_shapes`` are
    expanded into it when the obs_data is parsed), t from the start of the sub-experiment that
    uses it -- as the solvers apply a ``params_to_change`` string."""
    traces = (protocol_info or {}).get('protocol_traces') or {}
    if name not in traces:
        raise PredictionPlotError(f'{where}: trace {name!r} is not in protocol_traces.')
    return (np.asarray(traces[name]['t'], dtype=float),
            np.asarray(traces[name]['values'], dtype=float))


def _applied(t, values, duration):
    """The points of a trace applied over a sub-experiment of ``duration`` seconds: its own
    points inside, and its value at 0 and at ``duration`` (linear between points, held past its
    ends, as ``myokit.TimeSeriesProtocol``)."""
    inside = (t > 0.0) & (t < duration)
    knots = np.concatenate([[0.0], t[inside], [duration]])
    vals = np.concatenate([[np.interp(0.0, t, values)], values[inside],
                           [np.interp(duration, t, values)]])
    return knots, vals


def _check_reduce(where, reduce):
    if reduce is None or reduce in REDUCE_MODES:
        return
    if (isinstance(reduce, dict) and set(reduce) == {'at'} and _is_number(reduce['at'])
            and reduce['at'] >= 0):
        return
    raise PredictionPlotError(
        f'{where}: reduce {reduce!r} is not one of {list(REDUCE_MODES)} or '
        f'{{"at": <seconds from the start of the sub-experiment, >= 0>}}.')


def reduce_trace(t, values, duration, reduce):
    """One number from a trace ``(t, values)`` applied over a sub-experiment of ``duration``
    seconds: its ``"final"`` value (at ``duration``), its ``"mean"`` over ``[0, duration]``, its
    ``"max"``/``"min"`` there, or its value ``{"at": seconds}`` from the start."""
    t, values, duration = np.asarray(t, float), np.asarray(values, float), float(duration)
    if isinstance(reduce, dict):
        return float(np.interp(float(reduce['at']), t, values))
    knots, vals = _applied(t, values, duration)
    if reduce == 'final':
        return float(vals[-1])
    if reduce == 'max':
        return float(np.max(vals))
    if reduce == 'min':
        return float(np.min(vals))
    if reduce == 'mean':
        if duration <= 0:
            return float(vals[0])
        return float(np.sum(np.diff(knots) * (vals[1:] + vals[:-1]) / 2.0) / duration)
    _check_reduce('reduce_trace', reduce)
    raise PredictionPlotError(f'reduce_trace: no reduce given for a trace ({reduce!r}).')


def _input_value(where, ref, protocol_info, experiments):
    """``{experiment: value}`` of an input reference: a number as it is, a trace made one number
    by the reference's ``reduce``, or None for a trace without one (that experiment has no
    point). Raises for an unknown key, an out-of-range sub-experiment or a malformed reduce."""
    key, sub, reduce = ref['params_to_change'], ref['subexperiment_idx'], ref.get('reduce')
    _check_reduce(where, reduce)
    rows = _rows(where, key, protocol_info)
    sim_times = protocol_info['sim_times']
    values = {}
    for exp in experiments:
        row = rows[exp]
        if not 0 <= sub < len(row):
            raise PredictionPlotError(
                f'{where}: subexperiment_idx {sub} is out of range for experiment {exp} '
                f'({len(row)} sub-experiment(s)).')
        value = row[sub]
        if _is_number(value):
            values[exp] = float(value)
        elif isinstance(value, str):
            duration = float(sim_times[exp][sub])
            if isinstance(reduce, dict) and reduce['at'] > duration:
                raise PredictionPlotError(
                    f'{where}: reduce at {reduce["at"]:g} s is after the end of sub-experiment '
                    f'{sub} of experiment {exp} ({duration:g} s long).')
            trace = _trace(where, protocol_info, value)
            values[exp] = None if reduce is None else reduce_trace(*trace, duration, reduce)
        else:
            raise PredictionPlotError(
                f'{where}: params_to_change[{key!r}][{exp}][{sub}] is {value!r}, neither a '
                f'number nor a trace name.')
    return values


def resolve_plot(plot, index, groups, protocol_info):
    """The experiments of one feature plot and how each is placed:
    ``(y_items, x_items_or_None, x_inputs_or_None, series_inputs_or_None)``, each by
    experiment; an input is None in an experiment where it is a trace with no ``reduce``.
    Raises PredictionPlotError for a plot whose references do not resolve; its keys and types
    are the obs_data schema's (``$defs/prediction_plot``)."""
    where = _where(plot, index)
    kind, x = plot['kind'], plot['x']
    if kind == 'feature_vs_feature' and not isinstance(x, str):
        raise PredictionPlotError(f'{where}: a feature_vs_feature plot names its x group as a '
                                  f'string, got {x!r}.')
    if kind == 'feature_vs_input' and not isinstance(x, dict):
        raise PredictionPlotError(f'{where}: a feature_vs_input plot gives x as an input '
                                  f'reference, got {x!r}.')

    y_items = _by_experiment(f'{where} y', plot['y'], groups)
    experiments = sorted(y_items)
    x_items = x_inputs = series = None
    if kind == 'feature_vs_feature':
        x_items = _by_experiment(f'{where} x', x, groups)
        if sorted(x_items) != experiments:
            raise PredictionPlotError(
                f'{where}: the x group {x!r} covers experiments {sorted(x_items)} and the y '
                f'group {plot["y"]!r} covers {experiments}; every point needs both.')
    else:
        x_inputs = _input_value(f'{where} x', x, protocol_info, experiments)
    if plot.get('series') is not None:
        series = _input_value(f'{where} series', plot['series'], protocol_info, experiments)
    return y_items, x_items, x_inputs, series


def resolve_time_course(plot, index, protocol_info):
    """``(input key, output, experiments)`` of an input_and_output or output_vs_input plot.
    Raises PredictionPlotError for an input key the protocol does not set, an experiment out
    of range, or an output that is not a variable name."""
    where = _where(plot, index)
    if not isinstance(plot.get('input'), dict) or not isinstance(plot.get('output'), str):
        raise PredictionPlotError(f'{where}: a {plot["kind"]} plot gives "input" '
                                  f'({{"params_to_change": <key>}}) and "output" (a name).')
    key = plot['input']['params_to_change']
    _rows(f'{where} input', key, protocol_info)
    num_experiments = len(protocol_info['sim_times'])
    experiments = plot.get('experiments')
    if experiments is None:
        experiments = range(num_experiments)
    bad = [e for e in experiments if not 0 <= int(e) < num_experiments]
    if bad:
        raise PredictionPlotError(f'{where}: experiments {bad} are out of range (the protocol '
                                  f'has {num_experiments}).')
    output = plot['output'].strip()
    if not output or any(not part.strip() for part in output.split('/')):
        raise PredictionPlotError(
            f'{where}: output {plot["output"]!r} is not a variable name. Write '
            f'<vessel>/<variable>, a nested vessel by its path (SN/soma/membrane/V), or a '
            f'global by its bare name.')
    return key, output, sorted({int(e) for e in experiments})


def check_prediction_plots(plots, gt_df, prediction_info, protocol_info):
    """The rules the schema cannot check: unique names, groups that exist and cover the same
    experiments, and input references and experiments the protocol has."""
    if plots is None:
        return []
    groups = item_groups(gt_df, prediction_info)
    seen = set()
    for i, plot in enumerate(plots):
        if plot['kind'] in TIME_COURSE_KINDS:
            resolve_time_course(plot, i, protocol_info)
        else:
            resolve_plot(plot, i, groups, protocol_info)
        if plot['name'] in seen:
            raise PredictionPlotError(f'prediction_plots[{i}]: name {plot["name"]!r} is used by '
                                      f'an earlier plot; names are unique.')
        seen.add(plot['name'])
    return plots


def _check_scalar(item, prediction_info, funcs, where):
    from libcuflynx.param_id.prediction_features import is_scalar_feature
    if item['source'] == 'prediction_items':
        ok = is_scalar_feature(prediction_info, item['index'], funcs)
    else:
        ok = item['data_type'] == 'constant'
    if not ok:
        raise PredictionPlotError(f'{where}: item {item["name"]!r} is not a scalar feature, so '
                                  f'it cannot be a point.')


def _skip_message(where, which, ref, exp, protocol_info):
    key, sub = ref['params_to_change'], ref['subexperiment_idx']
    name = protocol_info['params_to_change'][key][exp][sub]
    return (f'{where}: experiment {exp} has no point: its {which} input params_to_change'
            f'[{key!r}] is the trace {name!r} in sub-experiment {sub}, which has no single '
            f'value. Give the {which} reference a "reduce" ("final", "mean", "max", "min" or '
            f'{{"at": <seconds>}}) to plot it.')


def plot_points(plot, index, groups, protocol_info, model_values, prediction_info=None,
                funcs=None):
    """``[{series, x, y, data_x, data, std, experiment_idx, data_references}]`` for one
    feature plot; ``data_references`` are the y item's and, for a feature x, the x item's.
    ``model_values`` is ``{data_item_name: model feature value}``. A data_item without one (no
    calibration to evaluate it) gives a data-only point, ``y`` None; ``data``/``std`` are None
    for an item without a value. An experiment whose x or series input is a trace with no
    ``reduce`` is left out, with a PredictionPlotWarning."""
    y_items, x_items, x_inputs, series = resolve_plot(plot, index, groups, protocol_info)
    where = _where(plot, index)

    def model(item):
        _check_scalar(item, prediction_info, funcs, where)
        if item['name'] in model_values:
            return float(model_values[item['name']])
        if item['source'] == 'data_items':
            return None
        raise PredictionPlotError(f'{where}: no model value for item {item["name"]!r}.')

    points = []
    for exp, y_item in sorted(y_items.items()):
        unset = [(which, plot[which]) for which, inputs in (('x', x_inputs), ('series', series))
                 if inputs is not None and inputs[exp] is None]
        if unset:
            warnings.warn(_skip_message(where, *unset[0], exp, protocol_info),
                          PredictionPlotWarning, stacklevel=2)
            continue
        y = model(y_item)
        references = [y_item['data_reference']]
        if x_items:
            references.append(x_items[exp]['data_reference'])
            x, data_x = model(x_items[exp]), _number(x_items[exp]['value'])
            x = data_x if x is None else x
            data_x = x if data_x is None else data_x
        else:
            x = data_x = x_inputs[exp]
        points.append({'series': series[exp] if series else None, 'x': x, 'y': y,
                       'data_x': data_x, 'data': _number(y_item['value']),
                       'std': _number(y_item['std']), 'experiment_idx': exp,
                       'data_references': references})
    return sorted(points, key=lambda p: (p['series'] is not None, p['series'] or 0, p['x']))


def _number(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0] if len(value) else None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else value


def file_stem(name):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_') or 'prediction_plot'


def _axis_label(plot, groups, which, unit_of):
    ref = plot[which]
    if isinstance(ref, str):
        return f'{ref} [{unit_of(ref)}]' if unit_of(ref) else ref
    label = f'{ref["params_to_change"]} (sub-experiment {ref["subexperiment_idx"]}'
    reduce = ref.get('reduce')
    if isinstance(reduce, dict):
        label += f', at {reduce["at"]:g} s'
    elif reduce:
        label += f', {reduce}'
    return label + ')'


# --- time-course kinds -------------------------------------------------------------------

def _model_output_name(sim_helper, output):
    """The name of ``output`` the solver resolves: as written, else a bare name as
    ``global/<name>``. None when it resolves neither (a helper with no resolver takes it as
    written)."""
    resolve = getattr(sim_helper, '_resolve_name', None)
    if resolve is None:
        return output
    for name in [output] + ([f'global/{output}'] if '/' not in output else []):
        if resolve(name)[0] not in (None, 'None'):
            return name
    return None


def record_time_courses(executor, protocol_info, plots, id_param_names=None,
                        id_param_vals=None):
    """``{output: {experiment: (t, values)}}`` for the time-course plots among ``plots``, t being
    the time since the end of the experiment's warm-up.

    One ``executor.run_protocol`` pass over the experiments they draw, with their outputs as
    the result variables -- how ``predictions.predict_on`` records prediction items -- and each
    experiment's sub-experiments joined by ``predictions.join_segments``.
    ``id_param_names``/``id_param_vals`` (a best fit) are set at the start of each experiment.
    Raises PredictionPlotError, before anything runs, for an output the model does not have.
    """
    from libcuflynx.protocol_runners.predictions import join_segments
    wanted, experiments = {}, set()
    for index, plot in enumerate(plots or []):
        if plot['kind'] in TIME_COURSE_KINDS:
            _, output, exps = resolve_time_course(plot, index, protocol_info)
            wanted.setdefault(output, []).append(plot['name'])
            experiments.update(exps)
    if not wanted:
        return {}
    names = {output: _model_output_name(executor.sim_helper, output) for output in wanted}
    missing = [f'{output!r} (plot {", ".join(repr(n) for n in wanted[output])})'
               for output, name in names.items() if name is None]
    if missing:
        raise PredictionPlotError(
            f'prediction_plots output(s) the model does not have: {"; ".join(missing)}. An '
            f'output is <vessel>/<variable>, a nested vessel by its path (SN/soma/membrane/V), '
            f'or a global by its bare name.')
    outputs = list(wanted)
    ok, by_sub, _, t_by_exp = executor.run_protocol(
        protocol_info, id_param_names=id_param_names, id_param_vals=id_param_vals,
        result_variables=[[names[output]] for output in outputs],
        exp_indices=sorted(experiments))
    if not ok:
        raise RuntimeError('the protocol simulation for prediction_plots failed')
    courses = {output: {} for output in outputs}
    for exp in sorted(experiments):
        sim_times = protocol_info['sim_times'][exp]
        joined = join_segments(
            [[result[0] for result in by_sub[(exp, sub)]] for sub in range(len(sim_times))],
            sim_times, executor.sim_helper.dt)
        for output, values in zip(outputs, joined):
            courses[output][exp] = (np.asarray(t_by_exp[exp], dtype=float), values)
    return courses


def _sub_starts(protocol_info, exp):
    """The start of each sub-experiment of ``exp`` and the end of the last, since the end of the
    warm-up; and the sub-experiments' lengths."""
    sim_times = [float(s) for s in protocol_info['sim_times'][exp]]
    return np.concatenate([[0.0], np.cumsum(sim_times)]), sim_times


def input_time_course(protocol_info, key, exp):
    """``(t, values)`` of ``params_to_change[key]`` over experiment ``exp``, t since the end of
    its warm-up: a number is held over its sub-experiment (a step at each boundary), a trace
    follows its waveform."""
    starts, sim_times = _sub_starts(protocol_info, exp)
    t, v = [], []
    for sub, value in enumerate(_row(protocol_info, key, exp)):
        if isinstance(value, str):
            knots, vals = _applied(*_trace(key, protocol_info, value), sim_times[sub])
        else:
            knots, vals = np.array([0.0, sim_times[sub]]), np.full(2, float(value))
        t.append(knots + starts[sub])
        v.append(vals)
    return np.concatenate(t), np.concatenate(v)


def input_at(protocol_info, key, exp, times):
    """``params_to_change[key]`` in experiment ``exp`` at ``times`` (since the end of the
    warm-up). A time on a boundary belongs to the sub-experiment it ends, as in the joined
    outputs."""
    starts, _ = _sub_starts(protocol_info, exp)
    row = _row(protocol_info, key, exp)
    times = np.asarray(times, dtype=float)
    subs = np.clip(np.searchsorted(starts[1:], times, side='left'), 0, len(row) - 1)
    values = np.empty_like(times)
    for sub, value in enumerate(row):
        mask = subs == sub
        if isinstance(value, str):
            t, v = _trace(key, protocol_info, value)
            values[mask] = np.interp(times[mask] - starts[sub], t, v)
        else:
            values[mask] = float(value)
    return values


def _has_operation(operation):
    if operation is None or (isinstance(operation, float) and np.isnan(operation)):
        return False
    return str(operation).strip() not in ('', 'None', 'nan')


def series_data(gt_df, output, exp, protocol_info):
    """``[(name, t, value, std_or_None)]``: the data_items of experiment ``exp`` that have
    ``output`` as a series (its first operand, no operation), t since the end of the warm-up."""
    from libcuflynx.utilities.parameter_names import output_model_name
    if gt_df is None or not len(gt_df):
        return []
    starts, _ = _sub_starts(protocol_info, exp)
    found = []
    for row in gt_df.to_dict('records'):
        operands = row.get('operands')
        if (row.get('data_type') != 'series' or int(row.get('experiment_idx', 0) or 0) != exp
                or not isinstance(operands, (list, tuple)) or not operands
                or output_model_name(operands[0]) != output_model_name(output)
                or _has_operation(row.get('operation')) or row.get('value') is None):
            continue
        value = np.asarray(row['value'], dtype=float)
        t = (starts[int(row.get('subexperiment_idx', 0) or 0)]
             + np.arange(len(value)) * float(row['obs_dt']))
        std = row.get('std')
        std = (None if std is None or np.all(np.isnan(np.asarray(std, dtype=float)))
               else np.broadcast_to(np.asarray(std, dtype=float), value.shape))
        found.append((str(row['data_item_name']), t, value, std))
    return found


def _experiment_title(protocol_info, exp):
    labels = protocol_info.get('experiment_labels') or []
    label = labels[exp] if exp < len(labels) else None
    if label is None or str(label).strip() in ('', 'None'):
        return f'experiment {exp}'
    return f'experiment {exp} ({label})'


def _draw_input_and_output(plot, index, protocol_info, time_courses, gt_df, out_dir, plt):
    """One figure per experiment, ``<out_dir>/<name>_exp_<idx>.png``; returns the paths."""
    key, output, experiments = resolve_time_course(plot, index, protocol_info)
    paths = []
    for exp in experiments:
        t, y = time_courses[output][exp]
        fig, ax_in = plt.subplots(figsize=(6, 4))
        ax_out = ax_in.twinx()
        t_in, v_in = input_time_course(protocol_info, key, exp)
        ax_in.plot(t_in, v_in, color='C0', label=key)
        ax_out.plot(t, y, color='C1', label=output)
        for name, t_data, data, std in series_data(gt_df, output, exp, protocol_info):
            ax_out.errorbar(t_data, data, yerr=std, fmt='s', ms=3, mfc='none', color='C1',
                            alpha=0.7, label=f'data ({name})')
        starts, _ = _sub_starts(protocol_info, exp)
        for boundary in starts[1:-1]:
            ax_in.axvline(boundary, color='0.6', linestyle='--', linewidth=0.8)
        ax_in.set_xlabel('time since the end of the warm-up [s]')
        ax_in.set_ylabel(key, color='C0')
        ax_out.set_ylabel(output, color='C1')
        ax_in.set_title(f'{plot["name"]}: {_experiment_title(protocol_info, exp)}')
        handles = [h for ax in (ax_in, ax_out) for h in ax.get_legend_handles_labels()[0]]
        ax_in.legend(handles, [h.get_label() for h in handles], fontsize=8, loc='best')
        fig.tight_layout()
        path = os.path.join(out_dir, f'{file_stem(plot["name"])}_exp_{exp}.png')
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def _draw_output_vs_input(plot, index, protocol_info, time_courses, out_dir, plt):
    """``<out_dir>/<name>.png``: the output against the input, one line per experiment."""
    key, output, experiments = resolve_time_course(plot, index, protocol_info)
    fig, ax = plt.subplots(figsize=(5, 4))
    colours = protocol_info.get('experiment_colors') or []
    for n, exp in enumerate(experiments):
        t, y = time_courses[output][exp]
        colour = colours[exp] if exp < len(colours) and colours[exp] else f'C{n % 10}'
        ax.plot(input_at(protocol_info, key, exp, t), y, color=colour,
                label=_experiment_title(protocol_info, exp))
    ax.set_xlabel(key)
    ax.set_ylabel(output)
    ax.set_title(plot['name'])
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{file_stem(plot["name"])}.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_prediction_plots(plots, gt_df, prediction_info, protocol_info, model_values, out_dir,
                          funcs=None, units=None, time_courses=None):
    """Draw every plot; returns the paths: ``<out_dir>/<name>.png``, and for input_and_output
    ``<out_dir>/<name>_exp_<idx>.png`` per experiment.

    ``units`` is ``{group label: unit}`` for the axis labels (optional). ``time_courses`` is
    ``record_time_courses``' result, which the time-course kinds are drawn from.
    """
    if not plots:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = item_groups(gt_df, prediction_info)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    # Names and labels are shown as written: paperPlotSetup's cmr10 has no "_" glyph (its slot
    # holds a dot accent), so "I_peak" would print as "I˙peak".
    with matplotlib.rc_context({'font.family': 'serif', 'font.serif': ['DejaVu Serif']}):
        for index, plot in enumerate(plots):
            if plot['kind'] not in TIME_COURSE_KINDS:
                paths.append(_draw(plot, index, groups, protocol_info, model_values, out_dir,
                                   prediction_info, funcs, units, plt))
                continue
            if time_courses is None:
                raise PredictionPlotError(
                    f'{_where(plot, index)}: a {plot["kind"]} plot is drawn from the outputs '
                    f'record_time_courses records; none were given.')
            if plot['kind'] == 'input_and_output':
                paths += _draw_input_and_output(plot, index, protocol_info, time_courses,
                                                gt_df, out_dir, plt)
            else:
                paths.append(_draw_output_vs_input(plot, index, protocol_info, time_courses,
                                                   out_dir, plt))
    return paths


def _draw(plot, index, groups, protocol_info, model_values, out_dir, prediction_info, funcs,
          units, plt):
    """One figure, ``<out_dir>/<name>.png``; returns its path."""
    points = plot_points(plot, index, groups, protocol_info, model_values,
                         prediction_info=prediction_info, funcs=funcs)
    fig, ax = plt.subplots(figsize=(5, 4))
    for colour, value in enumerate(dict.fromkeys(p['series'] for p in points)):
        line = [p for p in points if p['series'] == value]
        label = None if value is None else f'{plot["series"]["params_to_change"]} = {value:g}'
        modelled = [p for p in line if p['y'] is not None]
        if modelled:
            ax.plot([p['x'] for p in modelled], [p['y'] for p in modelled], 'o-',
                    color=f'C{colour}', label=label)
        data = [p for p in line if p['data'] is not None]
        if data:
            ax.errorbar([p['data_x'] for p in data], [p['data'] for p in data],
                        yerr=[p['std'] or 0.0 for p in data], fmt='s', mfc='none',
                        color=f'C{colour}', capsize=3, label='data' if value is None else None)
    unit_of = (units or {}).get
    ax.set_xlabel(_axis_label(plot, groups, 'x', unit_of))
    ax.set_ylabel(_axis_label(plot, groups, 'y', unit_of))
    ax.set_title(plot['name'])
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    # the source of the data drawn, from the items' data_reference
    source_line = data_reference_line(r for p in points if p['data'] is not None
                                      for r in p['data_references'])
    if source_line:
        fig.text(0.01, 0.01, source_line, fontsize=7, ha='left', va='bottom')
    fig.tight_layout(rect=(0, 0.04, 1, 1) if source_line else None)
    path = os.path.join(out_dir, f'{file_stem(plot["name"])}.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def group_units(gt_df, prediction_info):
    """``{group label: unit}`` (the first item's)."""
    units = {}
    if gt_df is not None and len(gt_df):
        for row in gt_df.to_dict('records'):
            units.setdefault(str(row['item_name_for_plotting']), row.get('unit'))
    info = prediction_info or {}
    for label, unit in zip(info.get('item_names_for_plotting') or [], info.get('units') or []):
        units.setdefault(str(label), unit)
    return units
