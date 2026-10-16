"""
``prediction_plots``: figures of one scalar feature against another across experiments (#568).

An obs_data may list figures at its top level::

    "prediction_plots": [
      {"name": "Peak I_Na vs step potential", "kind": "feature_vs_input",
       "x": {"params_to_change": "clamp/V_cmd", "subexperiment_idx": 1},
       "y": "I_peak", "series": null}]

A *group* is the items (data_items and prediction_items) that share an
``item_name_for_plotting`` -- the explicit label or its default. A point is one experiment: its y
is the y group's item for that experiment, its x the x group's item for the same experiment
(``feature_vs_feature``) or the number ``params_to_change[key]`` is set to in that
sub-experiment (``feature_vs_input``). ``series`` (an input reference too) draws one line per
distinct value of that input. Points are sorted by x within a series; an item with a
``value``/``std`` adds the data as error bars.

``check_prediction_plots`` runs the structural rules when the obs_data is parsed.
``plot_prediction_plots`` checks that every item is a scalar feature (which needs the operation
funcs) and draws the figures from the model's feature values.
"""

import os
import re

import numpy as np

KINDS = ('feature_vs_feature', 'feature_vs_input')


class PredictionPlotError(ValueError):
    '''A prediction_plots entry that cannot be drawn.'''


def item_groups(gt_df, prediction_info):
    """``{label: [item]}``, an item being a dict with ``source`` (``data_items`` /
    ``prediction_items``), ``index``, ``name``, ``experiment_idx``, ``value``, ``std`` and
    ``data_type``."""
    groups = {}
    if gt_df is not None and len(gt_df):
        for i, row in enumerate(gt_df.to_dict('records')):
            groups.setdefault(str(row['item_name_for_plotting']), []).append({
                'source': 'data_items', 'index': i, 'name': str(row['data_item_name']),
                'experiment_idx': int(row.get('experiment_idx', 0) or 0),
                'value': row.get('value'), 'std': row.get('std'),
                'data_type': row.get('data_type')})
    info = prediction_info or {}
    names = info.get('data_item_names') or []
    for i, label in enumerate(info.get('item_names_for_plotting') or []):
        groups.setdefault(str(label), []).append({
            'source': 'prediction_items', 'index': i, 'name': str(names[i]),
            'experiment_idx': int(info['experiment_idxs'][i]),
            'value': (info.get('values') or [None] * len(names))[i],
            'std': (info.get('stds') or [None] * len(names))[i],
            'data_type': (info.get('data_types') or [None] * len(names))[i]})
    return groups


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


def _input_value(where, ref, protocol_info, experiments):
    """``{experiment: value}`` of an input reference; raises unless it is a number in each."""
    key, sub = ref['params_to_change'], ref['subexperiment_idx']
    params = (protocol_info or {}).get('params_to_change') or {}
    if key not in params:
        raise PredictionPlotError(
            f'{where}: {key!r} is not a params_to_change key ({sorted(params)}).')
    values = {}
    for exp in experiments:
        row = params[key][exp]
        if not 0 <= sub < len(row):
            raise PredictionPlotError(
                f'{where}: subexperiment_idx {sub} is out of range for experiment {exp} '
                f'({len(row)} sub-experiment(s)).')
        value = row[sub]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise PredictionPlotError(
                f'{where}: params_to_change[{key!r}][{exp}][{sub}] is {value!r}, not a number '
                f'(a trace or shape has no single value to plot against).')
        values[exp] = float(value)
    return values


def resolve_plot(plot, index, groups, protocol_info):
    """The experiments of one plot and how each is placed:
    ``(y_items, x_items_or_None, x_inputs_or_None, series_inputs_or_None)``, each by
    experiment. Raises PredictionPlotError for a plot whose references do not resolve; its
    keys and types are the obs_data schema's (``$defs/prediction_plot``)."""
    where = f'prediction_plots[{index}] ({plot["name"]!r})'
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


def check_prediction_plots(plots, gt_df, prediction_info, protocol_info):
    """The rules the schema cannot check: unique names, groups that exist and cover the same
    experiments, and input references that resolve to numbers."""
    if plots is None:
        return []
    groups = item_groups(gt_df, prediction_info)
    seen = set()
    for i, plot in enumerate(plots):
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


def plot_points(plot, index, groups, protocol_info, model_values, prediction_info=None,
                funcs=None):
    """``[{series, x, y, data_x, data, std, experiment_idx}]`` for one plot. ``model_values`` is
    ``{data_item_name: model feature value}``. A data_item without one (no calibration to
    evaluate it) gives a data-only point, ``y`` None; ``data``/``std`` are None for an item
    without a value."""
    y_items, x_items, x_inputs, series = resolve_plot(plot, index, groups, protocol_info)
    where = f'prediction_plots[{index}] ({plot["name"]!r})'

    def model(item):
        _check_scalar(item, prediction_info, funcs, where)
        if item['name'] in model_values:
            return float(model_values[item['name']])
        if item['source'] == 'data_items':
            return None
        raise PredictionPlotError(f'{where}: no model value for item {item["name"]!r}.')

    points = []
    for exp, y_item in sorted(y_items.items()):
        y = model(y_item)
        if x_items:
            x, data_x = model(x_items[exp]), _number(x_items[exp]['value'])
            x = data_x if x is None else x
            data_x = x if data_x is None else data_x
        else:
            x = data_x = x_inputs[exp]
        points.append({'series': series[exp] if series else None, 'x': x, 'y': y,
                       'data_x': data_x, 'data': _number(y_item['value']),
                       'std': _number(y_item['std']), 'experiment_idx': exp})
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
    return f'{ref["params_to_change"]} (sub-experiment {ref["subexperiment_idx"]})'


def plot_prediction_plots(plots, gt_df, prediction_info, protocol_info, model_values, out_dir,
                          funcs=None, units=None):
    """Draw every plot as ``<out_dir>/<name>.png``; returns the paths.

    ``units`` is ``{group label: unit}`` for the axis labels (optional).
    """
    if not plots:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = item_groups(gt_df, prediction_info)
    os.makedirs(out_dir, exist_ok=True)
    # Names and labels are shown as written: paperPlotSetup's cmr10 has no "_" glyph (its slot
    # holds a dot accent), so "I_peak" would print as "I˙peak".
    with matplotlib.rc_context({'font.family': 'serif', 'font.serif': ['DejaVu Serif']}):
        return [_draw(plot, index, groups, protocol_info, model_values, out_dir,
                      prediction_info, funcs, units, plt)
                for index, plot in enumerate(plots)]


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
    fig.tight_layout()
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
