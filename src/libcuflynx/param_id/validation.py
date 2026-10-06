"""
Validation of a calibrated model against held-out data.

Held-out data lives in the obs_data it validates, as ``prediction_items`` that carry a
``value`` (and ``data_type``, ``std``, and ``obs_dt`` for a series). A prediction item is
never scored during calibration; after it, the model's prediction is compared with the item's
data here.

    results = validation_results(prediction_info, time_per_exp, prediction_per_item,
                                 time_per_item=None)
    write_validation_results(results, output_dir)      # -> validation_results.json

Each item is compared over its own segment -- its ``(experiment_idx, subexperiment_idx)``, the
sub-experiment defaulting to the experiment's last. ``time_per_item[i]`` is the time of item
``i``'s recorded samples (or ``time_per_exp[exp_idx]``, for callers that record one segment per
experiment), and ``prediction_per_item[i]`` its operand series on that grid: a list with one
series per operand, or a bare array for its one operand. Time is taken **from the start of the
segment**, the convention data_items use for a series in a sub-experiment: sample k of a
held-out series is at ``k * obs_dt`` from the start of its sub-experiment. A series is compared
at the observation times the simulation reaches, the model linearly interpolated onto them (as
paramID does for data_items). A constant is compared with the model's value at the end of the
segment.

An item with an ``operation`` is compared with the operation of its operands:

- a scalar item (``data_type: constant``) with ``operation(*operands, **operation_kwargs)``, a
  feature such as the max of a pressure over the segment (``t`` is then the segment's end);
- a series item with the operation's series: ``operation(..., series_output=True)`` for a
  ``@series_to_constant`` operation, else the operation's own (series) result -- the same rule
  data_items follow.

Per item the result holds the scores and the series for plotting:

    rmse          root mean square of model - data
    nrmse         rmse / the data's range (max - min; its magnitude for a constant)
    mean_abs_z    mean |model - data| / std          (None without std)
    within_2std   fraction of points with |model - data| <= 2 std  (None without std)
    t, data, std, model, operation
"""
import json
import os

import numpy as np

VALIDATION_RESULTS_FILE = 'validation_results.json'


def _as_array(x):
    return None if x is None else np.atleast_1d(np.asarray(x, dtype=float))


def _operand_series(prediction):
    """An item's operand series: a list/tuple is one series per operand, an array is one."""
    if isinstance(prediction, (list, tuple)):
        return [np.asarray(series, dtype=float).ravel() for series in prediction]
    return [np.asarray(prediction, dtype=float).ravel()]


def _item_result(name, operand, unit, data_type, value, std, obs_dt, t_sim, model,
                 operation=None, feature=None):
    data = _as_array(value)
    std = _as_array(std)
    t_sim = np.asarray(t_sim, dtype=float)
    t_sim = t_sim - t_sim[0]                 # from the start of the segment
    model = np.asarray(model, dtype=float).ravel()
    if feature is not None:
        # a scalar feature: operation(operands) over the run, compared with the constant
        t_obs = np.array([t_sim[-1]])
        data = data[:1]
        model_at = np.array([feature], dtype=float)
    elif data_type == 'series':
        t_obs = np.arange(data.size) * float(obs_dt)
        keep = t_obs <= t_sim[-1] + 1e-12 * max(1.0, abs(t_sim[-1]))
        t_obs, data = t_obs[keep], data[keep]
        if std is not None and std.size > 1:
            std = std[:keep.size][keep]
        model_at = np.interp(t_obs, t_sim, model)
    else:
        t_obs = np.array([t_sim[-1]])
        data = data[:1]
        model_at = model[-1:]
    if std is not None and std.size == 1 and data.size > 1:
        std = np.full(data.shape, float(std[0]))
    diff = model_at - data
    rmse = float(np.sqrt(np.mean(diff ** 2))) if data.size else None
    span = float(np.ptp(data)) if data.size > 1 else float(np.max(np.abs(data))) if data.size else 0.0
    nrmse = rmse / span if rmse is not None and span > 0 else None
    z = np.abs(diff) / std if std is not None and np.all(std > 0) else None
    return {
        'data_item_name': name,
        'operand': operand,
        'operation': operation,
        'unit': unit,
        'data_type': data_type,
        'n_points': int(data.size),
        'rmse': rmse,
        'nrmse': nrmse,
        'mean_abs_z': float(np.mean(z)) if z is not None and z.size else None,
        'within_2std': float(np.mean(z <= 2.0)) if z is not None and z.size else None,
        't': t_obs.tolist(),
        'data': data.tolist(),
        'std': std.tolist() if std is not None else None,
        'model': model_at.tolist(),
    }


def _series_output(func, operands, kwargs):
    """What an operation gives a series item: its series_output for a @series_to_constant
    operation, else its own result (as for a data_item series)."""
    if getattr(func, 'series_to_constant', False):
        return func(*operands, series_output=True, **kwargs)
    return func(*operands, **kwargs)


def validation_results(prediction_info, time_per_exp, prediction_per_item,
                       operation_funcs_dict=None, time_per_item=None):
    """The validation of every prediction item that carries data (see the module docstring).

    ``prediction_info`` is the parser's (``values``, ``data_types``, ``stds``, ``obs_dts``,
    ``operands``, ``units``, ``data_item_names``, ``experiment_idxs``, ``operations``,
    ``operation_kwargs``). Items without data are left out, so an obs_data with no held-out
    data gives an empty ``items`` list. ``operation_funcs_dict`` is the operation registry the
    run uses (user funcs included); the built-in one when omitted.
    """
    from libcuflynx.param_id import prediction_features

    from libcuflynx.param_id.operation_funcs import resolve_operation_kwargs

    values = prediction_info.get('values') or []
    operations = prediction_info.get('operations') or [None] * len(values)
    data_types = prediction_info.get('data_types') or [None] * len(values)
    features, series_outputs = {}, {}
    op_idxs = [i for i, op in enumerate(operations) if op is not None]
    if op_idxs:
        if operation_funcs_dict is None:
            from libcuflynx.parsers.PrimitiveParsers import scriptFunctionParser
            operation_funcs_dict = scriptFunctionParser().get_operation_funcs_dict('numpy')
        names = prediction_info['data_item_names']
        temp_results = {}
        # every item with an operation in item order, so an operation_kwargs reference to an
        # earlier one resolves whether or not that one carries data
        for i in op_idxs:
            operands = _operand_series(prediction_per_item[i])
            if data_types[i] == 'series':
                func = operation_funcs_dict.get(operations[i])
                if func is None:
                    raise ValueError(f'prediction item {names[i]!r}: operation '
                                     f'{operations[i]!r} is not a registered operation func.')
                kwargs = resolve_operation_kwargs(
                    (prediction_info.get('operation_kwargs') or [{}] * len(values))[i] or {},
                    func, operation_name=operations[i], data_item_name=str(names[i]),
                    temp_results=temp_results, num_operands=len(operands))
                result = np.asarray(_series_output(func, operands, kwargs), dtype=float).ravel()
                if result.size != operands[0].size:
                    raise ValueError(
                        f'prediction item {names[i]!r} is a series, but operation '
                        f'{operations[i]!r} returned {result.size} value(s) for a trace of '
                        f'{operands[0].size}. A series item needs an operation that returns a '
                        f'series (or has series_output).')
                series_outputs[i] = result
                temp_results[str(names[i])] = result
            elif prediction_features.is_scalar_feature(prediction_info, i,
                                                       operation_funcs_dict):
                features[i] = prediction_features.evaluate_feature(
                    prediction_info, i, operation_funcs_dict, operands,
                    temp_results=temp_results)
                temp_results[str(names[i])] = features[i]

    items = []
    for i, value in enumerate(values):
        if value is None:
            continue
        exp_idx = int(prediction_info['experiment_idxs'][i])
        operands = prediction_info['operands'][i]
        model = series_outputs.get(i)
        if model is None:
            model = _operand_series(prediction_per_item[i])[0]
        t_sim = time_per_item[i] if time_per_item is not None else time_per_exp[exp_idx]
        items.append(_item_result(
            prediction_info['data_item_names'][i], str(operands[0]) if len(operands) else '',
            prediction_info['units'][i], prediction_info['data_types'][i], value,
            (prediction_info.get('stds') or [None] * len(values))[i],
            (prediction_info.get('obs_dts') or [None] * len(values))[i],
            t_sim, model, operation=operations[i], feature=features.get(i)))
    return {'items': items}


def write_validation_results(results, output_dir):
    """Writes ``validation_results.json`` in ``output_dir``; returns its path (None when there
    is nothing to validate, so no file claims a validation that did not happen)."""
    if not results.get('items'):
        return None
    path = os.path.join(output_dir, VALIDATION_RESULTS_FILE)
    with open(path, 'w') as f:
        json.dump(results, f, indent=1)
    return path
