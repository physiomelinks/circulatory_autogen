"""
Validation of a calibrated model against held-out data.

Held-out data lives in the obs_data it validates, as ``prediction_items`` that carry a
``value`` (and ``data_type``, ``std``, and ``obs_dt`` for a series). The parser checks a ``std``
as it does a data item's -- one positive number, or for a series one per point -- so every
point of an item with a std gets a z-score. A prediction item is
never scored during calibration; after it, the model's prediction is compared with the item's
data here.

    results = validation_results(prediction_info, time_per_exp, prediction_per_item)
    write_validation_results(results, output_dir)      # -> validation_results.json

``time_per_exp[exp_idx]`` is the experiment's simulation time (from the start of the
experiment, pre_time removed) and ``prediction_per_item[i]`` the model series of prediction
item ``i`` on that grid. A series is compared at its observation times ``k * obs_dt`` that the
simulation reaches, the model linearly interpolated onto them (as paramID does for data_items).
A constant is compared with the model's value at the end of the experiment.

Per item the result holds the scores and the series for plotting:

    rmse          root mean square of model - data
    nrmse         rmse / the data's range (max - min; its magnitude for a constant)
    mean_abs_z    mean |model - data| / std          (None without std)
    within_2std   fraction of points with |model - data| <= 2 std  (None without std)
    t, data, std, model
"""
import json
import os

import numpy as np

VALIDATION_RESULTS_FILE = 'validation_results.json'


def _as_array(x):
    return None if x is None else np.atleast_1d(np.asarray(x, dtype=float))


def _item_result(name, operand, unit, data_type, value, std, obs_dt, t_sim, model):
    data = _as_array(value)
    std = _as_array(std)
    t_sim = np.asarray(t_sim, dtype=float)
    model = np.asarray(model, dtype=float).ravel()
    if data_type == 'series':
        t_obs = np.arange(data.size) * float(obs_dt)
        keep = t_obs <= t_sim[-1] + 1e-12 * max(1.0, abs(t_sim[-1]))
        t_obs, data = t_obs[keep], data[keep]
        if std is not None:
            std = std[keep]     # the parser made it one positive std per point
        model_at = np.interp(t_obs, t_sim, model)
    else:
        t_obs = np.array([t_sim[-1]])
        data = data[:1]
        model_at = model[-1:]
    diff = model_at - data
    rmse = float(np.sqrt(np.mean(diff ** 2))) if data.size else None
    span = float(np.ptp(data)) if data.size > 1 else float(np.max(np.abs(data))) if data.size else 0.0
    nrmse = rmse / span if rmse is not None and span > 0 else None
    z = np.abs(diff) / std if std is not None else None
    return {
        'data_item_name': name,
        'operand': operand,
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


def validation_results(prediction_info, time_per_exp, prediction_per_item):
    """The validation of every prediction item that carries data (see the module docstring).

    ``prediction_info`` is the parser's (``values``, ``data_types``, ``stds``, ``obs_dts``,
    ``operands``, ``units``, ``data_item_names``, ``experiment_idxs``). Items without data
    are left out, so an obs_data with no held-out data gives an empty ``items`` list.
    """
    values = prediction_info.get('values') or []
    items = []
    for i, value in enumerate(values):
        if value is None:
            continue
        exp_idx = int(prediction_info['experiment_idxs'][i])
        operands = prediction_info['operands'][i]
        items.append(_item_result(
            prediction_info['data_item_names'][i], str(operands[0]) if len(operands) else '',
            prediction_info['units'][i], prediction_info['data_types'][i], value,
            (prediction_info.get('stds') or [None] * len(values))[i],
            (prediction_info.get('obs_dts') or [None] * len(values))[i],
            time_per_exp[exp_idx], prediction_per_item[i]))
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
