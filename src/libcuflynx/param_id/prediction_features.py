"""Prediction items as scalar features.

A ``prediction_item`` is never scored in calibration. When it states an ``operation`` (plus
optional ``operation_kwargs``, with the same vocabulary and checks as a data_item's) it
reduces its operands to one number -- a *prediction feature*, e.g. the max of a pressure over
an experiment. Such a feature is:

* compared with the item's held-out ``value`` after calibration (``param_id.validation``);
* an extra output of sensitivity analysis with ``sa_options.include_prediction_items: true``;
* an extra output of a trained emulator with ``emulator_settings.include_prediction_items:
  true``.

Every item is a scalar or a series, never both. A prediction item is a scalar feature when it
has an operation and is scalar: ``data_type: constant``, or no data_type and an operation that
reduces a series to a number (the ``@series_to_constant`` ones: max, min, mean, ...). Items
without an operation, and series items (with or without one), are skipped with a
:class:`PredictionFeatureWarning` naming them. A feature whose operation does not actually
return a single number raises :class:`NonScalarPredictionFeatureError`.

**Which trace is reduced.** The operands are recorded over the item's ``(experiment_idx,
subexperiment_idx)`` segment; ``subexperiment_idx`` defaults to the experiment's last
sub-experiment. ``save_prediction_data`` and the validation use the same segment, so the
feature an SA or an emulator reports is the number the validation reports.

This module is the one place that rule and the scalar check live, so SA, the emulator trainer,
the emulator at use time and the validation cannot disagree about what a prediction feature is.
"""
import warnings

import numpy as np

from libcuflynx.param_id.operation_funcs import resolve_operation_kwargs

#: The key both ``sa_options`` and ``emulator_settings`` read. Off by default.
INCLUDE_PREDICTION_ITEMS = 'include_prediction_items'


class PredictionFeatureWarning(UserWarning):
    """Some prediction items were left out of the features because they have no operation."""


class NonScalarPredictionFeatureError(ValueError):
    """A prediction item's operation returned something other than a single number."""


def include_prediction_items(options):
    """Whether an ``sa_options`` / ``emulator_settings`` block asks for prediction features."""
    return bool((options or {}).get(INCLUDE_PREDICTION_ITEMS, False))


def _column(prediction_info, key, n):
    values = (prediction_info or {}).get(key)
    if values is None:
        return [None] * n
    return list(values)


def _num_items(prediction_info):
    return len((prediction_info or {}).get('data_item_names') or [])


def _default_funcs():
    from libcuflynx.parsers.PrimitiveParsers import scriptFunctionParser
    return scriptFunctionParser().get_operation_funcs_dict('numpy')


def is_scalar_feature(prediction_info, idx, operation_funcs_dict=None):
    """Whether prediction item ``idx`` is a scalar feature (see the module docstring)."""
    n = _num_items(prediction_info)
    operation = _column(prediction_info, 'operations', n)[idx]
    data_type = _column(prediction_info, 'data_types', n)[idx]
    if operation is None or data_type == 'series':
        return False
    if data_type == 'constant':
        return True
    funcs = operation_funcs_dict if operation_funcs_dict is not None else _default_funcs()
    func = funcs.get(operation)
    if func is None:
        name = _column(prediction_info, 'data_item_names', n)[idx]
        raise ValueError(
            f'prediction item {name!r}: operation {operation!r} is not a registered operation '
            f'func. Register it (operation_funcs_external_path, or add_user_operation_func) or '
            f'use a built-in one.')
    return bool(getattr(func, 'series_to_constant', False))


def prediction_feature_indices(prediction_info, operation_funcs_dict=None, warn=True,
                               context='the analysis'):
    """Indices of the prediction items that are scalar features.

    The others are skipped; with ``warn`` a :class:`PredictionFeatureWarning` names them, so a
    user who expected a trace to appear as a feature is told why it did not.
    """
    n = _num_items(prediction_info)
    names = _column(prediction_info, 'data_item_names', n)
    if n and operation_funcs_dict is None:
        operation_funcs_dict = _default_funcs()
    keep = [i for i in range(n) if is_scalar_feature(prediction_info, i, operation_funcs_dict)]
    skipped = [str(names[i]) for i in range(n) if i not in keep]
    if warn and skipped:
        message = (
            f'include_prediction_items: {len(skipped)} prediction item(s) are not a scalar and '
            f'are not included in {context}: {skipped}. Only a prediction item with an '
            f'operation that reduces it to one number is a feature: data_type "constant", or '
            f'an operation such as "max" or "mean" on an item without a data_type. Items '
            f'without an operation, and series, are left out.')
        warnings.warn(message, PredictionFeatureWarning, stacklevel=2)
    return keep


def prediction_feature_names(prediction_info, indices):
    """The features' names: each item's ``data_item_name`` (unique across the obs_data)."""
    names = _column(prediction_info, 'data_item_names', _num_items(prediction_info))
    return [str(names[i]) for i in indices]


def feature_subexperiment(protocol_info, exp_idx):
    """The sub-experiment a prediction feature of experiment ``exp_idx`` is reduced over."""
    protocol_info = protocol_info or {}
    num_sub = protocol_info.get('num_sub_per_exp')
    if num_sub is None:
        sim_times = protocol_info.get('sim_times') or [[None]]
        num_sub = [len(times) for times in sim_times]
    return max(int(num_sub[int(exp_idx)]) - 1, 0)


def feature_segments(prediction_info, indices, protocol_info):
    """``(experiment, sub-experiment)`` per item: its ``subexperiment_idx``, else the
    experiment's last sub-experiment."""
    n = _num_items(prediction_info)
    exps = _column(prediction_info, 'experiment_idxs', n)
    subs = _column(prediction_info, 'subexperiment_idxs', n)
    return [(int(exps[i]),
             int(subs[i]) if subs[i] is not None else feature_subexperiment(protocol_info,
                                                                            exps[i]))
            for i in indices]


def feature_experiments(prediction_info, indices, protocol_info):
    """The experiments the features in ``indices`` need simulated."""
    return sorted({exp for exp, _ in feature_segments(prediction_info, indices, protocol_info)})


def labels_with_segments(prediction_info, indices, protocol_info):
    """``"<data_item_name> (Exp<e>, Sub<s>)"`` per feature -- the same form as a data_item's
    Sobol column, with the sub-experiment the operation is applied to."""
    names = prediction_feature_names(prediction_info, indices)
    return [f'{name} (Exp{exp}, Sub{sub})'
            for name, (exp, sub) in zip(names, feature_segments(prediction_info, indices,
                                                                 protocol_info))]


def result_variables(prediction_info, indices):
    """The operands to record for the features, one list per feature (for ``get_results``)."""
    operands = _column(prediction_info, 'operands', _num_items(prediction_info))
    return [list(operands[i]) for i in indices]


def as_scalar(value, name, operation):
    """``value`` as a float, or :class:`NonScalarPredictionFeatureError` naming the item."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise NonScalarPredictionFeatureError(
            f'prediction item {name!r}: operation {operation!r} returned {type(value).__name__}, '
            f'not a number. A prediction feature must be a scalar.') from error
    if array.size != 1:
        raise NonScalarPredictionFeatureError(
            f'prediction item {name!r}: operation {operation!r} returned {array.size} values '
            f'(shape {array.shape}), not a scalar. Only an operation that reduces the operands '
            f'to one number (max, min, mean, ...) makes a prediction item a feature.')
    return float(array.reshape(-1)[0])


def evaluate_feature(prediction_info, idx, operation_funcs_dict, operand_values,
                     temp_results=None):
    """``operation(*operands, **operation_kwargs)`` for prediction item ``idx``, as a float.

    ``temp_results`` maps earlier prediction items' names to their values, which is what an
    ``operation_kwargs`` reference resolves against (the parser only allows earlier ones).
    """
    n = _num_items(prediction_info)
    name = str(_column(prediction_info, 'data_item_names', n)[idx])
    operation = _column(prediction_info, 'operations', n)[idx]
    func = operation_funcs_dict.get(operation)
    if func is None:
        raise ValueError(
            f'prediction item {name!r}: operation {operation!r} is not a registered operation '
            f'func. Register it (operation_funcs_external_path, or add_user_operation_func) or '
            f'use a built-in one.')
    raw_kwargs = _column(prediction_info, 'operation_kwargs', n)[idx] or {}
    kwargs = resolve_operation_kwargs(
        raw_kwargs, func, operation_name=operation, data_item_name=name,
        temp_results=temp_results, num_operands=len(operand_values))
    return as_scalar(func(*operand_values, **kwargs), name, operation)


def features_from_segments(prediction_info, indices, operation_funcs_dict, protocol_info,
                           outputs_by_segment, offset=0):
    """Every feature from one protocol run's recorded outputs.

    ``outputs_by_segment[(exp, sub)]`` is ``get_results(...)`` of that segment -- one entry per
    requested variable list -- and feature ``k``'s operands are entry ``offset + k``. A segment
    that failed (missing or None) gives ``nan`` for its features.
    """
    names = prediction_feature_names(prediction_info, indices)
    temp_results = {}
    out = []
    for k, (idx, segment) in enumerate(zip(indices,
                                           feature_segments(prediction_info, indices,
                                                            protocol_info))):
        outputs = outputs_by_segment.get(segment)
        if outputs is None:
            out.append(float('nan'))
            continue
        value = evaluate_feature(prediction_info, idx, operation_funcs_dict,
                                 list(outputs[offset + k]), temp_results=temp_results)
        temp_results[names[k]] = value
        out.append(value)
    return out


def simulate_features(pid, param_vals, prediction_info, indices):
    """One simulation of a param-id engine at ``param_vals``: ``(data_features, features)``.

    ``data_features`` is exactly :func:`param_id.fd_backend.observable_features` (the numbers
    the cost is built from), ``features`` the prediction features in ``indices`` order. Either
    is ``None`` when the simulation failed. Over an emulator the prediction features are read
    from it by name; it must have been trained with ``include_prediction_items``.
    """
    from libcuflynx.param_id import fd_backend

    param_vals = np.asarray(param_vals, dtype=float)
    names = prediction_feature_names(prediction_info, indices)
    if getattr(pid, 'emulates_features', False):
        data = fd_backend.observable_features(pid, param_vals)
        helper = pid.sim_helper
        helper.set_theta(param_vals)
        return data, list(helper.get_predicted_prediction_features(names))

    from libcuflynx.parsers.PrimitiveParsers import cost_experiment_idxs
    # The cost's experiments plus any validation-only one a feature is measured in.
    exp_idxs = sorted(set(cost_experiment_idxs(pid.protocol_info))
                      | set(feature_experiments(prediction_info, indices, pid.protocol_info)))
    _, operands_list, pred_list = pid.get_cost_obs_and_pred_from_params(
        param_vals, reset=True, pred_names=result_variables(prediction_info, indices),
        exp_idxs=exp_idxs)
    if not operands_list:
        return None, None
    data = fd_backend.features_from_operands(pid, operands_list)

    num_sub_per_exp = pid.protocol_info['num_sub_per_exp']
    by_segment = {}
    flat = 0
    for exp, num_sub in enumerate(num_sub_per_exp):
        for sub in range(num_sub):
            if flat < len(pred_list):
                by_segment[(exp, sub)] = pred_list[flat]
            flat += 1
    features = features_from_segments(prediction_info, indices, pid.operation_funcs_dict,
                                      pid.protocol_info, by_segment)
    return data, features


def feature_sensitivities(pid, param_vals, prediction_info, indices, h=1e-3):
    """d(prediction feature)/d(param) by central finite differences.

    ``{data_item_name: {param_label: derivative or None}}``, the shape of
    ``ParamID.get_observable_sensitivities``. Always FD, whatever arm computed the data_item
    sensitivities: the analytic arms differentiate the cost's observables, which prediction
    items are not. Costs 2M extra simulations (or emulator evaluations) for M parameters.
    Also returns the nominal feature values, for the relative normalisation.
    """
    from libcuflynx.param_id.fd_backend import _step
    from libcuflynx.parsers.PrimitiveParsers import param_entry_labels

    param_vals = np.asarray(param_vals, dtype=float)
    labels = param_entry_labels(pid.param_id_info)
    mins = np.asarray(pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(pid.param_id_info['param_maxs'], dtype=float)
    names = prediction_feature_names(prediction_info, indices)
    out = {name: {} for name in names}

    _, nominal = simulate_features(pid, param_vals, prediction_info, indices)
    if nominal is None:
        raise RuntimeError('Local sensitivity nominal simulation failed to converge.')

    for j, pname in enumerate(labels):
        step = _step(float(param_vals[j]), mins[j], maxs[j], h)
        p_plus, p_minus = param_vals.copy(), param_vals.copy()
        p_plus[j] += step
        p_minus[j] -= step
        _, f_plus = simulate_features(pid, p_plus, prediction_info, indices)
        _, f_minus = simulate_features(pid, p_minus, prediction_info, indices)
        for k, name in enumerate(names):
            if f_plus is None or f_minus is None:
                out[name][pname] = None
                continue
            d = (f_plus[k] - f_minus[k]) / (2.0 * step)
            out[name][pname] = float(d) if np.isfinite(d) else None
    return out, dict(zip(names, nominal))


def check_emulator_has_features(bundle, names, live_fingerprint, min_r2=None):
    """Refuse an emulator that cannot answer for these prediction features.

    An emulator trained without ``include_prediction_items`` has none, and one trained for
    different prediction items (or a changed operation) would answer about something else --
    the same reason the data_item fingerprint is checked (#333). ``live_fingerprint`` must
    include ``prediction_sha256`` (``fingerprint(..., prediction_info, indices)``).
    """
    bundle.check_matches(live_fingerprint, prediction_feature_labels=list(names))
    if min_r2 is not None and names:
        bundle.check_quality(min_r2, labels=list(names))
