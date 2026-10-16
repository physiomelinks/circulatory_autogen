"""
predictions.py -- run an obs_data's prediction items on a model, without a calibration.

    from libcuflynx.protocol_runners import run_prediction_items

    result = run_prediction_items(model, obs, dt, solver_info)
    result.t_by_exp    # per experiment, time from the end of its warm-up
    result.traces      # {data_item_name: (t, values)}: items without an operation
    result.features    # {data_item_name: float}: scalar feature items
    result.segments    # {(exp, sub): {operand: values, 'time': t}}
    result.validation  # validation.validation_results(...) for the items that carry a value

``predict_on`` is the same over an executor the caller already has; ``save_prediction_data``
uses it after a calibration. ``join_segments`` joins one experiment's sub-experiment results
the way the solvers step them, and ``ProtocolRunner.run_protocols`` uses it too.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredictionResult:
    t_by_exp: list
    traces: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    segments: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    plots: list = field(default_factory=list)


def segment_length(sim_time, dt):
    """Samples a solver records over one sub-experiment: ``int(sim_time/dt)`` steps plus the
    first sample."""
    return int(sim_time / dt) + 1


def join_segments(sub_results, sim_times, dt):
    """Join one experiment's per-sub-experiment results into one series per variable.

    ``sub_results[s]`` is sub-experiment ``s``'s list of results (one per variable), a series
    or a scalar. A scalar is padded to its own segment's length, and every sub-experiment after
    the first drops its first sample, the one it shares with the previous segment's last. A
    sub-experiment with no results (None) is left out.
    """
    joined = None
    for sub, (results, sim_time) in enumerate(zip(sub_results, sim_times)):
        if results is None:
            continue
        n = segment_length(sim_time, dt)
        columns = []
        for value in results:
            series = np.asarray(value, dtype=float).ravel()
            if series.size == 1 and n > 1:
                series = np.full(n, series[0])
            columns.append(series if sub == 0 else series[1:])
        joined = columns if joined is None else [
            np.concatenate([a, b]) for a, b in zip(joined, columns)]
    return joined


def _unresolved(sim_helper, names):
    """The names the helper cannot resolve (none checked if it has no resolver)."""
    resolve = getattr(sim_helper, '_resolve_name', None)
    if resolve is None:
        return []
    return [name for name in names
            if name != 'time' and resolve(name)[0] in (None, 'None')]


def predict_on(executor, protocol_info, prediction_info, funcs,
               id_param_names=None, id_param_vals=None):
    """Run the experiments that have prediction items, in one executor pass, and evaluate the
    items: traces, scalar features and validation (see the module docstring).

    ``funcs`` is the operation registry. ``id_param_names``/``id_param_vals`` are set at the
    start of each experiment, as a calibration's are. Every operand is checked before anything
    runs, and the ones the model does not have are reported together.
    """
    from libcuflynx.param_id import prediction_features as pf
    from libcuflynx.param_id import validation
    from libcuflynx.utilities.obs_data_helpers import obs_operand_lists

    operand_lists = [[str(op) for op in ops] for ops in obs_operand_lists(prediction_info)]
    n = len(operand_lists)
    names = [str(name) for name in prediction_info.get('data_item_names') or []]
    unresolved = _unresolved(executor.sim_helper,
                             list(dict.fromkeys(op for ops in operand_lists for op in ops)))
    if unresolved:
        raise ValueError(f'prediction item operand(s) the model does not have: {unresolved}')

    segments = pf.feature_segments(prediction_info, list(range(n)), protocol_info)
    feature_idxs = pf.prediction_feature_indices(prediction_info, funcs, warn=False)
    asked = operand_lists + pf.result_variables(prediction_info, feature_idxs)
    ok, by_sub, time_by_sub, t_by_exp = executor.run_protocol(
        protocol_info, id_param_names=id_param_names, id_param_vals=id_param_vals,
        result_variables=asked, extra_result_variables=['time'],
        exp_indices=sorted({exp for exp, _ in segments}))
    if not ok:
        raise RuntimeError('the protocol simulation failed')

    result = PredictionResult(t_by_exp=t_by_exp)
    for segment in sorted(set(segments)):
        outputs = by_sub[segment]
        recorded = {'time': np.asarray(time_by_sub[segment][0], dtype=float).ravel()}
        for k, ops in enumerate(operand_lists):
            if segments[k] == segment:
                for op, values in zip(ops, outputs[k]):
                    recorded[op] = np.asarray(values, dtype=float).ravel()
        result.segments[segment] = recorded

    operations = prediction_info.get('operations') or [None] * n
    for k in range(n):
        if operations[k] is None:
            seg = result.segments[segments[k]]
            result.traces[names[k]] = (seg['time'], seg[operand_lists[k][0]])
    values = pf.features_from_segments(prediction_info, feature_idxs, funcs, protocol_info,
                                       by_sub, offset=n)
    result.features = {names[i]: value for i, value in zip(feature_idxs, values)}

    per_item = [[result.segments[segments[k]][op] for op in operand_lists[k]] for k in range(n)]
    result.validation = validation.validation_results(
        prediction_info, None, per_item, operation_funcs_dict=funcs,
        time_per_item=[result.segments[segments[k]]['time'] for k in range(n)])
    return result


def run_prediction_items(model, obs, dt, solver_info=None, solver='CVODE_myokit',
                         operation_funcs_path=None, param_names=None, param_vals=None,
                         model_type=None, plot_dir=None):
    """Run ``obs``'s protocol on ``model`` and evaluate its prediction items, with no calibration.

    ``obs`` is an obs_data path or dict, parsed as calibration parses it (shapes, legacy keys and
    item checks apply). ``operation_funcs_path`` adds user operations, as
    ``operation_funcs_external_path`` does. ``param_names``/``param_vals`` (for example a
    calibrated parameter set) are applied at the start of each experiment. ``model_type`` is
    needed for a model that is not CellML. With ``plot_dir``, the obs_data's
    ``prediction_plots`` are drawn there (``result.plots`` lists them); a data_item point shows
    its data only, as there is no calibration to evaluate it.
    """
    from libcuflynx.param_id.operation_funcs import get_operation_funcs_dict_for_mode
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
    from libcuflynx.protocol_runners.protocol_runner import ProtocolRunner

    parser = ObsAndParamDataParser()
    if isinstance(obs, dict):
        parsed = parser.parse_obs_data_json(obs_data_dict=obs)
    else:
        parsed = parser.parse_obs_data_json(param_id_obs_path=str(obs))
    protocol_info, prediction_info = parsed['protocol_info'], parsed['prediction_info']
    runner = ProtocolRunner(model, inp_data_dict={'dt': dt, 'solver_info': solver_info or {}},
                            solver=solver, model_type=model_type)
    runner._apply_protocol_info(protocol_info)
    funcs = get_operation_funcs_dict_for_mode('numpy', operation_funcs_path)
    result = predict_on(runner._executor, protocol_info, prediction_info, funcs,
                        id_param_names=param_names, id_param_vals=param_vals)
    if plot_dir is not None and parsed['prediction_plots']:
        from libcuflynx.param_id import prediction_plots
        result.plots = prediction_plots.plot_prediction_plots(
            parsed['prediction_plots'], parsed['gt_df'], prediction_info, protocol_info,
            result.features, plot_dir, funcs=funcs,
            units=prediction_plots.group_units(parsed['gt_df'], prediction_info))
    return result
