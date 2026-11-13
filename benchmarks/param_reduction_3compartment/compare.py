"""Compare the calibrations of the parameter-reduction study (run_study.py).

For every ``fit_<method>_s<seed>`` step of ``<out>/run``:

* the seven parameters it ends with (the calibrated ones, the rest at their parameters-file
  defaults), and each one's log10 error against the truth;
* the six features there, and their residuals in obs_data stds (z), and chi2 = sum w z^2;
* the Gauss-Newton Fisher information of the calibrated parameters (log scale) from central
  finite differences of the features: its condition number, the Laplace std of each log10
  parameter and the largest correlation;
* the error of four traces against the truth's, as RMSE over the truth's range:
  ``heart/u_lv``, ``heart/q_lv``, ``aortic_root/v`` and the unobserved ``venous_svc/u``.

Writes ``<out>/compare/summary.json`` and figures. Usage: ``python compare.py <out>``
"""

import json
import os
import sys

import numpy as np

from make_synthetic_obs import OBS_FILE, PARAMS_FOR_ID_FILE, build_param_id, features_at

TRACES = ['heart/u_lv', 'heart/q_lv', 'aortic_root/v', 'venous_svc/u']
METHODS = ('full', 'pca', 'simple')
FD_STEP = 1e-3     # relative, in log space


def traces_at(pid, values, names=TRACES):
    """``{name: trace}`` over the logged window at the seven parameter ``values``."""
    from libcuflynx.parsers.PrimitiveParsers import expand_modifier_param_vals
    helper = pid.param_id.sim_helper
    helper.set_param_vals(pid.param_id_info['param_names'],
                          expand_modifier_param_vals(pid.param_id_info, np.asarray(values)))
    helper.reset_states()
    if not helper.run():
        raise RuntimeError(f'simulation failed at {values}')
    results = helper.get_results(names)
    helper.reset_and_clear()
    return {name: np.asarray(trace, float).ravel() for name, trace in zip(names, results)}


def fisher(pid, values, calibrated_idx, obs, stds, weights):
    """Gauss-Newton Fisher information, in log parameters, of the parameters at
    ``calibrated_idx``, from central differences of the features (2 runs each)."""
    columns = []
    for i in calibrated_idx:
        up, down = np.array(values, float), np.array(values, float)
        up[i] *= np.exp(FD_STEP)
        down[i] *= np.exp(-FD_STEP)
        f_up, f_down = np.array(features_at(pid, up)), np.array(features_at(pid, down))
        columns.append((f_up - f_down) / (2 * FD_STEP))
    jac = np.sqrt(weights)[:, None] * np.array(columns).T / stds[:, None]
    return jac.T @ jac


def summarise_fisher(fim, names):
    eig = np.linalg.eigvalsh(fim)
    cov = np.linalg.pinv(fim)
    std_ln = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(std_ln, std_ln)
    off = np.abs(corr - np.eye(len(names)))
    worst = np.unravel_index(np.nanargmax(off), off.shape) if len(names) > 1 else (0, 0)
    return {'eigenvalues': eig.tolist(),
            'condition_number': float(eig[-1] / eig[0]) if eig[0] > 0 else float('inf'),
            'laplace_std_log10': dict(zip(names, (std_ln / np.log(10)).tolist())),
            'max_abs_correlation': float(off[worst]) if len(names) > 1 else 0.0,
            'most_correlated': [names[worst[0]], names[worst[1]]] if len(names) > 1 else []}


def main(out):
    inputs = os.path.join(out, 'inputs')
    run_dir = os.path.join(out, 'run')
    compare_dir = os.path.join(out, 'compare')
    os.makedirs(compare_dir, exist_ok=True)
    with open(os.path.join(inputs, 'truth.json')) as f:
        truth_doc = json.load(f)
    with open(os.path.join(inputs, OBS_FILE)) as f:
        items = json.load(f)['data_items']
    with open(os.path.join(inputs, PARAMS_FOR_ID_FILE)) as f:
        names = [p['targets'][0] for p in json.load(f)['params']]
    obs = np.array([i['value'] for i in items])
    stds = np.array([i['std'] for i in items])
    weights = np.array([i.get('weight', 1.0) for i in items])
    truth = np.array([truth_doc['truth'][n] for n in names])
    defaults = np.array([truth_doc['defaults'][n] for n in names])

    pid = build_param_id(inputs, os.path.join(inputs, OBS_FILE),
                         os.path.join(inputs, PARAMS_FOR_ID_FILE),
                         os.path.join(out, 'generated_models'), os.path.join(compare_dir, 'pid'))
    truth_traces = traces_at(pid, truth)
    fim_truth = fisher(pid, truth, range(len(names)), obs, stds, weights)

    def reduction(step_id):
        path = os.path.join(run_dir, step_id, 'step_result.json')
        with open(path) as f:
            return json.load(f)

    reductions = {m: reduction(f'reduce_{m}') for m in ('pca', 'simple')}
    fits = []
    for step_id in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, step_id, 'step_result.json')
        if not step_id.startswith('fit_') or not os.path.isfile(path):
            continue
        with open(path) as f:
            result = json.load(f)
        method, seed = step_id[len('fit_'):].rsplit('_s', 1)
        values = defaults.copy()
        calibrated = []
        for row in result['calibrated']:
            name = f"{row['vessel']}/{row['param']}"
            values[names.index(name)] = row['value']
            calibrated.append(name)
        idx = [names.index(n) for n in calibrated]
        features = np.array(features_at(pid, values))
        z = (features - obs) / stds
        traces = traces_at(pid, values)
        trace_error = {n: float(np.sqrt(np.mean((traces[n] - truth_traces[n]) ** 2))
                                / np.ptp(truth_traces[n])) for n in TRACES}
        fits.append({
            'step': step_id, 'method': method, 'seed': int(seed),
            'calibrated': calibrated, 'best_cost': result.get('best_cost'),
            'values': dict(zip(names, values.tolist())),
            'log10_error': dict(zip(names, np.log10(values / truth).tolist())),
            'features': dict(zip([i['data_item_name'] for i in items], features.tolist())),
            'z': dict(zip([i['data_item_name'] for i in items], z.tolist())),
            'chi2': float(np.sum(weights * z ** 2)),
            'fisher': summarise_fisher(fisher(pid, values, idx, obs, stds, weights),
                                       calibrated),
            'trace_error': trace_error, 'mean_trace_error': float(np.mean(
                list(trace_error.values())))})
        print(f'{step_id}: chi2 {fits[-1]["chi2"]:.3g}, cond '
              f'{fits[-1]["fisher"]["condition_number"]:.3g}, trace error '
              f'{fits[-1]["mean_trace_error"]:.3g}', flush=True)

    summary = {'parameters': names, 'truth': truth.tolist(), 'defaults': defaults.tolist(),
               'features': [i['data_item_name'] for i in items], 'obs': obs.tolist(),
               'stds': stds.tolist(), 'weights': weights.tolist(),
               'fisher_at_truth_all_seven': summarise_fisher(fim_truth, names),
               'reductions': {m: {'selected': r['selected'], 'reduction': r['reduction']}
                              for m, r in reductions.items()},
               'fits': fits, 'traces': TRACES}
    with open(os.path.join(run_dir, 'screen', 'step_result.json')) as f:
        summary['screen'] = {k: v for k, v in json.load(f).items()
                             if k in ('sensitivity_dir', 'sobol_indices', 'settings')}
    with open(os.path.join(run_dir, 'reduce_pca', 'param_reduction.json')) as f:
        summary['pca_report'] = json.load(f)
    with open(os.path.join(compare_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    # traces of seed 0 of each method, for the page's figure
    example = {'truth': {n: truth_traces[n].tolist() for n in TRACES}}
    for fit in fits:
        if fit['seed'] == min(f['seed'] for f in fits if f['method'] == fit['method']):
            example[fit['method']] = {n: v.tolist() for n, v in traces_at(
                pid, [fit['values'][n] for n in names]).items()}
    example['time'] = np.asarray(pid.param_id.sim_helper.get_time()).ravel().tolist() \
        if hasattr(pid.param_id.sim_helper, 'get_time') else None
    with open(os.path.join(compare_dir, 'traces.json'), 'w') as f:
        json.dump(example, f)
    print(f'wrote {compare_dir}/summary.json')


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]))
