"""Compare the calibrations of one model of the parameter-reduction study:
``python compare.py <model> <out>``. Writes ``<out>/compare/summary.json``.

For every fit in ``<out>/fits``:

* the parameters it ends with (the calibrated ones, the rest at the model's defaults) and each
  one's log10 error against the truth;
* chi2 = sum w ((model - data) / std)^2 over the series it was fitted to, and the RMSE of each
  series against the truth's divided by the truth's range;
* the Gauss-Newton Fisher information of the calibrated parameters (log scale) from central
  differences of the binned features: its condition number, the Laplace std of each log10
  parameter and the largest correlation.
"""

import json
import os
import sys

import numpy as np

from models import SPECS, build_param_id, outputs_at, paths

FD_STEP = 1e-3     # relative, in log space


def fisher(pid, values, idx, stds, weights):
    columns = []
    for i in idx:
        up, down = np.array(values, float), np.array(values, float)
        up[i] *= np.exp(FD_STEP)
        down[i] *= np.exp(-FD_STEP)
        columns.append((np.array(outputs_at(pid, up)[0]) - np.array(outputs_at(pid, down)[0]))
                       / (2 * FD_STEP))
    jac = np.sqrt(weights)[:, None] * np.array(columns).T / stds[:, None]
    return jac.T @ jac


def summarise_fisher(fim, names):
    eig = np.linalg.eigvalsh(fim)
    cov = np.linalg.pinv(fim)
    std_ln = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(std_ln, std_ln)
    off = np.nan_to_num(np.abs(corr - np.eye(len(names))))
    worst = np.unravel_index(np.argmax(off), off.shape) if len(names) > 1 else (0, 0)
    return {'eigenvalues': eig.tolist(),
            'condition_number': float(eig[-1] / eig[0]) if eig[0] > 0 else float('inf'),
            'laplace_std_log10': dict(zip(names, (std_ln / np.log(10)).tolist())),
            'max_abs_correlation': float(off[worst]) if len(names) > 1 else 0.0,
            'most_correlated': [names[worst[0]], names[worst[1]]] if len(names) > 1 else []}


def main(model, out):
    spec = SPECS[model]
    p = paths(out)
    os.makedirs(p['compare'], exist_ok=True)
    with open(p['truth']) as f:
        truth_doc = json.load(f)
    with open(p['obs']) as f:
        series_items = json.load(f)['data_items']
    with open(p['sa_obs']) as f:
        bins = json.load(f)['data_items']
    names = spec.targets()
    truth = np.array([truth_doc['truth'][n] for n in names])
    defaults = np.array([truth_doc['defaults'][n] for n in names])
    bin_stds = np.array([b['std'] for b in bins])
    bin_weights = np.array([b['weight'] for b in bins])

    series_pid = build_param_id(spec, out, p['obs'], p['params_for_id'],
                                os.path.join(p['compare'], 'series_pid'))
    bin_pid = build_param_id(spec, out, p['sa_obs'], p['params_for_id'],
                             os.path.join(p['compare'], 'bin_pid'))
    dt = spec.settings['dt']

    def series_at(values):
        traces = outputs_at(series_pid, values)[1]
        return [trace[::int(round(float(item['obs_dt']) / dt))][:len(item['value'])]
                for item, trace in zip(series_items, traces)]

    truth_series = series_at(truth)
    truth_fim = summarise_fisher(fisher(bin_pid, truth, range(len(names)), bin_stds,
                                        bin_weights), names)

    reductions = {}
    for m in ('pca', 'simple'):
        with open(os.path.join(p['reductions'], m, 'param_reduction.json')) as f:
            reductions[m] = json.load(f)

    fits = []
    for fit_name in sorted(os.listdir(p['fits'])):
        path = os.path.join(p['fits'], fit_name, 'result.json')
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            result = json.load(f)
        values = defaults.copy()
        for name, value in zip(result['calibrated'], result['best']):
            values[names.index(name)] = value
        idx = [names.index(n) for n in result['calibrated']]
        model_series = series_at(values)
        chi2, series_error = 0.0, {}
        for item, sim, true in zip(series_items, model_series, truth_series):
            data = np.asarray(item['value'], float)
            std = np.broadcast_to(np.asarray(item['std'], float), data.shape)
            chi2 += float(item.get('weight', 1.0) * np.sum(((sim - data) / std) ** 2))
            series_error[item['data_item_name']] = float(
                np.sqrt(np.mean((sim - true) ** 2)) / np.ptp(true))
        log10_error = np.log10(values / truth)
        fits.append({
            'fit': fit_name, 'method': result['method'], 'seed': result['seed'],
            'calibrated': result['calibrated'], 'values': dict(zip(names, values.tolist())),
            'log10_error': dict(zip(names, log10_error.tolist())),
            'max_abs_log10_error': float(np.max(np.abs(log10_error))),
            'max_abs_log10_error_calibrated': float(np.max(np.abs(log10_error[idx]))),
            'chi2': chi2, 'series_error': series_error,
            'fisher': summarise_fisher(fisher(bin_pid, values, idx, bin_stds, bin_weights),
                                       result['calibrated'])})
        f = fits[-1]
        print(f'{fit_name}: chi2 {chi2:.3g}, cond {f["fisher"]["condition_number"]:.3g}, '
              f'max |log10 err| {f["max_abs_log10_error"]:.3f}', flush=True)

    summary = {'model': model, 'name': spec.name, 'prefix': spec.prefix,
               'parameters': names, 'extras': list(spec.extras), 'reasons': spec.reason,
               'truth': truth.tolist(), 'defaults': defaults.tolist(),
               'settings': spec.settings, 'bin_points': spec.bin_points,
               'n_series_points': sum(len(i['value']) for i in series_items),
               'n_bins': len(bins), 'fisher_at_truth': truth_fim,
               'reductions': reductions, 'fits': fits}
    with open(os.path.join(p['compare'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'wrote {p["compare"]}/summary.json')


if __name__ == '__main__':
    main(sys.argv[1], os.path.abspath(sys.argv[2]))
