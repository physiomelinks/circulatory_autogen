'''
Standard figures of a calibration workflow run, one function per figure.

Copy this script beside a workflow with
``libcuflynx.calibration_workflow.figures.write_standard_figures(<dir>)`` and name a function in a
figure step. Each function's parameters are the input names its step must give, plus
``out_dir``::

    {"id": "z_traces", "kind": "figure",
     "function": "figures/standard_figures.py:sweep_traces",
     "inputs": {"sweep": {"sweep": "scan"}}}

==================  =======================================================  ===================
function            draws                                                    inputs (name: kind)
==================  =======================================================  ===================
sweep_traces        every output over time, a line per point, per experiment sweep: sweep
sweep_features      each feature against the swept parameters                sweep: sweep
prediction_plots    the obs_data's prediction_plots, from one simulation     simulation: sweep,
                                                                             obs_data: obs_data
sobol_indices       S1 and ST per output, and a heatmap of ST                sensitivity:
                                                                             sensitivity
calibration_        the best cost and each normalised parameter per          progress: progress
progress            generation (per iteration of each start)
emulator_           the emulator against the simulator at its held-out       emulator:
validation          points, and the residuals against each parameter        emulator_validation
posterior           a histogram per parameter, its mean and 5-95% band       distribution:
                                                                             distribution
laplace             the normal density of each parameter over +-4 std        laplace: laplace
==================  =======================================================  ===================

Run on its own, on a finished run (``--step`` picks among several figure steps)::

    python standard_figures.py <run directory> [--step <id>] [--out-dir DIR]
    python standard_figures.py <run directory> --function sweep_traces --input sweep=sweep:scan
'''
import os
import re

import numpy as np

#: line and bar colours, in order
PALETTE = ('#5b9bd5', '#ed7d31', '#70ad47', '#ffc000', '#a142f4', '#e84a5f')
#: the most points of a sweep named in a legend
LEGEND_POINTS = 8
DPI = 120


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _stem(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'figure'


def _save(figure, out_dir, name):
    path = os.path.join(out_dir, name)
    figure.savefig(path, dpi=DPI)
    _pyplot().close(figure)
    return path


def _point_label(names, values):
    return ', '.join(f'{name} = {value:.4g}' for name, value in zip(names, values))


def _grid(count, width=3.2, height=2.6, columns=3):
    plt = _pyplot()
    columns = min(columns, count)
    rows = -(-count // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows),
                                squeeze=False)
    for ax in axes.ravel()[count:]:
        ax.set_visible(False)
    return figure, axes.ravel()[:count]


# --------------------------------------------------------------------------------------------
# sweeps and simulations
# --------------------------------------------------------------------------------------------

def sweep_traces(sweep, out_dir):
    '''``traces_exp_<e>.png``: per experiment, a panel per output with a line per point that
    did not fail, coloured by point (named in the legend up to LEGEND_POINTS points).'''
    outputs = [str(name) for name in sweep['outputs']]
    if not outputs:
        raise ValueError('the sweep recorded no outputs over time; give its "outputs".')
    names = [str(name) for name in sweep['parameters']]
    points, failed = sweep['points'], sweep['failed']
    good = [i for i in range(len(points)) if not failed[i]]
    plt = _pyplot()
    cmap = plt.get_cmap('viridis')
    experiments = sorted(int(key.split('_')[1]) for key in sweep if key.startswith('trace_'))
    paths = []
    for e in experiments:
        time, traces = sweep[f'time_{e}'], sweep[f'trace_{e}']
        figure, axes = _grid(len(outputs))
        for o, (ax, output) in enumerate(zip(axes, outputs)):
            for n, i in enumerate(good):
                colour = (PALETTE[n % len(PALETTE)] if len(good) <= len(PALETTE)
                          else cmap(n / max(len(good) - 1, 1)))
                ax.plot(time, traces[i, o], color=colour, linewidth=1.2,
                        label=_point_label(names, points[i]) if names else None)
            ax.set_title(output, fontsize=9)
            ax.set_xlabel('time [s]', fontsize=8)
            ax.tick_params(labelsize=7)
        if names and len(good) <= LEGEND_POINTS:
            axes[0].legend(fontsize=6)
        figure.suptitle(f'experiment {e}', fontsize=10)
        figure.tight_layout()
        paths.append(_save(figure, out_dir, f'traces_exp_{e}.png'))
    return paths


def sweep_features(sweep, out_dir):
    '''Each feature (an obs_data prediction item) against the swept parameters, as the sweep
    step draws them: ``<feature> vs <parameter>.png`` per feature and plot group.'''
    from libcuflynx.calibration_workflow.sweep import plot_features
    if not len(sweep['features']):
        raise ValueError('the sweep has no features: its obs_data has no prediction items.')
    groups = sweep['manifest'].get('plot_groups')
    if groups is None:
        raise ValueError('the sweep manifest has no plot_groups; run the sweep step again.')
    if not groups:
        raise ValueError('the sweep\'s design gives no feature plots (a grid of more than two '
                         'parameters).')
    return plot_features(groups, sweep['parameters'], sweep['points'], sweep, out_dir)


def prediction_plots(simulation, obs_data, out_dir):
    '''The obs_data's ``prediction_plots``, drawn from a simulate step (or a sweep of one point)
    that ran on that obs_data: its features give the feature kinds' model points, its traces
    the time-course kinds' outputs, which must be among the step's ``outputs``.'''
    from libcuflynx.param_id import prediction_plots as pp
    from libcuflynx.param_id.operation_funcs import get_operation_funcs_dict_for_mode
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser

    if len(simulation['points']) != 1:
        raise ValueError(f'prediction_plots reads one simulation (a simulate step); this step '
                         f'has {len(simulation["points"])} points.')
    if simulation['failed'][0]:
        raise ValueError('prediction_plots reads one simulation, and it failed.')
    parsed = ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=obs_data)
    plots, protocol = parsed.get('prediction_plots'), parsed['protocol_info']
    if not plots:
        raise ValueError('the obs_data has no prediction_plots.')
    features = {str(name): float(value)
                for name, value in zip(simulation['features'], simulation['feature_values'][0])}
    outputs = [str(name) for name in simulation['outputs']]
    courses = {}
    for index, plot in enumerate(plots):
        if plot['kind'] not in pp.TIME_COURSE_KINDS:
            continue
        _, output, experiments = pp.resolve_time_course(plot, index, protocol)
        if output not in outputs:
            raise ValueError(f'prediction plot {plot["name"]!r} draws {output!r}, which the '
                             f'step did not record; add it to its outputs ({outputs}).')
        courses[output] = {e: (simulation[f'time_{e}'], simulation[f'trace_{e}'][0,
                                                                              outputs.index(output)])
                           for e in experiments}
    prediction_info = parsed['prediction_info']
    # the step recorded exactly the scalar features, so an operation of one is series-to-constant
    funcs = dict(get_operation_funcs_dict_for_mode('numpy'))
    for name, operation in zip(prediction_info.get('data_item_names') or [],
                               prediction_info.get('operations') or []):
        if name in features and operation is not None:
            funcs[operation] = _Scalar
    return pp.plot_prediction_plots(plots, parsed['gt_df'], prediction_info, protocol, features,
                                    out_dir, funcs=funcs,
                                    units=pp.group_units(parsed['gt_df'], prediction_info),
                                    time_courses=courses)


class _Scalar:
    series_to_constant = True


# --------------------------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------------------------

def sobol_indices(sensitivity, out_dir):
    '''``sobol_<output>.png``: S1 and ST of each parameter for each output; and
    ``sobol_ST_heatmap.png``: ST, parameters by outputs, on a 0-1 scale.'''
    plt = _pyplot()
    outputs = list(sensitivity['ST'])
    if not outputs:
        raise ValueError('the sensitivity step wrote no indices.')
    parameters = list(sensitivity['ST'][outputs[0]])
    paths, x, width = [], np.arange(len(parameters)), 0.38
    for output in outputs:
        figure, ax = plt.subplots(figsize=(max(4.0, 0.6 * len(parameters) + 1.5), 3.2))
        for shift, index, colour in ((-width / 2, 'S1', PALETTE[0]),
                                     (width / 2, 'ST', PALETTE[1])):
            values = [sensitivity[index].get(output, {}).get(p, np.nan) for p in parameters]
            ax.bar(x + shift, values, width, color=colour, label=index)
        ax.set_xticks(x, parameters, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
        ax.set_title(output, fontsize=9)
        ax.legend(fontsize=7)
        figure.tight_layout()
        paths.append(_save(figure, out_dir, f'sobol_{_stem(output)}.png'))
    table = np.array([[sensitivity['ST'][o].get(p, np.nan) for o in outputs] for p in parameters])
    figure, ax = plt.subplots(figsize=(max(4.0, 2.0 + 0.8 * len(outputs)),
                                       max(3.0, 1.5 + 0.4 * len(parameters))))
    image = ax.imshow(np.clip(table, 0, 1), cmap='viridis', vmin=0, vmax=1, aspect='auto')
    for (row, column), value in np.ndenumerate(table):
        ax.text(column, row, f'{value:.2f}', ha='center', va='center', fontsize=7,
                color='white' if value < 0.6 else 'black')
    ax.set_xticks(range(len(outputs)), outputs, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(parameters)), parameters, fontsize=7)
    ax.set_title('total-order Sobol indices (ST)', fontsize=9)
    figure.colorbar(image, ax=ax)
    figure.tight_layout()
    paths.append(_save(figure, out_dir, 'sobol_ST_heatmap.png'))
    return paths


def posterior(distribution, out_dir):
    '''``posterior.png``: per parameter a 40-bin histogram of the samples, the 5-95% band and
    the mean, titled "mean +- std, 90% CI [q05, q95]".'''
    samples, names = np.asarray(distribution['samples']), distribution['names']
    figure, axes = _grid(len(names))
    for ax, name, column in zip(axes, names, samples.T):
        mean, std = float(np.mean(column)), float(np.std(column))
        q05, q95 = np.quantile(column, [0.05, 0.95])
        ax.axvspan(q05, q95, color=PALETTE[2], alpha=0.18, linewidth=0)
        ax.hist(column, bins=40, histtype='stepfilled', color=PALETTE[0], alpha=0.45)
        ax.hist(column, bins=40, histtype='step', color=PALETTE[0])
        ax.axvline(mean, color=PALETTE[3], linewidth=1.5)
        ax.set_xlabel(name, fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        ax.set_title(f'{mean:.3g} ± {std:.2g}, 90% CI [{q05:.3g}, {q95:.3g}]', fontsize=7)
    figure.tight_layout()
    return [_save(figure, out_dir, 'posterior.png')]


def calibration_progress(progress, out_dir):
    '''``cost_history.png``: the best cost per generation (iteration) on a log axis, with the
    band up to the worst of the recorded best costs when a row holds several (a genetic
    algorithm's top 10), or one line per start of a multi-start run; and
    ``param_history.png``: each parameter's value normalised by its params_for_id bounds (0 at
    its min, 1 at its max) per generation, or per iteration of each start.'''
    plt = _pyplot()
    labels, starts = list(progress['param_labels']), progress['starts']
    costs = [np.asarray(row, dtype=float) for row in progress['cost_history'] if len(row)]
    if not costs and not starts:
        raise ValueError('the calibration recorded no cost history.')
    figure, ax = plt.subplots(figsize=(5.0, 3.4))
    if starts:
        for n, start in enumerate(starts):
            ax.plot(start['cost'], color=_shade(PALETTE[0], n, len(starts)), linewidth=1.2,
                    label=f'start {n}' if len(starts) <= LEGEND_POINTS else None)
        ax.set_xlabel('iteration', fontsize=8)
    else:
        best = np.array([row[0] for row in costs])
        ax.plot(best, color=PALETTE[0], linewidth=2, label='best')
        if any(row.size > 1 for row in costs):
            worst = np.array([row.max() for row in costs])
            ax.fill_between(np.arange(len(best)), best, worst, color=PALETTE[0], alpha=0.15,
                            linewidth=0, label='recorded best costs')
        ax.set_xlabel('generation', fontsize=8)
    if all(np.all(np.asarray(s['cost']) > 0) for s in starts) and all(np.all(r > 0) for r in costs):
        ax.set_yscale('log')
    ax.set_ylabel('cost', fontsize=8)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    figure.tight_layout()
    paths = [_save(figure, out_dir, 'cost_history.png')]

    histories = ([np.asarray(s['params'], dtype=float) for s in starts] if starts else
                 [np.asarray(progress['param_history_norm'], dtype=float)])
    histories = [h for h in histories if h.ndim == 2 and h.size]
    if not histories:
        return paths
    if starts:
        # a multi-start history holds actual values: normalise by the params_for_id bounds
        low = np.asarray(progress['param_mins'], dtype=float)
        span = np.asarray(progress['param_maxs'], dtype=float) - low
        histories = [(h - low) / np.where(span > 0, span, 1.0) for h in histories]
    figure, ax = plt.subplots(figsize=(5.0, 3.4))
    for p in range(histories[0].shape[1]):
        for n, history in enumerate(histories):
            ax.plot(history[:, p], linewidth=1.2,
                    color=_shade(PALETTE[p % len(PALETTE)], n, len(histories)),
                    label=(labels[p] if p < len(labels) else f'p{p}') if n == 0 else None)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('iteration' if starts else 'generation', fontsize=8)
    ax.set_ylabel('normalised value (0 at min, 1 at max)', fontsize=8)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)
    figure.tight_layout()
    paths.append(_save(figure, out_dir, 'param_history.png'))
    return paths


def emulator_validation(emulator, out_dir):
    '''``emulator_parity.png``: per feature, the emulator's prediction against the simulated
    value at each held-out point, on a shared range with the y = x line, titled with the R2 and
    NRMSE; and ``emulator_residuals_<feature>.png``: per feature, the normalised residual
    (prediction - truth) / truth against each parameter (divided by the range of the truths
    when a truth is 0, and raw when that range is 0 too), symmetric about 0.'''
    theta, y_true, y_pred = emulator['theta'], emulator['y_true'], emulator['y_pred']
    features, parameters = emulator['feature_labels'], emulator['param_entry_labels']
    if not len(theta):
        raise ValueError('the emulator saved no held-out points.')
    plt = _pyplot()
    figure, axes = _grid(len(features), width=3.0, height=3.0)
    for f, (ax, feature) in enumerate(zip(axes, features)):
        true, pred = y_true[:, f], y_pred[:, f]
        low, high = min(true.min(), pred.min()), max(true.max(), pred.max())
        pad = 0.05 * (high - low or abs(high) or 1.0)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color='0.6', linestyle='--',
                linewidth=1)
        ax.scatter(true, pred, s=9, color=PALETTE[0], alpha=0.75)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('simulated', fontsize=8)
        ax.set_ylabel('emulator', fontsize=8)
        ax.tick_params(labelsize=7)
        scores = [f'{name} {values[f]:.3g}'
                  for name, values in (('R2', emulator.get('feature_r2')),
                                       ('NRMSE', emulator.get('feature_nrmse')))
                  if values is not None and f < len(values) and values[f] is not None]
        ax.set_title(feature + (f'\n{", ".join(scores)}' if scores else ''), fontsize=8)
    figure.tight_layout()
    paths = [_save(figure, out_dir, 'emulator_parity.png')]
    for f, feature in enumerate(features):
        true, residual = y_true[:, f], y_pred[:, f] - y_true[:, f]
        if np.all(true != 0):
            normalised, label = residual / true, '(emulator - simulated) / simulated'
        elif np.ptp(true) > 0:
            normalised, label = residual / np.ptp(true), '(emulator - simulated) / range'
        else:
            normalised, label = residual, 'emulator - simulated'
        limit = float(np.max(np.abs(normalised))) or 1.0
        figure, axes = _grid(len(parameters))
        for p, (ax, parameter) in enumerate(zip(axes, parameters)):
            ax.axhline(0.0, color='0.6', linestyle='--', linewidth=1)
            ax.scatter(theta[:, p], normalised, s=9, color=PALETTE[0], alpha=0.75)
            ax.set_ylim(-1.05 * limit, 1.05 * limit)
            ax.set_xlabel(parameter, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel(label, fontsize=7)
        figure.suptitle(feature, fontsize=10)
        figure.tight_layout()
        paths.append(_save(figure, out_dir, f'emulator_residuals_{_stem(feature)}.png'))
    return paths


def _shade(colour, n, count):
    '''``colour`` lightened toward white by up to 60% as ``n`` goes from 0 to ``count - 1``.'''
    from matplotlib.colors import to_rgb
    fraction = 0.6 * n / max(count - 1, 1)
    return tuple(c + (1 - c) * fraction for c in to_rgb(colour))


def laplace(laplace, out_dir):
    '''``laplace.png``: per parameter the normal density of the Laplace approximation over
    +-4 std; a parameter with no finite positive std is marked not identifiable.'''
    names = list(laplace['parameters']) or [f'p{i}' for i in range(len(laplace['mean']))]
    figure, axes = _grid(len(names))
    for ax, name, mean, std in zip(axes, names, laplace['mean'], laplace['std']):
        ax.set_xlabel(name, fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        if not (np.isfinite(std) and std > 0):
            ax.text(0.5, 0.5, 'not identifiable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_xticks([])
            continue
        x = np.linspace(mean - 4 * std, mean + 4 * std, 201)
        density = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax.fill_between(x, density, color=PALETTE[0], alpha=0.45, linewidth=0)
        ax.axvspan(mean - 1.645 * std, mean + 1.645 * std, color=PALETTE[2], alpha=0.18,
                   linewidth=0)
        ax.axvline(mean, color=PALETTE[3], linewidth=1.5)
        ax.set_title(f'{mean:.3g} ± {std:.2g}', fontsize=7)
    figure.tight_layout()
    return [_save(figure, out_dir, 'laplace.png')]


FIGURES = [sweep_traces, sweep_features, prediction_plots, sobol_indices, calibration_progress,
           emulator_validation, posterior, laplace]


def main(argv=None):
    from libcuflynx.calibration_workflow.figures import figure_main
    return figure_main(FIGURES, argv)


if __name__ == '__main__':
    main()
