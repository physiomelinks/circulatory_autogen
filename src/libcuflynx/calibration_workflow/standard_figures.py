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
validation          held-out prediction items against the model: scores,     validation:
                    a parity plot of the scalars and a plot per series        validation
protocol_inputs    each params_to_change entry over time, per experiment    obs_data: obs_data
sobol_indices       S1 and ST per output, and a heatmap of ST                sensitivity:
                                                                             sensitivity
local_sensitivity   a heatmap of a local analysis's relative sensitivities   sensitivity:
                                                                             sensitivity
pca_modes           per PCA mode its feature loadings and the ST of each     reduction:
                    parameter, and a heatmap of ST per mode                  reduction
calibration_        the best cost and each normalised parameter per          progress: progress
progress            generation (per iteration of each start)
fit_errors          the best fit's error per data_item, in percent and in    errors: fit_errors
                    noise stds
mcmc_diagnostics   each walker's trace, running mean and autocorrelation    chain: mcmc_chain
emulator_           the emulator against the simulator at its held-out       emulator:
validation          points, and the residuals against each parameter        emulator_validation
posterior           a histogram per parameter, its mean and 5-95% band       distribution:
                                                                             distribution
laplace             the normal density of each parameter over +-4 std        laplace: laplace
posterior_          a uq step's posterior predictive check: each scalar      predictive:
predictive          observable's model intervals against its data, and the  posterior_predictive
                    kept traces against the recorded ones
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


#: the scores of validation_scores.csv, in order
VALIDATION_SCORES = ('rmse', 'nrmse', 'mean_abs_z', 'within_2std')


def validation(validation, out_dir):
    '''The prediction items that carry a value against the model, from a simulate step's
    validation: ``validation_scores.csv``, per item its n_points and VALIDATION_SCORES (empty
    where an item has no std); ``validation_parity.png``, the scalar items' held-out data
    (+- std) against the model with the y = x line; and ``validation_<item>.png`` per series
    item, the data (+- std) and the model at the observation times, from the start of the
    item's sub-experiment.'''
    import csv
    items = validation['items']
    if not items:
        raise ValueError('the validation holds no item: no prediction item carries a value.')
    paths = [os.path.join(out_dir, 'validation_scores.csv')]
    with open(paths[0], 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(('item', 'operation', 'unit', 'n_points') + VALIDATION_SCORES)
        for item in items:
            writer.writerow([item['data_item_name'], item.get('operation') or '',
                             item.get('unit') or '', item['n_points']] +
                            ['' if item.get(score) is None else f'{item[score]:.6g}'
                             for score in VALIDATION_SCORES])
    plt = _pyplot()
    scalars = [item for item in items if item['data_type'] != 'series']
    if scalars:
        model = np.array([item['model'][0] for item in scalars], dtype=float)
        data = np.array([item['data'][0] for item in scalars], dtype=float)
        std = np.array([(item['std'] or [0.0])[0] for item in scalars], dtype=float)
        low = min(model.min(), (data - std).min())
        high = max(model.max(), (data + std).max())
        pad = 0.05 * (high - low or abs(high) or 1.0)
        figure, ax = plt.subplots(figsize=(4.2, 4.0))
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color='0.6', linestyle='--',
                linewidth=1)
        ax.errorbar(model, data, yerr=std, fmt='o', color=PALETTE[0], markersize=4, capsize=3)
        for item, x, y in zip(scalars, model, data):
            ax.annotate(item['data_item_name'], (x, y), xytext=(4, 4),
                        textcoords='offset points', fontsize=6)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel('model', fontsize=8)
        ax.set_ylabel('held-out data', fontsize=8)
        ax.tick_params(labelsize=7)
        figure.tight_layout()
        paths.append(_save(figure, out_dir, 'validation_parity.png'))
    for item in (item for item in items if item['data_type'] == 'series'):
        t = np.asarray(item['t'], dtype=float)
        figure, ax = plt.subplots(figsize=(5.0, 3.2))
        ax.errorbar(t, item['data'], yerr=item['std'], fmt='o', color=PALETTE[0],
                    markersize=3, capsize=2, label='held-out data')
        ax.plot(t, item['model'], color=PALETTE[1], linewidth=1.5, label='model')
        unit = f' [{item["unit"]}]' if item.get('unit') else ''
        ax.set_title(item['data_item_name'], fontsize=9)
        ax.set_xlabel('time from the sub-experiment start [s]', fontsize=8)
        ax.set_ylabel((item.get('operand') or item['data_item_name']) + unit, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        figure.tight_layout()
        paths.append(_save(figure, out_dir, f'validation_{_stem(item["data_item_name"])}.png'))
    return paths


def protocol_inputs(obs_data, out_dir):
    '''``inputs_exp_<e>.png``: per experiment, a panel per ``params_to_change`` entry over the
    logged time (the sub-experiments end to end, the pre-pass left out). A number is held over
    its sub-experiment; a trace name is drawn from ``protocol_traces`` (``protocol_shapes``
    expanded), from the sub-experiment's start, held at its ends as Myokit holds it.'''
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
    protocol = ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=obs_data)[
        'protocol_info']
    changes = protocol.get('params_to_change') or {}
    if not changes:
        raise ValueError('the obs_data changes no parameter: its protocol_info has no '
                         'params_to_change.')
    traces = protocol.get('protocol_traces') or {}
    labels = protocol.get('experiment_labels') or []
    paths = []
    for e, durations in enumerate(protocol['sim_times']):
        starts = np.concatenate([[0.0], np.cumsum(durations)])
        figure, axes = _grid(len(changes))
        for ax, (name, per_experiment) in zip(axes, changes.items()):
            for k, value in enumerate(per_experiment[e]):
                start, end = starts[k], starts[k + 1]
                if isinstance(value, str):
                    trace = traces[value]
                    order = np.argsort(trace['t'], kind='stable')
                    t, v = _trace_points(np.asarray(trace['t'], dtype=float)[order],
                                         np.asarray(trace['values'], dtype=float)[order],
                                         end - start)
                    ax.plot(start + t, v, color=PALETTE[0], linewidth=1.5)
                else:
                    ax.plot([start, end], [float(value)] * 2, color=PALETTE[0], linewidth=1.5)
            for boundary in starts[1:-1]:
                ax.axvline(boundary, color='0.75', linestyle=':', linewidth=1)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel('time [s]', fontsize=8)
            ax.tick_params(labelsize=7)
        label = labels[e] if e < len(labels) and labels[e] else None
        figure.suptitle(f'experiment {e}' + (f': {label}' if label else ''), fontsize=10)
        figure.tight_layout()
        paths.append(_save(figure, out_dir, f'inputs_exp_{e}.png'))
    return paths


def _trace_points(t, values, length):
    '''A trace's points over ``[0, length]``: those inside it, with repeated times kept as
    jumps, and its value interpolated (held at the trace's ends) at 0 and at ``length``.'''
    inside = (t > 0.0) & (t < length)
    times = np.concatenate([[0.0], t[inside], [length]])
    held = np.concatenate([[np.interp(0.0, t, values)], values[inside],
                           [np.interp(length, t, values)]])
    if np.any(t <= 0.0):
        held[0] = values[np.flatnonzero(t <= 0.0)[-1]]
    return times, held


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


#: the most chain steps and walkers an MCMC panel draws
MCMC_POINTS, MCMC_WALKERS = 400, 12
#: the autocorrelation a mixed chain stays within over the last ACF_TAIL of its lags
ACF_LIMIT, ACF_TAIL = 0.1, 0.2


def mcmc_diagnostics(chain, out_dir):
    '''A calibration's MCMC chain, a panel per parameter, up to MCMC_WALKERS walkers in
    PALETTE colours and MCMC_POINTS steps: ``mcmc_chain.png``, each walker's trace;
    ``mcmc_cumulative_mean.png``, each walker's running mean from step 0 (solid) and from the
    burn-in (dashed); ``mcmc_autocorrelation.png``, each walker's autocorrelation against lag
    (emcee's ``function_1d``) with guides at +-ACF_LIMIT, titled "within" when the walkers'
    mean stays inside them over the last ACF_TAIL of the lags, else "still correlated".'''
    from emcee.autocorr import function_1d
    samples = np.asarray(chain['chain'], dtype=float)
    steps, walkers, count = samples.shape
    if steps < 2:
        raise ValueError(f'the MCMC chain has {steps} step(s); there is nothing to draw.')
    names = list(chain['param_labels']) or [f'p{i}' for i in range(count)]
    start = min(max(int(chain['burn_in_index']), 0), steps - 1)
    shown = np.unique(np.linspace(0, steps - 1, min(steps, MCMC_POINTS)).astype(int))
    walkers = min(walkers, MCMC_WALKERS)
    colours = [PALETTE[w % len(PALETTE)] for w in range(walkers)]
    paths = []

    figure, axes = _grid(count)
    for p, ax in enumerate(axes):
        for w in range(walkers):
            ax.plot(shown, samples[shown, w, p], color=colours[w], linewidth=0.8, alpha=0.8)
        ax.axvline(start, color='0.6', linestyle=':', linewidth=1)
        _mcmc_axes(ax, names[p], 'step')
    figure.tight_layout()
    paths.append(_save(figure, out_dir, 'mcmc_chain.png'))

    figure, axes = _grid(count)
    for p, ax in enumerate(axes):
        for w in range(walkers):
            values = samples[:, w, p]
            from_zero = np.cumsum(values) / np.arange(1, steps + 1)
            after = np.cumsum(values[start:]) / np.arange(1, steps - start + 1)
            ax.plot(shown, from_zero[shown], color=colours[w], linewidth=1.0)
            later = shown[shown >= start]
            ax.plot(later, after[later - start], color=colours[w], linewidth=1.0,
                    linestyle='--')
        _mcmc_axes(ax, names[p], 'step')
    figure.tight_layout()
    paths.append(_save(figure, out_dir, 'mcmc_cumulative_mean.png'))

    figure, axes = _grid(count)
    lags = np.unique(np.linspace(0, steps - 1, min(steps, MCMC_POINTS)).astype(int))
    tail = lags[lags >= int((1 - ACF_TAIL) * (steps - 1))]
    for p, ax in enumerate(axes):
        acfs = np.array([_acf(function_1d, samples[:, w, p]) for w in range(walkers)])
        for w in range(walkers):
            ax.plot(lags, acfs[w, lags], color=colours[w], linewidth=0.8, alpha=0.8)
        for level in (-ACF_LIMIT, ACF_LIMIT):
            ax.axhline(level, color=PALETTE[5], linestyle='--', linewidth=1)
        ax.axhline(0.0, color='0.6', linewidth=0.8)
        mixed = bool(np.all(np.abs(np.mean(acfs[:, tail], axis=0)) <= ACF_LIMIT))
        _mcmc_axes(ax, f'{names[p]}: '
                       f'{"within" if mixed else "still correlated"}', 'lag')
    figure.tight_layout()
    paths.append(_save(figure, out_dir, 'mcmc_autocorrelation.png'))
    return paths


def _acf(function_1d, values):
    '''The autocorrelation of ``values`` (all zeros for a walker that never moved).'''
    if np.ptp(values) == 0:
        return np.zeros(len(values))
    return function_1d(values)


def _mcmc_axes(ax, title, xlabel):
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)


def local_sensitivity(sensitivity, out_dir):
    '''``local_sensitivity_heatmap.png``: a local analysis's relative sensitivities
    (|p| d(feature)/dp / |feature|), parameters by outputs, on a blue-white-red scale symmetric
    about zero.'''
    local = (sensitivity.get('local') or {}).get('relative')
    if not local:
        raise ValueError('the sensitivity step wrote no local sensitivities (sa_options '
                         'method "local").')
    plt = _pyplot()
    outputs = list(local)
    parameters = list(local[outputs[0]])
    table = np.array([[local[o].get(p, np.nan) for o in outputs] for p in parameters])
    limit = float(np.nanmax(np.abs(table))) if np.isfinite(table).any() else 1.0
    limit = limit or 1.0
    figure, ax = plt.subplots(figsize=(max(4.0, 2.0 + 0.8 * len(outputs)),
                                       max(3.0, 1.5 + 0.4 * len(parameters))))
    image = ax.imshow(table, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    for (row, column), value in np.ndenumerate(table):
        ax.text(column, row, f'{value:.2f}', ha='center', va='center', fontsize=7,
                color='white' if abs(value) > 0.6 * limit else 'black')
    ax.set_xticks(range(len(outputs)), outputs, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(parameters)), parameters, fontsize=7)
    ax.set_title('relative local sensitivity', fontsize=9)
    figure.colorbar(image, ax=ax)
    figure.tight_layout()
    return [_save(figure, out_dir, 'local_sensitivity_heatmap.png')]


def pca_modes(reduction, out_dir):
    '''Per PCA mode of a PCA reduction, ``pca_mode_<k>.png``: the mode's loading on each
    feature (signed, sorted by size) and the ST of each parameter for the mode (sorted); and
    ``pca_mode_ST_heatmap.png``: ST, modes by parameters, on a 0-1 scale (* marks a mode below
    the noise floor).'''
    modes = reduction.get('modes')
    if not modes:
        raise ValueError(f'the reduction has no PCA modes (method '
                         f'{reduction.get("reduction", {}).get("method")!r}); pca_modes draws '
                         f'a PCA reduction.')
    plt = _pyplot()
    ST = reduction['mode_indices']['ST']
    parameters = list(ST[f'mode_{modes[0]["mode"]}'])
    paths = []
    for mode in modes:
        k = mode['mode']
        loadings = sorted(mode['loadings'].items(), key=lambda item: -abs(item[1]))
        indices = sorted(ST[f'mode_{k}'].items(), key=lambda item: -item[1])
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(8.0, max(2.6, 0.9 + 0.3 * max(len(loadings), len(indices)))))
        left.barh(range(len(loadings)), [v for _, v in loadings],
                  color=[PALETTE[0] if v >= 0 else PALETTE[1] for _, v in loadings])
        left.axvline(0.0, color='0.4', linewidth=1)
        left.set_yticks(range(len(loadings)), [name for name, _ in loadings], fontsize=7)
        left.invert_yaxis()
        left.set_title('loading per feature', fontsize=9)
        right.barh(range(len(indices)), [v for _, v in indices], color=PALETTE[0])
        right.set_xlim(0, max(1.0, right.get_xlim()[1]))
        right.set_yticks(range(len(indices)), [name for name, _ in indices], fontsize=7)
        right.invert_yaxis()
        right.set_title('ST of each parameter', fontsize=9)
        for ax in (left, right):
            ax.tick_params(labelsize=7)
        note = '' if mode['informative'] else ', below the noise floor'
        figure.suptitle(f'mode {k} ({mode["variance_share"]:.0%} of the variance{note})',
                        fontsize=10)
        figure.tight_layout()
        paths.append(_save(figure, out_dir, f'pca_mode_{k}.png'))
    table = np.array([[ST[f'mode_{m["mode"]}'].get(p, np.nan) for p in parameters]
                      for m in modes])
    figure, ax = plt.subplots(figsize=(1.2 + 0.9 * len(parameters), 1.2 + 0.5 * len(modes)))
    image = ax.imshow(np.clip(table, 0, 1), vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(parameters)), parameters, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(modes)),
                  [f'mode {m["mode"]} ({m["variance_share"]:.0%})'
                   f'{"" if m["informative"] else " *"}' for m in modes], fontsize=7)
    ax.set_title('ST per PCA mode (* below the noise floor)', fontsize=9)
    figure.colorbar(image, ax=ax)
    figure.tight_layout()
    paths.append(_save(figure, out_dir, 'pca_mode_ST_heatmap.png'))
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


#: the model intervals the posterior predictive figure draws, widest first
PREDICTIVE_LEVELS = (0.95, 0.8)


def posterior_predictive(predictive, out_dir):
    '''A uq step's posterior predictive check, the model simulated at draws of the posterior:
    ``posterior_predictive_intervals.csv``, per scalar observable its data, std, the draws'
    median and PREDICTIVE_LEVELS intervals, and whether the data is inside the 95% one;
    ``posterior_predictive_intervals.png``, each observable's intervals in noise stds from its
    data (0), over the +-2 std band of the data, titled with the coverage; and
    ``posterior_predictive_<segment>_<variable>.png`` per kept trace, the draws over time with
    their median and the recorded trace of that variable, if the obs_data has one.'''
    import csv
    predictions = np.asarray(predictive['predictions'], dtype=float)
    data = np.asarray(predictive['ground_truth'], dtype=float)
    std = np.asarray(predictive['std'], dtype=float)
    labels = list(predictive['labels'])
    with np.errstate(all='ignore'):
        median = np.nanmedian(predictions, axis=0)
        bounds = {level: (np.nanpercentile(predictions, 50 * (1 - level), axis=0),
                          np.nanpercentile(predictions, 50 * (1 + level), axis=0))
                  for level in PREDICTIVE_LEVELS}
    scaled = np.isfinite(std) & (std > 0)
    paths = [os.path.join(out_dir, 'posterior_predictive_intervals.csv')]
    with open(paths[0], 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['observable', 'data', 'std', 'median'] +
                        [f'{side}_{round(100 * level)}' for level in PREDICTIVE_LEVELS
                         for side in ('low', 'high')] + ['inside_95'])
        low95, high95 = bounds[0.95]
        for k, label in enumerate(labels):
            writer.writerow([label] + [f'{v:.6g}' for v in (data[k], std[k], median[k])] +
                            [f'{bounds[level][side][k]:.6g}' for level in PREDICTIVE_LEVELS
                             for side in (0, 1)] +
                            [str(bool(low95[k] <= data[k] <= high95[k])).lower()])
    plt = _pyplot()
    shown = [k for k in range(len(labels)) if scaled[k]]
    if shown:
        figure, ax = plt.subplots(figsize=(7.0, max(2.4, 1.2 + 0.32 * len(shown))))
        ax.axvspan(-2, 2, color='0.9', linewidth=0, label='data ± 2 std')
        ax.axvline(0, color='0.5', linewidth=1)
        rows = np.arange(len(shown))[::-1]
        for n, level in enumerate(PREDICTIVE_LEVELS):
            low, high = ((bounds[level][side][shown] - data[shown]) / std[shown]
                         for side in (0, 1))
            ax.hlines(rows, low, high, color=_shade(PALETTE[0], len(PREDICTIVE_LEVELS) - 1 - n,
                                                    len(PREDICTIVE_LEVELS)),
                      linewidth=4 + 3 * n, label=f'model {round(100 * level)}%')
        ax.plot((median[shown] - data[shown]) / std[shown], rows, 'o', color=PALETTE[1],
                markersize=4, label='model median')
        ax.set_yticks(rows)
        ax.set_yticklabels([labels[k] for k in shown], fontsize=7)
        ax.set_ylim(-0.7, len(shown) - 0.3)
        ax.set_xlabel('(model - data) / std', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        coverage = (predictive.get('coverage') or {}).get('coverage', {})
        levels = sorted(coverage.get('levels', {}).items(), key=lambda kv: -float(kv[0]))
        title = 'data inside the model interval, of %d observables: %s' % (
            coverage.get('num_observables', len(labels)),
            ', '.join(f'{100 * row["predictive_coverage"]:.0f}% at {100 * float(level):.0f}%'
                      for level, row in levels))
        left_out = [labels[k] for k in range(len(labels)) if not scaled[k]]
        if left_out:
            title += '\nnot drawn (no std): ' + ', '.join(left_out)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        figure.tight_layout()
        paths.append(_save(figure, out_dir, 'posterior_predictive_intervals.png'))
    series = predictive.get('series')
    if series:
        meta = series.get('meta') or {}
        segment_names = meta.get('experiment_labels') or []
        for key, traces in series['traces'].items():
            segment, _, variable = key.partition('|')
            t = np.asarray(series['time'][segment], dtype=float)
            figure, ax = plt.subplots(figsize=(5.0, 3.2))
            for n, trace in enumerate(np.asarray(traces, dtype=float)):
                ax.plot(t, trace, color=PALETTE[0], alpha=0.3, linewidth=0.8,
                        label='posterior draws' if n == 0 else None)
            with np.errstate(all='ignore'):
                ax.plot(t, np.nanmedian(traces, axis=0), color=PALETTE[0], linewidth=1.8,
                        label='median')
            for observable in meta.get('observables', []):
                if observable.get('kind') == 'series' and \
                        str(observable['segment']) == segment and \
                        observable['variable'] == variable:
                    ax.plot(observable['time'], observable['values'], color=PALETTE[1],
                            linewidth=1.2, label=observable['label'])
            exp_sub = (meta.get('segments') or {}).get(segment)
            where = f'segment {segment}'
            if exp_sub:
                name = (segment_names[exp_sub[0]] if exp_sub[0] < len(segment_names)
                        else f'experiment {exp_sub[0]}')
                where = f'{name}, sub-experiment {exp_sub[1]}'
            ax.set_title(f'{variable} ({where})', fontsize=9)
            ax.set_xlabel('time from the sub-experiment start [s]', fontsize=8)
            ax.set_ylabel(variable, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            figure.tight_layout()
            paths.append(_save(figure, out_dir,
                               f'posterior_predictive_{_stem(segment)}_{_stem(variable)}.png'))
    return paths


def fit_errors(errors, out_dir):
    '''The best fit's error per data_item, from a calibration step: ``fit_errors.csv`` (name,
    percent and std error, data_reference); ``fit_percent_error.png``, a bar per item of
    100 (model - data) / data; and ``fit_std_error.png``, a bar per item in noise stds, over
    the +-1 and +-2 std bands.'''
    import csv
    names = list(errors['names'])
    percent = np.asarray(errors['percent_error'], dtype=float)
    std = np.asarray(errors['std_error'], dtype=float)
    if not names:
        raise ValueError('the calibration scored no data_item.')
    references = list(errors.get('data_references') or [''] * len(names))
    paths = [os.path.join(out_dir, 'fit_errors.csv')]
    with open(paths[0], 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(('data_item', 'percent_error', 'std_error', 'data_reference'))
        for row in zip(names, percent, std, references):
            writer.writerow([row[0], f'{row[1]:.6g}', f'{row[2]:.6g}', row[3]])
    plt = _pyplot()
    rows = np.arange(len(names))[::-1]
    height = max(2.4, 1.0 + 0.3 * len(names))
    for values, name, label, bands in (
            (percent, 'fit_percent_error.png', 'error [%]: 100 (model - data) / data', ()),
            (std, 'fit_std_error.png', 'error [noise stds]: (model - data) / std', (2, 1))):
        figure, ax = plt.subplots(figsize=(6.0, height))
        for n, band in enumerate(bands):
            ax.axvspan(-band, band, color=str(0.92 - 0.06 * n), linewidth=0)
        ax.axvline(0, color='0.5', linewidth=1)
        colours = [PALETTE[0] if not bands or abs(v) <= 2 else PALETTE[1] for v in values]
        ax.barh(rows, values, color=colours, height=0.6)
        ax.set_yticks(rows)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.set_xlabel(label, fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        figure.tight_layout()
        paths.append(_save(figure, out_dir, name))
    return paths


FIGURES = [sweep_traces, sweep_features, prediction_plots, validation, protocol_inputs,
           sobol_indices,
           local_sensitivity, pca_modes, calibration_progress, fit_errors, mcmc_diagnostics,
           emulator_validation, posterior, laplace, posterior_predictive]


def main(argv=None):
    from libcuflynx.calibration_workflow.figures import figure_main
    return figure_main(FIGURES, argv)


if __name__ == '__main__':
    main()
