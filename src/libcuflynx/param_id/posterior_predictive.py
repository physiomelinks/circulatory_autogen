"""Draw from a posterior, run the model at each draw, and check the result covers the data.

A chain says what the parameters could be. It does not say whether the model, at
those parameters, reproduces what was measured -- and a calibration that fits
badly can still produce a tidy-looking posterior. The check is to push samples
back through the forward model and ask how often the data lands inside the
predictions.

Two coverage numbers, because they answer different questions and are easy to
confuse:

``predictive_coverage``
    How often the measured value falls inside the model's own central interval at
    each level. Nominal: an 80% interval should contain about 80% of the
    observations. Well below means the posterior is too narrow or biased; well
    above means it is too wide to say much.

``sample_interval_coverage``
    What fraction of the posterior draws land inside the measurement's own
    interval, ``value +/- z*std``, averaged over observables. Each observable's
    error bar is treated as a fixed window and the posterior mass inside it is
    counted, so an observable whose draws straddle the window scores partly
    rather than being called a hit or a miss on its median alone. A model
    centred on the measurement with the measurement's own spread scores the
    nominal level. ``per_observable`` carries the individual fractions.

Run it against the **solver**, not an emulator, unless you only want a smoke test:
an emulator scoring its own predictions against the data cannot tell you that the
emulator is wrong, and it is the emulator's error you most want this to catch.
``posterior_predictive`` therefore builds its engine with ``use_emulator`` forced
off by default.

Usage::

    from libcuflynx.param_id.posterior_predictive import posterior_predictive

    result = posterior_predictive(inp_data_dict, num_samples=100)
    result.save(output_dir)
    print(result.summary())
"""
import json
import os

import numpy as np

#: Levels reported unless the caller asks for others.
DEFAULT_LEVELS = (0.8, 0.95)

CHAIN_FILE = 'mcmc_chain.npy'
SAMPLES_FILE = 'posterior_predictive.npz'
SERIES_FILE = 'posterior_predictive_series.npz'

#: Traces are kept for this many draws unless asked otherwise. A fan of twenty
#: thin lines is indistinguishable from a fan of a hundred, and a hundred draws
#: of every segment at the solver's own step is hundreds of megabytes.
DEFAULT_SERIES_DRAWS = 20
#: Points per trace after decimation. Far more than a plot can resolve, and small
#: enough that the whole study is a few megabytes.
DEFAULT_SERIES_POINTS = 2000
COVERAGE_FILE = 'posterior_predictive_coverage.json'


class PosteriorPredictiveError(ValueError):
    """The posterior predictive check cannot be run as asked."""


# ── the chain ──────────────────────────────────────────────────────────────
def load_chain(output_dir):
    """The saved chain, shaped ``(steps, walkers, params)``."""
    path = os.path.join(output_dir, CHAIN_FILE)
    if not os.path.isfile(path):
        raise PosteriorPredictiveError(
            'no %s in %s -- run the UQ stage (do_uq: true) first'
            % (CHAIN_FILE, output_dir))
    chain = np.load(path, allow_pickle=True)
    if chain.ndim != 3:
        raise PosteriorPredictiveError(
            '%s has shape %s; expected (steps, walkers, params)'
            % (path, chain.shape))
    return chain


def sample_parameters(chain, num_samples=100, burn_in=0.5, random_seed=0):
    """``num_samples`` draws from the chain, after burn-in.

    Drawn without replacement where the chain is long enough, and at random
    rather than by thinning at a fixed stride: a stride that happens to match a
    period in the walkers' motion samples a slice of the posterior rather than
    the posterior.

    ``burn_in`` below 1 is a fraction of the steps; 1 or above is a number of
    steps. The walkers start scattered over the prior box, so the early steps
    describe where they were initialised.
    """
    n_steps = chain.shape[0]
    start = int(n_steps * burn_in) if burn_in < 1 else int(burn_in)
    start = min(max(start, 0), max(n_steps - 1, 0))

    flat = chain[start:].reshape(-1, chain.shape[2])
    if flat.shape[0] == 0:
        raise PosteriorPredictiveError(
            'burn_in of %s leaves no samples in a chain of %d steps'
            % (burn_in, n_steps))

    rng = np.random.default_rng(random_seed)
    replace = flat.shape[0] < num_samples
    idx = rng.choice(flat.shape[0], size=num_samples, replace=replace)
    return flat[idx], {
        'n_steps': int(n_steps),
        'burn_in_steps': int(start),
        'pool': int(flat.shape[0]),
        'drawn_with_replacement': bool(replace),
    }


# ── traces ─────────────────────────────────────────────────────────────────
def trace_operand(obs_info, obs_idx):
    """``(index, name)`` of the operand that is the observable's own trace.

    Not always operand 0: a spike-frequency item is ``(time, V)``, so taking the
    first would plot time against time. The first operand that is not the time
    base is the one being measured.
    """
    operands = (obs_info.get('operands') or [])
    names = operands[obs_idx] if obs_idx < len(operands) else None
    if not names:
        return 0, None
    if isinstance(names, str):
        names = [names]
    for index, name in enumerate(names):
        if str(name).strip().lower() != 'time':
            return index, str(name)
    return 0, str(names[0])


def decimate(values, points):
    """``values`` thinned to at most ``points``, keeping the ends.

    Evenly spaced indices rather than a stride: a stride drops the last sample
    whenever the length is not a multiple of it, and the end of a trace is where
    a steady state is read off.
    """
    values = np.asarray(values, dtype=float).ravel()
    if points is None or len(values) <= points:
        return values
    idx = np.unique(np.linspace(0, len(values) - 1, points).round().astype(int))
    return values[idx]


def segment_time_axis(protocol_info, exp_idx, sub_idx, n_points):
    """The time grid for one sub-experiment, on the same clock as plot_outputs.

    Sub-experiments run back to back within an experiment, so each starts where
    the previous one ended -- plotting them all from zero would stack a step
    response on top of its own baseline.
    """
    sim_times = protocol_info['sim_times'][exp_idx]
    start = float(np.sum(sim_times[:sub_idx]))
    duration = float(sim_times[sub_idx])
    return np.linspace(start, start + duration, int(n_points))


# ── the forward model ──────────────────────────────────────────────────────
def predicted_constants(engine, obs_info, protocol_info, theta,
                        collect_series=False, series_points=DEFAULT_SERIES_POINTS):
    """The model's scalar observables at ``theta``, aligned with ``ground_truth_const``.

    One evaluation per sub-experiment segment, then each observable read from its
    own: a data_item names the experiment and sub-experiment it belongs to, and
    reading it out of another segment is reading the wrong number. Same mapping
    ``plot_outputs.emulator_error_vectors`` uses, for the same reason.
    """
    num_const = len(obs_info['ground_truth_const'])
    out = np.full(num_const, np.nan)
    series = {} if collect_series else None

    _, operands_list = engine.get_cost_and_obs_from_params(np.asarray(theta, dtype=float))
    if not operands_list:
        return (out, series) if collect_series else out

    num_sub_per_exp = protocol_info['num_sub_per_exp']
    by_segment = {}
    for const_idx, obs_idx in enumerate(obs_info['const_idx_to_obs_idx']):
        exp = int(obs_info['experiment_idxs'][obs_idx])
        sub = int(obs_info['subexperiment_idxs'][obs_idx])
        flat = sum(num_sub_per_exp[:exp]) + sub
        if flat >= len(operands_list) or operands_list[flat] is None:
            continue
        if flat not in by_segment:
            by_segment[flat] = np.asarray(
                engine.get_obs_output_dict(operands_list[flat])['const'], dtype=float)
        consts = by_segment[flat]
        if const_idx < len(consts):
            out[const_idx] = consts[const_idx]

        if collect_series:
            # The operand behind this observable, in the segment it belongs to.
            # Every item carries a trace in every segment, but only its own
            # segment's is the one it was measured from.
            index, name = trace_operand(obs_info, obs_idx)
            if name is None:
                continue
            key = (flat, name)
            if key in series:
                continue
            item = operands_list[flat][obs_idx] if obs_idx < len(operands_list[flat]) else None
            if item is None or index >= len(item):
                continue
            trace = np.asarray(item[index], dtype=float).ravel()
            if trace.size > 1:
                series[key] = decimate(trace, series_points)

    return (out, series) if collect_series else out


def simulate_samples(client, thetas, progress_every=10, comm=None,
                     series_draws=0, series_points=DEFAULT_SERIES_POINTS):
    """Run the forward model once per row of ``thetas``, spread across ranks.

    Each draw is one full evaluation of the protocol and the draws are entirely
    independent, so this is the part worth parallelising: on a real study a draw
    costs ~13s, which is an hour for a few hundred of them on one rank and a few
    minutes on ten.

    Under ``mpiexec`` each rank simulates a contiguous block and rank 0
    reassembles them in order; with no launcher the one-rank stub makes that the
    same serial loop it always was. Every rank must be handed the *same*
    ``thetas`` -- they are drawn deterministically from the chain for exactly
    that reason.

    Returns ``(predictions, failures)`` on rank 0 and ``(None, failures)``
    elsewhere. A draw that fails to simulate becomes a row of NaN rather than
    stopping the sweep -- posterior draws reach corners of the parameter box the
    calibration never visited, and losing the whole check to one of them helps
    nobody. The count is summed across ranks and reported.
    """
    from libcuflynx.emulators.emulator_trainer import _block_for_rank
    from libcuflynx.utilities.mpi_utils import get_MPI

    if comm is None:
        comm = get_MPI().COMM_WORLD
    rank, num_procs = comm.Get_rank(), comm.Get_size()

    engine = client.param_id
    obs_info = client.obs_info
    protocol_info = client.protocol_info
    num_const = len(obs_info['ground_truth_const'])

    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    start, end = _block_for_rank(len(thetas), rank, num_procs)

    rows = []
    failures = 0
    traces = {}
    for offset, theta in enumerate(thetas[start:end]):
        draw = start + offset
        # Traces are kept for the first few draws only -- see DEFAULT_SERIES_DRAWS.
        # Chosen by draw index rather than by rank so the same draws are kept
        # however many ranks the sweep is spread over.
        want_series = draw < series_draws
        try:
            result = predicted_constants(
                engine, obs_info, protocol_info, theta,
                collect_series=want_series, series_points=series_points)
            if want_series:
                row, drawn = result
                rows.append(row)
                for key, trace in (drawn or {}).items():
                    traces.setdefault(key, []).append((draw, trace))
            else:
                rows.append(result)
        except Exception as exc:  # noqa: BLE001 - one bad draw must not end the sweep
            failures += 1
            rows.append(np.full(num_const, np.nan))
            if failures <= 3:
                print('  [warn] rank %d: sample %d did not simulate: %s'
                      % (rank, start + offset, exc), flush=True)
        if progress_every and (offset + 1) % progress_every == 0:
            print('  rank %d simulated %d/%d of its posterior samples'
                  % (rank, offset + 1, end - start), flush=True)

    block = np.vstack(rows) if rows else np.empty((0, num_const))
    gathered = comm.gather((start, block, failures, traces), root=0)
    if rank != 0:
        return None, failures, None

    predictions = np.full((len(thetas), num_const), np.nan)
    total_failures = 0
    collected = {}
    for block_start, block_rows, block_failures, block_traces in gathered:
        predictions[block_start:block_start + len(block_rows)] = block_rows
        total_failures += block_failures
        for key, entries in (block_traces or {}).items():
            collected.setdefault(key, []).extend(entries)
    if total_failures == len(thetas) and len(thetas):
        # Not a partial result. Every draw failing means the run was misconfigured
        # -- a missing params_for_id leaves param_id_info None and raises on every
        # evaluation -- and reporting "0 observables" for that reads as a finding
        # about the fit rather than a broken setup.
        raise PosteriorPredictiveError(
            'every one of the %d posterior draws failed to simulate; the run is '
            'misconfigured rather than badly fitted. The warning above names the '
            'first few failures.' % len(thetas))
    if total_failures:
        print('  [warn] %d of %d posterior samples did not simulate'
              % (total_failures, len(thetas)), flush=True)

    # Sorted by draw index so the rows mean the same thing on any rank layout.
    stacked = {}
    for key, entries in collected.items():
        entries.sort(key=lambda item: item[0])
        lengths = {len(trace) for _, trace in entries}
        if len(lengths) != 1:
            # A segment whose traces came back different lengths cannot be
            # stacked; dropping it is better than padding it into a shape that
            # would plot as a real trace.
            print('  [warn] %s: traces of differing length %s; not kept'
                  % (str(key), sorted(lengths)), flush=True)
            continue
        stacked[key] = np.vstack([trace for _, trace in entries])
    return predictions, total_failures, stacked


# ── coverage ───────────────────────────────────────────────────────────────
def _z_for(level):
    """Two-sided normal quantile, without pulling scipy in for one number."""
    from math import erf, sqrt

    lo, hi = 0.0, 10.0
    target = (1.0 + level) / 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def coverage(predictions, ground_truth, std, levels=DEFAULT_LEVELS):
    """Both coverage numbers, per level, over the observables that simulated.

    ``predictions`` is ``(num_samples, num_const)``; NaN columns and NaN samples
    are excluded rather than counted as misses.
    """
    predictions = np.asarray(predictions, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    std = np.asarray(std, dtype=float)

    usable = ~np.all(np.isnan(predictions), axis=0) & np.isfinite(ground_truth)
    result = {'num_observables': int(usable.sum()),
              'num_observables_skipped': int((~usable).sum()),
              'levels': {}}
    if not usable.any():
        return result

    preds = predictions[:, usable]
    truth = ground_truth[usable]
    sigma = std[usable]

    for level in levels:
        lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
        lo = np.nanpercentile(preds, lo_q, axis=0)
        hi = np.nanpercentile(preds, hi_q, axis=0)
        inside_predictive = (truth >= lo) & (truth <= hi)

        # Every draw against its own observable's window, not the median against
        # it: collapsing to a median throws away the shape of the posterior, and
        # an observable whose draws straddle the error bar is neither a clean hit
        # nor a clean miss.
        z = _z_for(level)
        with np.errstate(invalid='ignore'):
            half = z * np.abs(sigma)
            inside = np.abs(preds - truth[None, :]) <= half[None, :]
        # A draw that did not simulate is not a miss; it is not a draw.
        counted = np.isfinite(preds)
        with np.errstate(invalid='ignore', divide='ignore'):
            per_observable = np.where(
                counted.sum(axis=0) > 0,
                (inside & counted).sum(axis=0) / np.maximum(counted.sum(axis=0), 1),
                np.nan)

        result['levels'][str(level)] = {
            'predictive_coverage': float(np.mean(inside_predictive)),
            'sample_interval_coverage': float(np.nanmean(per_observable)),
            'per_observable': [None if not np.isfinite(v) else float(v)
                               for v in per_observable],
            'z': float(z),
        }
    return result


# ── the whole check ────────────────────────────────────────────────────────
class PosteriorPredictiveResult:
    """Everything the check produced, ready to save, plot or print."""

    def __init__(self, thetas, predictions, ground_truth, std, labels,
                 coverage_summary, chain_info, failures, used_emulator,
                 series=None, series_meta=None):
        self.thetas = thetas
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.std = std
        self.labels = list(labels)
        self.coverage = coverage_summary
        self.chain_info = chain_info
        self.failures = failures
        self.used_emulator = used_emulator
        #: ``{(segment, variable): (draws, points)}`` for the draws traces were
        #: kept for, and the metadata a plot needs to place them.
        self.series = series or {}
        self.series_meta = series_meta or {}

    def intervals(self, level=0.95):
        """``(lo, median, hi)`` per observable at ``level``."""
        lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
        return (np.nanpercentile(self.predictions, lo_q, axis=0),
                np.nanmedian(self.predictions, axis=0),
                np.nanpercentile(self.predictions, hi_q, axis=0))

    def summary(self):
        lines = [
            'Posterior predictive: %d samples, %d observables%s'
            % (self.predictions.shape[0], self.coverage['num_observables'],
               ' (against the EMULATOR, not the solver)' if self.used_emulator else ''),
            '  chain: %d steps, %d dropped as burn-in, pool %d'
            % (self.chain_info['n_steps'], self.chain_info['burn_in_steps'],
               self.chain_info['pool']),
        ]
        if self.failures:
            lines.append('  %d sample(s) did not simulate' % self.failures)
        if self.coverage['num_observables_skipped']:
            lines.append('  %d observable(s) skipped (never simulated)'
                         % self.coverage['num_observables_skipped'])
        for level, row in sorted(self.coverage['levels'].items()):
            lines.append(
                '  %s: %.0f%% of data inside the model interval (nominal %.0f%%); '
                '%.0f%% of draws inside value +/- %.2f*std'
                % (level, 100 * row['predictive_coverage'], 100 * float(level),
                   100 * row['sample_interval_coverage'], row['z']))
        return '\n'.join(lines)

    def save(self, output_dir):
        """Write the samples and the coverage summary; returns the paths."""
        os.makedirs(output_dir, exist_ok=True)
        samples_path = os.path.join(output_dir, SAMPLES_FILE)
        np.savez(
            samples_path,
            thetas=self.thetas,
            predictions=self.predictions,
            ground_truth=self.ground_truth,
            std=self.std,
            labels=np.array(self.labels, dtype=object),
        )
        if self.series:
            arrays = {'__meta__': np.array(json.dumps(self.series_meta))}
            for (segment, variable), block in self.series.items():
                arrays['y|%d|%s' % (segment, variable)] = block
            for segment, axis in (self.series_meta.get('time') or {}).items():
                arrays['t|%s' % segment] = np.asarray(axis, dtype=float)
            np.savez_compressed(
                os.path.join(output_dir, SERIES_FILE), **arrays)

        coverage_path = os.path.join(output_dir, COVERAGE_FILE)
        with open(coverage_path, 'w') as file:
            json.dump({
                'coverage': self.coverage,
                'chain': self.chain_info,
                'num_samples': int(self.predictions.shape[0]),
                'samples_that_failed_to_simulate': int(self.failures),
                'used_emulator': bool(self.used_emulator),
            }, file, indent=2)
        return samples_path, coverage_path


def observable_labels(obs_info):
    """A readable name per constant observable, in ``ground_truth_const`` order."""
    names = (obs_info.get('data_item_names') or obs_info.get('names_for_plotting')
             or obs_info.get('obs_names'))
    return [str(names[obs_idx]) for obs_idx in obs_info['const_idx_to_obs_idx']]


def _resolve_model_path(config):
    """Fill in ``model_path`` when the config does not carry one.

    A user_inputs.yaml written by hand usually does not: the generation stage adds
    it at runtime, so a config that has only ever been through
    ``cuflynx-generate`` in the same process has it and one loaded fresh from disk
    does not. Without this, calling the check from a yaml fails inside
    ``init_from_dict`` on a missing positional argument, which says nothing about
    what to add.
    """
    if config.get('model_path'):
        return config['model_path']

    prefix = config.get('file_prefix')
    generated = config.get('generated_models_dir')
    if not (prefix and generated):
        raise PosteriorPredictiveError(
            'the config has no model_path, and no file_prefix/generated_models_dir '
            'to work one out from')

    candidate = os.path.join(generated, prefix, prefix + '.cellml')
    if os.path.isfile(candidate):
        return candidate

    # The layout a param-id run writes, when generation was driven by the obs file.
    obs_path = config.get('param_id_obs_path') or ''
    obs_stem = os.path.splitext(os.path.basename(obs_path))[0]
    if obs_stem:
        with_obs = os.path.join(
            generated, '%s_%s' % (prefix, obs_stem), prefix + '.cellml')
        if os.path.isfile(with_obs):
            return with_obs

    raise PosteriorPredictiveError(
        'no model_path in the config and no generated model at %s -- generate the '
        'model first, or set model_path' % candidate)


def series_metadata(client, series, ground_truth, std, predictions=None):
    """Everything a plot needs to place the traces: axes, and what to draw on them.

    The observables are carried alongside because a trace on its own says
    nothing about whether it is right -- what makes the figure readable is the
    measured value drawn across it in the style its ``plot_type`` asks for.

    ``predictions`` (draws x observables) adds the *model's* value of the same
    statistic beside the measured one. Without it a reader can see that the
    traces sit above the measured max but not by how much the reduced number
    disagrees, which is the quantity the cost is actually built from.
    """
    if not series:
        return {}

    obs_info = client.obs_info
    protocol_info = client.protocol_info
    num_sub_per_exp = protocol_info['num_sub_per_exp']

    flat_to_exp_sub = {}
    flat = 0
    for exp_idx, n_sub in enumerate(num_sub_per_exp):
        for sub_idx in range(n_sub):
            flat_to_exp_sub[flat] = (exp_idx, sub_idx)
            flat += 1

    time_axes = {}
    for (segment, _), block in series.items():
        key = str(segment)
        if key in time_axes:
            continue
        exp_idx, sub_idx = flat_to_exp_sub.get(segment, (0, 0))
        time_axes[key] = segment_time_axis(
            protocol_info, exp_idx, sub_idx, block.shape[1]).tolist()

    labels = observable_labels(obs_info)
    plot_types = obs_info.get('plot_type') or []
    observables = []
    for const_idx, obs_idx in enumerate(obs_info['const_idx_to_obs_idx']):
        exp_idx = int(obs_info['experiment_idxs'][obs_idx])
        sub_idx = int(obs_info['subexperiment_idxs'][obs_idx])
        segment = sum(num_sub_per_exp[:exp_idx]) + sub_idx
        _, variable = trace_operand(obs_info, obs_idx)
        observables.append({
            'segment': int(segment),
            'experiment': exp_idx,
            'subexperiment': sub_idx,
            'variable': variable,
            'label': labels[const_idx] if const_idx < len(labels) else str(const_idx),
            'value': float(ground_truth[const_idx]),
            'std': float(std[const_idx]),
            'plot_type': str(plot_types[obs_idx]) if obs_idx < len(plot_types) else 'None',
            'kind': 'constant',
            # What the model made of the same statistic. The median over draws
            # rather than the mean: one diverged draw should not move the line
            # the reader compares against.
            'operation': str((obs_info.get('operations') or [None] * (obs_idx + 1))[obs_idx]
                             or ''),
            'model_value': (float(np.median(predictions[:, const_idx]))
                            if predictions is not None and const_idx < predictions.shape[1]
                            else None),
        })

    # Recorded traces, which are what a simulated trace can actually be compared
    # against -- the scalar observables above reduce to a horizontal line, and a
    # line says nothing about whether the shape is right. Usually carried at
    # weight 0 (drawn, not fitted), and included whatever the weight: this is a
    # plotting payload, and a fitted series is if anything more worth seeing.
    recorded = obs_info.get('ground_truth_series') or []
    series_map = obs_info.get('series_idx_to_obs_idx') or []
    series_weights = obs_info.get('weight_series_vec')
    series_names = obs_info.get('data_item_names') or obs_info.get('trace_names_for_plotting') or []
    for series_idx, obs_idx in enumerate(series_map):
        if series_idx >= len(recorded):
            break
        values = np.asarray(recorded[series_idx], dtype=float).ravel()
        if values.size < 2:
            continue
        exp_idx = int(obs_info['experiment_idxs'][obs_idx])
        sub_idx = int(obs_info['subexperiment_idxs'][obs_idx])
        segment = sum(num_sub_per_exp[:exp_idx]) + sub_idx
        _, variable = trace_operand(obs_info, obs_idx)
        weight = 0.0
        if series_weights is not None and series_idx < len(series_weights):
            weight = float(np.max(np.abs(np.asarray(series_weights[series_idx], dtype=float))))
        observables.append({
            'segment': int(segment),
            'experiment': exp_idx,
            'subexperiment': sub_idx,
            'variable': variable,
            'label': (str(series_names[obs_idx]) if obs_idx < len(series_names)
                      else 'series %d' % series_idx),
            # Its own axis, not the block's: a recording is sampled at obs_dt and
            # the kept draws are decimated, so the two rarely share a length.
            'time': segment_time_axis(
                protocol_info, exp_idx, sub_idx, values.size).tolist(),
            'values': values.tolist(),
            'weight': weight,
            'plot_type': 'series',
            'kind': 'series',
        })

    return {
        'time': time_axes,
        'observables': observables,
        'experiment_labels': list(protocol_info.get('experiment_labels') or []),
        'segments': {str(k): list(v) for k, v in flat_to_exp_sub.items()},
    }


def posterior_predictive(inp_data_dict=None, num_samples=100, burn_in=0.5,
                         random_seed=0, levels=DEFAULT_LEVELS,
                         use_emulator=False, output_dir=None, save=True,
                         client=None, series_draws=DEFAULT_SERIES_DRAWS,
                         series_points=DEFAULT_SERIES_POINTS):
    """Sample the posterior, run the model at each draw, and report coverage.

    ``use_emulator`` is False by default and that is the point: an emulator
    scoring its own predictions against the data cannot tell you the emulator is
    wrong, and its error is the thing this check most needs to catch. Set it True
    only for a fast smoke test, and the summary will say so.

    Pass ``client`` when you already have a ``CVS0DParamID`` -- a run that has
    just finished sampling has one, and building a second compiles the model
    again. The caller owns what that engine was built with, so ``use_emulator``
    then only labels the summary; it does not change what is evaluated.

    Under ``mpiexec`` the draws are shared out across ranks; the result comes
    back on rank 0 and ``None`` elsewhere, as ``EmulatorTrainer.train()`` does.
    """
    if client is None:
        from libcuflynx.param_id.paramID import CVS0DParamID

        if inp_data_dict is None:
            raise PosteriorPredictiveError(
                'pass either a configuration or an already-built client')
        config = dict(inp_data_dict)
        config['model_path'] = _resolve_model_path(config)
        config['use_emulator'] = bool(use_emulator)
        if not use_emulator:
            # Otherwise init_from_dict resolves a bundle we are not using.
            config.pop('emulator_dir', None)
        client = CVS0DParamID.init_from_dict(config)
    else:
        use_emulator = bool(
            getattr(getattr(client, 'param_id', None), 'emulates_features', False))

    from libcuflynx.utilities.mpi_utils import get_MPI

    comm = get_MPI().COMM_WORLD

    # Rank 0 resolves the directory, reads the chain and draws. CVS0DParamID only
    # sets output_dir on rank 0, so the other ranks have nothing to resolve; and
    # every rank needs the *same* draws, which a broadcast guarantees outright
    # rather than relying on each of them seeding identically.
    payload = None
    if comm.Get_rank() == 0:
        resolved_dir = output_dir or client.output_dir
        chain = load_chain(resolved_dir)
        thetas, chain_info = sample_parameters(
            chain, num_samples=num_samples, burn_in=burn_in,
            random_seed=random_seed)
        payload = (resolved_dir, thetas, chain_info)
    if comm.Get_size() > 1:
        payload = comm.bcast(payload, root=0)
    resolved_dir, thetas, chain_info = payload

    if comm.Get_rank() == 0:
        print('Posterior predictive: simulating %d draws%s across %d rank(s)'
              % (len(thetas), ' on the emulator' if use_emulator else '',
                 comm.Get_size()), flush=True)
    predictions, failures, series = simulate_samples(
        client, thetas, comm=comm,
        series_draws=min(series_draws, len(thetas)), series_points=series_points)
    if comm.Get_rank() != 0:
        # Only rank 0 has the assembled predictions, so only rank 0 can score or
        # save them. Same contract as EmulatorTrainer.train().
        return None

    obs_info = client.obs_info
    ground_truth = np.asarray(obs_info['ground_truth_const'], dtype=float)
    std = np.asarray(obs_info['std_const_vec'], dtype=float)

    result = PosteriorPredictiveResult(
        thetas=thetas, predictions=predictions, ground_truth=ground_truth,
        std=std, labels=observable_labels(obs_info),
        coverage_summary=coverage(predictions, ground_truth, std, levels),
        chain_info=chain_info, failures=failures, used_emulator=bool(use_emulator),
        series=series,
        series_meta=series_metadata(client, series, ground_truth, std,
                                    predictions=predictions))

    if save:
        result.save(resolved_dir)
    return result
