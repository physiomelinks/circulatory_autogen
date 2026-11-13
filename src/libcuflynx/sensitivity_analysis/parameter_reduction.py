"""Choosing the parameters to calibrate from a Sobol sensitivity analysis.

A reduction reads a Sobol run's directory and writes the entries of the analysed params_for_id
for the parameters it keeps, so a later calibration fits only them::

    reduction = ParamReduction.from_options({'method': 'pca'})   # param_reduction_options
    kept = reduction.reduce('sa_results/', 'candidates_params_for_id.json',
                            'reduction/reduced_params_for_id.json')
    # [{'parameter': 'heart/E_lv_A', 'score': 0.6, 'mode': 1, ...}, ...], in the order kept

Two methods (``param_reduction_options.method``; options in
``ANALYSIS_OPTIONS['param_reduction']``, parsers.PrimitiveParsers):

* :class:`PCASobolReduction` (``'pca'``) -- PCA of the Sobol outputs scaled by their obs_data
  std and weight, Sobol indices of each mode's score, then one parameter per informative mode.
* :class:`SobolReduction` (``'sobol'``) -- the ``n_params`` parameters with the highest Sobol
  index over the outputs.

The report (``param_reduction.json``, and for ``pca`` the modes and their indices) is written
beside the reduced params_for_id. A calibration workflow's ``reduction`` step is one of these.
"""

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

from libcuflynx.sensitivity_analysis.sobol_ranking import (CRITERIA, rank_sobol_parameters,
                                                           read_sobol_indices)

INDEX_TYPES = ('ST', 'S1')
REPORT_FILE = 'param_reduction.json'
REDUCED_PARAMS_FOR_ID_FILE = 'reduced_params_for_id.json'
PCA_MODES_FILE = 'pca_modes.csv'
PCA_MODE_INDICES_FILE = 'pca_mode_sobol_indices.csv'
PCA_MODE_HEATMAP_FILE = 'pca_mode_sobol_ST.png'
SOBOL_INDICES_GLOB = 'all_outputs_n*_Sobol_indices.csv'
#: options that say where to read and write, not how to reduce
PATH_OPTIONS = ('sensitivity_dir', 'output_dir')


class SensitivityFilesError(ValueError):
    """A Sobol run directory without the files a reduction reads."""


def option_descriptors():
    """``{name: descriptor}`` of ``param_reduction_options``."""
    from libcuflynx.parsers.PrimitiveParsers import analysis_options
    return {o['name']: o for o in analysis_options('param_reduction')}


def sobol_run_files(sensitivity_dir):
    """The files of the Sobol run in ``sensitivity_dir``: ``indices``, and when the run saved
    them ``samples``, ``outputs`` and ``problem`` (else None). A SensitivityFilesError when there
    are no indices, or the indices of more than one run."""
    from libcuflynx.sensitivity_analysis.sobolSA import SOBOL_PROBLEM_FILE
    found = sorted(glob.glob(os.path.join(sensitivity_dir, SOBOL_INDICES_GLOB)))
    if not found:
        raise SensitivityFilesError(f'{sensitivity_dir} has no Sobol indices '
                                    f'({SOBOL_INDICES_GLOB}); run the Sobol analysis first.')
    if len(found) > 1:
        raise SensitivityFilesError(f'{sensitivity_dir} has the indices of {len(found)} Sobol '
                                    f'runs ({[os.path.basename(p) for p in found]}); give one '
                                    f'run its own output_dir.')
    stem = found[0][:-len('_indices.csv')]

    def existing(path):
        return path if os.path.isfile(path) else None
    return {'indices': found[0], 'samples': existing(f'{stem}_samples.csv'),
            'outputs': existing(f'{stem}_outputs.csv'),
            'problem': existing(os.path.join(sensitivity_dir, SOBOL_PROBLEM_FILE))}


class ParamReduction:
    """A way of choosing the parameters to calibrate; build one with :meth:`from_options`."""

    method = None
    #: the param_reduction_options this method reads, besides method and the paths
    OPTIONS = ()

    @classmethod
    def from_options(cls, options):
        """The reduction ``param_reduction_options`` describes. Unknown options are an error, as
        is an option another method reads set to other than its default; the paths
        (``sensitivity_dir``, ``output_dir``) are ignored here."""
        descriptors = option_descriptors()
        options = dict(options or {})
        unknown = sorted(set(options) - set(descriptors))
        if unknown:
            raise ValueError(f'param_reduction_options has no {unknown}; the options are '
                             f'{sorted(descriptors)}.')
        method = options.get('method') or descriptors['method']['default']
        if method not in REDUCTIONS:
            raise ValueError(f'param_reduction_options.method is one of {sorted(REDUCTIONS)}, '
                             f'not {method!r}.')
        reduction_cls = REDUCTIONS[method]
        kwargs = {}
        for name, value in options.items():
            if name == 'method' or name in PATH_OPTIONS:
                continue
            if name in reduction_cls.OPTIONS:
                if value is not None:
                    kwargs[name] = value
            elif value != descriptors[name]['default']:
                raise ValueError(f'param_reduction_options.{name} is not read by method '
                                 f'{method!r}; its options are {list(reduction_cls.OPTIONS)}.')
        return reduction_cls(**kwargs)

    def as_dict(self):
        """The options that rebuild this reduction with :meth:`from_options`."""
        return dict({'method': self.method},
                    **{name: getattr(self, name) for name in self.OPTIONS})

    @staticmethod
    def write_params_for_id(source_path, out_path, keep):
        """Write the entries of the params_for_id ``source_path`` whose parameter is in
        ``keep``, whole and in its order, to ``out_path``. Returns the number written."""
        from libcuflynx.parsers.PrimitiveParsers import (ObsAndParamDataParser,
                                                         param_entry_labels)
        from libcuflynx.utilities import params_for_id_file as pfi
        doc = pfi.read_params_for_id(source_path, keep_text=True)
        entries = list(doc.get('params') or [])
        labels = param_entry_labels(ObsAndParamDataParser().get_param_id_info(source_path))
        if len(labels) != len(entries):
            raise ValueError(f'{source_path} has {len(entries)} entries but {len(labels)} '
                             f'parameters; a reduction keeps whole entries, one parameter each.')
        unknown = sorted(set(keep) - set(labels))
        if unknown:
            raise ValueError(f'{source_path} has no parameter {unknown}.')
        kept = [entry for entry, label in zip(entries, labels) if label in keep]
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        pfi.write_params_for_id(out_path, dict(doc, params=kept))
        return len(kept)

    def select(self, files, report_dir):
        """``[{'parameter', 'score', ...}]`` of the kept parameters, in the order kept, from
        the Sobol run ``files`` (:func:`sobol_run_files`). May write files to ``report_dir``."""
        raise NotImplementedError

    def reduce(self, sensitivity_dir, params_for_id_path, out_path):
        """Choose from the Sobol run in ``sensitivity_dir`` and write the kept entries of
        ``params_for_id_path`` (the params_for_id that run analysed) to ``out_path``, and the
        report (``param_reduction.json``) beside it. Returns the selection (see
        :meth:`select`); a ValueError when nothing is kept."""
        report_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(report_dir, exist_ok=True)
        selected = self.select(sobol_run_files(sensitivity_dir), report_dir)
        if not selected:
            raise ValueError(f'the {self.method} reduction kept no parameter '
                             f'({self.as_dict()}).')
        self.write_params_for_id(params_for_id_path, out_path,
                                 [row['parameter'] for row in selected])
        with open(os.path.join(report_dir, REPORT_FILE), 'w') as f:
            json.dump({'reduction': self.as_dict(), 'sensitivity_dir': sensitivity_dir,
                       'params_for_id': params_for_id_path, 'reduced_params_for_id': out_path,
                       'selected': selected, **self.report_extra()}, f, indent=1)
        return selected

    def report_extra(self):
        """More keys for ``param_reduction.json``."""
        return {}


def _check_index_type(index_type):
    if str(index_type).upper() not in INDEX_TYPES:
        raise ValueError(f'index_type is one of {INDEX_TYPES}, not {index_type!r}.')
    return str(index_type).upper()


def _check_number(name, value, low=0.0, high=None, integer=False):
    kind = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kind) or value < low or \
            (high is not None and value > high):
        bound = f'between {low} and {high}' if high is not None else f'>= {low}'
        raise ValueError(f'{name} is a{"n integer" if integer else " number"} {bound}, '
                         f'not {value!r}.')
    return value


class SobolReduction(ParamReduction):
    """Keep the ``n_params`` parameters by Sobol index over the raw outputs.

    Args:
        n_params: How many parameters to keep, highest score first.
        index_type: ``'ST'`` (total, default) or ``'S1'`` (first order).
        criterion: A parameter's score over the outputs: ``'max'`` (default) or ``'mean'``.
        threshold: Parameters scoring below it are dropped, even within the n_params.
    """

    method = 'sobol'
    OPTIONS = ('n_params', 'index_type', 'criterion', 'threshold')

    def __init__(self, n_params=None, index_type='ST', criterion='max', threshold=None):
        if n_params is None:
            raise ValueError('a "sobol" reduction needs n_params, the number of parameters '
                             'to keep.')
        self.n_params = _check_number('n_params', n_params, low=1, integer=True)
        self.index_type = _check_index_type(index_type)
        if criterion not in CRITERIA:
            raise ValueError(f'criterion is one of {CRITERIA}, not {criterion!r}.')
        self.criterion = criterion
        self.threshold = None if threshold is None else _check_number('threshold', threshold)

    def rank(self, indices):
        """``[(parameter, score)]`` of the kept parameters, highest first. ``indices`` is a
        saved indices CSV or ``read_sobol_indices``' dict."""
        if isinstance(indices, str):
            indices = read_sobol_indices(indices)
        return rank_sobol_parameters(indices, self.n_params, self.index_type, self.criterion,
                                     self.threshold)

    def select(self, files, report_dir):
        return [{'parameter': name, 'score': float(score)}
                for name, score in self.rank(files['indices'])]


class PCASobolReduction(ParamReduction):
    """One parameter per informative PCA mode of the Sobol outputs.

    1. Scale each output ``j`` to ``sqrt(w_j) (y_j - mean y_j) / std_j`` with its obs_data
       ``std`` and ``weight`` (the metric of the gaussian_MLE cost), so a mode's variance is in
       units of the data's noise. An output without a std (a prediction feature, a series) is
       scaled by its standard deviation over the samples, with a warning.
    2. PCA (SVD) of the scaled outputs. A mode is informative when its standard deviation is at
       least ``min_mode_std`` and it is among the leading modes explaining ``variance_fraction``
       of the variance.
    3. Sobol indices of each informative mode's score, from the same Saltelli samples.
    4. Over the informative modes, largest first: keep the parameter not yet kept with the
       largest index for that mode, unless that index is below ``min_index`` (no free parameter
       drives the mode). Stop after ``n_params`` (default: one per informative mode).

    The number of informative modes bounds how many parameters the data can tell apart, and
    keeping one parameter per mode stops two parameters that move the data the same way from
    both being kept.
    """

    method = 'pca'
    OPTIONS = ('n_params', 'index_type', 'min_mode_std', 'variance_fraction', 'min_index')

    def __init__(self, n_params=None, index_type='ST', min_mode_std=1.0,
                 variance_fraction=0.99, min_index=0.05):
        self.n_params = None if n_params is None else _check_number('n_params', n_params, low=1,
                                                                     integer=True)
        self.index_type = _check_index_type(index_type)
        self.min_mode_std = float(_check_number('min_mode_std', min_mode_std))
        self.variance_fraction = float(_check_number('variance_fraction', variance_fraction,
                                                     low=0.0, high=1.0))
        self.min_index = float(_check_number('min_index', min_index, high=1.0))
        self._extra = {}

    # ------------------------------------------------------------------ analysis

    @staticmethod
    def scale_outputs(outputs, stds=None, weights=None, names=None):
        """``(Z, scales)``: the centred outputs divided by ``std / sqrt(weight)`` per column.
        A missing or non-positive std falls back to the column's sample std (warned); a
        constant column is left at zero."""
        outputs = np.asarray(outputs, dtype=float)
        n_out = outputs.shape[1]
        stds = list(stds) if stds is not None else [None] * n_out
        weights = list(weights) if weights is not None else [None] * n_out
        names = list(names) if names is not None else [f'output {j}' for j in range(n_out)]
        centred = outputs - np.nanmean(outputs, axis=0)
        sample_std = np.nanstd(outputs, axis=0, ddof=1)
        scales, fallback = np.ones(n_out), []
        for j in range(n_out):
            std = stds[j]
            if std is None or not np.isfinite(std) or std <= 0:
                fallback.append(names[j])
                std = sample_std[j]
            weight = 1.0 if weights[j] is None else float(weights[j])
            scales[j] = std / np.sqrt(weight) if weight > 0 else np.inf
        if fallback:
            warnings.warn(f'no obs_data std for {fallback}: scaled by their spread over the '
                          f'Sobol samples, so a mode of them is informative relative to that '
                          f'spread, not to the noise.', stacklevel=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            scaled = np.where(np.isfinite(scales) & (scales > 0), centred / scales, 0.0)
        return np.nan_to_num(scaled), scales

    @staticmethod
    def modes(scaled):
        """``(variances, loadings, scores)`` of the PCA of ``scaled`` (samples x outputs):
        mode k's variance, its unit loading vector over the outputs (rows of ``loadings``) and
        its score per sample (columns of ``scores``), largest variance first."""
        _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
        variances = singular ** 2 / max(scaled.shape[0] - 1, 1)
        # a fixed sign per mode, so runs and plots agree: largest loading positive
        signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1.0
        loadings = vt * signs[:, None]
        return variances, loadings, scaled @ loadings.T

    def informative(self, variances):
        """Boolean per mode: std at least ``min_mode_std``, within ``variance_fraction``."""
        total = float(np.sum(variances))
        if total <= 0:
            return np.zeros(len(variances), dtype=bool)
        before = np.concatenate([[0.0], np.cumsum(variances)[:-1]]) / total
        return (np.sqrt(variances) >= self.min_mode_std) & (before < self.variance_fraction)

    @staticmethod
    def mode_sobol_indices(problem, scores, calc_second_order=True):
        """``(S1, ST)``, modes x parameters, of each column of ``scores`` (one row per Saltelli
        sample of ``problem``)."""
        from SALib.analyze import sobol
        problem = {'num_vars': len(problem['names']),
                   # a Series: SALib mishandles a list of names (SALib #671)
                   'names': pd.Series(list(problem['names'])),
                   'bounds': [list(b) for b in problem['bounds']]}
        n_modes = scores.shape[1]
        S1 = np.zeros((n_modes, problem['num_vars']))
        ST = np.zeros_like(S1)
        for k in range(n_modes):
            result = sobol.analyze(problem, np.ascontiguousarray(scores[:, k]),
                                   calc_second_order=calc_second_order, print_to_console=False)
            S1[k], ST[k] = result['S1'], result['ST']
        return np.nan_to_num(S1), np.nan_to_num(ST)

    def choose(self, variances, informative, indices, names):
        """``(selected, skipped)``: the greedy one-parameter-per-mode choice from ``indices``
        (modes x parameters) over the informative modes, and the modes skipped because no
        free parameter reaches ``min_index``."""
        informative_idx = [k for k in np.argsort(-variances) if informative[k]]
        total = float(np.sum(variances[informative_idx])) or 1.0
        weighted = (variances[informative_idx, None] * indices[informative_idx]).sum(axis=0)
        weighted = weighted / total
        selected, skipped, taken = [], [], set()
        for k in informative_idx:
            if self.n_params is not None and len(selected) >= self.n_params:
                break
            free = [p for p in range(len(names)) if p not in taken]
            if not free:
                break
            best = max(free, key=lambda p: indices[k, p])
            if indices[k, best] < self.min_index:
                skipped.append({'mode': int(k) + 1,
                                'largest_free_index': float(indices[k, best])})
                continue
            taken.add(best)
            selected.append({'parameter': names[best], 'score': float(weighted[best]),
                             'mode': int(k) + 1,
                             f'{self.index_type}_in_mode': float(indices[k, best]),
                             'mode_variance_share': float(variances[k] / np.sum(variances)),
                             'mode_std': float(np.sqrt(variances[k]))})
        return selected, skipped

    def select(self, files, report_dir):
        missing = [key for key in ('samples', 'outputs', 'problem') if not files[key]]
        if missing:
            raise SensitivityFilesError(
                f'a "pca" reduction reads the Sobol samples and outputs, and the run that wrote '
                f'{files["indices"]} saved no {missing}; rerun the Sobol analysis.')
        with open(files['problem']) as f:
            problem = json.load(f)
        outputs_df = pd.read_csv(files['outputs'])
        records = {r['output']: r for r in problem.get('outputs', [])}
        names = list(outputs_df.columns)
        stds = [records.get(n, {}).get('std') for n in names]
        weights = [records.get(n, {}).get('weight') for n in names]
        scaled, scales = self.scale_outputs(outputs_df.to_numpy(), stds, weights, names)
        variances, loadings, scores = self.modes(scaled)
        informative = self.informative(variances)
        S1, ST = self.mode_sobol_indices(problem, scores,
                                         problem.get('calc_second_order', True))
        indices = ST if self.index_type == 'ST' else S1
        params = list(problem['names'])
        selected, skipped = self.choose(variances, informative, indices, params)
        self._write_report_files(report_dir, variances, informative, loadings, names, S1, ST,
                                 params)
        self._extra = {
            'modes': [{'mode': k + 1, 'variance': float(v), 'std': float(np.sqrt(v)),
                       'variance_share': float(v / np.sum(variances)),
                       'informative': bool(informative[k])}
                      for k, v in enumerate(variances)],
            'n_informative_modes': int(np.sum(informative)),
            'skipped_modes': skipped,
            'output_scales': dict(zip(names, map(float, scales))),
            'files': {'modes': PCA_MODES_FILE, 'mode_indices': PCA_MODE_INDICES_FILE,
                      'heatmap': PCA_MODE_HEATMAP_FILE}}
        return selected

    def report_extra(self):
        return dict(self._extra)

    def _write_report_files(self, report_dir, variances, informative, loadings, names, S1, ST,
                            params):
        modes = pd.DataFrame({'mode': np.arange(1, len(variances) + 1), 'variance': variances,
                              'variance_share': variances / np.sum(variances),
                              'informative': informative})
        for j, name in enumerate(names):
            modes[f'loading_{name}'] = loadings[:, j]
        modes.to_csv(os.path.join(report_dir, PCA_MODES_FILE), index=False)
        # the layout of the Sobol indices CSV, so read_sobol_indices reads it
        table = pd.DataFrame({'Parameter': params})
        for k in range(S1.shape[0]):
            table[f'S1_mode_{k + 1}'] = S1[k]
            table[f'ST_mode_{k + 1}'] = ST[k]
        table.to_csv(os.path.join(report_dir, PCA_MODE_INDICES_FILE), index=False)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(params), 1.2 + 0.5 * len(variances)))
            image = ax.imshow(ST, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(params)), params, rotation=45, ha='right')
            ax.set_yticks(range(len(variances)),
                          [f'mode {k + 1} ({v:.0%}){"" if inf else " *"}' for k, (v, inf) in
                           enumerate(zip(variances / np.sum(variances), informative))])
            ax.set_title('ST per PCA mode (* below the noise floor)')
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, PCA_MODE_HEATMAP_FILE), dpi=120)
            plt.close(fig)
        except Exception as exc:   # a plot is a convenience; the CSVs are the result
            warnings.warn(f'could not draw the PCA mode heatmap: {exc}', stacklevel=2)


REDUCTIONS = {'pca': PCASobolReduction, 'sobol': SobolReduction}


def run_param_reduction(inp_data_dict):
    """Run the reduction ``inp_data_dict['param_reduction_options']`` describes on the Sobol
    run in its ``sensitivity_dir``, for the params_for_id that run analysed
    (``params_for_id_path``). Writes ``reduced_params_for_id.json`` and the report into its
    ``output_dir``. Returns the selection."""
    options = dict(inp_data_dict.get('param_reduction_options') or {})
    reduction = ParamReduction.from_options(options)
    sensitivity_dir = options.get('sensitivity_dir') or inp_data_dict['sa_options']['output_dir']
    output_dir = options.get('output_dir') or os.path.join(sensitivity_dir, 'param_reduction')
    params_for_id_path = inp_data_dict.get('params_for_id_path')
    if not params_for_id_path or not os.path.isfile(params_for_id_path):
        raise ValueError('a parameter reduction writes entries of the params_for_id file the '
                         f'Sobol run analysed, and there is none at {params_for_id_path!r}.')
    out = os.path.join(output_dir, REDUCED_PARAMS_FOR_ID_FILE)
    return reduction.reduce(sensitivity_dir, params_for_id_path, out)
