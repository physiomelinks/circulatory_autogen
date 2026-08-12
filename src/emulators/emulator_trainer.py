"""Fitting an emulator of the forward model (issue #333).

The expensive analyses -- Sobol SA at ``num_samples*(2M+2)`` runs, MCMC at tens of thousands,
identifiability at a Hessian's worth -- all reach the model through one narrow interface:
theta in, scalar observable features out. This trains a surrogate of exactly that function, so
those analyses can evaluate it instead.

The training targets come from :func:`param_id.fd_backend.observable_features`, which is the
same function the cost is computed from. That is deliberate and load-bearing: a second
implementation of "reduce a run to its features" would let the emulator be accurate about
something the calibration is not fitting.

The N training runs are paid up front, so the win is real for Sobol, MCMC and identifiability,
and is *not* real for a single GA calibration, which may well be cheaper run directly.
"""
import os
import warnings

import numpy as np
from scipy.stats import qmc

from emulators.emulator_bundle import EmulatorBundle, fingerprint

AUTOEMULATE_MISSING_MESSAGE = (
    'autoemulate is not installed. Install it with `pip install '
    '"circulatory_autogen[emulation]"` (it needs Python >=3.10,<3.13 and pulls in torch, '
    'gpytorch and lightgbm).')

SAMPLE_TYPES = ('sobol', 'latin_hypercube', 'random')


def autoemulate_available():
    """Whether the optional backend is installed, without importing it.

    Deliberately a spec lookup rather than a try/except import: this module is reachable from
    ``solver_wrappers``, which every CA run imports, and importing autoemulate drags in torch
    -- seconds of startup for a run that may have nothing to do with emulation.
    """
    import importlib.util
    return importlib.util.find_spec('autoemulate') is not None


def _load_autoemulate():
    """Import and return the ``AutoEmulate`` class, or raise with the install instructions."""
    if not autoemulate_available():
        raise RuntimeError(AUTOEMULATE_MISSING_MESSAGE)
    from autoemulate import AutoEmulate
    return AutoEmulate


def emulator_model_names():
    """The emulator names ``emulator_settings.models`` accepts, for a settings form.

    Discovered from autoemulate's registry rather than hardcoded, in the same spirit as
    ``cost_func_metadata()``; empty when autoemulate is not installed, so a tool can show the
    setting as unavailable instead of showing a stale list.
    """
    if not autoemulate_available():
        return []
    return sorted(_load_autoemulate().list_emulators(default_only=False)['Emulator'].tolist())


def require_autoemulate():
    if not autoemulate_available():
        raise RuntimeError(AUTOEMULATE_MISSING_MESSAGE)


class EmulatorTrainer:
    """Design -> simulate -> fit -> validate -> persist, for one param-id engine.

    Args:
        param_id: an ``OpencorParamID`` built against the **truth** solver. Passing one whose
            sim helper is itself an emulator is rejected: it would fit a surrogate of a
            surrogate.
        emulator_settings: the ``emulator_settings`` block (see ``ANALYSIS_OPTIONS``).
        comm: an MPI communicator; ``None`` means "discover one, or run serially".

    There is deliberately no DEBUG shrinking here, unlike the optimiser and MCMC options.
    ``num_train_samples`` is the one setting that decides whether the emulator is any good,
    and quietly cutting it would produce an emulator that answers everything and is wrong --
    the failure mode this whole feature is built to prevent. A cheap run asks for it by name.
    """

    def __init__(self, param_id, emulator_settings, comm=None):
        if getattr(param_id.sim_helper, 'emulates_features', False):
            raise ValueError(
                'EmulatorTrainer was handed a param-id engine that is already running on an '
                'emulator. Build it with use_emulator: false so training runs the real solver.')
        self.pid = param_id
        self.settings = dict(emulator_settings or {})
        self.comm = comm if comm is not None else _discover_comm()
        self.rank = self.comm.Get_rank() if self.comm is not None else 0
        self.num_procs = self.comm.Get_size() if self.comm is not None else 1
        self._check_observables_are_scalar()

    # ------------------------------------------------------------------ construction

    @classmethod
    def init_from_dict(cls, inp_data_dict, comm=None):
        """Build the truth-solver engine this config describes, then a trainer over it."""
        from param_id.paramID import CVS0DParamID
        inp = dict(inp_data_dict)
        # The one line that stops the trainer emulating an emulator: whatever the config says
        # about using one, training always runs the solver named by `solver:`.
        inp['use_emulator'] = False
        inp['do_ad'] = False
        engine = CVS0DParamID.init_from_dict(inp)
        if inp_data_dict.get('obs_data_dict') is not None:
            engine.set_ground_truth_data(inp_data_dict['obs_data_dict'])
        if inp_data_dict.get('params_for_id') is not None:
            engine.set_params_for_id(inp_data_dict['params_for_id'])
        trainer = cls(engine.param_id, inp_data_dict.get('emulator_settings') or {}, comm=comm)
        trainer.output_dir = resolve_emulator_dir(inp_data_dict)
        return trainer

    def _check_observables_are_scalar(self):
        """Refuse non-scalar data_items up front, where it is a config error rather than a
        confusing shape mismatch a hundred simulations later."""
        bad = {jj: dtype for jj, dtype in enumerate(self.pid.obs_info['data_types'])
               if dtype != 'constant'}
        if bad:
            raise ValueError(
                f'the emulator predicts scalar data_item features only, but obs_data.json has '
                f'data_type(s) {sorted(set(bad.values()))} at data_item index(es) {sorted(bad)}. '
                f'Those need the full simulated trace ("series") or its FFT ("frequency"). '
                f'Remove them from the obs_data used for emulation, or run without an emulator.')

    # ------------------------------------------------------------------ settings

    def _setting(self, name, default):
        value = self.settings.get(name, default)
        return default if value is None else value

    @property
    def num_train_samples(self):
        return int(self._setting('num_train_samples', 128))

    @property
    def feature_labels(self):
        """The emulator's outputs, named exactly as the run that will use it names them."""
        from param_id.paramID import emulated_feature_labels
        return emulated_feature_labels(self.pid.obs_info)

    # ------------------------------------------------------------------ stages

    def design(self):
        """Training points over the ``params_for_id`` box, shape ``(n_samples, num_params)``.

        Deterministic given the seed, so every rank builds the identical design and no
        broadcast is needed to agree on who evaluates which sample.
        """
        mins = np.asarray(self.pid.param_id_info['param_mins'], dtype=float)
        maxs = np.asarray(self.pid.param_id_info['param_maxs'], dtype=float)
        num_params = mins.size
        n_samples = self.num_train_samples
        sample_type = self._setting('sample_type', 'sobol')
        seed = int(self._setting('random_seed', 0))

        if sample_type == 'sobol':
            with warnings.catch_warnings():
                # Sobol warns for counts that aren't a power of two; the balance properties
                # lost matter less here than letting the user pick a round sample count.
                warnings.simplefilter('ignore')
                unit = qmc.Sobol(d=num_params, scramble=True, seed=seed).random(n_samples)
        elif sample_type == 'latin_hypercube':
            unit = qmc.LatinHypercube(d=num_params, seed=seed).random(n_samples)
        elif sample_type == 'random':
            unit = np.random.default_rng(seed).random((n_samples, num_params))
        else:
            raise ValueError(f'unknown sample_type "{sample_type}", expected one of '
                             f'{", ".join(SAMPLE_TYPES)}')

        if bool(self._setting('log_scale_params', False)):
            if np.any(mins <= 0):
                raise ValueError(
                    'log_scale_params needs every parameter min to be positive; '
                    f'{[self.pid.param_id_info["param_mins"][i] for i in np.where(mins <= 0)[0]]} '
                    'are not. Set it false, or raise those mins above zero.')
            return np.exp(np.log(mins) + unit * (np.log(maxs) - np.log(mins)))
        return mins + unit * (maxs - mins)

    def evaluate(self, design):
        """Run the truth model at every design point; returns ``(x, y)`` on rank 0.

        Contiguous block split across ranks then a gather, the same shape as the Sobol
        sampler's parallel loop. A sample whose simulation fails is dropped rather than
        imputed: an imputed training target is a fabricated observation, and the emulator
        would learn it as fact.
        """
        # Imported here rather than at module scope: this module is reachable from
        # solver_wrappers, which every CA run imports, and fd_backend pulls in the parsers.
        from param_id.fd_backend import observable_features

        n_samples = len(design)
        start, end = _block_for_rank(n_samples, self.rank, self.num_procs)
        local_rows = []
        for local_idx, theta in enumerate(design[start:end]):
            features = observable_features(self.pid, theta)
            if features is None or not np.all(np.isfinite(features)):
                print(f'[emulator rank {self.rank}] sample {start + local_idx} failed; dropping it')
                continue
            local_rows.append((start + local_idx, np.asarray(features, dtype=float)))
            if self.rank == 0 and (local_idx + 1) % 10 == 0:
                print(f'[emulator] rank 0 has run {local_idx + 1}/{end - start} of its samples')

        gathered = self.comm.gather(local_rows, root=0) if self.comm is not None else [local_rows]
        if self.rank != 0:
            return None, None
        rows = sorted((row for chunk in gathered for row in chunk), key=lambda item: item[0])
        if not rows:
            raise RuntimeError('every training simulation failed; no emulator can be fitted. '
                               'Check that the model runs at the params_for_id bounds.')
        n_failed = n_samples - len(rows)
        if n_failed:
            print(f'[emulator] {n_failed}/{n_samples} training simulations failed and were dropped')
        self._n_failed = n_failed
        x = np.asarray([design[idx] for idx, _ in rows], dtype=float)
        y = np.asarray([features for _, features in rows], dtype=float)
        return x, y

    def fit(self, x, y):
        """Fit and compare emulators; returns ``(model, r2, rmse, name, x_scale, y_scale)``.

        Both x and y are mapped onto a well-conditioned range first. CA parameters routinely
        span a compliance near 1e-9 and a resistance near 1e8, and autoemulate works in float32
        torch, where that spread alone is enough to ruin a kernel fit.

        A test split is held out **here**, before fitting, and the reported R2/RMSE are scored
        on it. Scoring on the training points instead would report a number that says how well
        the emulator memorised the design, which is precisely the reassurance a bad emulator
        would give.
        """
        require_autoemulate()
        x_scale = EmulatorBundle.make_scale(x)
        y_scale = EmulatorBundle.make_scale(y)
        x_scaled = (x - np.asarray(x_scale['shift'])) / np.asarray(x_scale['span'])
        y_scaled = (y - np.asarray(y_scale['shift'])) / np.asarray(y_scale['span'])

        seed = int(self._setting('random_seed', 0))
        x_train, y_train, x_test, y_test = _train_test_split(
            x_scaled, y_scaled, float(self._setting('test_fraction', 0.2)), seed)

        kwargs = dict(n_iter=int(self._setting('n_iter', 10)),
                      n_splits=int(self._setting('n_splits', 5)),
                      random_seed=seed)
        models = _parse_models(self._setting('models', 'default'))
        if models is not None:
            kwargs['models'] = models

        emulation = _load_autoemulate()(x_train, y_train, test_data=(x_test, y_test), **kwargs)
        result = emulation.best_result()
        r2, rmse = _per_feature_scores(result.model, x_test, y_test, y_scale)
        return result.model, r2, rmse, _result_model_name(result), x_scale, y_scale

    def train(self):
        """The whole pipeline. Returns the bundle on rank 0, ``None`` elsewhere."""
        require_autoemulate()
        design = self.design()
        if self.rank == 0:
            print(f'[emulator] training on {len(design)} samples across {self.num_procs} rank(s)')
        x, y = self.evaluate(design)
        if self.rank != 0:
            return None

        model, r2, rmse, model_name, x_scale, y_scale = self.fit(x, y)
        meta = {
            'param_entry_labels': [str(label) for label in _param_labels(self.pid)],
            'param_mins': [float(v) for v in self.pid.param_id_info['param_mins']],
            'param_maxs': [float(v) for v in self.pid.param_id_info['param_maxs']],
            'param_names': _jsonable_names(self.pid.param_id_info['param_names']),
            # A modifier entry's one theta slot expands to theta * baseline per target before
            # it reaches a solver, so the helper cannot recover theta from what it is handed
            # and must be given it directly. It needs to know that from the metadata.
            'has_modifiers': bool(self.pid.param_id_info.get('modifiers')),
            'param_defaults': self._baseline_snapshot(),
            'feature_labels': self.feature_labels,
            'feature_r2': [float(v) for v in r2],
            'feature_rmse': [float(v) for v in rmse],
            'x_scale': x_scale,
            'y_scale': y_scale,
            'model_name': model_name,
            'design': {'sample_type': self._setting('sample_type', 'sobol'),
                       'num_train_samples': int(len(design)),
                       'num_used': int(len(x)),
                       'num_failed': int(getattr(self, '_n_failed', 0)),
                       'random_seed': int(self._setting('random_seed', 0)),
                       'log_scale_params': bool(self._setting('log_scale_params', False))},
            'fingerprint': fingerprint(self.pid.param_id_info, self.pid.obs_info,
                                       self.pid.protocol_info, self.pid.model_path),
            'provenance': _provenance(self.pid),
        }
        bundle = EmulatorBundle(model, meta, x_train=x, y_train=y)
        output_dir = getattr(self, 'output_dir', None) or os.getcwd()
        bundle.save(output_dir)
        print(f'[emulator] saved to {output_dir}')
        for label, score in zip(bundle.feature_labels, meta['feature_r2']):
            print(f'    held-out R2 {score:8.4f}   {label}')
        return bundle

    def _baseline_snapshot(self):
        """Parameter defaults recorded from the real model, for the emulator to serve later.

        ``resolve_modifier_baselines`` and the optimiser's x0 both read parameter defaults
        before any simulation runs. The emulator has no model to read them from, so they are
        captured here, while a real solver is still in hand.
        """
        helper = self.pid.sim_helper
        names = list(self.pid.param_id_info['param_names'])
        for modifier in self.pid.param_id_info.get('modifiers', []) or []:
            names.extend([[target] for target in modifier.get('targets', [])])
            names.extend([[qname] for qname in (modifier.get('inputs') or {}).values()
                          if isinstance(qname, str)])
        reader = getattr(helper, 'get_default_param_vals', None) or helper.get_init_param_vals
        snapshot = {}
        for entry in names:
            entry_names = entry if isinstance(entry, (list, tuple)) else [entry]
            try:
                values = reader([list(entry_names)])[0]
            except Exception as error:                       # pragma: no cover - model dependent
                print(f'[emulator] could not record a default for {entry_names}: {error}')
                continue
            values = values if isinstance(values, (list, tuple)) else [values]
            for name, value in zip(entry_names, values):
                snapshot[str(name)] = float(value)
        return snapshot


# ---------------------------------------------------------------------- helpers

def resolve_emulator_dir(inp_data_dict):
    """Where this config's emulator lives, defaulting beside the param-id output."""
    settings = inp_data_dict.get('emulator_settings') or {}
    if settings.get('emulator_dir'):
        return settings['emulator_dir']
    # Same fallback CVS0DParamID uses for param_id_output_dir, so training and the run that
    # uses the emulator land on the same directory even when neither names one.
    root = inp_data_dict.get('param_id_output_dir') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'param_id_output')
    prefix = inp_data_dict.get('file_prefix', 'model')
    obs_path = inp_data_dict.get('param_id_obs_path') or ''
    obs_prefix = os.path.splitext(os.path.basename(obs_path))[0] if obs_path else 'obs'
    return os.path.join(root, 'emulators', f'{prefix}_{obs_prefix}')


def _block_for_rank(n_samples, rank, num_procs):
    """The contiguous [start, end) block of samples this rank evaluates."""
    per_rank, remainder = divmod(n_samples, num_procs)
    if rank < remainder:
        start = rank * (per_rank + 1)
        return start, start + per_rank + 1
    start = rank * per_rank + remainder
    return start, start + per_rank


def _discover_comm():
    """The world communicator when running under a launcher, else None (serial).

    Imported lazily and only when mpi4py is already loaded or a launcher is present, so merely
    training an emulator in a plain script does not open MPI.
    """
    try:
        from utilities.mpi_utils import get_MPI          # available once #396 lands
        return get_MPI().COMM_WORLD
    except ImportError:
        pass
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:                                  # pragma: no cover - env dependent
        return None


def _parse_models(models):
    """``'default'`` -> autoemulate's own default set; otherwise a list of emulator names."""
    if models in (None, '', 'default'):
        return None
    if models == 'all':
        return emulator_model_names()
    if isinstance(models, str):
        return [name.strip() for name in models.split(',') if name.strip()]
    return list(models)


def _param_labels(pid):
    from parsers.PrimitiveParsers import param_entry_labels
    return param_entry_labels(pid.param_id_info)


def _jsonable_names(param_names):
    return [[str(name) for name in (entry if isinstance(entry, (list, tuple)) else [entry])]
            for entry in param_names]


def _provenance(pid):
    import subprocess
    provenance = {'model_path': str(pid.model_path), 'model_type': str(pid.model_type),
                  'solver': str((pid.solver_info or {}).get('solver'))}
    try:
        from importlib.metadata import version
        provenance['autoemulate_version'] = version('autoemulate')
    except Exception:                                     # pragma: no cover - env dependent
        pass
    try:
        provenance['ca_git_sha'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                     # pragma: no cover - not always a repo
        pass
    return provenance


def _result_model_name(result):
    for attr in ('model_name', 'name'):
        if hasattr(result, attr):
            return str(getattr(result, attr))
    return type(getattr(result, 'model', result)).__name__


def _train_test_split(x, y, test_fraction, seed):
    """A seeded split, with at least one test point and at least two training points."""
    n_samples = len(x)
    n_test = int(round(test_fraction * n_samples))
    n_test = max(1, min(n_test, n_samples - 2))
    order = np.random.default_rng(seed).permutation(n_samples)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def _per_feature_scores(model, x_test, y_test, y_scale):
    """Held-out R2 and RMSE, **per feature**.

    autoemulate's summary reports one score per model, aggregated over outputs. A single
    aggregate hides exactly the case this check exists for -- five features fitted well and one
    badly -- since the good ones average the bad one away. RMSE is converted back to the
    feature's real units so it can be read against the observation it approximates.
    """
    from emulators.emulator_bundle import _as_backend_input, _as_numpy_mean
    y_true = np.asarray(y_test, dtype=float).reshape(len(y_test), -1)
    n_features = y_true.shape[1]
    try:
        predicted = _as_numpy_mean(
            model.predict(_as_backend_input(model, x_test))).reshape(len(x_test), -1)
    except Exception as error:                            # pragma: no cover - backend dependent
        print(f'[emulator] could not score the fitted emulator per feature: {error}')
        return [float('nan')] * n_features, [float('nan')] * n_features

    span = np.asarray(y_scale['span'], dtype=float)
    r2, rmse = [], []
    for col in range(n_features):
        truth, pred = y_true[:, col], predicted[:, col]
        residual = float(np.sum((truth - pred) ** 2))
        total = float(np.sum((truth - np.mean(truth)) ** 2))
        # A degenerate test column (every value equal) has no variance to explain; report nan
        # rather than a 1.0 that would read as a perfect fit.
        r2.append(1.0 - residual / total if total > 0 else float('nan'))
        rmse.append(float(np.sqrt(residual / len(truth)) * span[col % span.size]))
    return r2, rmse
