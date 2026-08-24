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
import json
import os
import warnings

import numpy as np
from scipy.stats import qmc

from libcuflynx.emulators.emulator_bundle import (METADATA_FILE, TRAINING_DATA_FILE, EmulatorBundle,
                                       EmulatorQualityError, EmulatorReuseError, fingerprint)

AUTOEMULATE_MISSING_MESSAGE = (
    'autoemulate is not installed. Install it with `pip install '
    '"libcuflynx[emulation]"` (it needs Python >=3.10,<3.13, and pulls in torch, gpytorch '
    'and lightgbm -- about 750 MB).')

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
    base = sorted(_load_autoemulate().list_emulators(default_only=False)['Emulator'].tolist())
    # The two-phase variants are offered here so a settings form can list them, but
    # they are not part of `all` -- see _parse_models. Two stages cost more to fit
    # than one and only pay off when the features really do have a floor.
    from libcuflynx.emulators.internal_emulators import two_phase_model_names
    return base + two_phase_model_names(base)


def base_emulator_model_names():
    """Only autoemulate's own names -- what ``models: all`` resolves to."""
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
        from libcuflynx.param_id.paramID import CVS0DParamID
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
        confusing shape mismatch a hundred simulations later.

        Zero-weighted ones are exempt. A weight of 0 drops an item from the cost entirely,
        so the emulator is never asked for it -- ``emulated_feature_labels`` is built from
        ``const_idx_to_obs_idx`` and excludes non-constants already. Carrying a recorded
        trace at weight 0 purely so it can be drawn behind the model is a real use (it is
        the only way to have anything to compare a simulated trace against on an
        output-vs-time plot), and refusing it forced a second obs_data file that existed
        only to keep the emulator happy.
        """
        obs_info = self.pid.obs_info
        unweighted = set()
        for kind in ('series', 'amp', 'phase'):
            weights = obs_info.get('weight_%s_vec' % kind)
            idx_map = obs_info.get('%s_idx_to_obs_idx' % kind)
            if weights is None or idx_map is None:
                continue
            for weight, obs_idx in zip(weights, idx_map):
                if not np.any(np.asarray(weight, dtype=float)):
                    unweighted.add(int(obs_idx))

        bad = {jj: dtype for jj, dtype in enumerate(obs_info['data_types'])
               if dtype != 'constant' and jj not in unweighted}
        if bad:
            raise ValueError(
                f'the emulator predicts scalar data_item features only, but obs_data.json has '
                f'data_type(s) {sorted(set(bad.values()))} at data_item index(es) {sorted(bad)}. '
                f'Those need the full simulated trace ("series") or its FFT ("frequency"). '
                f'Remove them from the obs_data used for emulation, give them weight 0 if they '
                f'are only there to be plotted, or run without an emulator.')

    # ------------------------------------------------------------------ settings

    def _setting(self, name, default):
        value = self.settings.get(name, default)
        return default if value is None else value

    @property
    def num_train_samples(self):
        return int(self._setting('num_train_samples', 128))

    @property
    def reuse_samples(self):
        """Whether to refit the samples already on disk instead of simulating new ones."""
        return bool(self._setting('reuse_samples', False))

    @property
    def feature_labels(self):
        """The emulator's outputs, named exactly as the run that will use it names them."""
        from libcuflynx.param_id.paramID import emulated_feature_labels
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
        from libcuflynx.param_id.fd_backend import observable_features

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

        # Imported here, not at module scope: internal_emulators asks this module
        # for the emulator list when it builds its two_phase_<name> helpers, so a
        # top-level import either way closes the cycle.
        from libcuflynx.emulators.internal_emulators import (  # noqa: PLC0415
            base_emulator_name, fit_two_phase, is_two_phase, two_phase_name)

        two_phase = [name for name in (models or []) if is_two_phase(name)]
        if two_phase:
            if len(models) > 1:
                raise ValueError(
                    f'a two-phase emulator is fitted on its own, but models is {models!r}. '
                    f'Ask for exactly one, e.g. models: {two_phase[0]}.')
            # The classifier and the second regressor are fitted here; autoemulate
            # still does both regressions, so every other setting applies unchanged.
            base_name = base_emulator_name(two_phase[0])
            kwargs.pop('models', None)
            model, result = fit_two_phase(
                x_train, y_train, x_test, y_test, base_name,
                _load_autoemulate(), kwargs)
            validation = _validation_report(model, x_test, y_test, x_scale, y_scale)
            return (model, validation, two_phase_name(base_name), x_scale, y_scale)

        emulation = _load_autoemulate()(x_train, y_train, test_data=(x_test, y_test), **kwargs)
        result = emulation.best_result()
        validation = _validation_report(result.model, x_test, y_test, x_scale, y_scale)
        return (result.model, validation, _result_model_name(result), x_scale, y_scale)

    def output_directory(self):
        """Where this trainer's bundle is written, and read back from when reusing samples."""
        return getattr(self, 'output_dir', None) or os.getcwd()

    def load_previous_samples(self):
        """The saved design and simulated features, checked against the run at hand.

        Returns ``(x, y, design_meta)`` -- the same pair :meth:`evaluate` would have produced
        and the ``meta['design']`` block describing where they came from.

        Every rank runs this rather than only rank 0. It is a metadata read and one small npz,
        so there is nothing to split, and a refusal seen only by rank 0 would leave the other
        ranks exiting as though the run had succeeded. Nothing here is collective, so the
        non-zero ranks that drop out immediately afterwards cannot be left waiting on a
        barrier rank 0 has skipped.
        """
        directory = self.output_directory()
        bundle = self._load_previous_bundle(directory)
        x, y = self._checked_previous_samples(bundle, directory)

        if self.rank == 0:
            print(f'[emulator] reuse_samples: refitting {len(x)} samples already simulated in '
                  f'{directory}; no simulation will be run')
            if len(x) != self.num_train_samples:
                # Never let the requested number stand as though it had been used: the design
                # is fixed when reusing, and a metadata line claiming 128 samples behind a fit
                # of 24 is exactly the kind of provenance nobody re-checks.
                print(f'[emulator] note: emulator_settings.num_train_samples is '
                      f'{self.num_train_samples}, but the saved design holds {len(x)} usable '
                      f'sample(s), and that is what is being fitted. num_train_samples, '
                      f'sample_type and log_scale_params describe the design, which '
                      f'reuse_samples does not rebuild. Set reuse_samples: false to run a new '
                      f'design of {self.num_train_samples}.')

        # The saved block already says how those samples were designed (sample_type, seed,
        # how many failed) and all of it is still true, so it is carried through rather than
        # overwritten with settings that had no effect on this run.
        design_meta = dict(bundle.meta.get('design') or {})
        design_meta.update({
            'num_used': int(len(x)),
            'reused_samples': True,
            # The design seed stays whatever drew the design; this is the one the *fit* and
            # its train/test split used, which reuse still honours and is worth varying.
            'fit_random_seed': int(self._setting('random_seed', 0)),
        })
        return x, y, design_meta

    def _load_previous_bundle(self, directory):
        """The previous bundle's metadata and samples, without loading its fitted model.

        Deliberately not ``EmulatorBundle.load``: the fitted emulator is the one part of the
        artefact reuse replaces, so requiring it to deserialise (torch, joblib, on every rank)
        would make a refit fail for the sake of an object about to be thrown away.
        """
        meta_path = os.path.join(directory, METADATA_FILE)
        if not os.path.isfile(meta_path):
            raise EmulatorReuseError(
                f'emulator_settings.reuse_samples is set, but there is no emulator to reuse the '
                f'samples of in {directory} ({METADATA_FILE} is missing). Reuse refits samples a '
                f'previous run simulated, so the first training run has to happen with '
                f'reuse_samples: false -- that is the run that pays for the simulations.')
        with open(meta_path) as file:
            meta = json.load(file)
        x_train = y_train = None
        data_path = os.path.join(directory, TRAINING_DATA_FILE)
        if os.path.isfile(data_path):
            with np.load(data_path) as data:
                x_train, y_train = data['x_train'], data['y_train']
        try:
            # A bundle with no model: never predicted from, only asked whether its samples
            # belong to this problem -- which is `check_matches`, the same comparison every
            # other consumer of a bundle makes.
            return EmulatorBundle(None, meta, x_train=x_train, y_train=y_train)
        except ValueError as error:
            raise EmulatorReuseError(
                f'the emulator metadata in {directory} cannot be reused: {error}. Retrain with '
                f'emulator_settings.reuse_samples: false.') from error

    def _checked_previous_samples(self, bundle, directory):
        """Refuse saved samples that describe a different problem, or that are not there.

        Samples are only meaningful for the parameter box, obs_data, protocol and model they
        were simulated against. Refitting stale ones produces an emulator that is confidently
        wrong about a study it was never trained for, and nothing downstream can tell.
        """
        if bundle.x_train is None or bundle.y_train is None:
            raise EmulatorReuseError(
                f'the emulator in {directory} has no saved training samples to reuse '
                f'({TRAINING_DATA_FILE} is missing, or it was trained by a CA version that did '
                f'not keep the samples). Retrain with emulator_settings.reuse_samples: false '
                f'to simulate and save a design, after which reuse works.')
        try:
            bundle.check_matches(
                fingerprint(self.pid.param_id_info, self.pid.obs_info, self.pid.protocol_info,
                            self.pid.model_path),
                # str() to match how train() writes them, or an equal set of labels of a
                # different type would read as a changed parameter list.
                param_entry_labels=[str(label) for label in _param_labels(self.pid)],
                feature_labels=self.feature_labels)
        except EmulatorQualityError as error:
            raise EmulatorQualityError(
                f'{error} emulator_settings.reuse_samples cannot be used here: the samples in '
                f'{directory} were simulated for a different problem, and refitting them would '
                f'produce an emulator that is confidently wrong about this one. Retrain with '
                f'reuse_samples: false, which re-runs the simulations for the current setup.'
            ) from error

        x = np.asarray(bundle.x_train, dtype=float)
        y = np.asarray(bundle.y_train, dtype=float)
        num_params = len(np.asarray(self.pid.param_id_info['param_mins'], dtype=float).reshape(-1))
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or x.shape[1] != num_params \
                or y.shape[1] != len(self.feature_labels):
            raise EmulatorReuseError(
                f'the saved training samples in {directory} do not fit this run: x_train is '
                f'{x.shape} and y_train is {y.shape}, but this run has {num_params} parameter(s) '
                f'and {len(self.feature_labels)} feature(s). Retrain with '
                f'emulator_settings.reuse_samples: false.')
        if len(x) < 3:
            # fit() holds out at least one point and needs two to train on.
            raise EmulatorReuseError(
                f'only {len(x)} usable training sample(s) were saved in {directory}, which is '
                f'not enough to fit and hold out a test set. Retrain with '
                f'emulator_settings.reuse_samples: false and a larger num_train_samples.')
        return x, y

    def train(self):
        """The whole pipeline. Returns the bundle on rank 0, ``None`` elsewhere.

        With ``reuse_samples`` set, the design and the simulations are skipped entirely and the
        samples a previous run saved are fitted instead -- see :meth:`load_previous_samples`.
        The rest of the path (fit, metadata, bundle, save) is the same one a fresh run takes,
        so the artefact is not a lesser kind of emulator.
        """
        require_autoemulate()
        if self.reuse_samples:
            x, y, design_meta = self.load_previous_samples()
            if self.rank != 0:
                return None
        else:
            design = self.design()
            if self.rank == 0:
                print(f'[emulator] training on {len(design)} samples across '
                      f'{self.num_procs} rank(s)')
            x, y = self.evaluate(design)
            if self.rank != 0:
                return None
            design_meta = {'sample_type': self._setting('sample_type', 'sobol'),
                           'num_train_samples': int(len(design)),
                           'num_used': int(len(x)),
                           'num_failed': int(getattr(self, '_n_failed', 0)),
                           'random_seed': int(self._setting('random_seed', 0)),
                           'log_scale_params': bool(self._setting('log_scale_params', False)),
                           # So provenance never has to be read as "presumably simulated here".
                           'reused_samples': False}

        model, validation, model_name, x_scale, y_scale = self.fit(x, y)
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
            'feature_r2': [float(v) for v in validation['r2']],
            'feature_rmse': [float(v) for v in validation['rmse']],
            # The rest of what the held-out set says, per feature: not summary
            # enough to replace a plot, but enough to rank features by how badly
            # the emulator does on them, which R2 alone does not (a feature can
            # score well and still be biased). See `error_stats`.
            'feature_mae': [float(v) for v in validation['mae']],
            'feature_bias': [float(v) for v in validation['bias']],
            'feature_max_abs_error': [float(v) for v in validation['max_abs_error']],
            'feature_nrmse': [float(v) for v in validation['nrmse']],
            'x_scale': x_scale,
            'y_scale': y_scale,
            'model_name': model_name,
            'design': design_meta,
            # The emulator_settings block this emulator was made with. Saved because
            # two of these settings -- min_r2 and fd_rel_step -- are read again when
            # the emulator is USED, by which time the caller may be a calibration /
            # SA / UQ run that never saw the emulation settings at all. Without this
            # they silently fell back to the schema defaults (0.9 and 1e-3), so a
            # user who set min_r2: 0.88 in emulator_settings was refused at 0.9 and
            # told it was "the configured min_r2". The emulator now carries its own
            # configuration; see _use_time_setting in param_id/paramID.py.
            'settings': _jsonable_settings(self.settings),
            'fingerprint': fingerprint(self.pid.param_id_info, self.pid.obs_info,
                                       self.pid.protocol_info, self.pid.model_path),
            'provenance': _provenance(self.pid),
        }
        bundle = EmulatorBundle(model, meta, x_train=x, y_train=y,
                                validation=validation)
        output_dir = self.output_directory()
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
    from libcuflynx.utilities.paths import default_param_id_output_dir
    root = inp_data_dict.get('param_id_output_dir') or default_param_id_output_dir()
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
    """The world communicator: the real one under a launcher, a one-rank stub otherwise.

    Through ``get_MPI`` and never ``from mpi4py import MPI``, so a serial training run never
    opens MPI at all -- that import registers an atexit MPI_Finalize which aborts on macOS
    when a NIC goes away (#396). The stub implements the collectives used here, so the split
    and gather below need no serial special case.

    Imported inside the function because this module is reachable from ``solver_wrappers``,
    which every CA run imports.
    """
    from libcuflynx.utilities.mpi_utils import get_MPI  # noqa: PLC0415

    return get_MPI().COMM_WORLD


def _parse_models(models):
    """``'default'`` -> autoemulate's own default set; otherwise a list of emulator names."""
    if models in (None, '', 'default'):
        return None
    if models == 'all':
        # Base emulators only: a two-phase variant is opt-in by name, never swept up
        # by 'all', because it fits a classifier and a second regressor on top.
        return base_emulator_model_names()
    if isinstance(models, str):
        return [name.strip() for name in models.split(',') if name.strip()]
    return list(models)


def _param_labels(pid):
    from libcuflynx.parsers.PrimitiveParsers import param_entry_labels
    return param_entry_labels(pid.param_id_info)


def _jsonable_names(param_names):
    return [[str(name) for name in (entry if isinstance(entry, (list, tuple)) else [entry])]
            for entry in param_names]


def _jsonable_settings(settings):
    """The emulator_settings block, reduced to what json can hold.

    Anything that will not serialise is dropped rather than failing the save: this
    is provenance the *use* path reads back, and losing an exotic value is better
    than losing the emulator that was just trained for it.
    """
    out = {}
    for key, value in (settings or {}).items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[str(key)] = value
        elif isinstance(value, (list, tuple)):
            out[str(key)] = [v for v in value
                             if isinstance(v, (bool, int, float, str)) or v is None]
    return out


def _provenance(pid):
    import subprocess
    provenance = {'model_path': str(pid.model_path), 'model_type': str(pid.model_type),
                  'solver': str((pid.solver_info or {}).get('solver'))}
    try:
        from importlib.metadata import version
        provenance['autoemulate_version'] = version('autoemulate')
    except Exception:                                     # pragma: no cover - env dependent
        pass
    # Only meaningful when running out of a checkout. Installed there is no CA git repo, and
    # asking git from inside site-packages would report whatever unrelated repo happens to be
    # above it (#431).
    from libcuflynx.utilities.paths import repo_root
    checkout = repo_root()
    if checkout is not None:
        try:
            provenance['ca_git_sha'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=checkout,
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:                                 # pragma: no cover - not always a repo
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


def _validation_report(model, x_test, y_test, x_scale, y_scale):
    """Everything the held-out set says about the emulator, per feature.

    Returns the per-feature statistics **and the held-out points themselves** --
    the parameters, the simulator's answer and the emulator's -- because the
    statistics cannot answer the question a user actually has. R2 says how well the
    emulator does on average over the design; a parity or residual plot says
    *where* it goes wrong, which is what decides whether the region you care about
    is one of the good ones (#333).

    These points are free: training already paid to simulate them and then
    deliberately did not fit to them. Everything is converted back to real units,
    so a consumer never has to know the emulator was fitted in a scaled space.
    """
    from libcuflynx.emulators.emulator_bundle import _as_backend_input, _as_numpy_mean

    y_true_scaled = np.asarray(y_test, dtype=float).reshape(len(y_test), -1)
    n_features = y_true_scaled.shape[1]
    empty = [float('nan')] * n_features
    try:
        predicted_scaled = _as_numpy_mean(
            model.predict(_as_backend_input(model, x_test))).reshape(len(x_test), -1)
    except Exception as error:  # pragma: no cover - backend dependent
        print(f'[emulator] could not score the fitted emulator per feature: {error}')
        return {'r2': empty, 'rmse': empty, 'mae': empty, 'bias': empty,
                'max_abs_error': empty, 'nrmse': empty,
                'theta': np.empty((0, 0)), 'y_true': np.empty((0, n_features)),
                'y_pred': np.empty((0, n_features))}

    y_span = np.asarray(y_scale['span'], dtype=float)
    y_shift = np.asarray(y_scale['shift'], dtype=float)
    x_span = np.asarray(x_scale['span'], dtype=float)
    x_shift = np.asarray(x_scale['shift'], dtype=float)

    y_true = y_true_scaled * y_span + y_shift
    y_pred = predicted_scaled * y_span + y_shift
    theta = np.asarray(x_test, dtype=float) * x_span + x_shift

    report = {'r2': [], 'rmse': [], 'mae': [], 'bias': [], 'max_abs_error': [],
              'nrmse': [], 'theta': theta, 'y_true': y_true, 'y_pred': y_pred}
    for col in range(n_features):
        truth, pred = y_true[:, col], y_pred[:, col]
        error = pred - truth
        residual = float(np.sum(error ** 2))
        total = float(np.sum((truth - np.mean(truth)) ** 2))
        # A degenerate test column (every value equal) has no variance to explain;
        # report nan rather than a 1.0 that would read as a perfect fit.
        report['r2'].append(1.0 - residual / total if total > 0 else float('nan'))
        rmse = float(np.sqrt(residual / len(truth)))
        report['rmse'].append(rmse)
        report['mae'].append(float(np.mean(np.abs(error))))
        # Signed, deliberately: a systematically high emulator and a noisy one can
        # share an RMSE, and only one of them shifts every downstream cost.
        report['bias'].append(float(np.mean(error)))
        report['max_abs_error'].append(float(np.max(np.abs(error))))
        # RMSE against the spread of the feature, so features in different units
        # can be ranked against each other -- which raw RMSE cannot do, and which
        # is the axis "which feature is the emulator worst at" needs.
        spread = float(np.max(truth) - np.min(truth))
        report['nrmse'].append(rmse / spread if spread > 0 else float('nan'))
    return report
