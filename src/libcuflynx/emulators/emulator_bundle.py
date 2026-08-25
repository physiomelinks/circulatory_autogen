"""The trained-emulator artefact: the fitted model, its metadata, and every refusal rule.

An emulator does not fail loudly. Handed parameters it was never trained on, or fitted to a
design too small for the response it is approximating, it returns plausible numbers, and every
Sobol index, cost and posterior computed from them inherits the error silently. So the artefact
carries what is needed to *disprove* itself -- held-out R2 per output, the parameter box it was
trained in, and a fingerprint of the model, parameters and protocol it was trained against --
and CA refuses to use it when any of those disagree with the run at hand (issue #333).

The bundle also owns the input/output scaling. CA parameters span many orders of magnitude
(compliances near 1e-9 next to resistances near 1e8) and autoemulate works in float32 torch,
where that spread destroys a kernel fit. Parameters are mapped to the unit box and features are
standardised here, before anything reaches the emulator, and the transforms are stored in the
metadata -- an emulator reloaded without them would predict in the wrong units.
"""
import hashlib
import json
import os
import pickle

import numpy as np

METADATA_FILE = 'emulator_metadata.json'
MODEL_FILE = 'emulator'          # autoemulate's saver appends .joblib
#: How the fitted model is pickled. The file is called ``emulator.joblib``
#: whichever is used -- that is the artefact's name, not a claim about the
#: container -- and the bundle records which one wrote it so it can be read back.
SERIALISERS = ('auto', 'joblib', 'cloudpickle', 'dill')
DEFAULT_SERIALISER = 'auto'
#: What ``auto`` tries, in order. joblib first because it is what autoemulate
#: itself writes and reads. cloudpickle before dill on measurement, not
#: reputation: on autoemulate 2.1.2 dill *fails* where joblib succeeds --
#: torch-backed emulators hold a ``PyCapsule`` it recurses on -- so the fix
#: proposed in #468 would have traded one broken case for a commoner one.
#: cloudpickle handled every emulator tried, and the unnameable objects joblib
#: refuses. dill stays last because it is what the issue reports working.
_AUTO_ORDER = ('joblib', 'cloudpickle', 'dill')
#: What a serialiser raises when it meets an object it cannot take apart, as
#: opposed to a disk that is full or a path that does not exist. Only these are
#: worth retrying with a different serialiser.
#:
#: Both families are needed and neither implies the other: #468 reports a
#: ``TypeError`` ("cannot pickle '_abc._abc_data' object"), raised from the C
#: pickler, while a plain unnameable object -- a closure, a class defined in a
#: function -- raises ``PicklingError`` from the Python one.
_PICKLING_FAILURES = (TypeError, AttributeError, NotImplementedError, ValueError,
                      pickle.PickleError)
TRAINING_DATA_FILE = 'training_data.npz'
#: The held-out points, the simulator's answer at each and the emulator's. The
#: per-feature statistics in the metadata say how wrong the emulator is on
#: average; only these say *where* -- which is the question that decides whether
#: the region a study cares about is one of the good ones (#333).
VALIDATION_FILE = 'emulator_validation.npz'

#: metadata keys that must be present for a bundle to be usable at all.
REQUIRED_META_KEYS = ('param_entry_labels', 'param_mins', 'param_maxs', 'feature_labels',
                      'feature_r2', 'x_scale', 'y_scale', 'fingerprint')


def weighted_non_scalar_obs(obs_info):
    """Data_items an emulator would have to predict but cannot: non-scalar *and* weighted.

    The emulator predicts scalar features, so a ``series`` or ``frequency`` item in the
    cost is a shape mismatch waiting to happen. A **zero-weighted** one is not in the cost
    at all -- ``emulated_feature_labels`` is built from ``const_idx_to_obs_idx`` and
    excludes non-constants already -- so refusing it stops a legitimate and useful thing:
    carrying a recorded trace purely so it can be drawn behind the model.

    Returns ``{obs index: data_type}`` for the ones that really do have to be refused.
    Shared by the trainer and by the use-time check in ``paramID``, which had this rule
    written out twice and drifted: the trainer learned about weights and the other did not.
    """
    unweighted = set()
    for kind in ('series', 'amp', 'phase'):
        weights = obs_info.get('weight_%s_vec' % kind)
        idx_map = obs_info.get('%s_idx_to_obs_idx' % kind)
        if weights is None or idx_map is None:
            continue
        for weight, obs_idx in zip(weights, idx_map):
            if not np.any(np.asarray(weight, dtype=float)):
                unweighted.add(int(obs_idx))

    return {jj: dtype for jj, dtype in enumerate(obs_info['data_types'])
            if dtype != 'constant' and jj not in unweighted}


def fingerprint(param_id_info, obs_info, protocol_info, model_path=None):
    """A stable digest of everything an emulator was trained against.

    Changing a parameter's bounds, adding a data_item, editing an operation, moving a
    sub-experiment or regenerating the model all change what theta -> features *means*. None of
    those changes make the old emulator raise on its own -- it would keep answering, about a
    different model. Comparing this digest is what turns that into an error.

    ``model_path`` is hashed by content when it exists, so a regenerated CellML invalidates the
    emulator even if the inputs that produced it are unchanged.
    """
    payload = {
        'param_labels': [str(x) for x in _param_entry_labels(param_id_info)],
        'param_mins': _as_list(param_id_info.get('param_mins')),
        'param_maxs': _as_list(param_id_info.get('param_maxs')),
        'operands': _jsonable(obs_info.get('operands')),
        'operations': _jsonable(obs_info.get('operations')),
        'operation_kwargs': _jsonable(obs_info.get('operation_kwargs')),
        'data_types': _jsonable(obs_info.get('data_types')),
        'experiment_idxs': _as_list(obs_info.get('experiment_idxs')),
        'subexperiment_idxs': _as_list(obs_info.get('subexperiment_idxs')),
        'pre_times': _jsonable(protocol_info.get('pre_times')),
        'sim_times': _jsonable(protocol_info.get('sim_times')),
        'params_to_change': _jsonable(protocol_info.get('params_to_change')),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    digest = {'inputs_sha256': hashlib.sha256(blob).hexdigest()}
    if model_path and os.path.isfile(model_path):
        with open(model_path, 'rb') as file:
            digest['model_sha256'] = hashlib.sha256(file.read()).hexdigest()
    return digest


def _param_entry_labels(param_id_info):
    # Imported here rather than at module scope: PrimitiveParsers imports a good deal of CA,
    # and this module is imported by the solver-wrapper factory.
    from libcuflynx.parsers.PrimitiveParsers import param_entry_labels
    return param_entry_labels(param_id_info)


def _as_list(values):
    if values is None:
        return None
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class EmulatorQualityError(RuntimeError):
    """The emulator is not accurate enough, or was trained against something else."""


class EmulatorBoundsError(RuntimeError):
    """Evaluation was asked for outside the box the emulator was trained in."""


class EmulatorReuseError(RuntimeError):
    """``reuse_samples`` was asked for with no previous samples that can be refitted."""


class EmulatorBundle:
    """A fitted emulator plus the metadata that makes it checkable.

    Args:
        model: the fitted emulator (an autoemulate ``Emulator``, or anything with a
            ``predict(x)`` returning an array-like of shape ``(n, n_features)``).
        meta: the metadata dict (see ``REQUIRED_META_KEYS``).
        x_train / y_train: the training design and targets, in real units. Kept so the emulator
            can be refitted, extended or audited without re-running the simulator.
        validation: the held-out report from :func:`emulators.emulator_trainer._validation_report`
            -- per-feature statistics plus the points they were computed from.
    """

    def __init__(self, model, meta, x_train=None, y_train=None, validation=None):
        missing = [key for key in REQUIRED_META_KEYS if key not in meta]
        if missing:
            raise ValueError(f'emulator metadata is missing {missing}')
        self.model = model
        self.meta = meta
        self.x_train = None if x_train is None else np.asarray(x_train, dtype=float)
        self.y_train = None if y_train is None else np.asarray(y_train, dtype=float)
        self.validation = validation or {}

        self.param_mins = np.asarray(meta['param_mins'], dtype=float)
        self.param_maxs = np.asarray(meta['param_maxs'], dtype=float)
        self.param_entry_labels = list(meta['param_entry_labels'])
        self.feature_labels = list(meta['feature_labels'])
        self._x_shift = np.asarray(meta['x_scale']['shift'], dtype=float)
        self._x_span = np.asarray(meta['x_scale']['span'], dtype=float)
        self._y_shift = np.asarray(meta['y_scale']['shift'], dtype=float)
        self._y_span = np.asarray(meta['y_scale']['span'], dtype=float)

    # ------------------------------------------------------------------ scaling

    @staticmethod
    def make_scale(values):
        """Shift/span for an affine map onto a well-conditioned range.

        A span of zero (a constant column -- a parameter pinned to one value, or a feature the
        model does not respond to) would divide by zero, so it becomes 1: the column maps to a
        constant, which is exactly what it is.
        """
        values = np.asarray(values, dtype=float)
        shift = np.nanmin(values, axis=0) if values.ndim > 1 else np.nanmin(values)
        span = (np.nanmax(values, axis=0) - shift) if values.ndim > 1 else np.nanmax(values) - shift
        span = np.where(np.asarray(span) > 0, span, 1.0)
        return {'shift': np.atleast_1d(shift).tolist(), 'span': np.atleast_1d(span).tolist()}

    def scale_x(self, theta):
        return (np.asarray(theta, dtype=float) - self._x_shift) / self._x_span

    def unscale_y(self, y_scaled):
        return np.asarray(y_scaled, dtype=float) * self._y_span + self._y_shift

    def scale_y(self, y):
        return (np.asarray(y, dtype=float) - self._y_shift) / self._y_span

    # ------------------------------------------------------------------ checks

    def check_quality(self, min_r2):
        """Refuse an emulator whose worst held-out R2 is below ``min_r2``.

        Named per feature, because "the emulator is bad" is not actionable while "max of
        aortic_root/u has R2 0.42" tells the user which observable to add samples for.
        """
        if min_r2 is None:
            return
        worst_label, worst_r2 = None, np.inf
        for label, r2 in zip(self.feature_labels, self.meta['feature_r2']):
            if r2 is None or not np.isfinite(r2):
                worst_label, worst_r2 = label, float('-inf')
                break
            if r2 < worst_r2:
                worst_label, worst_r2 = label, float(r2)
        if worst_r2 < min_r2:
            raise EmulatorQualityError(
                f'emulator held-out R2 for feature {worst_label!r} is {worst_r2:.4g}, below the '
                f'configured min_r2 of {min_r2}. Increase emulator_settings.num_train_samples, '
                f'widen emulator_settings.models, or lower min_r2 if you accept the error.')

    def check_bounds(self, theta, policy='error'):
        """Apply the out-of-training-box policy, returning the theta to evaluate.

        An emulator is an interpolant; outside its design it is an extrapolation with no
        error estimate at all. Refusing is the default for that reason.
        """
        theta = np.asarray(theta, dtype=float)
        below = theta < self.param_mins
        above = theta > self.param_maxs
        if not (below.any() or above.any()):
            return theta
        offenders = [
            f'{self.param_entry_labels[i]}={theta[i]:.6g} outside '
            f'[{self.param_mins[i]:.6g}, {self.param_maxs[i]:.6g}]'
            for i in range(theta.size) if below[i] or above[i]]
        message = ('emulator evaluated outside the parameter box it was trained in: '
                   + '; '.join(offenders))
        if policy == 'clip':
            return np.clip(theta, self.param_mins, self.param_maxs)
        if policy == 'warn':
            print(f'WARNING: {message}. Predictions there are extrapolation.')
            return theta
        raise EmulatorBoundsError(
            message + '. Retrain over the wider box, or set '
                      'emulator_settings.out_of_bounds to "warn" or "clip".')

    def check_matches(self, live_fingerprint, param_entry_labels=None, feature_labels=None):
        """Refuse a bundle trained against a different model, parameter set or protocol."""
        if param_entry_labels is not None and list(param_entry_labels) != self.param_entry_labels:
            raise EmulatorQualityError(
                f'emulator was trained for parameters {self.param_entry_labels} but this run '
                f'calibrates {list(param_entry_labels)}. Retrain the emulator.')
        if feature_labels is not None and list(feature_labels) != self.feature_labels:
            raise EmulatorQualityError(
                f'emulator was trained for observables {self.feature_labels} but this run uses '
                f'{list(feature_labels)}. Retrain the emulator.')
        stored = self.meta['fingerprint']
        for key, value in live_fingerprint.items():
            if key in stored and stored[key] != value:
                raise EmulatorQualityError(
                    f'emulator is stale: {key} has changed since it was trained '
                    f'({stored[key][:12]}... -> {value[:12]}...). The model, parameter bounds, '
                    f'obs_data operations or protocol differ -- including the run window, '
                    f'since pre_time/sim_time are part of the protocol. Retrain it with '
                    f'do_emulation: true and emulator_settings.reuse_samples: false; the saved '
                    f'samples describe the old setup, so refitting them would answer about a '
                    f'different problem.')

    # ------------------------------------------------------------------ error

    def error_stats(self):
        """Per-feature held-out error, as rows ready to tabulate or plot.

        One row per feature with every statistic the metadata carries, because no
        single one of them is sufficient on its own: R2 can be high while the
        emulator is systematically off (``bias``), RMSE cannot be compared between
        features in different units (``nrmse`` can), and an emulator that is good
        almost everywhere still misleads a calibration that walks through the one
        place it is not (``max_abs_error``).
        """
        rows = []
        for i, label in enumerate(self.feature_labels):
            rows.append({
                'label': label,
                'r2': _stat(self.meta.get('feature_r2'), i),
                'rmse': _stat(self.meta.get('feature_rmse'), i),
                'mae': _stat(self.meta.get('feature_mae'), i),
                'bias': _stat(self.meta.get('feature_bias'), i),
                'max_abs_error': _stat(self.meta.get('feature_max_abs_error'), i),
                'nrmse': _stat(self.meta.get('feature_nrmse'), i),
            })
        return rows

    def error_points(self):
        """The held-out points: ``{theta, y_true, y_pred, residual, feature_labels,
        param_entry_labels}``, all in real units, or ``None`` if none were saved.

        This is what a parity plot (``y_pred`` against ``y_true``) and a residual
        plot (``residual`` against a column of ``theta``) are drawn from. The
        residual is included rather than left to the caller so that every consumer
        agrees on its sign: **prediction minus truth**, so a positive residual
        means the emulator reads high.
        """
        theta = self.validation.get('theta')
        if theta is None or not len(np.asarray(theta)):
            return None
        y_true = np.asarray(self.validation['y_true'], dtype=float)
        y_pred = np.asarray(self.validation['y_pred'], dtype=float)
        return {
            'theta': np.asarray(theta, dtype=float),
            'y_true': y_true,
            'y_pred': y_pred,
            'residual': y_pred - y_true,
            'feature_labels': list(self.feature_labels),
            'param_entry_labels': list(self.param_entry_labels),
        }

    # ------------------------------------------------------------------ predict

    def predict(self, theta, out_of_bounds='error'):
        """Predicted scalar features for one theta (1-D) or many (2-D)."""
        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        rows = theta.reshape(1, -1) if single else theta
        checked = np.vstack([self.check_bounds(row, out_of_bounds) for row in rows])
        y_scaled = self._predict_scaled(self.scale_x(checked))
        features = self.unscale_y(y_scaled)
        return features[0] if single else features

    # ------------------------------------------------------------------ persistence

    def save(self, directory):
        """Write model, metadata and training data to ``directory``. Returns the directory."""
        os.makedirs(directory, exist_ok=True)
        # Recorded in the metadata rather than inferred on the way back in: the
        # two containers are not distinguishable from the bytes, and guessing
        # wrong produces an unpickling error about the *model* rather than about
        # the file. Same reason min_r2 and fd_rel_step travel with the bundle --
        # by load time the caller may be a calibration run that never saw the
        # emulator settings.
        self.meta['model_serialiser'] = _save_model(
            self.model, os.path.join(directory, MODEL_FILE),
            serialiser=(self.meta.get('settings') or {}).get('model_serialiser'))
        with open(os.path.join(directory, METADATA_FILE), 'w') as file:
            json.dump(self.meta, file, indent=2, default=str)
        if self.x_train is not None and self.y_train is not None:
            # Kept so the design can be extended or refitted without paying for the
            # simulations again -- the expensive half of training is the runs, not the fit.
            np.savez(os.path.join(directory, TRAINING_DATA_FILE),
                     x_train=self.x_train, y_train=self.y_train)
        theta = self.validation.get('theta')
        if theta is not None and len(np.asarray(theta)):
            # Written even though the metadata already carries the statistics, and
            # in real units: a parity plot, a residual-against-parameter plot and
            # "which held-out point is worst" all need the points, and a consumer
            # that had to recompute them would need the emulator, the simulator and
            # the split -- i.e. would need to be circulatory_autogen.
            np.savez(
                os.path.join(directory, VALIDATION_FILE),
                theta=np.asarray(self.validation['theta'], dtype=float),
                y_true=np.asarray(self.validation['y_true'], dtype=float),
                y_pred=np.asarray(self.validation['y_pred'], dtype=float),
                feature_labels=np.asarray(self.feature_labels, dtype=object),
                param_entry_labels=np.asarray(self.param_entry_labels, dtype=object),
            )
        return directory

    @classmethod
    def load(cls, directory):
        meta_path = os.path.join(directory, METADATA_FILE)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f'no emulator found in {directory} ({METADATA_FILE} is missing). Train one '
                f'first with do_emulation: true (./run_emulator_training.sh N).')
        with open(meta_path) as file:
            meta = json.load(file)
        model = _load_model(os.path.join(directory, MODEL_FILE),
                            serialiser=meta.get('model_serialiser'))
        x_train = y_train = None
        data_path = os.path.join(directory, TRAINING_DATA_FILE)
        if os.path.isfile(data_path):
            with np.load(data_path) as data:
                x_train, y_train = data['x_train'], data['y_train']
        return cls(model, meta, x_train=x_train, y_train=y_train,
                   validation=_load_validation(directory))

    def _predict_scaled(self, x_scaled):
        """The one place the emulator backend is actually called.

        autoemulate emulators return either a tensor or a torch distribution depending on
        whether they are probabilistic, so both are reduced to a mean array here; a plain
        object with ``predict`` (what the tests use) passes straight through.
        """
        raw = self.model.predict(_as_backend_input(self.model, x_scaled))
        return _as_numpy_mean(raw).reshape(len(x_scaled), -1)


def _stat(values, index):
    """One statistic, as a float, or None when the emulator did not record it.

    None rather than nan: an older bundle simply has no value for a statistic
    added later, and that is different from a value that could not be computed.
    """
    try:
        value = float(values[index])
    except (TypeError, IndexError, ValueError):
        return None
    return value


def _load_validation(directory):
    """The held-out points, or ``{}`` when the bundle predates them / has none."""
    path = os.path.join(directory, VALIDATION_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with np.load(path, allow_pickle=True) as data:
            return {
                'theta': data['theta'],
                'y_true': data['y_true'],
                'y_pred': data['y_pred'],
            }
    except Exception:  # noqa: BLE001 - a damaged extra is not a damaged emulator
        return {}


def _serialiser(name):
    """The named module, or a refusal that says how to get it."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(
            f'the emulator is set to be saved with {name}, which is not installed. '
            f'Install it with `pip install {name}` (it comes with '
            f'`pip install "libcuflynx[emulation]"`), or set '
            f'emulator_settings.model_serialiser to one of '
            f'{", ".join(SERIALISERS)}.') from error


def _dump(module, model, path):
    """joblib takes a path; the pickle-alikes take a file object."""
    if module.__name__ == 'joblib':
        module.dump(model, path)
    else:
        with open(path, 'wb') as file:
            module.dump(model, file)


def _undump(module, path):
    if module.__name__ == 'joblib':
        return module.load(path)
    with open(path, 'rb') as file:
        return module.load(file)


def _save_model(model, path_without_suffix, serialiser=None):
    """Pickle the fitted model. Returns the serialiser that actually wrote it.

    joblib is the default because it is the container autoemulate's own saver
    uses. Some fitted emulators cannot go through it: an object holding an
    uninitialised C-extension descriptor -- ``_abc._abc_data`` is the one seen in
    the wild -- makes ``joblib.dump`` raise ``cannot pickle '_abc._abc_data'
    object``, and the training run dies after paying for every simulation
    (issue #468).

    ``auto`` therefore works down :data:`_AUTO_ORDER` until one succeeds, so that
    failure costs a warning rather than the run. Falling *back* rather than
    switching outright matters: on autoemulate 2.1.2 a torch-backed emulator
    pickles with joblib and fails under dill, so a blanket switch would break the
    common case to fix the rare one.
    """
    name = serialiser or DEFAULT_SERIALISER
    if name not in SERIALISERS:
        raise ValueError(
            f'emulator_settings.model_serialiser is {name!r}; '
            f'expected one of {", ".join(SERIALISERS)}.')
    path = f'{path_without_suffix}.joblib'

    if name != 'auto':
        _dump(_serialiser(name), model, path)
        return name

    first_error = None
    for candidate in _AUTO_ORDER:
        try:
            module = _serialiser(candidate)
        except RuntimeError as error:      # not installed; try the next one
            first_error = first_error or error
            continue
        try:
            _dump(module, model, path)
        except _PICKLING_FAILURES as error:
            first_error = first_error or error
            # A half-written file would be loaded in preference to nothing at all.
            if os.path.isfile(path):
                os.remove(path)
            continue
        if candidate != _AUTO_ORDER[0]:
            # Silently changing container would be worse than failing: the file
            # now needs that library wherever it is read.
            print(f'[emulator] {_AUTO_ORDER[0]} could not pickle this model '
                  f'({first_error}); saved with {candidate} instead. Set '
                  f'emulator_settings.model_serialiser to choose explicitly.')
        return candidate

    raise RuntimeError(
        f'could not save the emulator: none of {", ".join(_AUTO_ORDER)} could '
        f'pickle it. The first failure was: {first_error}. Every training '
        f'simulation has already run -- if any of those libraries is missing, '
        f'install it and set emulator_settings.model_serialiser to it, and the '
        f'design can be reused with reuse_samples: true.')


def _load_model(path_without_suffix, serialiser=None):
    """Read the model back, preferring the serialiser that wrote it.

    The others are still tried, because a bundle written before the choice
    existed records nothing, and because the containers cannot be told apart
    from their bytes.
    """
    path = f'{path_without_suffix}.joblib'
    if not os.path.isfile(path):
        raise FileNotFoundError(f'emulator model file {path} is missing')

    order = list(_AUTO_ORDER)
    if serialiser in order:
        order.remove(serialiser)
        order.insert(0, serialiser)

    first_error = None
    for name in order:
        try:
            module = _serialiser(name)
        except RuntimeError as error:
            first_error = first_error or error
            continue
        try:
            return _undump(module, path)
        except ModuleNotFoundError as error:
            # The common case is a bundle fitted with autoemulate being loaded in an
            # environment that does not have it -- say so, rather than reporting a bare
            # missing module. Not worth trying another container: the module the pickle
            # names is absent whichever reads it.
            raise RuntimeError(
                f'could not load the emulator at {path}: {error}. If it was fitted with '
                f'autoemulate, install it with `pip install "libcuflynx[emulation]"` '
                f'(needs Python >=3.10,<3.13).') from error
        except Exception as error:  # noqa: BLE001 - try the other containers first
            first_error = first_error or error

    raise RuntimeError(
        f'could not load the emulator at {path}: {first_error}. It was saved with '
        f'{serialiser or "an unrecorded serialiser"}, and none of '
        f'{", ".join(_AUTO_ORDER)} could read it back.')


def _as_backend_input(model, x_scaled):
    """torch tensor for a torch-backed emulator, plain array otherwise."""
    if type(model).__module__.startswith('autoemulate'):
        import torch
        return torch.as_tensor(np.asarray(x_scaled, dtype=np.float32))
    return np.asarray(x_scaled, dtype=float)


def _as_numpy_mean(raw):
    """The predicted values, whether the emulator returned a tensor or a distribution.

    A probabilistic emulator (every Gaussian process, i.e. the default) returns a torch
    distribution whose ``.mean`` is a *property* holding the prediction; a deterministic one
    returns a tensor, whose ``.mean`` is a *method*. Distinguishing them by callability rather
    than by type keeps this working for both without importing torch to ask.
    """
    mean = getattr(raw, 'mean', None)
    if mean is not None and not callable(mean):
        raw = mean
    if hasattr(raw, 'detach'):
        raw = raw.detach().cpu().numpy()
    return np.asarray(raw, dtype=float)
