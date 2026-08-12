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

import numpy as np

METADATA_FILE = 'emulator_metadata.json'
MODEL_FILE = 'emulator'          # autoemulate's saver appends .joblib
TRAINING_DATA_FILE = 'training_data.npz'

#: metadata keys that must be present for a bundle to be usable at all.
REQUIRED_META_KEYS = ('param_entry_labels', 'param_mins', 'param_maxs', 'feature_labels',
                      'feature_r2', 'x_scale', 'y_scale', 'fingerprint')


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
    from parsers.PrimitiveParsers import param_entry_labels
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


class EmulatorBundle:
    """A fitted emulator plus the metadata that makes it checkable.

    Args:
        model: the fitted emulator (an autoemulate ``Emulator``, or anything with a
            ``predict(x)`` returning an array-like of shape ``(n, n_features)``).
        meta: the metadata dict (see ``REQUIRED_META_KEYS``).
        x_train / y_train: the training design and targets, in real units. Kept so the emulator
            can be refitted, extended or audited without re-running the simulator.
    """

    def __init__(self, model, meta, x_train=None, y_train=None):
        missing = [key for key in REQUIRED_META_KEYS if key not in meta]
        if missing:
            raise ValueError(f'emulator metadata is missing {missing}')
        self.model = model
        self.meta = meta
        self.x_train = None if x_train is None else np.asarray(x_train, dtype=float)
        self.y_train = None if y_train is None else np.asarray(y_train, dtype=float)

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
                    f'obs_data operations or protocol differ. Retrain the emulator, or set '
                    f'emulator_settings.retrain_if_stale to true.')

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
        _save_model(self.model, os.path.join(directory, MODEL_FILE))
        with open(os.path.join(directory, METADATA_FILE), 'w') as file:
            json.dump(self.meta, file, indent=2, default=str)
        if self.x_train is not None and self.y_train is not None:
            # Kept so the design can be extended or refitted without paying for the
            # simulations again -- the expensive half of training is the runs, not the fit.
            np.savez(os.path.join(directory, TRAINING_DATA_FILE),
                     x_train=self.x_train, y_train=self.y_train)
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
        model = _load_model(os.path.join(directory, MODEL_FILE))
        x_train = y_train = None
        data_path = os.path.join(directory, TRAINING_DATA_FILE)
        if os.path.isfile(data_path):
            with np.load(data_path) as data:
                x_train, y_train = data['x_train'], data['y_train']
        return cls(model, meta, x_train=x_train, y_train=y_train)

    def _predict_scaled(self, x_scaled):
        """The one place the emulator backend is actually called.

        autoemulate emulators return either a tensor or a torch distribution depending on
        whether they are probabilistic, so both are reduced to a mean array here; a plain
        object with ``predict`` (what the tests use) passes straight through.
        """
        raw = self.model.predict(_as_backend_input(self.model, x_scaled))
        return _as_numpy_mean(raw).reshape(len(x_scaled), -1)


def _save_model(model, path_without_suffix):
    """joblib, the same container autoemulate's own serialiser uses."""
    import joblib
    joblib.dump(model, f'{path_without_suffix}.joblib')


def _load_model(path_without_suffix):
    import joblib
    path = f'{path_without_suffix}.joblib'
    if not os.path.isfile(path):
        raise FileNotFoundError(f'emulator model file {path} is missing')
    try:
        return joblib.load(path)
    except ModuleNotFoundError as error:
        # The common case is a bundle fitted with autoemulate being loaded in an environment
        # that does not have it -- say so, rather than reporting a bare missing module.
        raise RuntimeError(
            f'could not load the emulator at {path}: {error}. If it was fitted with '
            f'autoemulate, install it with `pip install "circulatory_autogen[emulation]"` '
            f'(needs Python >=3.10,<3.13).') from error


def _as_backend_input(model, x_scaled):
    """torch tensor for a torch-backed emulator, plain array otherwise."""
    if type(model).__module__.startswith('autoemulate'):
        import torch
        return torch.as_tensor(np.asarray(x_scaled, dtype=np.float32))
    return np.asarray(x_scaled, dtype=float)


def _as_numpy_mean(raw):
    if hasattr(raw, 'mean') and not isinstance(raw, np.ndarray):
        raw = raw.mean                     # a torch distribution (GaussianLike)
    if hasattr(raw, 'detach'):
        raw = raw.detach().cpu().numpy()
    return np.asarray(raw, dtype=float)
