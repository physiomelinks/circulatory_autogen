"""Emulators CA builds itself, wrapping autoemulate rather than replacing it.

autoemulate's emulators are regressors, and a regressor is the wrong shape for an
observable that spends most of its range pinned at one value. A spike count is the
example that motivated this: below rheobase it is exactly zero, above it jumps to
one spike per window, and there is no value in between the model can produce. Fit a
smooth regressor to that and it splits the difference -- predicting a fraction of a
spike across the floor, and undershooting everywhere above it.

The response surface is a floor, a cliff and a staircase, which the cardiac
electrophysiology literature calls a bifurcation: the model changes behaviour as a
parameter crosses a threshold, and a stationary kernel has to pick one length scale
for both sides of it. The standard answer is two stages -- learn *where* the
boundary is with a classifier, and learn the magnitude with a regressor fitted only
on the far side, so the floor cannot drag it down.

``TwoPhaseEmulator`` does that and keeps autoemulate for both regression halves, so
every setting that works for a plain run works here. The variants are exposed as
``two_phase_<name>`` for each autoemulate emulator -- ``two_phase_MLP``,
``two_phase_RandomForest`` and so on -- and are **deliberately absent from the
``all`` set**: two stages cost more to fit than one and are only worth it when the
features really do have a floor, so they are opt-in by name.

    emulator_settings:
      models: two_phase_MLP

Nothing about spiking is hardcoded. A "floor" is whatever single value a feature
returns unusually often, found from the training data.
"""
import numpy as np

#: Prefix that marks a two-phase variant of an autoemulate emulator.
TWO_PHASE_PREFIX = 'two_phase_'

#: A feature is treated as having a floor when at least this share of the design
#: returns one single value. Well below the 82% seen on real spike-frequency
#: features, and well above the repetition a genuinely continuous feature shows.
DEFAULT_FLOOR_SHARE = 0.25

#: Below this many off-floor points there is nothing to fit a second regressor on,
#: so the feature falls back to the single-stage path.
MIN_ACTIVE_ROWS = 8


def is_two_phase(name):
    """Whether ``name`` asks for a two-phase variant."""
    return isinstance(name, str) and name.startswith(TWO_PHASE_PREFIX)


def base_emulator_name(name):
    """``two_phase_MLP`` -> ``MLP``. Returns ``name`` unchanged if not two-phase."""
    return name[len(TWO_PHASE_PREFIX):] if is_two_phase(name) else name


def two_phase_name(name):
    """``MLP`` -> ``two_phase_MLP``."""
    return name if is_two_phase(name) else TWO_PHASE_PREFIX + name


def two_phase_model_names(base_names):
    """A ``two_phase_`` variant for each name in ``base_names``.

    Built from whatever autoemulate advertises rather than a list kept here, so a
    new emulator on their side gets a two-phase variant without a change on ours.
    """
    return [two_phase_name(name) for name in base_names]


def floor_mask(y, floor_share=DEFAULT_FLOOR_SHARE):
    """Which features have a floor, and what value it is.

    Returns ``(has_floor, floor_value)``, both length ``n_features``. A feature has
    a floor when one value accounts for at least ``floor_share`` of the design --
    the shape a threshold produces, and not one a continuous response takes.
    """
    y = np.asarray(y, dtype=float)
    n_rows, n_features = y.shape
    has_floor = np.zeros(n_features, dtype=bool)
    floor_value = np.zeros(n_features, dtype=float)
    for index in range(n_features):
        column = y[:, index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        values, counts = np.unique(np.round(finite, 12), return_counts=True)
        best = int(np.argmax(counts))
        if counts[best] / float(finite.size) >= floor_share:
            has_floor[index] = True
            floor_value[index] = float(values[best])
    return has_floor, floor_value


class TwoPhaseEmulator:
    """Classify which side of the cliff, then regress the magnitude.

    Fitted from :func:`fit_two_phase`, which owns the autoemulate calls. Predicting
    is the interesting half and lives here: for a feature with a floor, the
    classifier decides floor-or-not and the active regressor supplies the value
    when not; every other feature comes straight from the base regressor.

    Presents ``predict`` and nothing else, which is all ``EmulatorBundle`` asks of a
    model, and pickles through joblib like autoemulate's own.
    """

    def __init__(self, base_model, active_model, classifier, has_floor, floor_value,
                 base_name):
        self.base_model = base_model
        self.active_model = active_model
        self.classifier = classifier
        self.has_floor = np.asarray(has_floor, dtype=bool)
        self.floor_value = np.asarray(floor_value, dtype=float)
        self.base_name = base_name

    @property
    def model_name(self):
        return two_phase_name(self.base_name)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out = _as_array(self.base_model.predict(_backend_input(self.base_model, x)),
                        len(x))

        if not self.has_floor.any() or self.classifier is None:
            return out

        active = np.asarray(self.classifier.predict(x))
        if active.ndim == 1:
            active = active.reshape(-1, 1)
        active = active.astype(bool)

        if self.active_model is not None:
            magnitudes = _as_array(
                self.active_model.predict(_backend_input(self.active_model, x)), len(x))
        else:
            magnitudes = out

        floor_indices = np.flatnonzero(self.has_floor)
        for column, index in enumerate(floor_indices):
            if column >= active.shape[1]:
                break
            on_floor = ~active[:, column]
            out[on_floor, index] = self.floor_value[index]
            out[~on_floor, index] = magnitudes[~on_floor, index]
        return out


def fit_two_phase(x_train, y_train, x_test, y_test, base_name, autoemulate_cls,
                  fit_kwargs, floor_share=DEFAULT_FLOOR_SHARE):
    """Fit the classifier and both regressors. Returns ``(model, base_result_name)``.

    Two autoemulate runs rather than one per feature: the second is fitted on the
    rows where the floored features are active, which is the subset that stops the
    floor dragging the magnitude down. Per-feature runs would be the more faithful
    reading of the method and cost one full comparison per observable -- 84 of them
    on the study this was written for.
    """
    y_train = np.asarray(y_train, dtype=float)
    has_floor, floor_value = floor_mask(y_train, floor_share)

    kwargs = dict(fit_kwargs)
    kwargs['models'] = [base_name]
    base = autoemulate_cls(x_train, y_train, test_data=(x_test, y_test), **kwargs)
    base_result = base.best_result()

    if not has_floor.any():
        # Nothing to gate. Returned as a plain two-phase object with no classifier
        # so the bundle records what was asked for rather than silently degrading.
        return (TwoPhaseEmulator(base_result.model, None, None, has_floor, floor_value,
                                 base_name),
                base_result)

    floor_indices = np.flatnonzero(has_floor)
    labels = np.column_stack([
        ~np.isclose(y_train[:, index], floor_value[index]) for index in floor_indices])

    classifier = _fit_classifier(x_train, labels)

    active_rows = labels.any(axis=1)
    active_model = None
    if int(active_rows.sum()) >= MIN_ACTIVE_ROWS and not active_rows.all():
        active_kwargs = dict(fit_kwargs)
        active_kwargs['models'] = [base_name]
        active = autoemulate_cls(x_train[active_rows], y_train[active_rows],
                                 test_data=(x_test, y_test), **active_kwargs)
        active_model = active.best_result().model

    return (TwoPhaseEmulator(base_result.model, active_model, classifier,
                             has_floor, floor_value, base_name),
            base_result)


def _fit_classifier(x, labels):
    """The boundary. One gradient-boosted classifier per floored feature.

    sklearn rather than autoemulate: autoemulate has no classifier at all, and this
    is the half of the method it does not cover. A column whose label never changes
    carries no boundary to learn, so it gets a constant instead of a fit that would
    raise on a single class.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    return _MultiOutputBinary([
        _fit_column(GradientBoostingClassifier, x, labels[:, column])
        for column in range(labels.shape[1])])


def _fit_column(cls, x, column):
    if column.all() or not column.any():
        return _Constant(bool(column.all()))
    model = cls(random_state=0)
    model.fit(x, column.astype(int))
    return model


class _Constant:
    """A column that is always active, or never. Nothing to learn."""

    def __init__(self, value):
        self.value = bool(value)

    def predict(self, x):
        return np.full(len(x), self.value, dtype=bool)


class _MultiOutputBinary:
    """One binary classifier per floored feature, answered together."""

    def __init__(self, models):
        self.models = list(models)

    def predict(self, x):
        if not self.models:
            return np.zeros((len(x), 0), dtype=bool)
        return np.column_stack([
            np.asarray(model.predict(x)).astype(bool) for model in self.models])


def _backend_input(model, x):
    """autoemulate emulators may want a tensor; a plain object takes the array."""
    try:
        from libcuflynx.emulators.emulator_bundle import _as_backend_input
        return _as_backend_input(model, x)
    except Exception:  # noqa: BLE001 - a plain predictor takes the array as-is
        return x


def _as_array(raw, n_rows):
    try:
        from libcuflynx.emulators.emulator_bundle import _as_numpy_mean
        values = _as_numpy_mean(raw)
    except Exception:  # noqa: BLE001
        values = np.asarray(raw, dtype=float)
    return np.asarray(values, dtype=float).reshape(n_rows, -1)


# --------------------------------------------------------------------------------
# ``two_phase_<name>`` helpers
#
# One function per autoemulate emulator, so a caller can reach a variant by name
# without knowing the prefix convention. Generated at import from whatever
# autoemulate advertises; absent, and the module still imports and the names above
# still work, because everything they need is the string.
# --------------------------------------------------------------------------------

def _make_helper(base):
    def helper():
        """Return the ``models`` value that asks for the two-phase variant."""
        return two_phase_name(base)
    helper.__name__ = two_phase_name(base)
    helper.__qualname__ = helper.__name__
    helper.__doc__ = (
        'The ``emulator_settings.models`` value for a two-phase %s: a classifier for '
        'the floor and %s for the magnitude either side of it.' % (base, base))
    return helper


def _install_helpers():
    """Define ``two_phase_<name>`` for every emulator autoemulate offers."""
    try:
        from libcuflynx.emulators.emulator_trainer import emulator_model_names
        names = [name for name in emulator_model_names() if not is_two_phase(name)]
    except Exception:  # noqa: BLE001 - no autoemulate, no helpers, still importable
        names = []
    installed = []
    for base in names:
        helper = _make_helper(base)
        globals()[helper.__name__] = helper
        installed.append(helper.__name__)
    return installed


AVAILABLE_TWO_PHASE = _install_helpers()
