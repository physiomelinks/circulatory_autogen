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

#: Prefix that marks a multi-phase variant: one emulator that treats counts, jumps
#: and smooth observables each in the way their own shape asks for.
MULTI_PHASE_PREFIX = 'multi_phase_'

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
    """``two_phase_MLP`` / ``multi_phase_MLP`` -> ``MLP``; anything else unchanged."""
    for prefix in (TWO_PHASE_PREFIX, MULTI_PHASE_PREFIX):
        if isinstance(name, str) and name.startswith(prefix):
            return name[len(prefix):]
    return name


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



# ================================================================================
# multi-phase: one emulator, three kinds of observable
#
# A study's observables are not all the same shape, and fitting them as though they
# were is what the two-phase emulator was a first answer to. This is the second.
# Three kinds are recognised, each from the training data rather than from a name:
#
#   count   an integer-valued observable -- a spike count in a window. The forward
#           model is deterministic and returns one integer, so what is learned is a
#           *classifier* over the integers it returns, and what is predicted is the
#           expected count under that classifier. Never negative (so a Poisson cost
#           never meets its clip), and smooth in theta because the class
#           probabilities are, so an optimiser still has a gradient to follow.
#
#   jump    a continuous observable that lives on two separated branches -- the peak
#           voltage of a trace that either spikes or does not. A classifier picks the
#           branch and **a regressor of the base family is fitted on each side**, so
#           the quiet branch is predicted rather than pinned to a constant. That is
#           the difference from ``two_phase_``, which substitutes the floor value on
#           the inactive side and so cannot follow it at all.
#
#   smooth  everything else, straight from the base regressor.
#
# One base family per emulator: ``multi_phase_MLP`` uses an MLP for every regressor
# it fits, ``multi_phase_RadialBasisFunctions`` an RBF. Mixing families inside one
# emulator would make the comparison between them meaningless.
# ================================================================================

#: A column is a count when every finite value is a non-negative integer and it takes
#: at least this many distinct values -- one value is a constant, not a count.
COUNT_MIN_CLASSES = 2

#: ...and at most this many. A count with hundreds of levels is a continuous quantity for
#: every practical purpose, and a classifier over it would be all variance.
#:
#: Left at 24 deliberately. Raising it to 64 (so that counts with 26-53 levels became
#: _ExpectedCount models rather than falling through to the regressor) does remove the
#: negative predictions of #498 -- but it roughly doubles the error on exactly those
#: features (RMSE +107% on ox1, +81% on cpvt, measured on 12000-sample bundles) and makes
#: the whole emulator worse: std(dcost) 2.675 -> 3.097 on ox1 and 3.385 -> 4.343 on cpvt.
#: The warning above is correct; a 42-class classifier really is all variance. The negatives
#: are handled by clamping in MultiPhaseEmulator.predict instead, which costs nothing.
COUNT_MAX_CLASSES = 24

#: A column is a jump when the largest gap between consecutive sorted values is at
#: least this share of its full range. Chosen well above the gap a smooth sample
#: produces and well below a true branch separation.
JUMP_GAP_FRAC = 0.20

#: Both sides of a jump need at least this many rows to fit a regressor on.
MIN_SIDE_ROWS = 8


def is_multi_phase(name):
    """Whether ``name`` asks for a multi-phase variant."""
    return isinstance(name, str) and name.startswith(MULTI_PHASE_PREFIX)


def multi_phase_name(name):
    """``MLP`` -> ``multi_phase_MLP``."""
    return name if is_multi_phase(name) else MULTI_PHASE_PREFIX + name


def multi_phase_model_names(base_names):
    """A ``multi_phase_`` variant for each name in ``base_names``."""
    return [multi_phase_name(name) for name in base_names]


def is_count_column(column):
    """Whether a column is an integer count.

    Tested on the **unscaled** values: the trainer maps y onto a well-conditioned
    range before fitting, and an affine map turns integers into a grid that no
    integer test recognises. So the caller has to classify before scaling.
    """
    finite = np.asarray(column, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    if np.any(finite < 0):
        return False
    if not np.allclose(finite, np.round(finite), atol=1e-9, rtol=0):
        return False
    return COUNT_MIN_CLASSES <= len(np.unique(np.round(finite))) <= COUNT_MAX_CLASSES


def is_count_like(column):
    """A non-negative integer quantity, however many levels it happens to show.

    :func:`is_count_column` additionally caps the number of levels, because that is what
    decides whether a *classifier* over them is sensible. This one asks only what the
    quantity is, which is what decides whether a negative prediction is meaningful (#498).
    """
    finite = np.asarray(column, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or np.any(finite < 0):
        return False
    if not np.allclose(finite, np.round(finite), atol=1e-9, rtol=0):
        return False
    return len(np.unique(np.round(finite))) >= COUNT_MIN_CLASSES


def jump_threshold(column, gap_frac=JUMP_GAP_FRAC, min_side=MIN_SIDE_ROWS):
    """Where a column separates into two branches, or ``None`` if it does not.

    The split is the midpoint of the widest gap between consecutive observed values,
    accepted only when that gap is a large share of the column's range and both sides
    carry enough rows to fit on. Deliberately not "the value repeated most often"
    (:func:`floor_mask`): a peak voltage that either spikes or does not has two
    *populations*, not a repeated constant, and the floor test does not see it.
    """
    finite = np.asarray(column, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 * min_side:
        return None
    values = np.unique(finite)
    if values.size < 2:
        return None
    spread = values[-1] - values[0]
    if spread <= 0:
        return None
    gaps = np.diff(values)
    widest = int(np.argmax(gaps))
    if gaps[widest] / spread < gap_frac:
        return None
    threshold = 0.5 * (values[widest] + values[widest + 1])
    low = int(np.sum(finite <= threshold))
    if low < min_side or (finite.size - low) < min_side:
        return None
    return float(threshold)


def classify_features(y):
    """Label every column ``'count'``, ``'floored_count'``, ``'jump'`` or ``'smooth'``.

    Call this on the **unscaled** training targets; see :func:`is_count_column`.

    ``floored_count`` is a count with more levels than a classifier should be asked to
    separate, but which still spends much of the design sitting on one value -- a spike
    count that is zero until the cell starts firing. Before #498 those fell through to the
    plain regressor, which rippled below the floor and returned negative counts: on the
    12000-sample SN_full bundles, 46-49% of predictions for such a column were negative and
    ~100% of those sat where the true count was exactly zero. They get the floor treatment
    instead -- a *binary* classifier, which is cheap and low-variance, plus a regressor for
    the magnitude above it.
    """
    y = np.asarray(y, dtype=float)
    has_floor, _ = floor_mask(y)
    kinds = []
    for index in range(y.shape[1]):
        column = y[:, index]
        if is_count_column(column):
            kinds.append('count')
        elif is_count_like(column) and has_floor[index]:
            kinds.append('floored_count')
        elif jump_threshold(column) is not None:
            kinds.append('jump')
        else:
            kinds.append('smooth')
    return np.asarray(kinds, dtype=object)


class _ExpectedCount:
    """A classifier over the values a count observable takes, read as its mean.

    The forward model is deterministic -- one theta gives one integer -- so the
    distribution here is the emulator's uncertainty about which integer, not
    variability in the model. Reading it as an expectation is a plug-in estimate of
    that integer, and it is the useful one: non-negative by construction, and
    continuous in theta where an argmax would be a staircase with no gradient.

    Where the emulator is unsure the expectation sits between classes, which costs
    slightly *more* than the true integer would under a Poisson likelihood. That bias
    is the safe direction: the alternative failure -- scoring zero cost for a
    confidently wrong silence -- is the one that pulls a chain somewhere the model
    never goes.
    """

    def __init__(self, model, classes):
        self.model = model
        #: The values the observable actually takes, in ascending order. The classifier
        #: is fitted on their *indices*: sklearn refuses float labels as "continuous",
        #: and after the trainer's affine scaling the counts are floats.
        self.classes = np.asarray(classes, dtype=float)

    def predict(self, x):
        if self.model is None:
            return np.full(len(x), float(self.classes[0]))
        probabilities = np.asarray(self.model.predict_proba(x), dtype=float)
        indices = np.asarray(getattr(self.model, 'classes_', range(probabilities.shape[1])),
                             dtype=int)
        return probabilities @ self.classes[indices]


class MultiPhaseEmulator:
    """Counts, jumps and smooth observables, each predicted the way its shape asks.

    Presents ``predict`` and nothing else, which is all ``EmulatorBundle`` asks of a
    model, and pickles through joblib like autoemulate's own.
    """

    def __init__(self, base_model, kinds, count_models, jump_groups, base_name,
                 count_columns=None, floor_groups=None):
        self.base_model = base_model
        self.kinds = np.asarray(kinds, dtype=object)
        #: Every column that is a count, including any the classifier declined to model.
        self._count_columns = list(count_columns) if count_columns is not None \
            else list(count_models)
        #: column index -> _ExpectedCount
        self.count_models = dict(count_models)
        #: list of {'columns', 'classifier', 'low', 'high'}
        self.jump_groups = list(jump_groups)
        #: list of {'columns', 'classifier', 'floor_value', 'magnitude'} -- counts whose
        #: level count is past what a classifier should separate, but which still have a
        #: floor to put exactly where it belongs rather than smoothing through it (#498).
        self._floor_groups = list(floor_groups) if floor_groups is not None else []
        self.base_name = base_name

    @property
    def model_name(self):
        return multi_phase_name(self.base_name)

    @property
    def floor_groups(self):
        """Floor groups, empty for a bundle pickled before #498."""
        return getattr(self, '_floor_groups', []) or []

    @property
    def count_columns(self):
        """Count columns, falling back to the modelled ones.

        A property rather than a plain attribute because joblib restores ``__dict__``
        directly and never calls ``__init__``: a bundle pickled before #498 has no such
        attribute at all, and unpickling one must not raise.
        """
        return getattr(self, '_count_columns', None) or list(self.count_models)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out = _as_array(self.base_model.predict(_backend_input(self.base_model, x)), len(x))

        for column, model in self.count_models.items():
            if column < out.shape[1]:
                out[:, column] = model.predict(x)

        # A count is non-negative whatever produced it. _ExpectedCount is non-negative by
        # construction, but a column that failed is_count_column -- too many levels, or an
        # integrality test the design broke -- is left on the plain regressor, which is not.
        # Clamping here covers every path rather than only the classified one (#498).
        if self.count_columns:
            idx = [c for c in self.count_columns if c < out.shape[1]]
            if idx:
                out[:, idx] = np.clip(out[:, idx], 0.0, None)

        for group in self.floor_groups:
            active = np.asarray(group['classifier'].predict(x)).astype(bool).reshape(len(x))
            values = None
            if group['magnitude'] is not None and active.any():
                values = _as_array(
                    group['magnitude'].predict(_backend_input(group['magnitude'], x[active])),
                    int(active.sum()))
            for column in group['columns']:
                if column >= out.shape[1]:
                    continue
                out[~active, column] = group['floor_value'][column]
                if values is not None and column < values.shape[1]:
                    out[active, column] = values[:, column]

        for group in self.jump_groups:
            side = np.asarray(group['classifier'].predict(x)).astype(bool).reshape(len(x))
            for which, model in (('low', group['low']), ('high', group['high'])):
                if model is None:
                    continue
                rows = ~side if which == 'low' else side
                if not rows.any():
                    continue
                values = _as_array(model.predict(_backend_input(model, x[rows])),
                                   int(rows.sum()))
                for column in group['columns']:
                    if column < out.shape[1] and column < values.shape[1]:
                        out[rows, column] = values[:, column]
        return out


def fit_multi_phase(x_train, y_train, x_test, y_test, base_name, autoemulate_cls,
                    fit_kwargs, kinds):
    """Fit the base regressor, the count classifiers and the per-branch regressors.

    ``kinds`` comes from :func:`classify_features` on the *unscaled* targets; ``y_train``
    here is scaled, as everything the trainer fits is. That is fine for both extra
    stages: a classifier only needs the classes to be distinct, and an expectation
    commutes with the affine scaling, so the mean of the scaled classes is the scaled
    mean of the counts.

    Jump columns are grouped by the rows they put on each side, and one pair of
    regressors is fitted per group. Columns that jump together -- which is what
    observables of one trace do -- then cost one pair between them rather than one
    pair each.
    """
    y_train = np.asarray(y_train, dtype=float)
    kinds = np.asarray(kinds, dtype=object)

    kwargs = dict(fit_kwargs)
    kwargs['models'] = [base_name]
    base = autoemulate_cls(x_train, y_train, test_data=(x_test, y_test), **kwargs)
    base_result = base.best_result()

    count_columns = [int(c) for c in np.flatnonzero(kinds == 'count')]
    count_models = {}
    for column in count_columns:
        count_models[column] = _fit_expected_count(x_train, y_train[:, column])

    jump_groups = _fit_jump_groups(x_train, y_train, y_test, x_test, kinds,
                                   base_name, autoemulate_cls, fit_kwargs,
                                   base_result.model)

    floor_groups = _fit_floor_groups(x_train, y_train, y_test, x_test, kinds,
                                     base_name, autoemulate_cls, fit_kwargs,
                                     base_result.model)
    count_columns += [int(c) for c in np.flatnonzero(kinds == 'floored_count')]

    return (MultiPhaseEmulator(base_result.model, kinds, count_models, jump_groups,
                               base_name, count_columns=count_columns,
                               floor_groups=floor_groups),
            base_result)


def _fit_expected_count(x, column):
    """One multiclass classifier over the values this count takes."""
    from sklearn.ensemble import GradientBoostingClassifier

    values = np.unique(column[np.isfinite(column)])
    if values.size < 2:
        return _ExpectedCount(None, values if values.size else np.array([0.0]))
    # Fitted on indices into `values`, not on the values themselves -- see _ExpectedCount.
    labels = np.searchsorted(values, column)
    try:
        model = GradientBoostingClassifier(random_state=0)
        model.fit(x, labels)
    except Exception as error:  # noqa: BLE001 - one column must not end the run
        print(f'[emulator] a count classifier could not be fitted '
              f'({type(error).__name__}: {error}); predicting its mean instead')
        return _ExpectedCount(None, np.array([float(np.mean(column))]))
    return _ExpectedCount(model, values)


def _fit_floor_groups(x_train, y_train, y_test, x_test, kinds, base_name,
                      autoemulate_cls, fit_kwargs, fallback_model):
    """On-floor/active classifier plus a magnitude regressor, for each floored count.

    The same two-stage shape :class:`TwoPhaseEmulator` uses, applied inside multi_phase to
    the counts that have too many levels for :class:`_ExpectedCount`. The classifier is
    binary -- which is the whole point, since a classifier over their 26-53 levels is all
    variance and measurably worse than a regressor (#498).

    The floor is recomputed on the *scaled* targets, as the jump split is: an affine map
    moves the value but not which rows sit on it.
    """
    columns = [int(c) for c in np.flatnonzero(kinds == 'floored_count')]
    if not columns:
        return []

    has_floor, floor_value = floor_mask(y_train)
    from sklearn.ensemble import GradientBoostingClassifier

    groups = {}
    for column in columns:
        if not has_floor[column]:
            continue
        active = y_train[:, column] != floor_value[column]
        groups.setdefault(active.tobytes(), []).append(column)

    fitted = []
    for members in groups.values():
        active = y_train[:, members[0]] != floor_value[members[0]]
        if int(active.sum()) < MIN_ACTIVE_ROWS:
            continue
        classifier = _fit_column(GradientBoostingClassifier, x_train, active)
        magnitude = _fit_side(x_train, y_train, x_test, y_test, active, base_name,
                              autoemulate_cls, fit_kwargs, fallback_model)
        fitted.append({'columns': members,
                       'classifier': _SingleOutputBinary(classifier),
                       'floor_value': {c: float(floor_value[c]) for c in members},
                       'magnitude': magnitude})
    return fitted


def _fit_jump_groups(x_train, y_train, y_test, x_test, kinds, base_name,
                     autoemulate_cls, fit_kwargs, fallback_model):
    """A classifier and a regressor per side, for each set of columns that jump together."""
    columns = [int(c) for c in np.flatnonzero(kinds == 'jump')]
    if not columns:
        return []

    # Split each jump column on the *scaled* values: the threshold is recomputed here
    # rather than carried from the unscaled pass, because an affine map moves it.
    sides = {}
    for column in columns:
        threshold = jump_threshold(y_train[:, column])
        if threshold is None:
            continue
        sides[column] = y_train[:, column] > threshold

    groups = {}
    for column, mask in sides.items():
        groups.setdefault(mask.tobytes(), []).append(column)

    from sklearn.ensemble import GradientBoostingClassifier

    fitted = []
    for members in groups.values():
        mask = sides[members[0]]
        classifier = _fit_column(GradientBoostingClassifier, x_train, mask)
        low = _fit_side(x_train, y_train, x_test, y_test, ~mask, base_name,
                        autoemulate_cls, fit_kwargs, fallback_model)
        high = _fit_side(x_train, y_train, x_test, y_test, mask, base_name,
                         autoemulate_cls, fit_kwargs, fallback_model)
        fitted.append({'columns': members, 'classifier': _SingleOutputBinary(classifier),
                       'low': low, 'high': high})
    return fitted


def _fit_side(x_train, y_train, x_test, y_test, rows, base_name, autoemulate_cls,
              fit_kwargs, fallback_model):
    """A regressor of the base family fitted on one side of a jump.

    Falls back to the model fitted on everything when a side cannot be fitted -- a
    worse answer for those rows than a dedicated regressor, and a much better one than
    the constant ``two_phase_`` would put there.

    Two ways a side fails. It can be too thin: autoemulate cross-validates with
    ``n_splits`` folds, so a side needs enough rows to *fold*, not merely enough to
    fit, and eight rows across five folds leaves one point per fold. And it can fail
    anyway -- every candidate erroring leaves autoemulate with no results at all and
    ``best_result()`` raising, which would take a whole training run down over one
    branch of one observable.
    """
    n_splits = int(fit_kwargs.get('n_splits', 5) or 5)
    needed = max(MIN_SIDE_ROWS, 2 * n_splits)
    available = int(np.sum(rows))
    if available < needed:
        print(f'[emulator] a jump side has {available} row(s), fewer than the {needed} '
              f'a {n_splits}-fold fit needs; using the all-rows regressor there')
        return fallback_model
    kwargs = dict(fit_kwargs)
    kwargs['models'] = [base_name]
    try:
        side = autoemulate_cls(x_train[rows], y_train[rows], test_data=(x_test, y_test),
                               **kwargs)
        return side.best_result().model
    except Exception as error:  # noqa: BLE001 - any fit failure degrades, never fatal
        print(f'[emulator] a jump side ({available} rows) could not be fitted '
              f'({type(error).__name__}: {error}); using the all-rows regressor there')
        return fallback_model


class _SingleOutputBinary:
    """One binary classifier, answered as a flat boolean vector."""

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return np.asarray(self.model.predict(x)).astype(bool).reshape(len(x))


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
