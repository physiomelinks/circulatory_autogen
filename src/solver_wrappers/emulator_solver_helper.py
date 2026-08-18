"""A `SimulationHelper` that answers from a trained emulator instead of integrating (issue #333).

Everything downstream of a simulation -- the cost, Sobol sampling, MCMC, the Laplace Hessian --
reaches the model through the helper interface, so putting the emulator behind that interface
is what lets all of them use it without any of them knowing. The helper is constructed by the
same factory as every other backend (``get_simulation_helper(..., use_emulator=True)``), so an
emulator can be used wherever a solver is.

What it emulates is the *scalar feature* of each obs_data data_item -- the value after the
``operation`` reduction -- not the trace it was reduced from. That is why ``emulates_features``
is True: the two places that would otherwise apply the operation read it and skip that step,
because applying ``max_minus_min`` to an already-reduced scalar would quietly return zero.
Anything that genuinely needs a trace (plotting, saving all outputs, prediction items) raises
rather than inventing one.

Time is still tracked exactly as the real backends track it. The protocol executor accumulates
``tSim`` across sub-experiments whatever ran underneath, so the bookkeeping has to be right even
though nothing is integrated.
"""
import numpy as np

from emulators.emulator_bundle import EmulatorBundle
from solver_wrappers.param_grouping import pair_names_with_values

TRACE_REFUSAL = (
    'needs the full simulated trace, which a feature emulator cannot produce -- it predicts '
    'the scalar data_item features only. Re-run with use_emulator: false to plot or save '
    'simulated outputs.')


class SimulationHelper:
    """Emulator-backed drop-in for the solver helpers.

    Args:
        emulator_dir: directory holding the trained bundle (``emulator_metadata.json`` etc).
        dt: output sampling step, kept only so the time bookkeeping matches the real backends.
        sim_time: logged duration.
        solver_info: accepted and ignored; there is no integrator to configure.
        pre_time: unlogged spin-up; absorbed into the emulator at training time.
        bundle: an already-loaded bundle, used instead of reading ``emulator_dir`` (tests, and
            callers that have validated one already).
        out_of_bounds: 'error' | 'warn' | 'clip' -- what to do off the training box.
    """

    emulates_features = True

    def __init__(self, emulator_dir, dt=0.01, sim_time=1.0, solver_info=None, pre_time=0.0,
                 bundle=None, out_of_bounds=None):
        self.emulator_dir = emulator_dir
        self.solver_info = solver_info or {}
        # Loaded eagerly, on every rank. MCMC's worker ranks block inside the pool and never
        # reach a lazy first call, so a rank-0-only load would leave them without an emulator.
        self.bundle = bundle if bundle is not None else EmulatorBundle.load(emulator_dir)
        # None means "whatever this emulator was trained under". The caller here is often a
        # calibration/SA/UQ run whose own settings say nothing about emulation, so defaulting
        # to 'error' at this seam would override a user's emulator_settings.out_of_bounds
        # without ever having read it. An explicit value still wins.
        if out_of_bounds is None:
            trained_with = (self.bundle.meta or {}).get('settings') or {}
            out_of_bounds = trained_with.get('out_of_bounds', 'error')
        self.out_of_bounds = out_of_bounds
        _limit_torch_threads()

        self.dt = dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.n_steps = 1
        self.pre_steps = 0
        self.stop_time = (pre_time or 0.0) + (sim_time or 0.0)
        self.tSim = np.linspace(pre_time or 0.0, self.stop_time, 2)

        self.param_names = self.bundle.meta.get('param_names') or []
        self.param_defaults = self.bundle.meta.get('param_defaults') or {}
        self.num_params = len(self.bundle.param_entry_labels)
        self._has_modifiers = bool(self.bundle.meta.get('has_modifiers'))
        self._theta = None
        self._theta_is_authoritative = False
        self._features = None
        self._protocol_info = None
        self._obs_map = None
        self._num_obs = None

    # ------------------------------------------------------------------ inputs

    def set_theta(self, theta):
        """Set the calibration vector directly, in ``params_for_id`` entry order.

        This, not ``set_param_vals``, is the emulator's real input. A modifier entry occupies
        one slot in theta but names several model parameters, and by the time the executor calls
        ``set_param_vals`` those slots have already been expanded to per-parameter values --
        which is the wrong thing to feed a surrogate trained on theta. CA's call sites set theta
        here before the protocol runs, so no inversion is ever needed.
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.num_params:
            raise ValueError(
                f'the emulator was trained on {self.num_params} parameters '
                f'({self.bundle.param_entry_labels}) but was given {theta.size} values.')
        if self._theta is None or not np.array_equal(theta, self._theta):
            self._features = None
        self._theta = theta
        # From here on the executor's set_param_vals for the calibrated set is redundant, and
        # for a modifier entry it is actively wrong (it carries theta * baseline, not theta).
        self._theta_is_authoritative = True

    def set_param_vals(self, param_names, param_vals, change_states=True):
        """Accept the executor's per-parameter values.

        With no modifiers in play these are theta itself, entry for entry, so they are recorded
        as such and the helper works for a caller that only knows the ordinary interface. With
        modifiers they are the expanded per-target values, which theta cannot be recovered from
        here -- ``set_theta`` must have been called, and this is then a consistency check.
        """
        names = list(param_names)
        values = list(param_vals)
        if len(names) != len(values):
            raise ValueError(f'{len(names)} parameter name entries against {len(values)} values')

        if self._is_theta_entry_list(names):
            if self._theta_is_authoritative:
                # theta came from set_theta for this evaluation; these are its expansion.
                return
            if self._has_modifiers:
                raise RuntimeError(
                    'this emulator has modifier parameters, whose theta cannot be recovered '
                    'from expanded parameter values (a modifier slot expands to theta * '
                    'baseline per target). Call set_theta(theta) first.')
            theta = [_single_value(entry_names, value)
                     for entry_names, value in zip(names, values)]
            self.set_theta(theta)
            self._theta_is_authoritative = False
            return

        # Not the calibrated set: the protocol's params_to_change. Those were fixed while the
        # emulator was trained (the protocol fingerprint enforces it), so there is nothing to
        # apply -- but a name that belongs to neither set means the caller thinks it is
        # changing something the emulator cannot see, which must not pass silently.
        known_protocol = set(map(str, ((self._protocol_info or {}).get('params_to_change') or {})))
        unknown = [str(name)
                   for entry, value in zip(names, values)
                   for name, _ in pair_names_with_values(entry, value)
                   if str(name) not in self.param_defaults and str(name) not in known_protocol]
        if unknown:
            raise ValueError(
                f'the emulator cannot change {sorted(set(map(str, unknown)))}: it was trained '
                f'over the params_for_id parameters only, with everything else held at the '
                f'values it was trained with.')

    def _is_theta_entry_list(self, names):
        """True when ``names`` is the calibrated parameter set the emulator was trained on."""
        return _jsonable_names(names) == self.param_names

    # ------------------------------------------------------------------ running

    def run(self):
        """Predict the feature vector for the current theta. Returns success, like a solver."""
        if self._theta is None:
            raise RuntimeError('no parameters have been set on the emulator helper; call '
                               'set_theta(theta) or set_param_vals(...) before run().')
        if self._features is None:
            self._features = self.bundle.predict(self._theta, out_of_bounds=self.out_of_bounds)
        return True

    def update_times(self, dt, start_time, sim_time, pre_time):
        """Time bookkeeping only -- nothing is integrated, but ``tSim`` must still be right.

        The protocol executor concatenates each sub-experiment's ``tSim`` onto a cumulative
        time vector and drops the duplicated first sample, so a missing or single-point axis
        would break the caller rather than the emulator. Mirrors the opencor backend.
        """
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.pre_steps = int(pre_time / dt) if dt else 0
        self.n_steps = max(int(sim_time / dt) if dt else 1, 1)
        self.stop_time = start_time + pre_time + self.n_steps * dt
        self.tSim = np.linspace(start_time + pre_time, self.stop_time, self.n_steps + 1)

    def get_time(self, include_pre_time=False):
        return self.tSim if include_pre_time else self.tSim - self.pre_time

    # ------------------------------------------------------------------ outputs

    def set_obs_map(self, const_idx_to_obs_idx, num_obs=None):
        """Tell the helper which data_item each trained feature belongs to.

        The emulator's outputs are ordered by ``obs_info['const_idx_to_obs_idx']`` while the
        consumers index by data_item. Rather than assume the two coincide, the caller states
        the mapping once at setup; without it they are taken to be the same, which is true
        whenever every data_item is a scalar one -- the only case the emulator supports.
        """
        self._obs_map = [int(i) for i in const_idx_to_obs_idx]
        self._num_obs = int(num_obs) if num_obs is not None else (max(self._obs_map) + 1)

    def get_predicted_features(self):
        """The predicted scalar feature per data_item, ``nan`` for any the emulator misses.

        Indexed by data_item so both reduction sites can read it positionally.
        """
        if self._features is None:
            self.run()
        mapping = self._obs_map if self._obs_map is not None else list(range(len(self._features)))
        n_items = self._num_obs if self._num_obs is not None else len(mapping)
        by_item = np.full(max(n_items, len(mapping)), np.nan)
        for k, obs_idx in enumerate(mapping):
            by_item[obs_idx] = self._features[k]
        return by_item

    def get_results(self, variables_list_of_lists, flatten=False):
        """The predicted features, in the shape the executor expects from a solver.

        Each operand slot holds a length-1 array carrying its data_item's predicted feature.
        The consumers that know about ``emulates_features`` read the value and skip the
        operation; anything else applying ``mean``/``max``/``min`` to it gets the same number
        back, which is the least surprising thing an unaware caller could receive.
        """
        if self._features is None:
            self.run()
        by_item = self.get_predicted_features()
        if self._num_obs is not None and len(variables_list_of_lists) != self._num_obs:
            # The only list the emulator can answer is the obs_data operands, one entry per
            # data_item. A different length means something else was asked for -- prediction
            # variables, most likely, which are traces the emulator does not have.
            raise NotImplementedError(
                f'the emulator can only return the {self._num_obs} obs_data data_item features '
                f'it was trained on, but {len(variables_list_of_lists)} variables were '
                f'requested. Prediction variables and other model outputs are traces, which '
                f'{TRACE_REFUSAL}')
        results = []
        for item_idx, operands in enumerate(variables_list_of_lists):
            operand_names = operands if isinstance(operands, (list, tuple)) else [operands]
            value = by_item[item_idx] if item_idx < by_item.size else np.nan
            results.append([np.array([value], dtype=float) for _ in operand_names])
        if flatten:
            return [array for item in results for array in item]
        return results

    def get_all_variable_names(self):
        """The emulator's outputs, named as the observables they are."""
        return list(self.bundle.feature_labels)

    def get_init_param_vals(self, param_names):
        """Parameter defaults, served from the snapshot taken when the emulator was trained.

        The optimiser's x0 and ``resolve_modifier_baselines`` both read these before anything
        is simulated. The emulator has no model to read them from, so training recorded them.
        """
        out = []
        for entry in param_names:
            names = entry if isinstance(entry, (list, tuple)) else [entry]
            values = []
            for name in names:
                if str(name) not in self.param_defaults:
                    raise ValueError(
                        f'the emulator has no recorded default for {name!r}. It records the '
                        f'params_for_id parameters and any modifier targets/inputs present when '
                        f'it was trained; retrain it if those have changed.')
                values.append(float(self.param_defaults[str(name)]))
            out.append(values if len(values) > 1 else values[0])
        return out

    def get_default_param_vals(self, param_names):
        """Same snapshot -- for an emulator the defaults never move, so the two coincide."""
        return self.get_init_param_vals(param_names)

    # ------------------------------------------------------------------ inert / refused

    def set_protocol_info(self, protocol_info):
        self._protocol_info = protocol_info

    def reset_states(self):
        return None

    def reset_and_clear(self, only_one_exp=-1):
        self._features = None
        return None

    def close_simulation(self):
        return None

    def run_offline_pre_and_set_default_state(self, t):
        # The emulator absorbed every pre-pass at training time; there is no state to settle.
        return None

    def get_all_results(self, flatten=False):
        raise NotImplementedError(f'get_all_results {TRACE_REFUSAL}')

    def get_all_results_dict(self):
        raise NotImplementedError(f'get_all_results_dict {TRACE_REFUSAL}')

    def modify_params_and_run_and_get_results(self, *args, **kwargs):
        raise NotImplementedError(f'modify_params_and_run_and_get_results {TRACE_REFUSAL}')


def _single_value(entry_names, value):
    """One theta value from an entry that may name several parameters sharing it."""
    pairs = pair_names_with_values(entry_names, value, context='emulator set_param_vals')
    values = [float(v) for _, v in pairs]
    if len(set(values)) > 1:
        raise ValueError(
            f'the emulator was trained on one shared value for {entry_names}, but was given '
            f'differing values {values}. Call set_theta(theta) instead.')
    return values[0]


def _jsonable_names(param_names):
    return [[str(name) for name in (entry if isinstance(entry, (list, tuple)) else [entry])]
            for entry in param_names]


def _limit_torch_threads():
    """One thread per rank.

    Under mpiexec every rank holds its own copy of the emulator, and torch's default is to
    take every core -- N ranks then oversubscribe the machine N-fold and each runs slower than
    it would alone.
    """
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                   # pragma: no cover - torch is optional
        pass
