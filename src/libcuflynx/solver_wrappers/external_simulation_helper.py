"""Backend for ``model_type: 'external_python'`` (solver ``'external'``).

This is the one way to bring your own Python model. The user supplies a Python file containing a
**class** that owns its own time stepping; this wrapper adapts it to the ``SimulationHelper``
surface every other backend implements, so the protocol runner, param_id, sensitivity analysis
and UQ work against it unchanged.

Owning the time loop is what makes this shape general. A model that already *has* a solver -- a
finite-element code, a compiled library behind a thin Python binding, a scheme whose time
stepping is the point -- cannot be squeezed through a per-step RHS callback; and a model that is
just an ODE loses nothing, because calling ``scipy.integrate.solve_ivp`` inside ``run()`` is four
lines (see ``funcs_user/example_model_scipy/``).

The contract the user's class must satisfy (see ``funcs_user/example_model_scipy/`` for a small
ODE and ``funcs_user/example_model_external/`` for a hand-marched PDE)::

    class MyModel:
        parameters = {"heat/k": 1.0, "heat/u_D": 0.0}     # name -> default, LITERAL values
        output_names = ["heat/T_p1", "heat/T_p2"]         # LITERAL list

        def init_solver(self, config): ...                # once; expensive setup here
        def update_times(self, dt, start_time, sim_time, pre_time): ...
        def set_param_vals(self, param_dict): ...         # {name: value}, no re-init
        def run(self): ...                                # full grid incl. pre_time; -> False if diverged
        def get_results(self): ...                        # {output_name: 1D array of length N+1}

        # optional
        def get_init_param_vals(self, names): ...
        def reset(self): ...
        def extra_plots(self): ...                        # -> [matplotlib Figure, ...]
        def close(self): ...

    SIM_HELPER = MyModel                                  # required, module level, the class

``parameters`` and ``output_names`` are class attributes with literal values on purpose: a tool
(e.g. CUFLynx) reads them straight out of the file by AST, without importing it and without
running any user code, to populate a parameter table before a simulation is ever set up.

**Who owns the clock.** The user class is told what grid to produce and is never asked to keep
track of it. This wrapper owns ``dt`` / ``pre_steps`` / ``n_steps`` / ``t_eval`` / ``tSim`` with
exactly the semantics of ``python_solver_helper.update_times``, including that ``tSim`` keeps the
pre-time offset -- ``protocol_executor`` subtracts ``pre_times[exp]`` itself, so shifting here
would subtract it twice and every logged trace would start at ``-pre_time``.
"""
import hashlib
import importlib.util
import os
import traceback

import numpy as np

from libcuflynx.solver_wrappers.param_grouping import as_name_list, pair_names_with_values

#: The module-level name the user's file must bind to their solver class. Explicit registration
#: rather than "the only class in the file": a user file is free to define helper classes, and
#: guessing which one is the model turns a typo into a confusing failure much later.
SIM_HELPER_ATTR = 'SIM_HELPER'

#: Methods the user class must provide. Checked up front, all at once, so a file missing three of
#: them says so once instead of failing three runs later.
_REQUIRED_METHODS = ('init_solver', 'update_times', 'set_param_vals', 'run', 'get_results')

#: Methods the wrapper uses when present and substitutes for when absent.
_OPTIONAL_METHODS = ('get_init_param_vals', 'reset', 'extra_plots', 'close')


def _load_module_from_path(path):
    """Import an arbitrary ``.py`` file as a module.

    The module name is derived from the absolute path (as in
    ``param_id.external_funcs._load_module_from_path``) rather than a fixed literal. A fixed name
    is a collision hazard here in a way it is not for a single generated model: two external
    models can be alive in one process (a comparison run, a test session), and under one shared
    name the second import would rebind the first's module object.
    """
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(f"external model file not found: {abspath}")
    modname = "ca_external_model_" + hashlib.md5(abspath.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(modname, abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load external model file: {abspath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_parameters(parameters, model_path):
    """``parameters`` must be a ``{'component/variable': number}`` dict."""
    if not isinstance(parameters, dict):
        raise ValueError(
            f"{model_path}: {SIM_HELPER_ATTR}.parameters must be a literal dict of "
            f"'component/variable' -> number, but is {type(parameters).__name__}. A tool reads "
            f"this attribute by AST without importing the file, so it has to be a literal.")
    validated = {}
    for name, value in parameters.items():
        if not isinstance(name, str) or '/' not in name:
            raise ValueError(
                f"{model_path}: parameter name {name!r} must be of the form "
                f"'component/variable' -- the params_for_id CSV addresses parameters as "
                f"vessel_name + param_name, and obs_data operands use the same qualified form.")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"{model_path}: default value for parameter {name!r} must be a number, but is "
                f"{value!r}. parameters maps a name to its default value.")
        validated[name] = float(value)
    return validated


def _validate_output_names(output_names, model_path):
    """``output_names`` must be a list of ``'component/variable'`` names."""
    if isinstance(output_names, str) or not isinstance(output_names, (list, tuple)):
        raise ValueError(
            f"{model_path}: {SIM_HELPER_ATTR}.output_names must be a literal list of "
            f"'component/variable' names, but is {type(output_names).__name__}.")
    validated = []
    for name in output_names:
        if not isinstance(name, str):
            raise ValueError(
                f"{model_path}: output name {name!r} must be a string of the form "
                f"'component/variable'.")
        if '/' not in name:
            raise ValueError(
                f"{model_path}: output name {name!r} lacks a '/'. Outputs are addressed as "
                f"'component/variable' by obs_data operands and model_out_names, so an "
                f"unqualified name cannot be referred to.")
        validated.append(name)
    if not validated:
        raise ValueError(
            f"{model_path}: {SIM_HELPER_ATTR}.output_names is empty, so the model produces "
            f"nothing that can be observed or calibrated against.")
    return validated


def _resolve_user_class(module, model_path):
    """Pull ``SIM_HELPER`` out of the imported module and check its declared surface."""
    user_class = getattr(module, SIM_HELPER_ATTR, None)
    if user_class is None:
        raise ValueError(
            f"module {model_path} does not define {SIM_HELPER_ATTR}. Add "
            f"`{SIM_HELPER_ATTR} = YourModelClass` at module level (the class object itself, not "
            f"an instance and not its name as a string).")
    if not isinstance(user_class, type):
        raise ValueError(
            f"{model_path}: {SIM_HELPER_ATTR} must be the solver class itself, but is "
            f"{user_class!r}. Assign the class object, not an instance of it.")

    missing = [name for name in _REQUIRED_METHODS
               if not callable(getattr(user_class, name, None))]
    if missing:
        raise ValueError(
            f"{model_path}: {SIM_HELPER_ATTR} class {user_class.__name__} is missing required "
            f"method(s) {missing}. The external-model contract is: "
            f"{list(_REQUIRED_METHODS)} required, {list(_OPTIONAL_METHODS)} optional.")

    if not hasattr(user_class, 'parameters'):
        raise ValueError(
            f"{model_path}: {SIM_HELPER_ATTR} class {user_class.__name__} does not declare a "
            f"`parameters` class attribute (a dict of 'component/variable' -> default value).")
    if not hasattr(user_class, 'output_names'):
        raise ValueError(
            f"{model_path}: {SIM_HELPER_ATTR} class {user_class.__name__} does not declare an "
            f"`output_names` class attribute (a list of 'component/variable' names).")

    parameters = _validate_parameters(user_class.parameters, model_path)
    output_names = _validate_output_names(user_class.output_names, model_path)
    return user_class, parameters, output_names


class SimulationHelper:
    """``SimulationHelper`` over a user-supplied external solver class.

    Constructed by [`get_simulation_helper`][solver_wrappers.get_simulation_helper] for
    ``model_type='external_python'`` / ``solver='external'``; the signature is the one every
    backend shares.

    Args:
        model_path: Path to the user's ``.py`` file containing ``SIM_HELPER``.
        dt: Output sampling step (s).
        sim_time: Logged simulation duration (s).
        solver_info: The CA solver_info dict. Its ``user_config`` entry is free-form and is
            handed to the user class untouched.
        pre_time: Unlogged spin-up duration (s). Samples over it are produced by the user class
            and discarded here, so the logged trace starts from a settled state.
    """

    def __init__(self, model_path, dt, sim_time, solver_info=None, pre_time=0.0):
        self.model_path = model_path
        self.solver_info = dict(solver_info or {})
        self.protocol_info = None
        # The one solver_info setting this backend declares. Free-form on purpose: what an
        # external solver needs to be told (a mesh file, a tolerance, a device) is not something
        # CA can enumerate, and inventing a schema for it would only constrain the user.
        self.user_config = (solver_info or {}).get('user_config')

        self.module = _load_module_from_path(model_path)
        self.user_class, self.parameters, self.output_names = _resolve_user_class(
            self.module, model_path)
        # What the user's parameters currently are, so get_init_param_vals can answer without
        # requiring the user class to track it (they may optionally do so; see below).
        self.default_param_vals = dict(self.parameters)
        self._current_param_vals = dict(self.parameters)

        try:
            self.user = self.user_class()
        except Exception as exc:
            raise RuntimeError(
                f"{model_path}: {SIM_HELPER_ATTR} class {self.user_class.__name__} could not be "
                f"instantiated. CA constructs it with no arguments; put configuration in "
                f"init_solver(config) instead of __init__. Original error: {exc!r}") from exc

        self._set_times(dt, 0.0, sim_time, pre_time)
        self._results = None
        self._last_results_dict = None
        self._has_run = False

        config = {
            'dt': self.dt,
            'sim_time': self.sim_time,
            'pre_time': self.pre_time,
            'start_time': self.start_time,
            'solver_info': self.solver_info,
            # Lifted out of solver_info as a convenience, so a user class does not have to
            # navigate CA's dict to reach its own options.
            'user_config': self.user_config,
        }
        self.user.init_solver(config)
        self.user.update_times(self.dt, self.start_time, self.sim_time, self.pre_time)

    # ---- timing ----
    def _set_times(self, dt, start_time, sim_time, pre_time):
        """The wrapper's own timeline bookkeeping, identical to python_solver_helper's."""
        self.dt = dt
        self.start_time = start_time
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time / dt)
        self.n_steps = int(sim_time / dt)
        # Built from a step count rather than np.arange(start, stop, dt) so the number of samples
        # is exactly what the user class is told to produce, with no floating-point edge case at
        # the final point.
        self.t_eval = start_time + np.arange(self.pre_steps + self.n_steps + 1) * dt
        # WITH the pre-time offset still in it: protocol_executor subtracts pre_times[exp] from
        # this vector itself, so removing it here would subtract it twice.
        self.tSim = self.t_eval[self.pre_steps:]

    def update_times(self, dt, start_time, sim_time, pre_time):
        """Reconfigure the simulation timing and tell the user class the new grid.

        Args:
            dt: Output sampling step (s).
            start_time: Start time of the simulation (s).
            sim_time: Logged simulation duration (s).
            pre_time: Unlogged spin-up duration (s).
        """
        self._set_times(dt, start_time, sim_time, pre_time)
        # Results on the old grid are not results on the new one.
        self._results = None
        self._has_run = False
        self.user.update_times(dt, start_time, sim_time, pre_time)

    def get_time(self, include_pre_time=False):
        """Return the output time vector.

        Args:
            include_pre_time: If True, keep the unlogged pre-time offset; otherwise return time
                relative to the end of pre-time.

        Returns:
            numpy.ndarray: The sampled time points.
        """
        if include_pre_time:
            return self.tSim
        return self.tSim - self.pre_time

    def set_protocol_info(self, protocol_info):
        """Store protocol metadata for the common helper API."""
        self.protocol_info = protocol_info

    # ---- parameters ----
    def _check_known(self, name):
        if name not in self.parameters:
            raise ValueError(
                f"{self.model_path}: parameter {name!r} is not declared by "
                f"{SIM_HELPER_ATTR}.parameters. Declared parameters are "
                f"{sorted(self.parameters)}.")

    def set_param_vals(self, param_names, param_vals, change_states=True):
        """Set the values of the named parameters on the user's solver.

        Args:
            param_names: List of parameter names; each entry may itself be a list of names
                sharing one value (a grouped params_for_id row).
            param_vals: Matching list of values.
            change_states: Accepted for API compatibility. An external solver's state is its own
                business -- the contract has no state-init parameters -- so there is nothing here
                for the flag to select.
        """
        updates = {}
        for idx, name_or_list in enumerate(param_names):
            for name, val in pair_names_with_values(name_or_list, param_vals[idx],
                                                    'set_param_vals'):
                if isinstance(val, str):
                    raise NotImplementedError(
                        f"'{name}' was given the protocol trace name '{val}', but the external "
                        f"backend cannot drive a variable from a time series. protocol_traces "
                        f"(and the protocol_shapes that expand into them) are only implemented "
                        f"for solver 'CVODE_myokit'.")
                self._check_known(name)
                updates[name] = float(val)

        if not updates:
            return
        self._current_param_vals.update(updates)
        self.user.set_param_vals(updates)
        # A parameter change invalidates the previous solve.
        self._results = None
        self._has_run = False

    def _grouped_values(self, param_names, source, user_fn=None):
        """Read values for a params_for_id-shaped name list.

        Each entry is a single name or a list of names sharing one value; the return mirrors
        that shape (a scalar for a single name, a list for a group), as every other backend's
        ``get_*_param_vals`` does.
        """
        vals = []
        for name_or_list in param_names:
            names = as_name_list(name_or_list)
            for name in names:
                self._check_known(name)
            if user_fn is not None:
                sub = list(user_fn(names))
                if len(sub) != len(names):
                    raise ValueError(
                        f"{self.model_path}: {SIM_HELPER_ATTR}.get_init_param_vals({names}) "
                        f"returned {len(sub)} value(s) for {len(names)} name(s). It must return "
                        f"one value per requested name, in order.")
            else:
                sub = [source[name] for name in names]
            vals.append(sub if len(sub) > 1 else sub[0])
        return vals

    def get_init_param_vals(self, param_names):
        """Read the current values of the named parameters.

        Delegates to the user class's optional ``get_init_param_vals(names)``; when it is not
        implemented, the wrapper answers from the values it has tracked itself (the declared
        defaults, updated by every ``set_param_vals``).

        Args:
            param_names: List of parameter names (each entry may be a list of names sharing a
                value).

        Returns:
            list: Value(s) for each requested entry.
        """
        user_fn = getattr(self.user, 'get_init_param_vals', None)
        return self._grouped_values(param_names, self._current_param_vals,
                                    user_fn=user_fn if callable(user_fn) else None)

    def get_default_param_vals(self, param_names):
        """Read the *declared* default values of the named parameters.

        Always the ``parameters`` class attribute, never the live values: a modifier parameter
        applies ``theta * baseline``, and reading the baseline from a live array that
        ``set_param_vals`` has already written would compound the factor every iteration.

        Args:
            param_names: List of parameter names (each entry may be a list of names sharing a
                value).

        Returns:
            list: Default value(s) for each requested entry.
        """
        return self._grouped_values(param_names, self.default_param_vals)

    # ---- simulation ----
    def run(self):
        """Run the user's solver over the configured grid.

        Returns:
            bool: True on success. False when the user's ``run()`` reports divergence or raises
            -- the same signal every other backend gives, which the cost function turns into
            ``inf`` rather than a crashed calibration.
        """
        try:
            success = self.user.run()
            if success is False:
                return False
            raw_results = self.user.get_results()
        except Exception:
            # A user solver blowing up on a bad parameter set is an ordinary event during
            # calibration, not a bug to abort on -- but it must be visible, so the traceback is
            # printed rather than swallowed.
            print(f"external model {self.model_path} failed during run():")
            print(traceback.format_exc())
            self._results = None
            self._has_run = False
            return False

        self._results = self._validated_results(raw_results)
        self._has_run = True
        return True

    def _validated_results(self, raw_results):
        """Check the user's results dict against the grid they were asked for.

        A contract violation raises rather than returning False: a wrong-length array is a bug in
        the user's class, not a diverged solve, and silently accepting it would put a
        misaligned trace into the cost function.
        """
        expected = self.pre_steps + self.n_steps + 1
        if not isinstance(raw_results, dict):
            raise ValueError(
                f"{self.model_path}: {SIM_HELPER_ATTR}.get_results() must return a dict of "
                f"{{output_name: 1D array}}, but returned {type(raw_results).__name__}.")

        missing = [name for name in self.output_names if name not in raw_results]
        if missing:
            raise ValueError(
                f"{self.model_path}: {SIM_HELPER_ATTR}.get_results() is missing output(s) "
                f"{missing}. It must return an entry for every name in output_names "
                f"({self.output_names}).")

        results = {}
        for name in self.output_names:
            arr = np.asarray(raw_results[name], dtype=float).reshape(-1)
            if arr.shape[0] != expected:
                raise ValueError(
                    f"{self.model_path}: {SIM_HELPER_ATTR}.get_results()['{name}'] has "
                    f"{arr.shape[0]} samples, expected {expected}. The grid last set by "
                    f"update_times is dt={self.dt}, pre_time={self.pre_time}, "
                    f"sim_time={self.sim_time}, i.e. int(pre_time/dt) + int(sim_time/dt) + 1 = "
                    f"{self.pre_steps} + {self.n_steps} + 1 = {expected} samples at "
                    f"start_time + i*dt. run() must produce the whole grid, pre_time included.")
            results[name] = arr
        return results

    # ---- results ----
    def get_all_variable_names(self):
        """Every name that can be asked for: the model outputs, its parameters, and ``'time'``."""
        return list(self.output_names) + list(self.parameters.keys()) + ['time']

    def _extract(self, name):
        if name == 'time':
            return self.tSim
        if self._results is None:
            raise RuntimeError(
                f"{self.model_path}: results were requested before a successful run(). Call "
                f"run() (and check that it returned True) before get_results().")
        if name in self._results:
            # Drop the pre-time samples: they are spin-up, and every other backend logs only
            # what follows them.
            return self._results[name][self.pre_steps:]
        if name in self._current_param_vals:
            # A parameter is constant over the run, but callers (plotting, get_all_results) ask
            # for it as a series alongside the outputs.
            return np.full(len(self.tSim), self._current_param_vals[name], dtype=float)
        raise ValueError(
            f"{self.model_path}: variable {name!r} not found. Available names are "
            f"{self.get_all_variable_names()}.")

    def get_results(self, variables_list_of_lists, flatten=False):
        """Return time-series results for the requested variables.

        Args:
            variables_list_of_lists: Variable names. Either a flat list of names, or a list of
                lists grouping the operands of one observable (which is what param_id passes,
                and what is splatted into the observable's operation func).
            flatten: If True, flatten the grouped result into a single list.

        Returns:
            list: One numpy array per requested variable (nested unless ``flatten=True``). Use
            ``'time'`` to request the time vector.
        """
        if len(variables_list_of_lists) > 0 and type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]
        results = []
        for variables_list in variables_list_of_lists:
            results.append([self._extract(name) for name in variables_list])
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_all_results(self, flatten=False):
        """Return time-series results for every available variable."""
        return self.get_results(self.get_all_variable_names(), flatten=flatten)

    def get_all_results_dict(self):
        """Return all results as a dict keyed by variable name.

        Returns:
            dict: ``{variable_name: numpy.ndarray}`` over the logged samples.

        Raises:
            RuntimeError: If the simulation has not been run yet.
        """
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
            return {name: np.asarray(val).copy() for name, val in self._last_results_dict.items()}
        if self._last_results_dict is not None:
            return {name: np.asarray(val).copy() for name, val in self._last_results_dict.items()}
        raise RuntimeError("Simulation has not been run yet.")

    def _collect_all_results_dict(self):
        names = self.get_all_variable_names()
        values = self.get_results(names, flatten=True)
        return {name: np.asarray(val) for name, val in zip(names, values)}

    def get_extra_figures(self):
        """Figures the user's class draws for itself, if it draws any.

        A field solver has views CA cannot guess at -- a mesh, a space-time map -- so the
        contract offers ``extra_plots()`` as the place to produce them. Optional: a class without
        it contributes nothing rather than failing.

        Returns:
            list: ``matplotlib.figure.Figure`` objects, empty when the hook is absent.
        """
        extra_plots = getattr(self.user, 'extra_plots', None)
        if not callable(extra_plots):
            return []
        try:
            figures = extra_plots()
        except Exception as error:                        # noqa: BLE001 - reported, not raised
            # "Optional" has to cover a hook that declines as well as one that is absent.
            # A field solver draws from state its last run built, so a run that diverged --
            # or that has not happened yet -- leaves it with nothing to draw and it raises.
            # Letting that propagate turns a legitimate "no fit at these parameters" into a
            # failed simulation, and the message the user sees names solver tolerances
            # rather than the missing run. Decorative output must not decide whether the
            # simulation succeeded.
            print(f'[external] {type(self.user).__name__}.extra_plots() did not draw: '
                  f'{type(error).__name__}: {error}')
            return []
        if figures is None:
            return []
        if not isinstance(figures, (list, tuple)):
            return [figures]
        return list(figures)

    # ---- reset / teardown ----
    def reset_states(self):
        """No-op: the contract requires ``run()`` to start from the initial condition every time,
        so there is no evolved state left over to reset."""

    def reset_and_clear(self, only_one_exp=-1):
        """Restore the declared parameter defaults and clear the cached results.

        Mirrors the SciPy backend, where this re-runs ``initialise_variables`` and so puts every
        constant back to its model default.
        """
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
        names = list(self.default_param_vals.keys())
        if names:
            self.set_param_vals(names, [self.default_param_vals[n] for n in names])
        user_reset = getattr(self.user, 'reset', None)
        if callable(user_reset):
            user_reset()
        self._results = None
        self._has_run = False

    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        """Not available for external models.

        Offline pre-time means running a warmup once and adopting the state it ends in as the
        starting state of every later run. That requires reaching into the solver's state and
        carrying it over, which the external contract deliberately does not expose -- ``run()``
        always starts from the initial condition. Use ``pre_time`` instead, which the user's
        ``run()`` performs as part of each solve.
        """
        raise NotImplementedError(
            "offline pre-time requires state carry-over between runs, which is not part of the "
            "external-model contract (run() always starts from the initial condition). Use "
            "pre_time, which is spun up inside each run and discarded from the logged output.")

    def close_simulation(self):
        """Release the user's solver resources, if it holds any."""
        user_close = getattr(self.user, 'close', None)
        if callable(user_close):
            user_close()
