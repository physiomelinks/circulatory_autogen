import importlib.util
import numpy as np
from scipy.integrate import solve_ivp
import copy
import sys
from enum import Enum
from .name_resolver import VariableNameResolver
DEBUG = False


class SimulationHelper:
    """SciPy-based solver for libCellML-generated Python modules.

    This implements the common `SimulationHelper` interface that every backend
    (OpenCOR, Myokit, SciPy, CasADi) shares. Obtain an instance via
    [`get_simulation_helper`][solver_wrappers.get_simulation_helper] rather than
    constructing a backend directly. The methods documented here are available on
    all backends.

    Typical usage::

        sim = get_simulation_helper_from_inp_data_dict(inp)
        sim.run()
        t = sim.get_time()
        y = sim.get_results(["component/var"], flatten=True)
    """

    def __init__(self, model_path, dt, sim_time, solver_info=None, pre_time=0.0):
        self.model_path = model_path
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.protocol_info = None
        self.solver_info = solver_info or {}
        # pull optional SciPy solve_ivp settings; default method RK45
        # Check both 'solver' (new) and 'method' (legacy) for backward compatibility
        solver_method = self.solver_info.get('method')
        self.solve_ivp_method = solver_method 
        self.solve_ivp_kwargs = {
            k: v
            for k, v in self.solver_info.items()
            if k in ["rtol", "atol", "max_step", "vectorized", "dense_output", "jac"]
        }
        self._load_model()
        self._init_state()
        self.update_times(dt, 0.0, sim_time, pre_time)
        self._has_run = False
        self._last_results_dict = None

    def get_time(self, include_pre_time=False):
        """Return the output time vector.

        Args:
            include_pre_time: If True, include the unlogged pre-time portion;
                otherwise return time relative to the end of pre-time.

        Returns:
            numpy.ndarray: The sampled time points.
        """
        if include_pre_time:
            return self.tSim
        else:
            return self.tSim - self.pre_time

    def set_protocol_info(self, protocol_info):
        """Store protocol metadata for a common helper API."""
        self.protocol_info = protocol_info

    def set_solve_ivp_method(self, method):
        self.solve_ivp_method = method 
        
    # ---- setup helpers ----
    @staticmethod
    def _is_user_defined_wrapper(module):
        """A user wrapper exposes rhs/PARAMETERS/STATES instead of the
        libCellML-generated STATE_COUNT/compute_rates surface."""
        return (callable(getattr(module, "rhs", None))
                and hasattr(module, "PARAMETERS")
                and hasattr(module, "STATES"))

    def _load_model(self):
        spec = importlib.util.spec_from_file_location("generated_model", self.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module

        if self._is_user_defined_wrapper(module):
            # model_type == 'python_user_defined': build a synthetic model from
            # the user's wrapper so the rest of this class works unchanged.
            self._load_user_defined_model(module)
            self._finalise_name_maps()
            return

        if not hasattr(module, 'STATE_COUNT') and not hasattr(module, 'VARIABLE_INFO'):
            raise ValueError(f"Module {self.model_path} Does have any states or variables. Model invalid")

        self.STATE_COUNT = module.STATE_COUNT
        self.VARIABLE_INFO = module.VARIABLE_INFO
        self.STATE_INFO = module.STATE_INFO

        self._finalise_name_maps()

    def _finalise_name_maps(self):
        """Build the resolver + convenience index maps from STATE_INFO/VARIABLE_INFO.
        Shared by the libCellML and user-defined load paths."""
        self._resolver = VariableNameResolver(self.STATE_INFO, self.VARIABLE_INFO)

        # Convenience maps (derived from resolver, kept for compatibility)
        self.state_name_to_idx = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "state"}
        self.var_name_to_idx   = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "var"}
        self.state_idx_to_name = {idx: name for name, idx in self.state_name_to_idx.items()}
        self.var_idx_to_name   = {idx: name for name, idx in self.var_name_to_idx.items()}

        # identify constants and algebraics
        self.constant_indices = [i for i, info in enumerate(self.VARIABLE_INFO) if info["type"].name in ["CONSTANT", "COMPUTED_CONSTANT"]]
        self.algebraic_indices = [i for i, info in enumerate(self.VARIABLE_INFO) if info["type"].name == "ALGEBRAIC"]

    def _load_user_defined_model(self, module):
        """Adapt a user wrapper (rhs/STATES/PARAMETERS/OUTPUT_NAMES) into the
        libCellML-style model surface (STATE_COUNT, STATE_INFO, VARIABLE_INFO,
        create_*_array, initialise_variables, compute_*) that the rest of this
        class relies on. Integration is still done by this class's run() via
        scipy solve_ivp on self.model.compute_rates."""
        parameters = dict(getattr(module, "PARAMETERS"))
        states = dict(getattr(module, "STATES"))
        output_names = list(getattr(module, "OUTPUT_NAMES", list(states.keys())))
        user_rhs = getattr(module, "rhs")
        user_compute_outputs = getattr(module, "compute_outputs", None)

        if not isinstance(parameters, dict):
            raise ValueError(f"{self.model_path}: PARAMETERS must be a dict of name->float")
        if not isinstance(states, dict):
            raise ValueError(f"{self.model_path}: STATES must be a dict of name->float")
        if not isinstance(output_names, (list, tuple)):
            raise ValueError(f"{self.model_path}: OUTPUT_NAMES must be a list of names")
        if user_compute_outputs is not None and not callable(user_compute_outputs):
            raise ValueError(f"{self.model_path}: compute_outputs must be callable if defined")

        class VariableType(Enum):
            STATE = 0
            CONSTANT = 1
            COMPUTED_CONSTANT = 2
            ALGEBRAIC = 3

        def _split(name):
            comp, _, var = name.rpartition("/")
            return comp, (var or name)

        state_names = list(states.keys())
        param_names = list(parameters.keys())
        # Outputs that are themselves states resolve as states; only the rest
        # need an algebraic variable slot.
        state_name_set = set(state_names)
        algebraic_names = [n for n in output_names if n not in state_name_set]

        self.STATE_COUNT = len(state_names)
        self.STATE_INFO = []
        for n in state_names:
            comp, var = _split(n)
            self.STATE_INFO.append({"name": var, "units": "dimensionless",
                                    "component": comp, "type": VariableType.STATE})

        # VARIABLE_INFO: parameters (constants) first, then algebraic outputs.
        self.VARIABLE_INFO = []
        for n in param_names:
            comp, var = _split(n)
            self.VARIABLE_INFO.append({"name": var, "units": "dimensionless",
                                       "component": comp, "type": VariableType.CONSTANT})
        for n in algebraic_names:
            comp, var = _split(n)
            self.VARIABLE_INFO.append({"name": var, "units": "dimensionless",
                                       "component": comp, "type": VariableType.ALGEBRAIC})

        state_idx = {n: i for i, n in enumerate(state_names)}
        param_var_idx = {n: i for i, n in enumerate(param_names)}
        algebraic_var_idx = {n: len(param_names) + i for i, n in enumerate(algebraic_names)}
        variable_count = len(param_names) + len(algebraic_names)

        state_init_vals = [float(states[n]) for n in state_names]
        param_default_vals = [float(parameters[n]) for n in param_names]

        def _params_from_variables(variables):
            return {n: variables[param_var_idx[n]] for n in param_names}

        class _UserModelAdapter:
            STATE_COUNT = self.STATE_COUNT
            VARIABLE_COUNT = variable_count

            @staticmethod
            def create_states_array():
                return np.zeros(len(state_names))

            @staticmethod
            def create_variables_array():
                return np.zeros(variable_count)

            @staticmethod
            def initialise_variables(states_arr, rates_arr, variables_arr):
                for i, v in enumerate(state_init_vals):
                    states_arr[i] = v
                for i, v in enumerate(param_default_vals):
                    variables_arr[i] = v

            @staticmethod
            def compute_computed_constants(variables_arr):
                pass

            @staticmethod
            def compute_rates(voi, states_arr, rates_arr, variables_arr):
                params = _params_from_variables(variables_arr)
                drvs = user_rhs(voi, list(states_arr), params)
                for i, d in enumerate(drvs):
                    rates_arr[i] = d

            @staticmethod
            def compute_variables(voi, states_arr, rates_arr, variables_arr):
                if user_compute_outputs is None:
                    return
                params = _params_from_variables(variables_arr)
                outs = user_compute_outputs(voi, list(states_arr), params) or {}
                for name, idx in algebraic_var_idx.items():
                    if name in outs:
                        variables_arr[idx] = outs[name]

        self.model = _UserModelAdapter()

        # The wrapper is integrated with solve_ivp; ensure a real method is set
        # even if the config left it unset or at the 'user_defined' placeholder.
        if self.solve_ivp_method in (None, 'user_defined'):
            self.solve_ivp_method = 'RK45'

    def _init_state(self):
        self.states = self.model.create_states_array()
        self.rates = self.model.create_states_array()
        self.variables = self.model.create_variables_array()
        self.model.initialise_variables(self.states, self.rates, self.variables)
        self.model.compute_computed_constants(self.variables)
        self._store_constants_defaults()

    def _store_constants_defaults(self):
        self.default_constants = [self.variables[i] for i in self.constant_indices]
        self.default_state_inits = [s for s in self.states]

    # ---- name resolution ----
    def _resolve_name(self, name: str):
        """Delegate to VariableNameResolver. Returns (kind, index) or (None, None)."""
        return self._resolver.resolve(name)

    # ---- timing helpers ----
    def update_times(self, dt, start_time, sim_time, pre_time):
        """Reconfigure the simulation timing.

        Args:
            dt: Output sampling step (s).
            start_time: Start time of the simulation (s).
            sim_time: Logged simulation duration (s).
            pre_time: Unlogged steady-state spin-up duration (s).
        """
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.start_time = start_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time/dt)
        self.n_steps = int(sim_time/dt)
        self.t_eval = np.arange(start_time, self.stop_time + dt/2, dt)
        # stored portion excludes pre_time
        self.tSim = self.t_eval[self.pre_steps:]

    # ---- parameter helpers ----
    def get_init_param_vals(self, param_names):
        """Read the initial values of the named parameters.

        Args:
            param_names: List of variable names (each entry may itself be a list
                of names sharing a value).

        Returns:
            list: Initial value(s) for each requested parameter.
        """
        vals = []
        for name_or_list in param_names:
            if not isinstance(name_or_list, list):
                name_or_list = [name_or_list]
            sub = []
            for name in name_or_list:
                kind, idx = self._resolve_name(name)
                if kind == "state":
                    sub.append(self.states[idx])
                elif kind == "var":
                    sub.append(self.variables[idx])
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
            vals.append(sub if len(sub) > 1 else sub[0])
        return vals

    def set_param_vals(self, param_names, param_vals):
        """Set the values of the named parameters.

        Args:
            param_names: List of variable names (each entry may be a list of
                names sharing a value).
            param_vals: Matching list of values to assign.
        """
        for idx, name_or_list in enumerate(param_names):
            vals = param_vals[idx]

            def _to_list(x):
                if isinstance(x, (list, tuple)):
                    return list(x)
                try:
                    import numpy as _np
                    if isinstance(x, _np.ndarray):
                        return x.tolist()
                except Exception:
                    pass
                return [x]

            name_or_list = _to_list(name_or_list)
            vals = _to_list(vals)

            for name, val in zip(name_or_list, vals):
                if type(val) == str:
                    raise NotImplementedError("Setting parameter values by name of protocol trace is not implemented for OpenCOR")
                elif type(val) not in [float, np.float64, int]:
                    raise ValueError(f"Parameter value {param_vals[JJ]} is not a valid type. {type(param_vals[JJ])}" + \
                                    "must be a float, np.float64, or int.")
                kind, idx_res = self._resolve_name(name)
                if kind == "state":
                    self.states[idx_res] = val
                elif kind == "var":
                    self.variables[idx_res] = val
                    resolved_name = self.var_idx_to_name.get(idx_res, "")
                    # If a constant follows the common "<state>_init" naming,
                    # keep the corresponding state/default-init synchronized.
                    # Use just the variable part (after '/') for the _init lookup.
                    var_part = resolved_name.split("/")[-1] if "/" in resolved_name else resolved_name
                    if var_part.endswith("_init"):
                        state_var = var_part[:-5]
                        # Try component/state_var first, then bare state_var
                        state_idx = self.state_name_to_idx.get(state_var)
                        if state_idx is None:
                            for qname, sidx in self.state_name_to_idx.items():
                                if (qname.split("/")[-1] if "/" in qname else qname) == state_var:
                                    state_idx = sidx
                                    break
                        if state_idx is not None:
                            self.states[state_idx] = val
                            self.default_state_inits[state_idx] = val
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
        self.model.compute_computed_constants(self.variables)

    # ---- simulation ----
    def _rhs(self, t, y):
        states = list(y)
        rates = [0.0]*self.STATE_COUNT
        # computed constants already in self.variables
        self.model.compute_rates(t, states, rates, self.variables)
        return rates

    def _post_process(self, sol):
        # store trajectories for requested variables
        self.state_traj = np.asarray(sol.y)
        self.time_full = sol.t
        self.var_traj = {}
        for name, idx in self.var_name_to_idx.items():
            self.var_traj[name] = []

        for ti, state_vec in zip(self.time_full, self.state_traj.T):
            states = list(state_vec)
            rates = [0.0]*self.STATE_COUNT
            vars_copy = copy.copy(self.variables)
            self.model.compute_rates(ti, states, rates, vars_copy)
            self.model.compute_variables(ti, states, rates, vars_copy)
            for name, idx in self.var_name_to_idx.items():
                val = vars_copy[idx]
                self.var_traj[name].append(val)

        # convert to numpy arrays
        for name in self.var_traj:
            self.var_traj[name] = np.asarray(self.var_traj[name])

    def run(self):
        """Run the simulation over the configured time window.

        Returns:
            bool: True on success, False if integration failed.
        """
        # integrate
        solve_kwargs = dict(
            method=self.solve_ivp_method,
            t_eval=self.t_eval,
            **self.solve_ivp_kwargs,
        )
        if DEBUG:
            sys_stdout = sys.__stdout__
            try:
                if sys_stdout:
                    sys_stdout.write(f"[PY_SOLVE] start t0={self.t_eval[0]} t1={self.t_eval[-1]} dt={self.dt}\n")
                    sys_stdout.flush()
            except Exception:
                pass

        try:
            sol = solve_ivp(
                self._rhs,
                (self.t_eval[0], self.t_eval[-1]),
                y0=self.states,
                **solve_kwargs,
            )
        except (OverflowError, FloatingPointError, ValueError, ArithmeticError):
            if DEBUG:
                try:
                    sys_stdout.write("[PY_SOLVE] fail integration raised in RHS\n")
                    sys_stdout.flush()
                except Exception:
                    pass
            return False
        if not sol.success:
            if DEBUG:
                try:
                    sys_stdout.write(f"[PY_SOLVE] fail message={sol.message}\n")
                    sys_stdout.flush()
                except Exception:
                    pass
            return False
        try:
            self._post_process(sol)
        except (OverflowError, FloatingPointError, ValueError, ArithmeticError):
            if DEBUG:
                try:
                    sys_stdout.write("[PY_SOLVE] fail post-process raised\n")
                    sys_stdout.flush()
                except Exception:
                    pass
            return False
        self._has_run = True
        # update current state to final
        self.states = list(sol.y[:, -1])
        if DEBUG:
            try:
                sys_stdout.write(f"[PY_SOLVE] end t_last={sol.t[-1]} len={len(sol.t)}\n")
                sys_stdout.flush()
            except Exception:
                pass

        return True

    # ---- results ----
    def get_all_variable_names(self):
        """Return the names of all state and algebraic/constant variables in the model."""
        return list(self.state_name_to_idx.keys()) + list(self.var_name_to_idx.keys())

    def _extract(self, name):
        if name == 'time':
            return self.tSim
        if name in self.state_name_to_idx:
            data = self.state_traj[self.state_name_to_idx[name], :]
            return data[self.pre_steps:]
        if name in self.var_name_to_idx:
            data = self.var_traj[name]
            return data[self.pre_steps:]
        # attempt to resolve common alternative namings (e.g., "a/b" vs "a_b")
        kind, idx_res = self._resolve_name(name)
        if kind == "state":
            resolved_name = self.state_idx_to_name[idx_res]
            data = self.state_traj[idx_res, :]
            return data[self.pre_steps:]
        if kind == "var":
            resolved_name = self.var_idx_to_name[idx_res]
            data = self.var_traj[resolved_name]
            return data[self.pre_steps:]
        raise ValueError(f"variable {name} not found")

    def get_results(self, variables_list_of_lists, flatten=False):
        """Return time-series results for the requested variables.

        Args:
            variables_list_of_lists: Variable names. Either a flat list of names,
                or a list of lists to group variables.
            flatten: If True, flatten the grouped result into a single list.

        Returns:
            list: One numpy array per requested variable (nested unless
            ``flatten=True``). Use ``'time'`` to request the time vector.
        """
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]
        results = []
        for variables_list in variables_list_of_lists:
            row = [self._extract(name) for name in variables_list]
            results.append(row)
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_all_results(self, flatten=False):
        """Return time-series results for every variable in the model."""
        return self.get_results(self.get_all_variable_names(), flatten=flatten)

    def get_all_results_dict(self):
        """Return all results as a dict keyed by variable name.

        Returns:
            dict: ``{variable_name: numpy.ndarray}`` for every variable.

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
        variable_names = self.get_all_variable_names()
        values = self.get_results(variable_names, flatten=True)
        return {name: np.asarray(val) for name, val in zip(variable_names, values)}

    # ---- reset helpers ----
    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        """Run unlogged warmup once; use end state as default for reset_states()."""
        offline_pre_time = float(offline_pre_time)
        if offline_pre_time <= 0:
            return
        self.update_times(self.dt, 0.0, offline_pre_time, 0.0)
        success = self.run()
        if not success:
            # Fallback to a robust explicit method for stiff startup transients.
            original_method = self.solve_ivp_method
            original_kwargs = copy.deepcopy(self.solve_ivp_kwargs)
            try:
                self.solve_ivp_method = "RK45"
                fallback_kwargs = copy.deepcopy(original_kwargs)
                fallback_kwargs["max_step"] = min(float(fallback_kwargs.get("max_step", self.dt)), self.dt)
                self.solve_ivp_kwargs = fallback_kwargs
                self._init_state()
                self.update_times(self.dt, 0.0, offline_pre_time, 0.0)
                success = self.run()
            finally:
                self.solve_ivp_method = original_method
                self.solve_ivp_kwargs = original_kwargs
        if not success:
            raise RuntimeError("Offline pre-time simulation failed")
        self.default_state_inits = copy.copy(self.states)
        self._has_run = False
        self.states = copy.copy(self.default_state_inits)
        self.model.compute_computed_constants(self.variables)

    def reset_and_clear(self, only_one_exp=-1):
        """Reset the model state to initial conditions, caching the last results."""
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
        self._init_state()
        self._has_run = False

    def reset_states(self):
        """Reset the state variables to their default initial values."""
        self.states = copy.copy(self.default_state_inits)
        self.model.compute_computed_constants(self.variables)

    def close_simulation(self):
        """Release simulation resources (no-op for the SciPy backend)."""
        # no-op for scipy solver
        pass

